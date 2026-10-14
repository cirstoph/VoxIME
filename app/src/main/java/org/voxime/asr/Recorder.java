package org.voxime.asr;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;
import androidx.core.app.ActivityCompat;
import androidx.preference.PreferenceManager;
import com.konovalov.vad.webrtc.Vad;
import com.konovalov.vad.webrtc.VadWebRTC;
import com.konovalov.vad.webrtc.config.FrameSize;
import com.konovalov.vad.webrtc.config.Mode;
import com.konovalov.vad.webrtc.config.SampleRate;
import org.voxime.R;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Microphone recorder with VAD-based end-of-speech detection.
 *
 * Lifecycle: one worker thread handles recording requests sequentially. stop()
 * only requests the end of the capture and never blocks - listeners are
 * notified from the worker thread when the final buffer is ready. stopAndWait()
 * waits on a state predicate with a bounded timeout for callers that must
 * synchronize. shutdown() ends the worker thread. Completion is signalled in
 * finally, so a denied permission or an AudioRecord error can never leave a
 * caller waiting forever.
 *
 * The capture loop reads fixed 480-sample frames and feeds VAD directly from
 * the freshly read bytes - the previous implementation copied the whole
 * capture buffer into a new byte array on every step (O(n^2) over the
 * recording, ~480 MB of copies for a 30 s dictation).
 */
public class Recorder {
    public interface RecorderListener {
        void onUpdateReceived(String message);
    }

    private static final String TAG = "Recorder";
    public static final String ACTION_STOP = "Stop";
    public static final String ACTION_RECORD = "Record";
    public static final String MSG_RECORDING = "Recording...";
    public static final String MSG_RECORDING_DONE = "Recording done...!";
    public static final String MSG_RECORDING_ERROR = "Recording error...";

    private enum State { IDLE, RECORDING }

    private static final int VAD_FRAME_SIZE = 480;
    private static final int CAPTURE_TIMEOUT_MS = 5000;

    private final Context mContext;
    private final AtomicBoolean mInProgress = new AtomicBoolean(false);
    private volatile State mState = State.IDLE;
    private RecorderListener mListener;
    private final Lock lock = new ReentrantLock();
    private final Condition hasTask = lock.newCondition();
    private final Condition recordingDone = lock.newCondition();
    private volatile boolean shouldStartRecording = false;
    private volatile boolean shutdown = false;
    private volatile boolean requestStopVAD = false;
    private boolean useVAD = false;
    private VadWebRTC vad = null;
    private SharedPreferences sp;
    private final Thread workerThread;

    public Recorder(Context context) {
        this.mContext = context;
        sp = PreferenceManager.getDefaultSharedPreferences(mContext);
        workerThread = new Thread(this::recordLoop, "voxime-recorder");
        workerThread.start();
    }

    public void setListener(RecorderListener listener) {
        this.mListener = listener;
    }

    /** Starts a new capture; ignored while one is already running. */
    public void start() {
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Recording is already in progress...");
            return;
        }
        lock.lock();
        try {
            Log.d(TAG, "Recording starts now");
            shouldStartRecording = true;
            hasTask.signal();
        } finally {
            lock.unlock();
        }
    }

    public void initVad() {
        int silenceDurationMs = sp.getInt("silenceDurationMs", 800);
        if (vad != null) {
            // reuse across dictations if compatible; rebuild otherwise
            if (vad.getSilenceDurationMs() == silenceDurationMs) {
                requestStopVAD = false;
                useVAD = true;
                Log.d(TAG, "VAD reused");
                return;
            }
            vad.close();
            vad = null;
        }
        vad = Vad.builder()
                .setSampleRate(SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(FrameSize.FRAME_SIZE_480)
                .setMode(Mode.VERY_AGGRESSIVE)
                .setSilenceDurationMs(silenceDurationMs)
                .setSpeechDurationMs(200)
                .build();
        useVAD = true;
        requestStopVAD = false;
        Log.d(TAG, "VAD initialized");
    }

    public void requestStopVad() {
        requestStopVAD = true;
    }

    /**
     * Requests the end of the running capture. Non-blocking; the worker thread
     * finishes the capture and notifies the listener. Callers on the main
     * thread receive MSG_RECORDING_DONE/MSG_RECORDING_ERROR asynchronously.
     */
    public void stop() {
        Log.d(TAG, "Recording stop requested");
        mInProgress.set(false);
    }

    /**
     * Requests the end of the capture and waits (bounded) until the worker has
     * finished it. Returns true when the capture completed, false on timeout.
     * Waits on the state predicate, so an early or spurious signal cannot
     * make this wait forever and a late signal cannot be missed.
     */
    public boolean stopAndWait(long timeoutMillis) {
        lock.lock();
        try {
            mInProgress.set(false);
            long deadline = System.nanoTime() + timeoutMillis * 1_000_000L;
            while (mState != State.IDLE) {
                long remaining = deadline - System.nanoTime();
                if (remaining <= 0 || !recordingDone.await(remaining, TimeUnit.NANOSECONDS)) break;
            }
            return mState == State.IDLE;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return mState == State.IDLE;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Stops the worker thread. The current capture (if any) is aborted; the
     * worker ends at the latest after one frame read. Never blocks.
     */
    public void shutdown() {
        shutdown = true;
        mInProgress.set(false);
        lock.lock();
        try {
            shouldStartRecording = false;
            hasTask.signalAll();
        } finally {
            lock.unlock();
        }
        workerThread.interrupt();
    }

    public boolean isInProgress() {
        return mInProgress.get();
    }

    private void sendUpdate(String message) {
        if (mListener != null)
            mListener.onUpdateReceived(message);
    }

    private void recordLoop() {
        while (!shutdown && !Thread.currentThread().isInterrupted()) {
            lock.lock();
            try {
                while (!shouldStartRecording && !shutdown) {
                    hasTask.await();
                }
                shouldStartRecording = false;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } finally {
                lock.unlock();
            }
            if (shutdown) return;
            try {
                recordAudio();
            } catch (Exception e) {
                Log.e(TAG, "Recording error...", e);
                sendUpdate(e.getMessage() != null ? e.getMessage() : MSG_RECORDING_ERROR);
            } finally {
                finishCapture();
            }
        }
    }

    private void finishCapture() {
        lock.lock();
        try {
            mState = State.IDLE;
            recordingDone.signalAll();
        } finally {
            lock.unlock();
        }
        mInProgress.set(false);
    }

    private void recordAudio() {
        mState = State.RECORDING;
        if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "AudioRecord permission is not granted");
            sendUpdate(mContext.getString(R.string.need_record_audio_permission));
            return;
        }
        int channels = 1;
        int bytesPerSample = 2;
        int sampleRateInHz = 16000;
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
        int audioSource = MediaRecorder.AudioSource.VOICE_RECOGNITION;
        int bufferSize = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat);
        if (bufferSize < VAD_FRAME_SIZE * 2) bufferSize = VAD_FRAME_SIZE * 2;
        AudioManager audioManager = (AudioManager) mContext.getSystemService(Context.AUDIO_SERVICE);
        if (sp.getBoolean("bluetooth", false)) {
            audioManager.startBluetoothSco();
            audioManager.setBluetoothScoOn(true);
        }
        AudioRecord audioRecord = null;
        try {
            audioRecord = new AudioRecord.Builder()
                    .setAudioSource(audioSource)
                    .setAudioFormat(new AudioFormat.Builder()
                            .setChannelMask(channelConfig)
                            .setEncoding(audioFormat)
                            .setSampleRate(sampleRateInHz)
                            .build())
                    .setBufferSizeInBytes(bufferSize)
                    .build();
            audioRecord.startRecording();
            int bytesForThirtySeconds = sampleRateInHz * bytesPerSample * channels * 30;
            // One preallocated buffer for the whole capture (30 s, ~960 KB).
            byte[] outputBuffer = new byte[bytesForThirtySeconds];
            byte[] audioData = new byte[VAD_FRAME_SIZE * 2];
            int totalBytesRead = 0;
            boolean isSpeech;
            boolean isRecording = false;
            while (mInProgress.get() && totalBytesRead < bytesForThirtySeconds && !shutdown) {
                int bytesRead = audioRecord.read(audioData, 0, VAD_FRAME_SIZE * 2);
                if (bytesRead > 0) {
                    if (totalBytesRead + bytesRead <= outputBuffer.length) {
                        System.arraycopy(audioData, 0, outputBuffer, totalBytesRead, bytesRead);
                    }
                    totalBytesRead += bytesRead;
                } else {
                    Log.d(TAG, "AudioRecord error, bytes read: " + bytesRead);
                    break;
                }
                if (useVAD && vad != null) {
                    // VAD reads the freshly captured frame directly; no full-buffer copy.
                    isSpeech = vad.isSpeech(audioData);
                    if (isSpeech && !requestStopVAD) {
                        if (!isRecording) {
                            Log.d(TAG, "VAD Speech detected: recording starts");
                            sendUpdate(MSG_RECORDING);
                        }
                        isRecording = true;
                    } else {
                        if (isRecording || requestStopVAD) {
                            isRecording = false;
                            mInProgress.set(false);
                        }
                    }
                } else {
                    if (!isRecording) sendUpdate(MSG_RECORDING);
                    isRecording = true;
                }
            }
            Log.d(TAG, "Total bytes recorded: " + totalBytesRead);
            // Hand the recorded data (bounded to 30 s) to the transcription buffer.
            RecordBuffer.setOutputBuffer(
                    Arrays.copyOf(outputBuffer, Math.min(totalBytesRead, bytesForThirtySeconds))
            );
            if (totalBytesRead > 6400) {  //min 0.2s
                sendUpdate(MSG_RECORDING_DONE);
            } else {
                sendUpdate(MSG_RECORDING_ERROR);
            }
        } finally {
            if (useVAD) {
                useVAD = false;
                if (vad != null) {
                    vad.close();
                    vad = null;
                    Log.d(TAG, "Closing VAD");
                }
            }
            if (audioRecord != null) {
                try {
                    audioRecord.stop();
                } catch (Exception ignored) {
                }
                audioRecord.release();
            }
            try {
                audioManager.stopBluetoothSco();
                audioManager.setBluetoothScoOn(false);
            } catch (Exception ignored) {
            }
        }
    }
}
