package org.voxime.asr;


import android.content.Context;
import android.util.Log;
import org.voxime.utils.AppLog;

import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Frontend for all input paths (app, IME, RecognitionService).
 *
 * The engine is only used when the model is complete (ModelFiles) and loaded
 * successfully; every state transition is observable via getState().
 */
public class Voxime {

    public enum ModelState { MISSING, LOADING, READY, ERROR }

    public interface VoximeListener {
        void onUpdateReceived(String message);
        void onResultReceived(VoximeResult result);
    }

    private static final String TAG = "Voxime";
    public static final String MSG_PROCESSING = "Processing...";
    public static final String MSG_PROCESSING_DONE = "Processing done...!";
    /** Signals that the dictation was cut off at the token limit (not a complete result). */
    public static final String MSG_TRUNCATED = "Result truncated: token limit reached";
    /** Emitted when model loading starts, followed by per-step MSG_LOAD_STEP_* messages. */
    public static final String MSG_LOADING = "Loading model...";
    public static final String MSG_LOAD_READY = "Model ready";

    private final AtomicBoolean mInProgress = new AtomicBoolean(false);
    private final Lock taskLock = new ReentrantLock(true);
    private final Condition hasTask = taskLock.newCondition();
    private volatile boolean taskAvailable = false;
    private volatile boolean shutdown = false;

    private org.voxime.asr.Transcribe.Action mAction;
    private String mLangCode = "";
    private VoximeListener mUpdateListener;
    private volatile VoxtralEngine mVoxtralEngine = null;
    private volatile ModelState mState = ModelState.MISSING;
    private volatile String mStateError = null;
    private Thread mWorkerThread = null;
    private Thread mLoaderThread = null;
    private final Context mContext;
    private long startTime;

    public Voxime(Context context) {
        mContext = context;
        File dir = mContext.getExternalFilesDir(null);
        if (dir != null && !dir.exists()) dir.mkdirs();
        mState = ModelState.MISSING;
    }

    /** True if all seven model files are present with the expected sizes. */
    public static boolean isModelInstalled(Context context) {
        return ModelFiles.isComplete(context.getExternalFilesDir(null));
    }

    public ModelState getState() { return mState; }
    public String getStateError() { return mStateError; }
    public boolean isReady() { return mState == ModelState.READY && mVoxtralEngine != null; }

    public void setListener(VoximeListener listener) {
        this.mUpdateListener = listener;
    }

    /**
     * Loads the Voxtral engine if and only if the model is complete.
     * Idempotent; loading happens on a background thread.
     */
    public synchronized void loadModel() {
        if (mState == ModelState.READY || mState == ModelState.LOADING) return;
        if (mVoxtralEngine != null) return;
        File dir = mContext.getExternalFilesDir(null);
        if (dir == null) {
            mState = ModelState.ERROR;
            mStateError = "External storage unavailable";
            return;
        }
        if (!ModelFiles.isComplete(dir)) {
            mState = ModelState.MISSING;
            Log.w(TAG, "Model incomplete, not loading engine");
            return;
        }
        mState = ModelState.LOADING;
        sendUpdate(MSG_LOADING);
        startWorkerIfNeeded();
        mLoaderThread = new Thread(() -> {
            VoxtralEngine engine = new VoxtralEngine(mContext);
            engine.setLoadProgressListener(message -> sendUpdate(message));
            try {
                engine.loadModel();
                if (shutdown) {
                    engine.unloadModel();
                    return;
                }
                mVoxtralEngine = engine;
                mState = ModelState.READY;
                mStateError = null;
                sendUpdate(MSG_LOAD_READY);
                AppLog.i(mContext, TAG, "state -> READY");
                Log.d(TAG, "Voxtral engine initialized");
            } catch (Exception e) {
                engine.unloadModel();
                mVoxtralEngine = null;
                mState = ModelState.ERROR;
                mStateError = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                sendUpdate("Model failed to load: " + mStateError);
                AppLog.e(mContext, TAG, "Voxtral init error: " + mStateError, e);
                Log.e(TAG, "Voxtral init error", e);
            }
        }, "voxtral-loader");
        mLoaderThread.start();
    }

    private void startWorkerIfNeeded() {
        if (mWorkerThread != null && mWorkerThread.isAlive()) return;
        taskLock.lock();
        try {
            shutdown = false;
        } finally {
            taskLock.unlock();
        }
        mWorkerThread = new Thread(this::processRecordBufferLoop, "voxime-worker");
        mWorkerThread.start();
    }

    /** Stops the worker, waits for the current transcription to finish, releases the engine. */
    public void unloadModel() {
        shutdown = true;
        taskLock.lock();
        try {
            taskAvailable = false;
            hasTask.signalAll();
        } finally {
            taskLock.unlock();
        }
        Thread w = mWorkerThread;
        if (w != null) {
            try { w.join(3000); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
            mWorkerThread = null;
        }
        Thread l = mLoaderThread;
        if (l != null) {
            try { l.join(3000); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
            mLoaderThread = null;
        }
        VoxtralEngine e = mVoxtralEngine;
        mVoxtralEngine = null;
        if (e != null) e.unloadModel();
        mState = ModelState.MISSING;
    }

    public void setAction(org.voxime.asr.Transcribe.Action action) {
        this.mAction = action;
    }

    public void setLanguage(String language) {
        this.mLangCode = language;
    }

    public void start() {
        if (!isReady()) {
            sendUpdate(mState == ModelState.LOADING ? "Engine still loading" : "Engine not ready");
            return;
        }
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Execution is already in progress...");
            return;
        }
        taskLock.lock();
        try {
            taskAvailable = true;
            hasTask.signal();
        } finally {
            taskLock.unlock();
        }
    }

    public void stop() {
        mInProgress.set(false);
    }

    public boolean isInProgress() {
        return mInProgress.get();
    }

    private void processRecordBufferLoop() {
        while (!shutdown && !Thread.currentThread().isInterrupted()) {
            boolean run = false;
            taskLock.lock();
            try {
                if (taskAvailable) {
                    run = true;
                    taskAvailable = false;
                } else {
                    hasTask.await();
                    run = taskAvailable;
                    taskAvailable = false;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } finally {
                taskLock.unlock();
            }
            if (run && !shutdown) {
                processRecordBuffer();
            }
        }
    }

    private void processRecordBuffer() {
        try {
            VoxtralEngine engine = mVoxtralEngine;
            if (engine == null) {
                sendUpdate("Engine not initialized or file path not set");
                return;
            }
            float[] samples = RecordBuffer.getSamples();
            if (samples == null) {
                sendUpdate("Engine not initialized or file path not set");
                return;
            }
            startTime = System.currentTimeMillis();
            AppLog.i(mContext, TAG, "transcribe start: lang=" + mLangCode + " samples=" + samples.length);
            sendUpdate(MSG_PROCESSING);
            VoxtralEngine.EngineResult res = engine.transcribe(samples,
                    "auto".equals(mLangCode) ? null : mLangCode,
                    128,
                    new VoxtralEngine.Listener() {
                        @Override
                        public void onUpdate(String message) { sendUpdate(message); }
                    });
            VoximeResult result = new VoximeResult(res.text, mLangCode, mAction);
            sendResult(result);
            if (res.truncated) sendUpdate(MSG_TRUNCATED);
            long timeTaken = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Time Taken for transcription: " + timeTaken + "ms");
            sendUpdate(MSG_PROCESSING_DONE);
        } catch (Exception e) {
            AppLog.e(mContext, TAG, "transcription failed", e);
            Log.e(TAG, "Error during transcription", e);
            sendUpdate("Transcription failed: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName()));
        } finally {
            mInProgress.set(false);
        }
    }

    private void sendUpdate(String message) {
        VoximeListener l = mUpdateListener;
        if (l != null) l.onUpdateReceived(message);
    }

    private void sendResult(VoximeResult whisperResult) {
        VoximeListener l = mUpdateListener;
        if (l != null) l.onResultReceived(whisperResult);
    }
}
