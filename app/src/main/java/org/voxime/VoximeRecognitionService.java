package org.voxime;

import static android.speech.SpeechRecognizer.ERROR_CLIENT;
import static android.speech.SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS;


import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.os.RemoteException;
import android.speech.RecognitionService;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;
import android.widget.Toast;

import androidx.core.content.ContextCompat;
import androidx.preference.PreferenceManager;

import com.github.houbb.opencc4j.util.ZhConverterUtil;
import org.voxime.utils.AppLog;
import org.voxime.asr.Recorder;
import org.voxime.asr.Voxime;
import org.voxime.asr.VoximeResult;
import org.voxime.utils.HapticFeedback;
import java.util.ArrayList;
import static org.voxime.asr.Transcribe.ACTION_TRANSCRIBE;

public class VoximeRecognitionService extends RecognitionService {
    private static final String TAG = "VoximeRecognitionService";
    private Recorder mRecorder = null;
    private Voxime mVoxime = null;
    private boolean recognitionCancelled = false;
    private SharedPreferences sp = null;

    @Override
    protected void onStartListening(Intent recognizerIntent, Callback callback) {
        String targetLang = recognizerIntent.getStringExtra(RecognizerIntent.EXTRA_LANGUAGE);
        ParcelFileDescriptor audioExtra = recognizerIntent.getParcelableExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE);
        if (audioExtra != null) {
            Log.w(TAG, "EXTRA_AUDIO_SOURCE not supported");
            try {
                callback.error(SpeechRecognizer.ERROR_CLIENT); // or define a custom error code
                new Handler(Looper.getMainLooper()).post(() ->
                        Toast.makeText(this, "EXTRA_AUDIO_SOURCE not supported", Toast.LENGTH_SHORT).show()
                );
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
            return; // Stop further processing
        }

        sp = PreferenceManager.getDefaultSharedPreferences(this);
        String langCode = sp.getString("recognitionServiceLanguage", "auto");
        Log.d(TAG,"default langCode " + langCode);

        if (targetLang != null) {
            Log.d(TAG,"StartListening in " + targetLang);
            langCode = targetLang.split("[-_]")[0].toLowerCase();   //support both de_DE and de-DE
        } else {
            Log.d(TAG,"StartListening, no language specified");
        }

        checkRecordPermission(callback);

        initModel(callback, langCode);

        mRecorder = new Recorder(this);
        mRecorder.setListener(message -> {
            if (message.equals(Recorder.MSG_RECORDING)){
                try {
                    callback.beginningOfSpeech();
                    callback.rmsChanged(10);
                } catch (RemoteException e) {
                    throw new RuntimeException(e);
                }
            } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                HapticFeedback.vibrate(this);
                try {
                    callback.rmsChanged(-20.0f);
                } catch (RemoteException e) {
                    throw new RuntimeException(e);
                }
                startTranscription();
            } else if (message.equals(Recorder.MSG_RECORDING_ERROR)) {
                try {
                    callback.error(ERROR_CLIENT);
                } catch (RemoteException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        if (!mVoxime.isInProgress()) {
            HapticFeedback.vibrate(this);
            startRecording();
            try {
                callback.readyForSpeech(new Bundle());
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
        }


    }

    private void stopRecording() {
        if (mRecorder != null && mRecorder.isInProgress()) {
            mRecorder.stop();
        }
    }

    @Override
    protected void onCancel(Callback callback) {
        Log.d(TAG,"cancel");
        stopRecording();
        recognitionCancelled = true;
    }

    @Override
    protected void onStopListening(Callback callback) {
        Log.d(TAG,"StopListening");
        stopRecording();
    }

    // Model initialization
    private void initModel(Callback callback, String langCode) {

        mVoxime = new Voxime(this);
        mVoxime.loadModel();
        mVoxime.setLanguage(langCode);
        Log.d(TAG, "Language token " + langCode);
        mVoxime.setListener(new Voxime.VoximeListener() {
            private boolean finished = false;
            @Override
            public void onUpdateReceived(String message) {
                if (message == null) return;
                if (message.startsWith("Transcription failed")) {
                    finishWithError(callback, message);
                } else if (message.equals("Engine not initialized or file path not set")
                        || message.equals("Engine still loading") || message.equals("Engine not ready")) {
                    finishWithError(callback, "VoxIME engine not ready");
                }
            }
            @Override
            public void onResultReceived(VoximeResult whisperResult) {
                if (finished) return;
                finished = true;
                try {
                    callback.endOfSpeech();
                    Bundle results = new Bundle();
                    ArrayList<String> resultList = new ArrayList<>();
                    String result = whisperResult.getResult();
                    if (whisperResult.getLanguage().equals("zh")){
                        boolean simpleChinese = sp.getBoolean("RecognitionServiceSimpleChinese",false);
                        result = simpleChinese ? ZhConverterUtil.toSimple(result) : ZhConverterUtil.toTraditional(result);
                    }
                    resultList.add(result.trim());
                    results.putStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION, resultList);
                    callback.results(results);
                } catch (RemoteException e) {
                    Log.e(TAG, "results() failed", e);
                }
                // keep the engine loaded for the next request (loading takes ~10-100s)
            }
        });
    }

    private void finishWithError(Callback callback, String message) {
        AppLog.e(this, TAG, "recognition error: " + message, null);
        Log.e(TAG, "RecognitionService error: " + message);
        try {
            callback.error(SpeechRecognizer.ERROR_RECOGNIZER_BUSY);
        } catch (RemoteException e) {
            Log.e(TAG, "error() failed", e);
        }
    }

    private void startRecording() {
        mRecorder.initVad();
        mRecorder.start();
        recognitionCancelled = false;
    }

    private void startTranscription() {
        if (!recognitionCancelled){
            Handler handler = new Handler(Looper.getMainLooper());
            handler.post(()-> {
                Toast toast = new Toast(this);
                toast.setDuration(Toast.LENGTH_SHORT);
                toast.setText(R.string.processing);
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    toast.addCallback(new Toast.Callback() {
                        @Override
                        public void onToastHidden() {
                            super.onToastHidden();
                            if (mVoxime!=null) toast.show();
                        }
                    });
                }
                toast.show();
            });
            mVoxime.setAction(ACTION_TRANSCRIBE);
            mVoxime.start();
            Log.d(TAG,"Start Transcription");
        }
    }

    @Override
    public void onDestroy (){
        deinitModel();
    }
    private void deinitModel() {
        if (mVoxime != null) {
            mVoxime.unloadModel();
            mVoxime = null;
        }
    }

    private void checkRecordPermission(Callback callback) {
        int permission = ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO);
        if (permission != PackageManager.PERMISSION_GRANTED){
            Log.d(TAG,getString(R.string.need_record_audio_permission));
            try {
                callback.error(ERROR_INSUFFICIENT_PERMISSIONS);
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
        }
    }

}
