package org.voxime.asr;


import android.content.Context;
import android.util.Log;
import org.voxime.utils.AppLog;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
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

    /**
     * Process-wide engine registry. Activity, IME and RecognitionService run in
     * the same process and each used to load its own copy of all three ONNX
     * sessions. The registry keeps exactly one engine resident; components
     * acquire it with an owner key and release the reference when destroyed.
     */
    private static final Map<String, Voxime> SHARED = new HashMap<>();
    private static final Map<String, Integer> REFCOUNTS = new HashMap<>();

    /** Returns the process-wide engine for owner, creating it if necessary. */
    public static synchronized Voxime getShared(Context context, String owner) {
        Voxime v = SHARED.get(owner);
        if (v == null) {
            v = new Voxime(context.getApplicationContext());
            SHARED.put(owner, v);
        }
        REFCOUNTS.merge(owner, 1, Integer::sum);
        return v;
    }

    /** Releases the caller's reference; the engine stays resident for the next owner. */
    public static synchronized void releaseShared(String owner) {
        Integer n = REFCOUNTS.get(owner);
        if (n == null) return;
        if (n <= 1) REFCOUNTS.remove(owner); else REFCOUNTS.put(owner, n - 1);
    }

    /**
     * Unloads the engine in every registered Voxime and clears the registry,
     * so the next getShared()/loadModel() starts from a clean state. Used when
     * a setting changes that is only read at engine load time (e.g. the
     * variable-length encoder preference) - without this the resident engine
     * would keep the old configuration until process death.
     */
    public static synchronized void unloadAllShared() {
        for (Voxime v : SHARED.values()) v.unloadModel();
        SHARED.clear();
        REFCOUNTS.clear();
    }

    /** True when the underlying engine is ready (tests/diagnostics). */
    public static synchronized boolean hasSharedEngine() {
        return !SHARED.isEmpty();
    }

    /** Number of live Voxime instances (tests/diagnostics). */
    public static synchronized int getSharedInstanceCount() {
        return SHARED.size();
    }

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
    private final AtomicBoolean mCancelRequested = new AtomicBoolean(false);
    private volatile int mActiveTaskId = 0;
    private volatile int mResultTaskId = 0;
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
        if (mState == ModelState.ERROR) return;
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
        final boolean varLenPref = mContext.getSharedPreferences(
                mContext.getPackageName() + "_preferences", android.content.Context.MODE_PRIVATE)
                .getBoolean("encoderVarLen", false);
        mLoaderThread = new Thread(() -> {
            VoxtralEngine engine = new VoxtralEngine(mContext);
            engine.setEncoderVarLenPref(varLenPref);
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
        unloadModel(false);
    }

    /**
     * Stops the worker and releases the engine. force interrupts the worker even
     * if it is inside a long-running ORT run(); the sessions are only closed after
     * the worker thread has verifiably terminated.
     */
    public void unloadModel(boolean force) {
        shutdown = true;
        mCancelRequested.set(true);
        taskLock.lock();
        try {
            taskAvailable = false;
            hasTask.signalAll();
        } finally {
            taskLock.unlock();
        }
        Thread w = mWorkerThread;
        if (w != null) {
            if (force) w.interrupt();
            try { w.join(3000); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
            if (w.isAlive()) {
                AppLog.w(mContext, TAG, "worker thread did not terminate within 3 s" + (force ? " after interrupt" : "") + "; sessions stay open");
                mStateError = "worker still running";
                return;
            }
            mWorkerThread = null;
        }
        Thread l = mLoaderThread;
        if (l != null) {
            if (force) l.interrupt();
            try { l.join(3000); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
            if (l.isAlive()) {
                AppLog.w(mContext, TAG, "loader thread did not terminate within 3 s; sessions stay open");
                return;
            }
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

    public synchronized int startTask() {
        if (!isReady()) {
            sendUpdate(mState == ModelState.LOADING ? "Engine still loading" : "Engine not ready");
            return -1;
        }
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Execution is already in progress...");
            return -1;
        }
        mActiveTaskId++;
        mCancelRequested.set(false);
        int taskId = mActiveTaskId;
        taskLock.lock();
        try {
            taskAvailable = true;
            hasTask.signal();
        } finally {
            taskLock.unlock();
        }
        return taskId;
    }

    public void start() {
        startTask();
    }

    /**
     * Requests cancellation of the active transcription. The worker checks the
     * flag between decoder steps and phases; a cancelled task delivers no
     * result. The next startTask() clears the flag.
     */
    public void stop() {
        mCancelRequested.set(true);
        mInProgress.set(false);
    }

    /** True when a cancellation was requested for the current task. */
    public boolean isCancelRequested() {
        return mCancelRequested.get();
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
                run = run && !mCancelRequested.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } finally {
                taskLock.unlock();
            }
            if (run && !shutdown) {
                processRecordBuffer(mActiveTaskId);
            }
        }
    }

    private void processRecordBuffer(int taskId) {
        StageTiming timing = new StageTiming();
        try {
            if (mCancelRequested.get()) {
                AppLog.i(mContext, TAG, "task " + taskId + " cancelled before start");
                return;
            }
            VoxtralEngine engine = mVoxtralEngine;
            if (engine == null) {
                sendUpdate("Engine not initialized or file path not set");
                return;
            }
            long pcmStart = System.nanoTime();
            float[] samples = RecordBuffer.getSamples();
            timing.addPhase(StageTiming.PCM_CONVERT, pcmStart);
            if (samples == null) {
                sendUpdate("Engine not initialized or file path not set");
                return;
            }
            AppLog.i(mContext, TAG, "transcribe start: lang=" + mLangCode + " samples=" + samples.length);
            sendUpdate(MSG_PROCESSING);
            VoxtralEngine.EngineResult res = engine.transcribe(samples,
                    "auto".equals(mLangCode) ? null : mLangCode,
                    128,
                    timing,
                    this::isCancelRequested,
                    new VoxtralEngine.Listener() {
                        @Override
                        public void onUpdate(String message) { sendUpdate(message); }
                    });
            if (mCancelRequested.get()) {
                AppLog.i(mContext, TAG, "task " + taskId + " cancelled, result discarded");
                return;
            }
            VoximeResult result = new VoximeResult(res.text, mLangCode, mAction);
            mResultTaskId = taskId;
            sendResult(result);
            if (res.truncated) sendUpdate(MSG_TRUNCATED);
            timing.logSummary("dictation:");
            sendUpdate(MSG_PROCESSING_DONE);
        } catch (VoxtralEngine.TranscriptionCancelledException e) {
            AppLog.i(mContext, TAG, "transcription cancelled by user");
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

    private void sendResult(VoximeResult result) {
        VoximeListener l = mUpdateListener;
        if (l != null) l.onResultReceived(result);
    }
}
