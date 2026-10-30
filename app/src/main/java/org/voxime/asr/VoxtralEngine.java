package org.voxime.asr;

import android.content.Context;
import android.util.Log;
import org.voxime.utils.AppLog;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Voxtral-Mini-3B-2507 ONNX inference engine.
 *
 * Three sessions (all located in the app's external files dir):
 *   audio_encoder_q4f16.onnx    (128, 3000) mel -> (375, 3072) audio features
 *   embed_tokens_q4.onnx        token ids -> embeddings
 *   decoder_model_merged_q4.onnx  decoder with merged KV cache
 */
public class VoxtralEngine {

    private static final String TAG = "VoxtralEngine";

    public static final int AUDIO_TOKEN_ID = 24;
    public static final int BEGIN_AUDIO_TOKEN_ID = 25;
    public static final int NUM_AUDIO_TOKENS = 375;
    public static final int BOS_TOKEN_ID = 1;
    public static final int EOS_TOKEN_ID = 2;
    public static final int NUM_LAYERS = 30;
    public static final int KV_HEADS = 8;
    public static final int HEAD_DIM = 128;
    public static final int HIDDEN = 3072;

    private final Context context;
    private OrtEnvironment env;
    private OrtSession encoderSession;
    /** True when the loaded encoder graph accepts a variable time axis (experiment). */
    private volatile boolean encoderVarLen = false;
    /** Preference: use the variable-length encoder (bundled graph, default on). */
    private volatile boolean encoderVarLenPref = true;
    /** Patched encoder graph shipped as an APK asset (see tools/patch_encoder_varlen.py). */
    public static final String VAR_ENCODER_ASSET = "audio_encoder_var_q4f16.onnx";
    /** Size of the asset, used to validate the extracted copy. */
    public static final long VAR_ENCODER_ASSET_SIZE = 419933L;
    /** Extracted graph file inside the model dir; its external data points at the original .onnx_data. */
    public static final String VAR_ENCODER_FILE = "audio_encoder_var_q4f16.onnx";

    /**
     * Extracts the bundled variable-length encoder graph into the model dir
     * (ORT needs it as a real file next to the external data). The graph's
     * external weights reference the original audio_encoder_q4f16.onnx_data,
     * so no extra 384 MB are needed. Returns false if extraction fails.
     */
    private boolean ensureVarEncoderGraph(File dir) {
        File out = new File(dir, VAR_ENCODER_FILE);
        if (out.isFile() && out.length() == VAR_ENCODER_ASSET_SIZE) return true;
        File tmp = new File(dir, VAR_ENCODER_FILE + ".tmp");
        try (InputStream in = context.getAssets().open(VAR_ENCODER_ASSET);
             OutputStream os = Files.newOutputStream(tmp.toPath(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            byte[] buf = new byte[65536];
            int n;
            while ((n = in.read(buf)) != -1) os.write(buf, 0, n);
        } catch (Exception e) {
            AppLog.w(context, TAG, "bundled varlen encoder unavailable: " + e.getMessage());
            tmp.delete();
            return false;
        }
        if (tmp.length() != VAR_ENCODER_ASSET_SIZE) {
            tmp.delete();
            return false;
        }
        if (out.isFile()) out.delete();
        return tmp.renameTo(out);
    }

    public void setEncoderVarLenPref(boolean enabled) {
        this.encoderVarLenPref = enabled;
    }

    public boolean isEncoderVarLen() {
        return encoderVarLen;
    }
    private OrtSession embedSession;
    private OrtSession decoderSession;

    private final Map<String, Integer> vocab = new HashMap<>();      // token string -> id
    private final List<String> idToToken = new ArrayList<>();       // id -> token string (byte-level)

    public interface Listener {
        default void onUpdate(String message) {}
    }

    /** Reports model-loading progress to the UI (called from the loader thread). */
    public interface LoadProgressListener {
        void onProgress(String message);
    }

    private volatile LoadProgressListener loadProgressListener;
    public void setLoadProgressListener(LoadProgressListener l) { this.loadProgressListener = l; }
    private void loadProgress(String message) {
        AppLog.i(context, TAG, message);
        LoadProgressListener l = loadProgressListener;
        if (l != null) l.onProgress(message);
    }


    /** Transcription output with a flag telling whether the token limit cut it off. */
    /** Thrown when a transcription is cancelled between phases or decoder steps. */
    public static final class TranscriptionCancelledException extends Exception {
        public TranscriptionCancelledException() { super("transcription cancelled"); }
    }

    public static final class EngineResult {
        public final String text;
        public final boolean truncated;
        public EngineResult(String text, boolean truncated) { this.text = text; this.truncated = truncated; }
    }

    public VoxtralEngine(Context context) {
        this.context = context;
    }

    public synchronized void loadModel() throws Exception {
        loadModel(null);
    }

    /** Loads all sessions; per-session load durations are recorded into timing when provided. */
    public synchronized void loadModel(StageTiming timing) throws Exception {
        if (decoderSession != null) return;
        File dir = context.getExternalFilesDir(null);
        env = OrtEnvironment.getEnvironment();

        int cores = Runtime.getRuntime().availableProcessors();
        int threads = Math.max(4, cores - 1);
        AppLog.i(context, TAG, "using " + threads + " inference threads (" + cores + " cores)");
        OrtSession.SessionOptions enc = new OrtSession.SessionOptions();
        enc.setIntraOpNumThreads(threads);
        boolean useVarLen = encoderVarLenPref && ModelFiles.isComplete(dir) && ensureVarEncoderGraph(dir);
        AppLog.i(context, TAG, useVarLen
                ? "variable-length encoder (bundled graph), using " + VAR_ENCODER_FILE
                : "using fixed 30 s encoder window (variable-length "
                    + (encoderVarLenPref ? "graph unavailable" : "disabled by setting") + ")");
        loadProgress("Loading encoder (384 MB)...");
        long t = System.currentTimeMillis();
        encoderSession = env.createSession(
                (useVarLen ? new File(dir, VAR_ENCODER_FILE) : new File(dir, "audio_encoder_q4f16.onnx")).getAbsolutePath(), enc);
        this.encoderVarLen = useVarLen;
        loadProgress("Encoder loaded in " + (System.currentTimeMillis() - t) / 1000 + " s");
        if (timing != null) timing.addNanos(StageTiming.LOAD_ENCODER_SESSION, (System.currentTimeMillis() - t) * 1000000L);
        closeQuietly(enc);

        OrtSession.SessionOptions emb = new OrtSession.SessionOptions();
        emb.setIntraOpNumThreads(2);
        loadProgress("Loading token embeddings (252 MB)...");
        t = System.currentTimeMillis();
        embedSession = env.createSession(new File(dir, "embed_tokens_q4.onnx").getAbsolutePath(), emb);
        loadProgress("Embeddings loaded in " + (System.currentTimeMillis() - t) / 1000 + " s");
        if (timing != null) timing.addNanos(StageTiming.LOAD_EMBED_SESSION, (System.currentTimeMillis() - t) * 1000000L);
        closeQuietly(emb);

        OrtSession.SessionOptions dec = new OrtSession.SessionOptions();
        dec.setIntraOpNumThreads(threads);
        loadProgress("Loading decoder (2.3 GB) - this takes a while...");
        t = System.currentTimeMillis();
        decoderSession = env.createSession(new File(dir, "decoder_model_merged_q4.onnx").getAbsolutePath(), dec);
        loadProgress("Decoder loaded in " + (System.currentTimeMillis() - t) / 1000 + " s");
        if (timing != null) timing.addNanos(StageTiming.LOAD_DECODER_SESSION, (System.currentTimeMillis() - t) * 1000000L);
        closeQuietly(dec);

        loadProgress("Loading vocabulary...");
        long tv = System.currentTimeMillis();
        loadVocab();
        if (timing != null) timing.addNanos(StageTiming.LOAD_VOCAB, (System.currentTimeMillis() - tv) * 1000000L);
        AppLog.i(context, TAG, "engine loaded (3 sessions + vocab)");
        Log.d(TAG, "Voxtral engine loaded");
    }

    public synchronized void unloadModel() {
        try { if (encoderSession != null) encoderSession.close(); } catch (Exception ignored) {}
        try { if (embedSession != null) embedSession.close(); } catch (Exception ignored) {}
        try { if (decoderSession != null) decoderSession.close(); } catch (Exception ignored) {}
        encoderSession = null; embedSession = null; decoderSession = null;
    }

    private static void closeQuietly(OrtSession.SessionOptions options) {
        try { if (options != null) options.close(); } catch (Exception ignored) {}
    }

    public boolean isLoaded() {
        return decoderSession != null;
    }

    private void loadVocab() throws Exception {
        InputStream is = context.getAssets().open("voxtral/tokenizer.json");
        byte[] buf = new byte[is.available()];
        is.read(buf);
        is.close();
        JSONObject root = new JSONObject(new String(buf, "UTF-8"));
        JSONObject model = root.getJSONObject("model");
        JSONObject vocabJson = model.getJSONObject("vocab");
        int maxId = 0;
        JSONArray pieces = vocabJson.names();
        for (int i = 0; i < pieces.length(); i++) {
            String piece = pieces.getString(i);
            int id = vocabJson.getInt(piece);
            vocab.put(piece, id);
            if (id > maxId) maxId = id;
        }
        idToToken.clear();
        for (int i = 0; i <= maxId; i++) idToToken.add(null);
        for (String piece : vocab.keySet()) {
            idToToken.set(vocab.get(piece), piece);
        }
        Log.d(TAG, "Vocab loaded: " + idToToken.size());
    }

    // ---------- Byte-level BPE helpers ----------

    private static final char[] BYTE_ENCODER = buildByteEncoder();
    private static final java.util.Map<Character, Integer> BYTE_DECODER = buildByteDecoder();

    private static char[] buildByteEncoder() {
        java.util.Set<Character> printable = new java.util.HashSet<>();
        for (int b = '!'; b <= '~'; b++) printable.add((char) b);
        for (int b = 161; b <= 172; b++) printable.add((char) b);
        for (int b = 174; b <= 255; b++) printable.add((char) b);
        char[] enc = new char[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (printable.contains((char) b)) {
                enc[b] = (char) b;
            } else {
                enc[b] = (char) (256 + n);
                n++;
            }
        }
        return enc;
    }

    private static java.util.Map<Character, Integer> buildByteDecoder() {
        java.util.Map<Character, Integer> dec = new java.util.HashMap<>();
        for (int b = 0; b < 256; b++) dec.put(BYTE_ENCODER[b], b);
        return dec;
    }

    public String decode(List<Long> ids) {
        StringBuilder chars = new StringBuilder();
        for (long id : ids) {
            if (id < 0 || id >= idToToken.size()) continue;
            String tok = idToToken.get((int) id);
            if (tok == null) continue;
            chars.append(tok);
        }
        // byte chars -> bytes -> UTF-8 text
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        for (int i = 0; i < chars.length(); i++) {
            Integer b = BYTE_DECODER.get(chars.charAt(i));
            if (b != null) bos.write(b);
        }
        return new String(bos.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }

    // ---------- Mel spectrogram (exact transformers.js Voxime feature extractor) ----------
    /**
     * 16 kHz mono float samples -> (128, 3000) log-mel, padded/truncated to 30 s.
     * Implemented in MelFrontend (FFT + cached tables); output is numerically
     * identical to the previous direct-DFT implementation (see MelFrontendTest).
     */
    public float[][] melSpectrogram(float[] samples) {
        return MelFrontend.compute(samples);
    }

    // ---------- Prompt building ----------
    // Precomputed with the reference tokenizer (see tools/voxtral_spike.py)
    // lang -> token ids of " lang:<iso> "
    private static final Map<String, int[]> LANG_IDS = new HashMap<>();
    static {
        LANG_IDS.put("en", new int[]{4376, 1058, 1262});
        LANG_IDS.put("fr", new int[]{4376, 1058, 7064});
        LANG_IDS.put("de", new int[]{4376, 1058, 1558});
        LANG_IDS.put("es", new int[]{4376, 1058, 1264});
        LANG_IDS.put("it", new int[]{4376, 1058, 1276});
        LANG_IDS.put("pt", new int[]{4376, 1058, 1515});
        LANG_IDS.put("nl", new int[]{4376, 24082, 1108});
        LANG_IDS.put("hi", new int[]{4376, 1058, 8101});
    }

    /**
     * Prompt: <s>[INST][BEGIN_AUDIO][AUDIO]*375 lang:<iso> [TRANSCRIBE][/INST]
     * If langCode is "auto" or unsupported, the lang part is omitted.
     */
    public long[] buildPrompt(String langCode) {
        return buildPrompt(langCode, NUM_AUDIO_TOKENS);
    }

    /**
     * Dynamic prompt: audio token count matches the real audio length
     * (1 token = 80 ms, i.e. 1280 samples at 16 kHz). Short dictations
     * get a proportionally cheaper prefill; verified against the full
     * prompt: identical transcription, ~9x faster prefill on 3 s audio.
     */
    public long[] buildPrompt(String langCode, int numAudioTokens) {
        int n = Math.max(1, Math.min(NUM_AUDIO_TOKENS, numAudioTokens));
        List<Long> tokens = new ArrayList<>();
        tokens.add((long) BOS_TOKEN_ID);
        tokens.add(3L);
        tokens.add((long) BEGIN_AUDIO_TOKEN_ID);
        for (int k = 0; k < n; k++) tokens.add((long) AUDIO_TOKEN_ID);
        int[] langIds = null;
        if (langCode != null) langIds = LANG_IDS.get(langCode.toLowerCase());
        if (langIds != null) for (int id : langIds) tokens.add((long) id);
        tokens.add(34L);
        tokens.add(4L);
        long[] out = new long[tokens.size()];
        for (int i2 = 0; i2 < out.length; i2++) out[i2] = tokens.get(i2);
        return out;
    }

    // ---------- Transcription ----------

    public EngineResult transcribe(float[] samples, String langCode, int maxNewTokens, Listener listener) throws Exception {
        return transcribe(samples, langCode, maxNewTokens, null, listener);
    }

    /** Same as transcribe(), recording stage timings into the provided instance. */
    public EngineResult transcribe(float[] samples, String langCode, int maxNewTokens, StageTiming externalTiming, Listener listener) throws Exception {
        return transcribe(samples, langCode, maxNewTokens, externalTiming, null, listener);
    }

    /**
     * Full transcription with cancellation. isCancelled is checked between
     * pipeline phases and between decoder steps; an in-flight ORT run() cannot
     * be terminated (no RunOptions termination support in this runtime
     * version), so the abort latency is bounded by the longest single run().
     */
    public EngineResult transcribe(float[] samples, String langCode, int maxNewTokens, StageTiming externalTiming, java.util.function.BooleanSupplier isCancelled, Listener listener) throws Exception {
        if (decoderSession == null) throw new IllegalStateException("Engine not loaded");

        // Silence margin: 8 tokens (0.64 s) of real silence appended so the model
        // sees the speech end followed by silence - the training-time chunk
        // distribution. Without it, German dictations with lang=auto could run
        // into hallucination loops (device log 2026-10-09 run 1); the full-375
        // prompt masked this because it always carried ~25 s of silence tokens.
        int silenceMarginSamples = 8 * 1280;
        float[] padded = samples;
        if (samples.length + silenceMarginSamples < 480000) {
            padded = new float[samples.length + silenceMarginSamples];
            System.arraycopy(samples, 0, padded, 0, samples.length);
        }
        StageTiming timing = (externalTiming != null) ? externalTiming : new StageTiming();
        final java.util.function.BooleanSupplier cancelled =
                (isCancelled != null) ? isCancelled : () -> false;
        if (cancelled.getAsBoolean()) throw new TranscriptionCancelledException();
        long t0 = System.nanoTime();
        // 1. mel
        // Variable-length experiment: the patched encoder (audio_encoder_var_q4f16.onnx)
        // processes only the real audio instead of the full 30 s window. The number of
        // mel frames must be a multiple of 8 (one audio token covers 8 frames).
        float[][] mel;
        int melFrames;
        if (encoderVarLen) {
            melFrames = Math.min(3000, ((padded.length + 159) / 160 + 7) & ~7);
            mel = MelFrontend.compute(MelFrontend.subBuffer(padded, (melFrames - 1) * 160 + 400), melFrames);
        } else {
            mel = melSpectrogram(padded);
        }
        timing.addPhase(StageTiming.MEL, t0);
        if (listener != null) listener.onUpdate("Mel done");

        // 2. encoder
        melFrames = mel[0].length;
        long[] melShape = {1, 128, melFrames};
        float[] flatMel = new float[128 * melFrames];
        for (int m = 0; m < 128; m++) System.arraycopy(mel[m], 0, flatMel, m * melFrames, melFrames);
        float[] audioFeatures;
        long encRunStart = System.nanoTime();
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatMel), melShape)) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("audio_values", t);
            try (OrtSession.Result res = encoderSession.run(feeds)) {
                timing.addPhase(StageTiming.ENCODER_RUN, encRunStart);
                long encCopyStart = System.nanoTime();
                float[][] feats = (float[][]) res.get(0).getValue();
                audioFeatures = new float[feats.length * feats[0].length];
                for (int i = 0; i < feats.length; i++)
                    System.arraycopy(feats[i], 0, audioFeatures, i * feats[0].length, feats[0].length);
                timing.addPhase(StageTiming.ENCODER_COPY, encCopyStart);
            }
        }
        if (cancelled.getAsBoolean()) throw new TranscriptionCancelledException();
        if (listener != null) listener.onUpdate("Encoder done");

        int numAudioTokens = encoderVarLen
                ? audioFeatures.length / HIDDEN
                : Math.max(1, Math.min(NUM_AUDIO_TOKENS, (padded.length + 1279) / 1280));
        if (numAudioTokens < 1 || numAudioTokens > NUM_AUDIO_TOKENS)
            throw new IllegalStateException("unexpected audio token count: " + numAudioTokens);
        AppLog.i(context, TAG, "audio tokens for prefill: " + numAudioTokens + " (" + samples.length + " samples + silence margin)");
        timing.setTokenCounts(numAudioTokens, -1, 0);
        long[] prompt = buildPrompt(langCode, numAudioTokens);
        long promptEmbedStart = System.nanoTime();
        long[][] promptIdsArr = {prompt};
        float[][][] promptEmbeds;
        try (OnnxTensor t = OnnxTensor.createTensor(env, promptIdsArr)) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", t);
            try (OrtSession.Result res = embedSession.run(feeds)) {
                promptEmbeds = (float[][][]) res.get(0).getValue();
            }
        }
        timing.addPhase(StageTiming.PROMPT_EMBED_RUN, promptEmbedStart);
        // scatter audio features into [AUDIO] positions
        int pos = 0;
        int audioIdx = 0;
        float[][] embeds2d = promptEmbeds[0];
        for (int i = 0; i < prompt.length; i++) {
            if (prompt[i] == AUDIO_TOKEN_ID && audioIdx < audioFeatures.length / HIDDEN) {
                System.arraycopy(audioFeatures, audioIdx * HIDDEN, embeds2d[i], 0, HIDDEN);
                audioIdx++;
            }
        }
        if (audioIdx != numAudioTokens) {
            throw new IllegalStateException("audio scatter mismatch: " + audioIdx + " != " + numAudioTokens);
        }
        if (cancelled.getAsBoolean()) throw new TranscriptionCancelledException();
        timing.addPhase(StageTiming.AUDIO_SCATTER, promptEmbedStart);
        timing.setTokenCounts(numAudioTokens, prompt.length, 0);
        // 4. decoder loop with KV cache
        // Ownership model: the current OrtSession.Result owns the present.* tensors.
        // It stays OPEN until the next decoder run has succeeded; only then is the
        // previous Result closed. Tensors are therefore never used after close.
        List<Long> generated = new ArrayList<>();
        float[][] curEmbeds = embeds2d;
        int curLen = 0;
        int repeatStreak = 0;
        OrtSession.Result prevResult = null;
        try {
            for (int step = 0; step < maxNewTokens; step++) {
                if (cancelled.getAsBoolean()) throw new TranscriptionCancelledException();
                int seqLen = curEmbeds.length;
                long tensorPrepStart = System.nanoTime();
                long[] attn = new long[curLen + seqLen];
                java.util.Arrays.fill(attn, 1L);
                long[] posIds = new long[seqLen];
                for (int i = 0; i < seqLen; i++) posIds[i] = curLen + i;
                Map<String, OnnxTensor> feeds = new HashMap<>();
                long[] embShape = {1, seqLen, HIDDEN};
                float[] flatEmb = new float[seqLen * HIDDEN];
                for (int i = 0; i < seqLen; i++) System.arraycopy(curEmbeds[i], 0, flatEmb, i * HIDDEN, HIDDEN);
                List<OnnxTensor> ownedInputs = new ArrayList<>();
                ownedInputs.add(OnnxTensor.createTensor(env, FloatBuffer.wrap(flatEmb), embShape));
                ownedInputs.add(OnnxTensor.createTensor(env, LongBuffer.wrap(attn), new long[]{1, attn.length}));
                ownedInputs.add(OnnxTensor.createTensor(env, LongBuffer.wrap(posIds), new long[]{1, seqLen}));
                feeds.put("inputs_embeds", ownedInputs.get(0));
                feeds.put("attention_mask", ownedInputs.get(1));
                feeds.put("position_ids", ownedInputs.get(2));
                boolean hasPast = prevResult != null;
                if (!hasPast) {
                    long[] emptyShape = {1, KV_HEADS, 0, HEAD_DIM};
                    FloatBuffer empty = FloatBuffer.allocate(0);
                    for (int l = 0; l < NUM_LAYERS; l++) {
                        feeds.put("past_key_values." + l + ".key", OnnxTensor.createTensor(env, empty, emptyShape));
                        feeds.put("past_key_values." + l + ".value", OnnxTensor.createTensor(env, empty, emptyShape));
                    }
                } else {
                    for (int l = 0; l < NUM_LAYERS; l++) {
                        feeds.put("past_key_values." + l + ".key", (OnnxTensor) prevResult.get("present." + l + ".key").get());
                        feeds.put("past_key_values." + l + ".value", (OnnxTensor) prevResult.get("present." + l + ".value").get());
                    }
                }
                boolean isPrefill = (step == 0);
                long runStart = System.nanoTime();
                OrtSession.Result res = decoderSession.run(feeds);
                long runNanos = System.nanoTime() - runStart;
                timing.recordTensorPrep(System.nanoTime() - tensorPrepStart - runNanos);
                if (isPrefill) {
                    timing.addNanos(StageTiming.PREFILL_RUN, runNanos);
                } else {
                    timing.recordDecodeRun(runNanos);
                }
                timing.setStep(step);
                long logitsStart = System.nanoTime();
                long nextId;
                // created input tensors are no longer needed once run() returned
                if (!hasPast) {
                    for (Map.Entry<String, OnnxTensor> e : feeds.entrySet()) {
                        if (e.getKey().startsWith("past_key_values.")) e.getValue().close();
                    }
                }
                for (OnnxTensor t : ownedInputs) t.close();
                try {
                    // Read only the LAST position's logits row through the flat
                    // buffer API. The old getValue() materialized the whole
                    // [1, S, 131072] float tensor on the Java heap (0.5 MiB per
                    // position; ~190 MiB for a 383-position prefill).
                    nextId = argmaxLastRow(res);
                } catch (Exception e) {
                    res.close();
                    throw e;
                }
                timing.recordLogitsEval(System.nanoTime() - logitsStart);
                if (isPrefill) timing.addNanos(StageTiming.PREFILL_LOGITS, System.nanoTime() - logitsStart);
                if (generated.isEmpty()) timing.markFirstToken();
                // new generation succeeded: release the previous one (exactly once)
                if (prevResult != null) prevResult.close();
                prevResult = res;
                if (nextId == EOS_TOKEN_ID) { timing.setStopReason("eos"); break; }
                // repetition guard: a stuck loop (same token over and over) wastes
                // minutes of compute; treat 6 consecutive identical tokens as stop
                if (!generated.isEmpty()) {
                    int last = generated.get(generated.size() - 1).intValue();
                    if (nextId == last) {
                        repeatStreak++;
                        if (repeatStreak >= 6) {
                            AppLog.w(context, TAG, "repetition guard triggered at step " + step + " (token " + nextId + ")");
                            timing.setStopReason("repetition_guard");
                            generated.add(nextId);
                            break;
                        }
                    } else {
                        repeatStreak = 0;
                    }
                }
                generated.add(nextId);
                curLen += seqLen;
                float[][] nextEmb1d;
                long embStart = System.nanoTime();
                try (OnnxTensor t = OnnxTensor.createTensor(env, new long[][]{{nextId}})) {
                    Map<String, OnnxTensor> f = new HashMap<>();
                    f.put("input_ids", t);
                    try (OrtSession.Result res2 = embedSession.run(f)) {
                        float[][][] e = (float[][][]) res2.get(0).getValue();
                        nextEmb1d = e[0];
                    }
                }
                timing.recordEmbedRun(System.nanoTime() - embStart);
                curEmbeds = nextEmb1d;
                if (listener != null && step % 8 == 0)
                    listener.onUpdate("Decoding " + step);
            }
        } finally {
            if (prevResult != null) {
                try { prevResult.close(); } catch (Exception ignored) {}
            }
        }
        boolean truncated = !generated.isEmpty() && generated.size() >= maxNewTokens;
        if (truncated) timing.setStopReason("token_limit");
        timing.setTokenCounts(numAudioTokens, prompt.length, generated.size());
        timing.logSummary("transcribe:");
        return new EngineResult(decode(generated), truncated);
    }

    /**
     * Argmax over the last position's logits row, read from the flat tensor
     * buffer. The OrtSession.Result stays open (caller owns it); the buffer is
     * copied row-wise (OnnxTensor does not expose a zero-copy view), but only
     * for the last position instead of the whole [1, S, vocab] tensor.
     */
    private static long argmaxLastRow(OrtSession.Result res) throws OrtException {
        OnnxValue v = res.get(0);
        long[] shape = ((OnnxTensor) v).getInfo().getShape();
        long seqLen = shape[1];
        long vocab = shape[2];
        java.nio.FloatBuffer fb = ((OnnxTensor) v).getFloatBuffer();
        long offset = (seqLen - 1) * vocab;
        int best = 0;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            float val = fb.get((int) offset + i);
            if (val > bestVal) {
                bestVal = val;
                best = i;
            }
        }
        return best;
    }

    private static long argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    

    

    
}
