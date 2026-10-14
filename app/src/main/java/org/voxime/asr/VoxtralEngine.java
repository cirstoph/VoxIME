package org.voxime.asr;

import android.content.Context;
import android.util.Log;
import org.voxime.utils.AppLog;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
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
        loadProgress("Loading encoder (384 MB)...");
        long t = System.currentTimeMillis();
        encoderSession = env.createSession(new File(dir, "audio_encoder_q4f16.onnx").getAbsolutePath(), enc);
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

    static float[] melFilterBank(int numMelFilters, int numFreqBins) {
        double melMin = hzToMelSlaney(0.0);
        double melMax = hzToMelSlaney(8000.0);
        double[] melFreqs = new double[numMelFilters + 2];
        for (int i = 0; i < numMelFilters + 2; i++) {
            melFreqs[i] = melMin + (melMax - melMin) * i / (numMelFilters + 1);
        }
        double[] filterFreqs = new double[numMelFilters + 2];
        for (int i = 0; i < numMelFilters + 2; i++) {
            filterFreqs[i] = melToHzSlaney(melFreqs[i]);
        }
        double[] fftFreqs = new double[numFreqBins];
        for (int i = 0; i < numFreqBins; i++) {
            fftFreqs[i] = 8000.0 * i / (numFreqBins - 1);
        }
        float[] fb = new float[numMelFilters * numFreqBins];
        for (int m = 0; m < numMelFilters; m++) {
            double a = filterFreqs[m], b = filterFreqs[m + 1], c = filterFreqs[m + 2];
            double enorm = 2.0 / (c - a);
            for (int k = 0; k < numFreqBins; k++) {
                double up = b != a ? (fftFreqs[k] - a) / (b - a) : 0.0;
                double down = c != b ? (c - fftFreqs[k]) / (c - b) : 0.0;
                double v = Math.max(0.0, Math.min(up, down));
                fb[m * numFreqBins + k] = (float) (v * enorm);
            }
        }
        return fb;
    }

    static double hzToMelSlaney(double f) {
        if (f >= 1000.0) return 15.0 + Math.log(f / 1000.0 + 1e-10) / Math.log(6.4) * 27.0;
        return 3.0 * f / 200.0;
    }

    static double melToHzSlaney(double m) {
        if (m >= 15.0) return 1000.0 * Math.exp(Math.log(6.4) / 27.0 * (m - 15.0));
        return 200.0 / 3.0 * m;
    }

    /**
     * 16 kHz mono float samples -> (128, 3000) log-mel, padded/truncated to 30 s.
     */
    public float[][] melSpectrogram(float[] samples) {
        final int nFft = 400, hop = 160, nMels = 128, nFreq = nFft / 2 + 1;
        int target = 480000;
        float[] audio = samples;
        if (audio.length < target) {
            audio = new float[target];
            System.arraycopy(samples, 0, audio, 0, samples.length);
        } else {
            audio = new float[target];
            System.arraycopy(samples, 0, audio, 0, target);
        }
        int nFrames = 1 + (audio.length - nFft) / hop;
        if (nFrames > 3000) nFrames = 3000;

        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            window[i] = (float) (0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / nFft));
        }

        float[] melFb = melFilterBank(nMels, nFreq);
        double[][] melOut = new double[3000][nMels];

        float[] frame = new float[nFft];
        double[] spec = new double[nFreq];
        for (int f = 0; f < nFrames; f++) {
            int off = f * hop;
            for (int i = 0; i < nFft; i++) frame[i] = audio[off + i] * window[i];
            powerSpectrum(frame, spec);
            for (int m = 0; m < nMels; m++) {
                double sum = 0.0;
                for (int k = 0; k < nFreq; k++) {
                    sum += spec[k] * melFb[m * nFreq + k];
                }
                melOut[f][m] = sum;
            }
        }

        double maxLog = -Double.MAX_VALUE;
        double[][] logSpec = new double[3000][nMels];
        for (int f = 0; f < 3000; f++) {
            for (int m = 0; m < nMels; m++) {
                double v = Math.log10(Math.max(1e-10, melOut[f][m]));
                logSpec[f][m] = v;
                if (v > maxLog) maxLog = v;
            }
        }
        float[][] out = new float[nMels][3000];
        for (int f = 0; f < 3000; f++) {
            for (int m = 0; m < nMels; m++) {
                double v = Math.max(logSpec[f][m], maxLog - 8.0);
                out[m][f] = (float) ((v + 4.0) / 4.0);
            }
        }
        return out;
    }

    private static final int FFT_N = 400;
    private static final int FREQ_BINS = FFT_N / 2 + 1;
    private static final double[] DFT_COS = new double[FREQ_BINS * FFT_N];
    private static final double[] DFT_SIN = new double[FREQ_BINS * FFT_N];
    static {
        for (int k = 0; k < FREQ_BINS; k++) {
            for (int t = 0; t < FFT_N; t++) {
                double ang = -2.0 * Math.PI * k * t / FFT_N;
                DFT_COS[k * FFT_N + t] = Math.cos(ang);
                DFT_SIN[k * FFT_N + t] = Math.sin(ang);
            }
        }
    }
    private static void powerSpectrum(float[] frame, double[] out) {
        for (int k = 0; k < FREQ_BINS; k++) {
            double re = 0.0, im = 0.0;
            int base = k * FFT_N;
            for (int t = 0; t < FFT_N; t++) {
                double v = frame[t];
                re += v * DFT_COS[base + t];
                im += v * DFT_SIN[base + t];
            }
            out[k] = re * re + im * im;
        }
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
        long t0 = System.nanoTime();
        // 1. mel
        float[][] mel = melSpectrogram(padded);
        timing.addPhase(StageTiming.MEL, t0);
        if (listener != null) listener.onUpdate("Mel done");

        // 2. encoder
        long[] melShape = {1, 128, 3000};
        float[] flatMel = new float[128 * 3000];
        for (int m = 0; m < 128; m++) System.arraycopy(mel[m], 0, flatMel, m * 3000, 3000);
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
        if (listener != null) listener.onUpdate("Encoder done");

        int numAudioTokens = Math.max(1, Math.min(NUM_AUDIO_TOKENS, (padded.length + 1279) / 1280));
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
                    float[][][] logits3 = (float[][][]) res.get(0).getValue();
                    float[] lastLogits = logits3[0][logits3[0].length - 1];
                    nextId = argmax(lastLogits);
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

    private static long argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    

    

    
}
