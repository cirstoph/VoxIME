package com.whisperonnx.asr;

import android.content.Context;
import android.util.Log;

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
    private final Map<String, long[]> textPromptCache = new HashMap<>();
    private final Map<String, String> langTokens = new HashMap<>(); // "en" -> " lang:en "

    private long[] promptIds;

    public interface Listener {
        void onUpdate(String message);
        void onResult(String text, String language);
    }

    public VoxtralEngine(Context context) {
        this.context = context;
    }

    public synchronized void loadModel() throws Exception {
        if (decoderSession != null) return;
        File dir = context.getExternalFilesDir(null);
        env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions enc = new OrtSession.SessionOptions();
        enc.setIntraThreadCount(2);
        encoderSession = env.createSession(new File(dir, "audio_encoder_q4f16.onnx").getAbsolutePath(), enc);

        OrtSession.SessionOptions emb = new OrtSession.SessionOptions();
        emb.setIntraThreadCount(1);
        embedSession = env.createSession(new File(dir, "embed_tokens_q4.onnx").getAbsolutePath(), emb);

        OrtSession.SessionOptions dec = new OrtSession.SessionOptions();
        dec.setIntraThreadCount(4);
        decoderSession = env.createSession(new File(dir, "decoder_model_merged_q4.onnx").getAbsolutePath(), dec);

        loadVocab();
        Log.d(TAG, "Voxtral engine loaded");
    }

    public synchronized void unloadModel() {
        try { if (encoderSession != null) encoderSession.close(); } catch (Exception ignored) {}
        try { if (embedSession != null) embedSession.close(); } catch (Exception ignored) {}
        try { if (decoderSession != null) decoderSession.close(); } catch (Exception ignored) {}
        encoderSession = null; embedSession = null; decoderSession = null;
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
        for (String piece : vocabJson.keySet()) {
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

    private static final String BYTE_ENCODER = buildByteEncoder();

    private static String buildByteEncoder() {
        List<Character> chars = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            int b = i;
            if (b >= '!' && b <= '~' || b >= 161 && b <= 172 || b >= 174 && b <= 255) {
                chars.add((char) b);
            } else {
                chars.add((char) (256 + chars.size()));
            }
        }
        StringBuilder sb = new StringBuilder();
        for (char c : chars) sb.append(c);
        return sb.toString();
    }

    private static char byteToChar(int b) {
        return BYTE_ENCODER.charAt(b & 0xFF);
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
            char c = chars.charAt(i);
            int idx = BYTE_ENCODER.indexOf(c);
            if (idx >= 0) bos.write(idx);
        }
        return new String(bos.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }

    // ---------- Mel spectrogram (exact transformers.js Whisper feature extractor) ----------

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
        for (int f = 0; f < nFrames; f++) {
            int off = f * hop;
            for (int i = 0; i < nFft; i++) frame[i] = audio[off + i] * window[i];
            // real FFT via simple DFT (power spectrum) - nFreq bins
            // use radix-2 FFT for speed
            double[] re = new double[nFft], im = new double[nFft];
            for (int i = 0; i < nFft; i++) { re[i] = frame[i]; }
            fft(re, im);
            for (int m = 0; m < nMels; m++) {
                double sum = 0.0;
                for (int k = 0; k < nFreq; k++) {
                    double p = re[k] * re[k] + im[k] * im[k];
                    sum += p * melFb[m * nFreq + k];
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

    private static void fft(double[] re, double[] im) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j |= bit;
            if (i < j) {
                double t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * Math.PI / len;
            double wr = Math.cos(ang), wi = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double cr = 1.0, ci = 0.0;
                for (int k = 0; k < len / 2; k++) {
                    double ur = re[i + k], ui = im[i + k];
                    double vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                    double vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                    re[i + k] = ur + vr; im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                    double ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
            }
        }
    }

    // ---------- Prompt building ----------
    // Precomputed with the reference tokenizer (see tools/voxtral_spike.py)
    // lang -> token ids of " lang:<iso> "
    private static final Map<String, int[]> LANG_IDS = new HashMap<>();
    static {
        LANG_IDS.put("en", new int[]{4376, 1058, 1262, 1032});
        LANG_IDS.put("fr", new int[]{4376, 1058, 7064, 1032});
        LANG_IDS.put("de", new int[]{4376, 1058, 1558, 1032});
        LANG_IDS.put("es", new int[]{4376, 1058, 1264, 1032});
        LANG_IDS.put("it", new int[]{4376, 1058, 1276, 1032});
        LANG_IDS.put("pt", new int[]{4376, 1058, 1515, 1032});
        LANG_IDS.put("nl", new int[]{4376, 24082, 1108, 1032});
        LANG_IDS.put("hi", new int[]{4376, 1058, 8101, 1032});
    }

    /**
     * Prompt: <s>[INST][BEGIN_AUDIO][AUDIO]*375 lang:<iso> [TRANSCRIBE][/INST]
     * If langCode is "auto" or unsupported, the lang part is omitted.
     */
    public long[] buildPrompt(String langCode) {
        int[] langIds = null;
        if (langCode != null) langIds = LANG_IDS.get(langCode.toLowerCase());
        int n = 1 + 1 + 1 + NUM_AUDIO_TOKENS + 1 + (langIds != null ? langIds.length + 1 : 0) + 1 + 1;
        long[] out = new long[n];
        int i = 0;
        out[i++] = BOS_TOKEN_ID;      // <s>
        out[i++] = 3;                 // [INST]
        out[i++] = BEGIN_AUDIO_TOKEN_ID;
        for (int k = 0; k < NUM_AUDIO_TOKENS; k++) out[i++] = AUDIO_TOKEN_ID;
        if (langIds != null) {
            for (int k = 0; k < langIds.length; k++) out[i++] = langIds[k];
            out[i++] = 34;            // [TRANSCRIBE]
        } else {
            out[i++] = 34;            // [TRANSCRIBE] (no lang: auto-detect)
        }
        out[i++] = 4;                 // [/INST]
        return out;
    }

    // ---------- Transcription ----------

    public String transcribe(float[] samples, String langCode, int maxNewTokens, Listener listener) throws Exception {
        if (decoderSession == null) throw new IllegalStateException("Engine not loaded");

        long t0 = System.currentTimeMillis();
        // 1. mel
        float[][] mel = melSpectrogram(samples);
        if (listener != null) listener.onUpdate("Mel done");
        long t1 = System.currentTimeMillis();

        // 2. encoder
        long[] melShape = {1, 128, 3000};
        float[] flatMel = new float[128 * 3000];
        for (int m = 0; m < 128; m++) System.arraycopy(mel[m], 0, flatMel, m * 3000, 3000);
        float[] audioFeatures;
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatMel), melShape)) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("audio_values", t);
            try (OrtSession.Result res = encoderSession.run(feeds)) {
                float[][] feats = (float[][]) res.get(0).getValue();
                audioFeatures = new float[feats.length * feats[0].length];
                for (int i = 0; i < feats.length; i++)
                    System.arraycopy(feats[i], 0, audioFeatures, i * feats[0].length, feats[0].length);
            }
        }
        if (listener != null) listener.onUpdate("Encoder done");
        long t2 = System.currentTimeMillis();

        // 3. prompt embeddings
        long[] prompt = buildPrompt(langCode);
        long[][] promptIdsArr = {prompt};
        float[][][] promptEmbeds;
        try (OnnxTensor t = OnnxTensor.createTensor(env, promptIdsArr)) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", t);
            try (OrtSession.Result res = embedSession.run(feeds)) {
                promptEmbeds = (float[][][]) res.get(0).getValue();
            }
        }
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
        long t3 = System.currentTimeMillis();

        // 4. decoder loop with KV cache
        List<Long> generated = new ArrayList<>();
        float[][] curEmbeds = embeds2d;
        int curLen = 0;
        Map<String, float[][][][]> past = emptyPast();

        for (int step = 0; step < maxNewTokens; step++) {
            int seqLen = curEmbeds.length;
            long[] attn = new long[curLen + seqLen];
            java.util.Arrays.fill(attn, 1L);
            long[] posIds = new long[seqLen];
            for (int i = 0; i < seqLen; i++) posIds[i] = curLen + i;

            Map<String, OnnxTensor> feeds = new HashMap<>();
            long[] embShape = {1, seqLen, HIDDEN};
            float[] flatEmb = new float[seqLen * HIDDEN];
            for (int i = 0; i < seqLen; i++) System.arraycopy(curEmbeds[i], 0, flatEmb, i * HIDDEN, HIDDEN);
            feeds.put("inputs_embeds", OnnxTensor.createTensor(env, FloatBuffer.wrap(flatEmb), embShape));
            feeds.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(attn), new long[]{1, attn.length}));
            feeds.put("position_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(posIds), new long[]{1, seqLen}));
            for (int l = 0; l < NUM_LAYERS; l++) {
                feeds.put("past_key_values." + l + ".key", pastTensor(past.get("key")[l]));
                feeds.put("past_key_values." + l + ".value", pastTensor(past.get("value")[l]));
            }

            long nextId;
            Map<String, float[][][][]> newPast = new HashMap<>();
            newPast.put("key", new float[NUM_LAYERS][][][]);
            newPast.put("value", new float[NUM_LAYERS][][][]);
            try (OrtSession.Result res = decoderSession.run(feeds)) {
                float[][][] logits3 = (float[][][]) res.get(0).getValue();
                float[] lastLogits = logits3[0][logits3[0].length - 1];
                nextId = argmax(lastLogits);
                for (int l = 0; l < NUM_LAYERS; l++) {
                    newPast.get("key")[l] = toJagged(res.get("present." + l + ".key").getValue(), KV_HEADS, HEAD_DIM);
                    newPast.get("value")[l] = toJagged(res.get("present." + l + ".value").getValue(), KV_HEADS, HEAD_DIM);
                }
            } finally {
                for (OnnxTensor t : feeds.values()) t.close();
            }

            if (nextId == EOS_TOKEN_ID) break;
            generated.add(nextId);
            curLen += seqLen;
            past = newPast;
            float[][] nextEmb1d;
            try (OnnxTensor t = OnnxTensor.createTensor(env, new long[][]{{nextId}})) {
                Map<String, OnnxTensor> f = new HashMap<>();
                f.put("input_ids", t);
                try (OrtSession.Result res = embedSession.run(f)) {
                    float[][][] e = (float[][][]) res.get(0).getValue();
                    nextEmb1d = e[0];
                }
            }
            curEmbeds = nextEmb1d;
            if (listener != null && step % 8 == 0)
                listener.onUpdate("Decoding " + step);
        }
        long t4 = System.currentTimeMillis();
        Log.d(TAG, "timings: mel=" + (t1 - t0) + " enc=" + (t2 - t1) + " prompt=" + (t3 - t2) + " decode=" + (t4 - t3) + " tokens=" + generated.size());

        return decode(generated);
    }

    private static long argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    private Map<String, float[][][][]> emptyPast() {
        Map<String, float[][][][]> past = new HashMap<>();
        past.put("key", new float[NUM_LAYERS][][][]);
        past.put("value", new float[NUM_LAYERS][][][]);
        return past;
    }

    private OnnxTensor pastTensor(float[][][] kv) {
        // kv: [heads][seq][dim] or empty
        int seq = kv.length == 0 ? 0 : kv[0].length;
        if (seq == 0) {
            return OnnxTensor.createTensor(env, FloatBuffer.allocate(0), new long[]{1, KV_HEADS, 0, HEAD_DIM});
        }
        float[] flat = new float[KV_HEADS * seq * HEAD_DIM];
        for (int h = 0; h < KV_HEADS; h++)
            for (int s = 0; s < seq; s++)
                System.arraycopy(kv[h][s], 0, flat, (h * seq + s) * HEAD_DIM, HEAD_DIM);
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{1, KV_HEADS, seq, HEAD_DIM});
    }

    private static float[][][] toJagged(Object value, int heads, int dim) {
        // ONNX Runtime returns float[][][][]: [batch][heads][seq][dim]
        float[][][][] v = (float[][][][]) value;
        return v[0];
    }
}
