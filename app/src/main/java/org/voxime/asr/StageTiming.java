package org.voxime.asr;

import android.os.SystemClock;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Structured per-dictation stage timing. One instance per transcription;
 * the engine records monotonic nanosecond durations per phase, plus
 * per-decoder-step samples, and reports a single JSON summary line at
 * the end of the dictation. Cheap enough to stay enabled in release:
 * a few long-field writes, no I/O.
 */
public final class StageTiming {
    private static final String TAG = "StageTiming";
    public static final int VERSION = 1;

    /** Phase keys, in pipeline order. */
    public static final String LOAD_ENCODER_SESSION = "load_encoder_session";
    public static final String LOAD_EMBED_SESSION = "load_embed_session";
    public static final String LOAD_DECODER_SESSION = "load_decoder_session";
    public static final String LOAD_VOCAB = "load_vocab";
    public static final String STOP_TO_AUDIO = "stop_to_audio";
    public static final String PCM_CONVERT = "pcm_convert";
    public static final String MEL = "mel";
    public static final String ENCODER_RUN = "encoder_run";
    public static final String ENCODER_COPY = "encoder_copy";
    public static final String PROMPT_EMBED_RUN = "prompt_embed_run";
    public static final String AUDIO_SCATTER = "audio_scatter";
    public static final String PREFILL_RUN = "prefill_run";
    public static final String PREFILL_LOGITS = "prefill_logits";
    public static final String PREFILL_TOTAL = "prefill_total";
    public static final String DECODE_TOTAL = "decode_total";

    private final long startNanos = SystemClock.elapsedRealtimeNanos();
    private final Map<String, Long> phases = new TreeMap<>();
    private long decodeRunNanos = 0;
    private int decodeRunCount = 0;
    private long embedRunNanos = 0;
    private int embedRunCount = 0;
    private long tensorPrepNanos = 0;
    private long logitsEvalNanos = 0;
    private long firstTokenNanos = -1;
    private int lastStep = -1;
    private String stopReason = "eos";
    private int numAudioTokens = -1;
    private int numPromptTokens = -1;
    private int numOutputTokens = -1;

    private static long now() {
        return SystemClock.elapsedRealtimeNanos();
    }

    public void addPhase(String key, long fromNanos) {
        addNanos(key, now() - fromNanos);
    }

    public void addNanos(String key, long nanos) {
        phases.merge(key, nanos, Long::sum);
    }

    public void recordDecodeRun(long nanos) {
        decodeRunNanos += nanos;
        decodeRunCount++;
    }

    public void recordEmbedRun(long nanos) {
        embedRunNanos += nanos;
        embedRunCount++;
    }

    public void recordTensorPrep(long nanos) {
        tensorPrepNanos += nanos;
    }

    public void recordLogitsEval(long nanos) {
        logitsEvalNanos += nanos;
    }

    public void markFirstToken() {
        if (firstTokenNanos < 0) firstTokenNanos = now();
    }

    public void setStep(int step) {
        lastStep = step;
    }

    public void setStopReason(String reason) {
        stopReason = reason;
    }

    public void setTokenCounts(int audio, int prompt, int output) {
        numAudioTokens = audio;
        numPromptTokens = prompt;
        numOutputTokens = output;
    }

    public long elapsedSince(long fromNanos) {
        return now() - fromNanos;
    }

    public long startNanos() {
        return startNanos;
    }

    /** Single-line JSON summary for the log; one logcat + one file line per dictation. */
    public String summaryJson() {
        JSONObject root = new JSONObject();
        try {
            root.put("v", VERSION);
            long end = now();
            root.put("total_nanos", end - startNanos);
            JSONObject p = new JSONObject();
            for (Map.Entry<String, Long> e : phases.entrySet()) {
                p.put(e.getKey(), e.getValue());
            }
            if (decodeRunCount > 0) {
                p.put(DECODE_TOTAL, decodeRunNanos);
                p.put("decode_run_per_step_nanos", decodeRunNanos / Math.max(1, decodeRunCount));
            }
            if (embedRunCount > 0) {
                p.put("embed_run_total_nanos", embedRunNanos);
                p.put("embed_run_per_step_nanos", embedRunNanos / Math.max(1, embedRunCount));
            }
            if (tensorPrepNanos > 0) p.put("tensor_prep_nanos", tensorPrepNanos);
            if (logitsEvalNanos > 0) p.put("logits_eval_nanos", logitsEvalNanos);
            root.put("phases", p);
            JSONObject counts = new JSONObject();
            counts.put("decode_steps", lastStep + 1);
            counts.put("decode_runs", decodeRunCount);
            counts.put("embed_runs", embedRunCount);
            counts.put("audio_tokens", numAudioTokens);
            counts.put("prompt_tokens", numPromptTokens);
            counts.put("output_tokens", numOutputTokens);
            root.put("counts", counts);
            root.put("stop_reason", stopReason);
            if (firstTokenNanos >= 0) root.put("first_token_nanos", firstTokenNanos - startNanos);
            root.put("per_token_nanos",
                    (numOutputTokens > 0 && decodeRunNanos > 0)
                            ? decodeRunNanos / numOutputTokens : 0);
        } catch (Exception ignored) {
        }
        return root.toString();
    }

    public void logSummary(String where) {
        String json = summaryJson();
        Log.d(TAG, where + " " + json);
        org.voxime.utils.AppLog.i(null, TAG, where + " " + json);
    }
}
