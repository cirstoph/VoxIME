package org.voxime.asr;

import android.util.Log;

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

    private final long startNanos = System.nanoTime();
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
        return System.nanoTime();
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
        StringBuilder sb = new StringBuilder(512);
        sb.append("{\"v\":").append(VERSION);
        sb.append(",\"total_nanos\":").append(now() - startNanos);
        sb.append(",\"phases\":{");
        boolean first = true;
        for (Map.Entry<String, Long> e : phases.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(e.getKey()).append("\":").append(e.getValue().toString());
        }
        if (decodeRunCount > 0) {
            if (!first) sb.append(',');
            sb.append('"').append(DECODE_TOTAL).append("\":").append(decodeRunNanos);
            sb.append(",\"decode_run_per_step_nanos\":").append(decodeRunNanos / Math.max(1, decodeRunCount));
        }
        if (embedRunCount > 0) {
            if (!first) sb.append(',');
            sb.append("\"embed_run_total_nanos\":").append(embedRunNanos);
            sb.append(",\"embed_run_per_step_nanos\":").append(embedRunNanos / Math.max(1, embedRunCount));
        }
        if (tensorPrepNanos > 0) {
            if (!first) sb.append(',');
            sb.append("\"tensor_prep_nanos\":").append(tensorPrepNanos);
        }
        if (logitsEvalNanos > 0) {
            if (!first) sb.append(',');
            sb.append("\"logits_eval_nanos\":").append(logitsEvalNanos);
        }
        sb.append("},\"counts\":{\"decode_steps\":").append(lastStep + 1);
        sb.append(",\"decode_runs\":").append(decodeRunCount);
        sb.append(",\"embed_runs\":").append(embedRunCount);
        sb.append(",\"audio_tokens\":").append(numAudioTokens);
        sb.append(",\"prompt_tokens\":").append(numPromptTokens);
        sb.append(",\"output_tokens\":").append(numOutputTokens);
        sb.append("},\"stop_reason\":\"").append(stopReason).append('\"');
        if (firstTokenNanos >= 0) sb.append(",\"first_token_nanos\":").append(firstTokenNanos - startNanos);
        sb.append(",\"per_token_nanos\":").append(
                (numOutputTokens > 0 && decodeRunNanos > 0) ? decodeRunNanos / numOutputTokens : 0);
        sb.append('}');
        return sb.toString();
    }

    public void logSummary(String where) {
        String json = summaryJson();
        Log.d(TAG, where + " " + json);
        org.voxime.utils.AppLog.i(null, TAG, where + " " + json);
    }
}
