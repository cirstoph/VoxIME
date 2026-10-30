package org.voxime.asr;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Unit tests for the shared-engine registry and cancellation semantics
 * (no Android runtime required for the pure-logic parts).
 */
public class VoximeTest {

    @Test
    public void sharedEngineIsReusedPerOwner() {
        // The registry is a pure static map; without a Context we can only
        // assert the refcount lifecycle, which is the regression risk.
        Voxime.releaseShared("unit-test-a");
        Voxime.releaseShared("unit-test-a");
        Voxime.releaseShared("unit-test-a");   // over-release must not throw
        assertEquals(0, Voxime.getSharedInstanceCount());
    }

    @Test
    public void stageTimingSummaryHasPhasesAndCounts() throws Exception {
        StageTiming t = new StageTiming();
        long start = t.startNanos();
        Thread.sleep(2);
        t.addNanos(StageTiming.MEL, 1_000_000L);
        t.addNanos(StageTiming.ENCODER_RUN, 2_000_000L);
        t.recordDecodeRun(3_000_000L);
        t.recordDecodeRun(5_000_000L);
        t.recordEmbedRun(500_000L);
        t.recordTensorPrep(100_000L);
        t.recordLogitsEval(50_000L);
        t.markFirstToken();
        t.setStep(1);
        t.setStopReason("eos");
        t.setTokenCounts(38, 44, 2);
        String json = t.summaryJson();
        StageTimingReport r = StageTimingReport.parse(json);
        assertEquals(1_000_000L, r.phaseNanos(StageTiming.MEL));
        assertEquals(2_000_000L, r.phaseNanos(StageTiming.ENCODER_RUN));
        assertEquals(2, r.count("decode_runs"));
        assertEquals(8_000_000L, r.phaseNanos(StageTiming.DECODE_TOTAL));
        assertEquals(4_000_000L, r.phaseNanos("decode_run_per_step_nanos"));
        assertEquals(38, r.count("audio_tokens"));
        assertEquals("eos", r.stopReason());
        assertTrue(r.firstTokenNanos() >= 0);
        assertTrue(r.totalNanos() >= 2_000_000L);
        assertTrue(r.hasAllPhases(StageTiming.MEL, StageTiming.ENCODER_RUN,
                StageTiming.DECODE_TOTAL, "embed_run_total_nanos", "tensor_prep_nanos", "logits_eval_nanos"));
    }
}
