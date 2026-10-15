package org.voxime.asr;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Random;

/**
 * Regression tests for the FFT-based mel frontend against a faithful
 * re-implementation of the previous production DFT pipeline
 * (pad to 480000 -> Hann window -> direct 201x400 DFT -> Slaney mel filter
 * bank -> log10/clip/normalize over all 3000 frames).
 */
public class MelFrontendTest {

    /** Direct-DFT power spectrum, identical to the old production code. */
    private static void powerSpectrumDft(double[] frame, double[] out) {
        final int n = 400, bins = 201;
        for (int k = 0; k < bins; k++) {
            double re = 0.0, im = 0.0;
            for (int t = 0; t < n; t++) {
                double ang = -2.0 * Math.PI * ((long) k * t) / n;
                re += frame[t] * Math.cos(ang);
                im += frame[t] * Math.sin(ang);
            }
            out[k] = re * re + im * im;
        }
    }

    /** Full reference pipeline (old VoxtralEngine.melSpectrogram). */
    private static float[][] referenceMel(float[] samples) {
        final int nFft = 400, hop = 160, nMels = 128, nFreq = 201;
        int target = 480000;
        float[] audio = new float[target];
        System.arraycopy(samples, 0, audio, 0, Math.min(samples.length, target));
        int nFrames = 1 + (audio.length - nFft) / hop;
        if (nFrames > 3000) nFrames = 3000;
        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            window[i] = (float) (0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / nFft));
        }
        float[] melFb = MelFrontend.melFilterBank(nMels, nFreq);
        double[][] melOut = new double[3000][nMels];
        double[] frame = new double[nFft];
        double[] spec = new double[nFreq];
        for (int f = 0; f < nFrames; f++) {
            int off = f * hop;
            for (int i = 0; i < nFft; i++) frame[i] = audio[off + i] * window[i];
            powerSpectrumDft(frame, spec);
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

    private static void assertParity(float[] samples, double tolerance) {
        float[][] ref = referenceMel(samples);
        float[][] fft = MelFrontend.compute(samples);
        assertEquals(128, fft.length);
        assertEquals(3000, fft[0].length);
        double maxAbs = 0.0;
        double sumAbs = 0.0;
        int n = 0;
        for (int m = 0; m < 128; m++) {
            for (int f = 0; f < 3000; f++) {
                double d = Math.abs(ref[m][f] - fft[m][f]);
                maxAbs = Math.max(maxAbs, d);
                sumAbs += d;
                n++;
            }
        }
        double meanAbs = sumAbs / n;
        assertTrue("maxAbs=" + maxAbs + " meanAbs=" + meanAbs, maxAbs <= tolerance);
        assertTrue("meanAbs too large: " + meanAbs, meanAbs <= tolerance / 10.0);
    }

    @Test
    public void fftPowerMatchesDftOnRandomFrames() {
        Random r = new Random(123);
        double maxRel = 0.0;
        for (int trial = 0; trial < 10; trial++) {
            double[] re = new double[400];
            double[] im = new double[400];
            for (int i = 0; i < 400; i++) {
                re[i] = r.nextGaussian();
                im[i] = 0.0;
            }
            double[] ref = new double[201];
            powerSpectrumDft(re, ref);
            MelFrontend.powerSpectrumFft(re, im);
            for (int k = 0; k < 201; k++) {
                double abs = Math.abs(re[k] - ref[k]);
                maxRel = Math.max(maxRel, abs / Math.max(1e-9, Math.abs(ref[k])));
            }
        }
        assertTrue("maxRel=" + maxRel, maxRel < 1e-8);
    }

    @Test
    public void parityShortSpeech() {
        // 1 s of speech-like signal (voiced formants + noise), then zero padding
        float[] s = new float[16000];
        Random r = new Random(5);
        for (int i = 0; i < s.length; i++) {
            s[i] = (float) (0.6 * Math.sin(2 * Math.PI * 220 * i / 16000.0)
                    + 0.3 * Math.sin(2 * Math.PI * 880 * i / 16000.0)
                    + 0.05 * r.nextGaussian());
        }
        assertParity(s, 1e-4);
    }

    @Test
    public void parityImpulseAndSilence() {
        float[] impulse = new float[16000];
        impulse[100] = 1.0f;
        assertParity(impulse, 1e-5);
        float[] silence = new float[480000];
        assertParity(silence, 1e-5);
    }

    @Test
    public void parityFullLength() {
        // 30 s: no zero-padding branch
        Random r = new Random(11);
        float[] s = new float[480000];
        for (int i = 0; i < s.length; i++) s[i] = (float) (0.2 * Math.sin(2 * Math.PI * 440 * i / 16000.0) + 0.02 * r.nextGaussian());
        assertParity(s, 1e-4);
    }

    @Test
    public void variableFrameCountMatchesFullComputation() {
        // The variable-length path must produce the same values as the default
        // path for the frames it computes (prefix property); extra frames are
        // zero-power like in the reference.
        Random r = new Random(21);
        float[] s = new float[16000];
        for (int i = 0; i < s.length; i++) s[i] = (float) r.nextGaussian();
        float[][] full = MelFrontend.compute(s);
        for (int nf : new int[]{8, 64, 104}) {   // multiples of 8 (audio-token grid)
            float[][] var = MelFrontend.compute(s, nf);
            assertEquals(nf, var[0].length);
            for (int m = 0; m < 128; m++) {
                for (int f = 0; f < nf; f++) {
                    assertEquals("nf=" + nf + " m=" + m + " f=" + f,
                            full[m][f], var[m][f], 0.0f);
                }
            }
        }
    }

    @Test
    public void parityBoundaryLengths() {
        // lengths around the zero-frame boundary (frame = 400, hop = 160)
        assertParity(new float[400], 1e-5);
        assertParity(new float[400 + 160 * 17], 1e-5);
        Random r = new Random(77);
        float[] s = new float[9973];
        for (int i = 0; i < s.length; i++) s[i] = (float) r.nextGaussian();
        assertParity(s, 1e-4);
    }
}
