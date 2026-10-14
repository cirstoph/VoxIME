package org.voxime.asr;

/**
 * Log-mel frontend for the Voxtral audio encoder.
 *
 * Produces the same (128, 3000) log-mel tensor as the previous direct-DFT
 * implementation, but with a real FFT and cached tables:
 *
 * - Hann window and Slaney mel filter bank are computed once (static).
 * - The 400-point real DFT (201 bins x 400 samples, ~241M multiply-accumulates
 *   per dictation) is replaced by a mixed-radix FFT (Cooley-Tukey 25 x 16 with
 *   precomputed twiddles): numerically verified against the DFT with a maximum
 *   relative power error of ~1e-11 over random frames, impulse and sine
 *   inputs (see MelFrontendTest).
 * - Frames whose entire window lies beyond the real audio (zero padding) have
 *   exactly zero power and are skipped analytically; frames overlapping real
 *   audio are still computed. Skipped frames keep mel = 0.0 and flow through
 *   the same log10/clip/normalize path as before.
 *
 * Semantics preserved: pad/truncate to 480000 samples, 3000 frames, global
 * log-maximum clipping at maxLog - 8, output (v + 4) / 4, shape [128][3000].
 */
final class MelFrontend {

    static final int FFT_N = 400;
    static final int FREQ_BINS = FFT_N / 2 + 1;
    static final int NUM_MELS = 128;
    static final int NUM_FRAMES = 3000;
    static final int TARGET_SAMPLES = 480000;
    private static final int HOP = 160;
    private static final int N1 = 16;
    private static final int N2 = 25;

    private static final float[] HANN_WINDOW = new float[FFT_N];
    private static final float[] MEL_FILTER_BANK = melFilterBank(NUM_MELS, FREQ_BINS);
    private static final double[] COS16 = buildCos(N1);
    private static final double[] SIN16 = buildSin(N1);
    private static final double[] COS25 = buildCos(N2);
    private static final double[] SIN25 = buildSin(N2);
    private static final double[] COS_TW = new double[N2 * N1];
    private static final double[] SIN_TW = new double[N2 * N1];

    static {
        for (int i = 0; i < FFT_N; i++) {
            HANN_WINDOW[i] = (float) (0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / FFT_N));
        }
        // cross-pass twiddles exp(-2 pi i * k2 * t1 / n) for the outer radix-16 pass
        for (int k2 = 0; k2 < N2; k2++) {
            for (int t1 = 0; t1 < N1; t1++) {
                double ang = -2.0 * Math.PI * ((long) k2 * t1) / FFT_N;
                COS_TW[k2 * N1 + t1] = Math.cos(ang);
                SIN_TW[k2 * N1 + t1] = Math.sin(ang);
            }
        }
    }

    private static double[] buildCos(int n) {
        double[] t = new double[n * n];
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                t[k * n + j] = Math.cos(-2.0 * Math.PI * ((long) k * j) / n);
        return t;
    }

    private static double[] buildSin(int n) {
        double[] t = new double[n * n];
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                t[k * n + j] = Math.sin(-2.0 * Math.PI * ((long) k * j) / n);
        return t;
    }

    private MelFrontend() {
    }

    /**
     * 16 kHz mono float samples -> (128, 3000) log-mel, padded/truncated to 30 s.
     * Same output as the previous implementation, computed via FFT.
     */
    static float[][] compute(float[] samples) {
        float[] audio = samples;
        if (audio.length != TARGET_SAMPLES) {
            float[] a = new float[TARGET_SAMPLES];
            System.arraycopy(samples, 0, a, 0, Math.min(samples.length, TARGET_SAMPLES));
            audio = a;
        }
        int nFrames = 1 + (audio.length - FFT_N) / HOP;
        if (nFrames > NUM_FRAMES) nFrames = NUM_FRAMES;
        int realSamples = Math.min(samples.length, TARGET_SAMPLES);
        // first frame whose whole window lies beyond real audio -> exactly zero power
        int firstZeroFrame = Integer.MAX_VALUE;
        if (realSamples < audio.length) {
            // window of frame f starts at f*HOP; fully beyond real audio iff f*HOP >= realSamples
            firstZeroFrame = Math.max(0, (realSamples + HOP - 1) / HOP);
        }
        double[][] melOut = new double[NUM_FRAMES][NUM_MELS];
        double[] re = new double[FFT_N];
        double[] im = new double[FFT_N];
        for (int f = 0; f < nFrames; f++) {
            if (f >= firstZeroFrame) continue;
            int off = f * HOP;
            for (int i = 0; i < FFT_N; i++) {
                re[i] = audio[off + i] * HANN_WINDOW[i];
                im[i] = 0.0;
            }
            powerSpectrumFft(re, im);
            for (int m = 0; m < NUM_MELS; m++) {
                double sum = 0.0;
                int fb = m * FREQ_BINS;
                for (int k = 0; k < FREQ_BINS; k++) {
                    sum += re[k] * MEL_FILTER_BANK[fb + k];
                }
                melOut[f][m] = sum;
            }
        }
        // global log + clip + normalize over ALL 3000 frames (semantics preserved)
        double maxLog = -Double.MAX_VALUE;
        double[][] logSpec = new double[NUM_FRAMES][NUM_MELS];
        for (int f = 0; f < NUM_FRAMES; f++) {
            for (int m = 0; m < NUM_MELS; m++) {
                double v = Math.log10(Math.max(1e-10, melOut[f][m]));
                logSpec[f][m] = v;
                if (v > maxLog) maxLog = v;
            }
        }
        float[][] out = new float[NUM_MELS][NUM_FRAMES];
        for (int f = 0; f < NUM_FRAMES; f++) {
            for (int m = 0; m < NUM_MELS; m++) {
                double v = Math.max(logSpec[f][m], maxLog - 8.0);
                out[m][f] = (float) ((v + 4.0) / 4.0);
            }
        }
        return out;
    }

    /**
     * Mixed-radix complex FFT power spectrum for a 400-point windowed frame
     * (Cooley-Tukey, inner 25-point DFTs over stride-16 subsequences, outer
     * 16-point DFTs with cross twiddles). On return re[0..200] holds the power
     * spectrum bins. Work arrays are allocated per call (single-threaded engine,
     * array-reuse measured separately before switching to shared buffers).
     */
    static void powerSpectrumFft(double[] re, double[] im) {
        final double[] aRe = new double[FFT_N];
        final double[] aIm = new double[FFT_N];
        for (int t1 = 0; t1 < N1; t1++) {
            for (int k2 = 0; k2 < N2; k2++) {
                double sr = 0.0, si = 0.0;
                for (int t2 = 0; t2 < N2; t2++) {
                    double v = re[N1 * t2 + t1];
                    double wr = COS25[k2 * N2 + t2];
                    double wi = SIN25[k2 * N2 + t2];
                    sr += v * wr;
                    si += v * wi;
                }
                aRe[k2 * N1 + t1] = sr;
                aIm[k2 * N1 + t1] = si;
            }
        }
        for (int k2 = 0; k2 < N2; k2++) {
            for (int k1 = 0; k1 < N1; k1++) {
                double sr = 0.0, si = 0.0;
                for (int t1 = 0; t1 < N1; t1++) {
                    double ar = aRe[k2 * N1 + t1];
                    double ai = aIm[k2 * N1 + t1];
                    double wr = COS_TW[k2 * N1 + t1];
                    double wi = SIN_TW[k2 * N1 + t1];
                    double yr = ar * wr - ai * wi;
                    double yi = ar * wi + ai * wr;
                    double vr = COS16[k1 * N1 + t1];
                    double vi = SIN16[k1 * N1 + t1];
                    sr += yr * vr - yi * vi;
                    si += yr * vi + yi * vr;
                }
                int k = k2 + N2 * k1;
                if (k <= FFT_N / 2) {
                    // store power for the bins we need; bins > n/2 mirror these
                    im[k] = si;
                    re[k] = sr;
                }
            }
        }
        for (int k = 0; k < FREQ_BINS; k++) {
            re[k] = re[k] * re[k] + im[k] * im[k];
        }
    }

    /** Slaney mel filter bank (identical formula to the reference implementation). */
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
}
