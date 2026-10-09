import java.util.*;

/**
 * Acceptance test for the dynamic prefill:
 * audio token count = ceil(samples / 1280), clamped to [1, 375].
 */
public class DynamicPromptTest {
    static final int NUM_AUDIO_TOKENS = 375;

    static int audioTokensFor(int samples) {
        return Math.max(1, Math.min(NUM_AUDIO_TOKENS, (samples + 1279) / 1280));
    }

    public static void main(String[] a) {
        int[][] cases = {
            // samples, expected tokens
            {1, 1}, {1280, 1}, {1281, 2}, {47040, 37}, {48000, 38},
            {480000, 375}, {600000, 375}, {0, 1}, {-5, 1}
        };
        int fails = 0;
        for (int[] c : cases) {
            int got = audioTokensFor(c[0]);
            boolean ok = got == c[1];
            System.out.printf("samples=%d -> %d tokens %s%n", c[0], got, ok ? "ok" : "FAIL (want " + c[1] + ")");
            if (!ok) fails++;
        }
        // prompt length consistency: BOS INST BEGIN n*audio TRANSCRIBE /INST
        for (int samples : new int[]{47040, 48000, 480000}) {
            int n = audioTokensFor(samples);
            int len = 1 + 1 + 1 + n + 1 + 1;
            System.out.printf("samples=%d: prompt_len=%d (audio=%d)%n", samples, len, n);
        }
        System.out.println(fails == 0 ? "DYNAMIC PROMPT TEST PASSED" : "DYNAMIC PROMPT TEST FAILED");
        if (fails > 0) System.exit(1);
    }
}
