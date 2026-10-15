import java.util.*;

public class BuildPromptTest {
    static final int AUDIO_TOKEN_ID = 24;
    static final int BEGIN_AUDIO_TOKEN_ID = 25;
    static final int NUM_AUDIO_TOKENS = 375;
    static final int BOS_TOKEN_ID = 1;
    static final Map<String, int[]> LANG_IDS = new HashMap<>();
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

    // EXACT copy of engine logic
    public static long[] buildPrompt(String langCode) {
        List<Long> tokens = new ArrayList<>();
        tokens.add((long) BOS_TOKEN_ID);
        tokens.add(3L);
        tokens.add((long) BEGIN_AUDIO_TOKEN_ID);
        for (int k = 0; k < NUM_AUDIO_TOKENS; k++) tokens.add((long) AUDIO_TOKEN_ID);
        int[] langIds = null;
        if (langCode != null) langIds = LANG_IDS.get(langCode.toLowerCase());
        if (langIds != null) for (int id : langIds) tokens.add((long) id);
        tokens.add(34L);
        tokens.add(4L);
        long[] out = new long[tokens.size()];
        for (int i = 0; i < out.length; i++) out[i] = tokens.get(i);
        return out;
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> cases = new LinkedHashMap<>();
        cases.put("auto", null);
        cases.put("en", "en"); cases.put("de", "de"); cases.put("fr", "fr");
        cases.put("es", "es"); cases.put("it", "it"); cases.put("pt", "pt");
        cases.put("nl", "nl"); cases.put("hi", "hi");
        // expected reference sequences from Python reference tokenizer (dumped to file)
        List<String> ref = new ArrayList<>();
        try (Scanner sc = new Scanner(new java.io.File(args[0]))) {
            while (sc.hasNextLine()) ref.add(sc.nextLine().trim());
        }
        int idx = 0; boolean ok = true;
        for (Map.Entry<String, String> e : cases.entrySet()) {
            long[] got = buildPrompt(e.getValue());
            String[] expStr = ref.get(idx).split(",");
            long[] exp = new long[expStr.length];
            for (int i = 0; i < exp.length; i++) exp[i] = Long.parseLong(expStr[i]);
            boolean match = Arrays.equals(got, exp);
            boolean endsWithInst = got[got.length-1] == 4;
            boolean noZero = true;
            for (long t : got) if (t == 0) noZero = false;
            System.out.printf("%-4s len=%d match=%b endsWith[/INST]=%b noZeroToken=%b%n",
                e.getKey(), got.length, match, endsWithInst, noZero);
            if (!match || !endsWithInst || !noZero) ok = false;
            idx++;
        }
        System.out.println(ok ? "PROMPT TEST PASSED" : "PROMPT TEST FAILED");
        if (!ok) System.exit(1);
    }
}
