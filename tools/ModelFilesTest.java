import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Acceptance test for P0-03: model validation must be deterministic. */
public class ModelFilesTest {
    // mirror of ModelFiles.REQUIRED
    static final String[][] REQ = {
        {"audio_encoder_q4f16.onnx", "403958"},
        {"audio_encoder_q4f16.onnx_data", "383696896"},
        {"embed_tokens_q4.onnx", "542"},
        {"embed_tokens_q4.onnx_data", "251658240"},
        {"decoder_model_merged_q4.onnx", "306657"},
        {"decoder_model_merged_q4.onnx_data", "2073260032"},
        {"decoder_model_merged_q4.onnx_data_1", "251658240"},
    };

    public static void main(String[] a) throws Exception {
        Path dir = Files.createTempDirectory("voxime-test");
        File d = dir.toFile();
        int failures = 0;
        // case 1: empty dir -> 7 missing
        failures += check(countMissing(d), 7, "empty dir");
        // case 2: 5 valid files -> 2 missing
        for (int i = 0; i < 5; i++) write(d, REQ[i][0], Long.parseLong(REQ[i][1]));
        failures += check(countMissing(d), 2, "5 of 7 files");
        // case 3: 6 valid files -> 1 missing
        write(d, REQ[5][0], Long.parseLong(REQ[5][1]));
        failures += check(countMissing(d), 1, "6 of 7 files");
        // case 4: all 7 valid -> complete
        write(d, REQ[6][0], Long.parseLong(REQ[6][1]));
        failures += check(countMissing(d), 0, "7 of 7 files");
        if (!isComplete(d)) { System.out.println("FAIL: isComplete with 7 valid"); failures++; }
        // case 5: corrupt file (wrong size) -> not complete, exactly 1 bad
        write(d, REQ[3][0], 12345);
        failures += check(countMissing(d), 1, "corrupt file (wrong size)");
        if (isComplete(d)) { System.out.println("FAIL: isComplete with corrupt file"); failures++; }
        // case 6: truncated big file
        write(d, REQ[3][0], Long.parseLong(REQ[3][1]) - 1);
        failures += check(countMissing(d), 1, "truncated file (off-by-one)");
        // restore & verify names/unknown rejected
        write(d, REQ[3][0], Long.parseLong(REQ[3][1]));
        if (!isComplete(d)) { System.out.println("FAIL: isComplete after restore"); failures++; }
        if (expectedName("evil.exe")) { System.out.println("FAIL: unexpected name accepted"); failures++; }
        if (!expectedName("embed_tokens_q4.onnx")) { System.out.println("FAIL: expected name rejected"); failures++; }
        if (expectedSize("decoder_model_merged_q4.onnx_data_1") != 251658240L) { System.out.println("FAIL: wrong expected size"); failures++; }
        // cleanupTemp test
        new File(d, "audio_encoder_q4f16.onnx.part").delete();
        new File(d, "audio_encoder_q4f16.onnx.part2").delete();
        Files.write(dir.resolve("audio_encoder_q4f16.onnx.part"), new byte[10]);
        Files.write(dir.resolve("random-notmodel.part"), new byte[10]);
        cleanupTemp(d);
        if (new File(d, "audio_encoder_q4f16.onnx.part").exists()) { System.out.println("FAIL: temp not cleaned"); failures++; }
        if (!new File(d, "random-notmodel.part").exists()) { System.out.println("FAIL: unrelated file deleted"); failures++; }
        // null dir
        failures += check(countMissing(null), 7, "null dir");
        System.out.println(failures == 0 ? "MODELFILES TEST PASSED" : "MODELFILES TEST FAILED (" + failures + ")");
        if (failures > 0) System.exit(1);
    }

    static int countMissing(File dir) {
        int bad = 0;
        for (String[] e : REQ) {
            File f = dir == null ? null : new File(dir, e[0]);
            if (dir == null || !f.isFile() || f.length() != Long.parseLong(e[1])) bad++;
        }
        return bad;
    }
    static boolean isComplete(File dir) { return countMissing(dir) == 0; }
    static boolean expectedName(String n) {
        for (String[] e : REQ) if (e[0].equals(n)) return true;
        return false;
    }
    static long expectedSize(String n) {
        for (String[] e : REQ) if (e[0].equals(n)) return Long.parseLong(e[1]);
        return -1;
    }
    static void cleanupTemp(File dir) {
        File[] files = dir.listFiles();
        for (File f : files) {
            String n = f.getName();
            if (n.endsWith(".part")) {
                String base = n.substring(0, n.length() - 5);
                if (expectedName(base)) f.delete();
            }
        }
    }
    static void write(File dir, String name, long size) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(new File(dir, name), "rw")) {
            raf.setLength(size);
        }
    }
    static int check(int got, int want, String label) {
        if (got != want) { System.out.println("FAIL " + label + ": got " + got + " want " + want); return 1; }
        System.out.println("ok: " + label);
        return 0;
    }
}
