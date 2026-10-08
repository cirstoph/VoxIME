package com.whisperonnx.asr;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Central model manifest and validation for the Voxtral ONNX files.
 *
 * The engine requires exactly seven files in the app's external files dir.
 * Sizes are the byte counts of the Hugging Face revision the app downloads from
 * (onnx-community/Voxtral-Mini-3B-2507-ONNX). A file counts as present only if
 * its size matches exactly; a size mismatch means a corrupt or partial download.
 */
public final class ModelFiles {

    public static final class Entry {
        public final String name;
        public final long size;
        Entry(String name, long size) { this.name = name; this.size = size; }
    }

    public static final List<Entry> REQUIRED = new ArrayList<>();
    static {
        REQUIRED.add(new Entry("audio_encoder_q4f16.onnx", 403958L));
        REQUIRED.add(new Entry("audio_encoder_q4f16.onnx_data", 383696896L));
        REQUIRED.add(new Entry("embed_tokens_q4.onnx", 542L));
        REQUIRED.add(new Entry("embed_tokens_q4.onnx_data", 251658240L));
        REQUIRED.add(new Entry("decoder_model_merged_q4.onnx", 306657L));
        REQUIRED.add(new Entry("decoder_model_merged_q4.onnx_data", 2073260032L));
        REQUIRED.add(new Entry("decoder_model_merged_q4.onnx_data_1", 251658240L));
    }

    /** Total expected download size in bytes. */
    public static long totalSize() {
        long t = 0;
        for (Entry e : REQUIRED) t += e.size;
        return t;
    }

    public static boolean isExpectedName(String name) {
        for (Entry e : REQUIRED) if (e.name.equals(name)) return true;
        return false;
    }

    public static long expectedSize(String name) {
        for (Entry e : REQUIRED) if (e.name.equals(name)) return e.size;
        return -1;
    }

    /**
     * Returns the list of files that are missing or have the wrong size.
     * Empty list means the model is complete and (by size) intact.
     */
    public static List<Entry> missingOrCorrupt(File dir) {
        List<Entry> bad = new ArrayList<>();
        if (dir == null) return new ArrayList<>(REQUIRED);
        for (Entry e : REQUIRED) {
            File f = new File(dir, e.name);
            if (!f.isFile() || f.length() != e.size) bad.add(e);
        }
        return bad;
    }

    /** True only if all seven files exist with exactly the expected sizes. */
    public static boolean isComplete(File dir) {
        return missingOrCorrupt(dir).isEmpty();
    }

    /** Returns true if at least one (even partial) model file exists. */
    public static boolean anyPresent(File dir) {
        if (dir == null) return false;
        File[] files = dir.listFiles();
        if (files == null) return false;
        for (File f : files) if (isExpectedName(f.getName())) return true;
        return false;
    }

    /** Deletes .part leftovers that do not belong to any expected file. */
    public static void cleanupTemp(File dir) {
        if (dir == null) return;
        File[] files = dir.listFiles();
        if (files == null) return;
        for (File f : files) {
            String n = f.getName();
            if (n.endsWith(".part") || n.endsWith(".tmp")) {
                // only remove temp files for expected model names
                String base = n.endsWith(".part") ? n.substring(0, n.length() - 5) : n.substring(0, n.length() - 4);
                if (isExpectedName(base)) f.delete();
            }
        }
    }

    private ModelFiles() {}
}
