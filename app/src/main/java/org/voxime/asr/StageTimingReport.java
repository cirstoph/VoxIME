package org.voxime.asr;

/**
 * Minimal reader for the single-line JSON summaries emitted by StageTiming,
 * so unit tests and tooling can assert on phase keys without duplicating the
 * format and without org.json (whose Android stub returns nulls in unit
 * tests). Not used by the production pipeline itself.
 */
public final class StageTimingReport {
    private final String json;

    private StageTimingReport(String json) {
        this.json = json;
    }

    public static StageTimingReport parse(String json) {
        return new StageTimingReport(json);
    }

    private String objectAt(String key) {
        int i = json.indexOf("\"" + key + "\":{");
        if (i < 0) return null;
        int start = i + key.length() + 3;
        int depth = 1;
        int j = start;
        while (j < json.length() && depth > 0) {
            char c = json.charAt(j);
            if (c == '{') depth++;
            if (c == '}') depth--;
            j++;
        }
        return json.substring(start, j - 1);
    }

    private long longAt(String object, String key) {
        if (object == null) return -1;
        int i = object.indexOf("\"" + key + "\":");
        if (i < 0) return -1;
        int j = i + key.length() + 3;
        int end = j;
        while (end < object.length()) {
            char c = object.charAt(end);
            if (c == ',' || c == '}') break;
            end++;
        }
        try {
            return Long.parseLong(object.substring(j, end).trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    public long phaseNanos(String key) {
        return longAt(objectAt("phases"), key);
    }

    public long totalNanos() {
        return longAt(json, "total_nanos");
    }

    public long firstTokenNanos() {
        return longAt(json, "first_token_nanos");
    }

    public int count(String key) {
        return (int) longAt(objectAt("counts"), key);
    }

    public String stopReason() {
        int i = json.indexOf("\"stop_reason\":\"");
        if (i < 0) return null;
        int start = i + "\"stop_reason\":\"".length();
        int end = json.indexOf('\"', start);
        return end > start ? json.substring(start, end) : null;
    }

    public boolean hasAllPhases(String... keys) {
        String phases = objectAt("phases");
        if (phases == null) return false;
        for (String k : keys) {
            if (!phases.contains("\"" + k + "\":")) return false;
        }
        return true;
    }

    public int numPhases() {
        String phases = objectAt("phases");
        if (phases == null) return 0;
        int count = 0;
        for (int i = 0; i < phases.length(); i++) {
            if (phases.charAt(i) == ',') count++;
        }
        return count + 1;
    }
}
