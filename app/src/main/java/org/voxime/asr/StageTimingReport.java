package org.voxime.asr;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Iterator;

/**
 * Parses the single-line JSON summaries emitted by StageTiming so that
 * unit tests and tooling can assert on phase keys without duplicating
 * the format. Not used by the production pipeline itself.
 */
public final class StageTimingReport {
    public final JSONObject root;

    private StageTimingReport(JSONObject root) {
        this.root = root;
    }

    public static StageTimingReport parse(String json) throws org.json.JSONException {
        return new StageTimingReport(new JSONObject(json));
    }

    public long phaseNanos(String key) throws org.json.JSONException {
        return root.getJSONObject("phases").optLong(key, -1);
    }

    public long totalNanos() throws org.json.JSONException {
        return root.getLong("total_nanos");
    }

    public long firstTokenNanos() throws org.json.JSONException {
        return root.optLong("first_token_nanos", -1);
    }

    public int count(String key) throws org.json.JSONException {
        return root.getJSONObject("counts").optInt(key, -1);
    }

    public String stopReason() throws org.json.JSONException {
        return root.getString("stop_reason");
    }

    public boolean hasAllPhases(String... keys) throws org.json.JSONException {
        JSONObject p = root.getJSONObject("phases");
        for (String k : keys) if (!p.has(k)) return false;
        return true;
    }

    public int numPhases() throws org.json.JSONException {
        return root.getJSONObject("phases").length();
    }
}
