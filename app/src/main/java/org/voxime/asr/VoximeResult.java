package org.voxime.asr;


public class VoximeResult {
    private final String result;
    private final String language;
    private final Transcribe.Action task;

    public VoximeResult(String result, String language, Transcribe.Action task){
        this.result = result;
        this.language = language;
        this.task = task;
    }

    public String getResult() {
        return result;
    }

    public String getLanguage() {
        return language;
    }

    public Transcribe.Action getTask() {
        return task;
    }
}
