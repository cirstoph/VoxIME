package org.voxime.asr;

/**
 * Minimal replacement for the removed RTranslator Recognizer constants.
 * Voxtral supports transcription into these languages (auto detection
 * happens when no language is forced in the prompt).
 */
public final class Transcribe {

    public enum Action { TRANSCRIBE }

    /** ISO codes of the languages Voxtral was trained on. */
    public static final String[] LANGUAGES = {
            "en", "fr", "de", "es", "it", "pt", "nl", "hi"
    };

    public static final Action ACTION_TRANSCRIBE = Action.TRANSCRIBE;

    private Transcribe() {}
}
