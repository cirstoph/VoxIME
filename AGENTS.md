# AGENTS.md — Logbuch für Agenten (VoxIME)

**Anweisung an jede zukünftige Sitzung:** Dieses Dokument weiterführen, nicht
neu erfinden. Vor dem Arbeiten zuerst lesen: `README.md`,
`docs/performance-0.5.4.md`, PR #2 (Kommentare — Session-Verlauf bis 0.5.2),
diese Datei. Nach jeder Sitzung **einen datierten Eintrag unter „Logbuch"
anhängen** (neuester zuletzt): was wurde untersucht/geändert (Commit /
APK-Version), welches Geräte-Log (Datum, Uhrzeit) zeigt was, was ist offen.
Unsicherheiten ausdrücklich als „unbestätigt" kennzeichnen. Nie von vorn
anfangen, alles aus dem Code zu erschließen — hier steht, was sich bereits
jemand erschlossen hat.

## 1. Projekt-Überblick

- **VoxIME**: Android-Offline-Spracherkennung (IME + System-Voice-Input +
  Diktier-App), Fork von woheller69/whisperIMEplus (GPLv3), Package
  `org.voxime`. Engine: **Voxtral-Mini-3B-2507 als ONNX (q4)**, komplett
  on-device, kein Cloud-Fallback.
- **Aktiver Entwicklungsbranch:** `vibe/finish-voxtral-apk` (PR #2, offen).
  `main` enthält nur ein README — alle Arbeit passiert auf dem PR-Branch.
- **Zielgerät:** Pixel 6 (Tensor G1, 8 Kerne, 6 GB RAM). Pro Diktat ~3,0–3,3 GB.
- **Modell:** 7 Dateien, ~2,7 GB, In-App-Download von
  `https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/onnx/`
  (`SetupActivity.java:70`), Ablage in `getExternalFilesDir(null)`, Manifest
  mit exakten Größen in `ModelFiles.java:26-32`. Download seit 0.5.7 mit
  Retry (6×, Backoff) + Range-Resume aus `.part`.
- **Versionen:** `app/build.gradle` (`versionCode 15` / `versionName "0.5.7"`),
  Changelog `fastlane/metadata/android/en-US/changelogs/<versionCode>.txt`,
  APK als Commit unter `apks/` (CI-Artefakt, debug-signiert), Download-Link in
  README. Keystore **nie im Repo**; Release-Build mit
  `-PstoreFile/-PstorePassword/-PkeyAlias/-PkeyPassword`. Neue Signatur ⇒
  alte App deinstallieren (versionCode-Downgrade verweigert Android ohnehin).
- **CI:** `.github/workflows/android.yml` — `testDebugUnitTest` + `assembleDebug`
  bei jedem Push/PR, Artefakt „VoxIME-debug".

- **Wichtige Links:** Original-Modell (Apache-2.0)
  <https://huggingface.co/mistralai/Voxtral-Mini-3B-2507>; ONNX-Export (primär,
  Dateien unter `/onnx`)
  <https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX>;
  Referenz-Inferenz-Logik (JS): transformers.js `VoxtralForConditionalGeneration`
  + `VoxtralProcessor`; Basis-App (GPLv3)
  <https://github.com/woheller69/whisperIMEplus>; ORT Android via Maven
  `com.microsoft.onnxruntime:onnxruntime-android`.

## 2. Architektur-Schnellreferenz

Pipeline (`VoxtralEngine.java`): 16-kHz-Mono-PCM → `MelFrontend`
(128 Mel-Bins, FFT 400, Hop 160, 3000 Frames = 30 s, Prefix-Eigenschaft per
Unit-Test) → Encoder → Audio-Features werden in die `[AUDIO]`-Positionen der
Prompt-Embeddings gestreut (Scatter-Guard wirft bei Mismatch) → Decoder-Loop
mit KV-Cache (`inputs_embeds`, `attention_mask`, `position_ids`, `past_*` →
`logits`, `present_*`; 30 Layer, 8 KV-Heads, head_dim 128, hidden 3072).

- **Prompt:** `<s>[INST][BEGIN_AUDIO][AUDIO]×n [ lang:xx] [TRANSCRIBE][/INST]`.
  Token-IDs: BOS=1, EOS=2, `[INST]`=3, `[/INST]`=4, `[AUDIO]`=24,
  `[BEGIN_AUDIO]`=25, `[TRANSCRIBE]`=34. Referenz: `tools/ref_prompts.txt`
  (neu generieren: `tools/gen_ref_prompts.py` aus
  `app/src/main/assets/voxtral/tokenizer.json`). `LANG_IDS` nicht von Hand
  editieren — wurde gegen die echte BPE verifiziert (`tools/BuildPromptTest.java`).
- **Dynamischer Prompt:** 1 Audio-Token = 80 ms = 1280 Samples; Audio-Token-Zahl
  folgt der echten Audiolänge (0.4.0), plus **8 Tokens Stille-Marge**
  (`silenceMarginSamples`, 0.5.0 — ohne diese lief das Modell nach dem
  Transkript in Halluzinationsschleifen). Repetition-Guard: 6 identische
  Tokens in Folge = Stop. `maxNewTokens` = 128 (`Voxime.java:342-344`).
- **Encoder-Varianten:** fix = `audio_encoder_q4f16.onnx` (immer 30 s);
  varlen = gepatchter Graph `audio_encoder_var_q4f16.onnx`
  (`tools/patch_encoder_varlen.py`), seit 0.5.7 als APK-Asset gebündelt und
  **default AN** (`VoxtralEngine.java:58,156`, `Voxime.java:153`,
  `SettingsActivity.java:163`), in den Einstellungen umschaltbar (wirkt sofort
  via Engine-Neuladen).
- **ORT:** `onnxruntime-android:1.22.0` + `onnxruntime-extensions:0.12.4`
  (`app/build.gradle:87-88`). Achtung: `< 1.20` lädt `embed_tokens_q4.onnx`
  nicht (`GatherBlockQuantized`).

## 3. Sandbox: was geht — und was nicht

- **Geht:** JVM-Unit-Tests (`app/src/test/.../MelFrontendTest.java`,
  `VoximeTest.java`), `tools/*Test.java`, statische Analyse, ONNX-Inferenz auf
  x86 via Python-onnxruntime (Referenz-Pipeline: `tools/voxtral_spike.py`).
- **Geht nicht / ist irreführend:** Kein arm64-Gerät in der Sandbox. **Laufzeit
  auf x86 überträgt sich nicht aufs Pixel 6** — XNNPACK war dort 65 %
  *langsamer*, Encoder-Thread-Cap schadete (0.4.1/0.5.1, PR #2). Nur
  Geräte-Messungen zählen: `voxime.log` (App-Datenordner, 2-MB-Rotation) /
  logcat, pro Diktat eine StageTiming-JSON-Zeile (Tag `StageTiming`),
  „Log teilen"/„Log kopieren" im Drei-Punkte-Menü.
- **Sandbox-Fallen (aus der Portierungs-Session):** Downloads mit `python3 -c "urllib.request.urlretrieve(...)"`
  (curl/wget blockiert); Build mit `./gradlew assembleDebug|assembleRelease`,
  JDK 17 + SDK unter `/workspace/jdk` bzw. `/workspace/sdk` (falls vorhanden);
  Signieren mit `/workspace/sdk/build-tools/34.0.0/apksigner`; APK-Artefakte
  nach `/workspace/artifacts/`.

## 4. Bekannte Fallen (aus der Session-Historie, PR #2)

- Manifest-Komponentennamen müssen zu den DEX-Klassen passen — der
  Whisper*→Voxime*-Rename verpasste 3 `android:name`-Einträge; alle
  Voice-Input-Wege crashen mit `ClassNotFoundException` (behoben in 0.5.6).
- Beim Einbauen von Logging nicht die `env.createSession()`-Aufrufe löschen
  (passierte einmal in 0.5.2, Crash „Engine not loaded"; CI fängt es jetzt).
- Große Modell-Downloads brechen mit `SocketException: Software caused
  connection abort` ab — Retry/Resume seit 0.5.7, nicht „fixen".
- `versionCode` strikt erhöhen; neue Signatur ⇒ Deinstallation nötig.
- Decoder-Schleife: KV-Cache durchgehend als `OnnxTensor` halten (zero-copy),
  `OrtSession.Result` erst nach erfolgreichem nächstem `run()` schließen.
  Logits nur letzte Zeile via `argmaxLastRow` (kein `[1,S,131072]`-Materialisieren).

## 5. Offene Fragen / Unsicherheiten (unbestätigt, Stand: 0.5.7)

- **Regression in 0.5.7:** Geräte-Log 2026-11-18 23:11: 3,09 s Audio
  (49.440 Samples) → 47 Audio-Tokens, **66 Output-Tokens, stop_reason=eos,
  29,2 s gesamt** (`decode_total` 19,1 s = 66 × 290 ms; `encoder_run` nur 3,2 s
  — varlen funktioniert speed-seitig). Modell „antwortet statt zu
  transkribieren" (korrekt wären ~10–11 Tokens, vgl. 0.5.4-Gerätemessung).
  **Verdacht (unbestätigt):** varlen-Encoder default an ohne Geräte-A/B —
  die Encoder-Attention ist nicht kausal, im Kurzfenster unterscheiden sich die
  Features vom stille-gepaddeten 30-s-Fenster („KNOWN SEMANTIC DIFFERENCE",
  `tools/patch_encoder_varlen.py:18-24`, `docs/performance-0.5.4.md:80-84`).
  Test: Settings-Toggle „Fast encoder" aus, gleiche Aufnahme vergleichen.
  Fix-Kandidat: Default wieder `false` (`Voxime.java:153`,
  `SettingsActivity.java:163`).
- Ungeklärt, ob varlen-Features überhaupt reparabel sind (kausale Attention /
  Mindestfenster — braucht Graph-Neu-Export), und ob ein varlen-Lauf ohne
  Stille-Marge-Rand erneut in Schleifen läuft.
- **~290 ms/Token Decode** ist der q4-CPU-Floor auf dem Gerät (0.5.4-Messung:
  282 ms); Hebel nur über weniger Output-Tokens.
- First-Token 6,1 s (mel 1,2 + encoder 3,2 kalt [384-MB-Data-Read] + prefill
  1,6 s); warmes Verhalten des varlen-Encoders auf dem Gerät unbekannt.
- Ausgabetext des 66-Token-Laufs wurde im Log nicht mitgespeichert — nächster
  Log soll den dekodierten Text enthalten, um „antwortet" zu belegen.

## 6. Logbuch

- **2026-11-18/19 — AGENTS.md angelegt (diese Sitzung).** Analyse der
  0.5.7-Session (Commits `0d0beb7`..`833a6e5` auf `vibe/finish-voxtral-apk`):
  Download-Retry + varlen-Encoder gebündelt/default-an. Geräte-Log
  2026-11-18 ausgewertet (s. §5): varlen bringt Encoder 26–29 s → 3,2 s,
  aber 66 Output-Tokens statt Transkript → Verdacht varlen-Features,
  A/B via Settings-Toggle empfohlen, (noch) kein Code-Change.
- **2026-11-19 — PLAN.md entfernt.** War reines Planungs-/Übergabe-Dokument der
  Portierung (Commit `c6e9a74`, PR #1); als lebendes Dokument seit AGENTS.md
  obsolet. Noch relevante Inhalte (Link-Tabelle, Sandbox-Traps) sind nach
  AGENTS.md übernommen; die Datei bleibt in der Git-Historie und auf dem Branch
  `vibe/add-plan-doc` (PR #1) erhalten.
