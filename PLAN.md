# VoxIME — Projektplan & Übergabe-Dokument (v2, final)

Für: Ausführende KI-Inferenz (z. B. GLM 5.3) in einer neuen Session.
Selbstenhaltend: Alle Links, Befehle, Abhängigkeiten, Entscheidungen. **Nicht loscoden, bevor §8 (offene Entscheidungen) vom Nutzer bestätigt ist.**

## 1. Projektziel

Die Android-App **Whisper+** ([woheller69/whisperIMEplus](https://github.com/woheller69/whisperIMEplus), GPLv3) — Offline-Spracherkennung als IME, System-Voice-Input und Diktier-App — soll echte Voxtral-Architektur nutzen:

- **Engine:** Voxtral-Mini-3B-2507 als ONNX (Audio-Encoder + Ministral-3B-Decoder mit KV-Cache)
- **App-Name:** VoxIME (statt „Voxtral“ — Markenkonflikt mit Mistral AI vermeiden)
- **Package-ID:** `org.voxime`
- **Lizenz:** GPLv3, Modell Apache-2.0 — rechtlich geklärt (§6)
- **Zielgerät:** 6 GB RAM → nur q4/q4f16-Quantisierung (§5)
- **Modi:** 1) Transkription (Standard, IME), 2) Audio-Verstehen (optional)
- **Kein Fallback-Weg (bewusst):** ONNX muss funktionieren; Stellschraube sind Quantisierungs-Varianten.

## 2. Wichtigste Links

| Ressource | URL |
|---|---|
| Basis-App (Quellcode, GPLv3) | https://github.com/woheller69/whisperIMEplus |
| Original-Modell (Apache-2.0) | https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 |
| ONNX-Export (**primär!**) | https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX |
| ONNX-Dateien | https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/tree/main/onnx |
| Referenz-Inferenz-Logik (JS) | https://github.com/huggingface/transformers.js — Klassen `VoxtralForConditionalGeneration` + `VoxtralProcessor` |
| ONNX Runtime Android (bereits Dependency) | Maven: `com.microsoft.onnxruntime:onnxruntime-android:1.19.0` |
| Temurin JDK 17 | https://github.com/adoptium/temurin17-binaries/releases — Asset `OpenJDK17U-jdk_x64_linux_hotspot_*.tar.gz` |
| Android cmdline-tools | https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip |

**Benötigte ONNX-Dateien (HF-Revision `2defcb7`):**
`audio_encoder_q4f16.onnx` + Data (384 MB), `decoder_model_merged_q4.onnx` + Data (2,07 GB + 252 MB), `embed_tokens_q4.onnx` + Data (252 MB), dazu `preprocessor_config.json`, Chat-Template, SentencePiece-Tokenizer aus dem Repo-Root (Liste vor Verwendung prüfen).

## 3. Bisheriger Stand (erledigt)

1. Repo geklont: `/workspace/whisperIMEplus` (master, Commit `0169c59`)
2. Rebranding fertig: „VoxIME“ in allen `values*/strings.xml`; `applicationId "org.voxime"`; `local.properties` → `sdk.dir=/workspace/sdk`
3. Build-Umgebung steht: JDK 17 (`/workspace/jdk`), Android SDK (`/workspace/sdk`: android-34/35, build-tools 34.0.0)
4. Erste APK gebaut+signiert (`VoxIME.apk`, noch Whisper-Engine — nicht das Endprodukt). Keystore `voxtral.keystore` (Passwort `voxtral123`, Alias `voxtral`) — weiterverwenden für Update-Fähigkeit
5. Lizenzen geklärt: `NOTICE.md` im Repo, README erweitert

**Sandbox-Fallen:**
- Kein `gradlew` → `java -cp gradle/wrapper/gradle-wrapper.jar org.gradle.wrapper.GradleWrapperMain assembleRelease --no-daemon` mit `JAVA_HOME=/workspace/jdk`, `ANDROID_HOME=/workspace/sdk`.
- Downloads via `python3 -c "import urllib.request; urllib.request.urlretrieve(url, file)"` (curl/wget blockiert; `api.adoptium.net` = 403 → Temurin via GitHub-Assets).
- Kein root.
- Signieren mit `/workspace/sdk/build-tools/34.0.0/apksigner` (s. §7).
- APK nach `/workspace/artifacts/` kopieren; Zustellungsweg mit Nutzer klären (Sandbox für ihn nicht einsehbar).

## 4. GitHub-Verbindung (Schritt 0)

1. `list_unauthenticated_connectors` (Query `github`) → `ask_enable_connector` mit der gelieferten ID; Nutzer folgt dem Auth-Flow.
2. Verifizieren: `gh auth status`.
3. Repo `voxime-android` anlegen (public/privat = fragen), Stand pushen. `.gitignore`: `build/`, `local.properties`, `*.keystore`, `*.apk` **NICHT** ins Repo.
4. Bekanntes Problem: Aktivierung schlug in der Vor-Session fehl („Sandbox not created“ / „Unknown connector ID“). Wenn wieder: nicht endlos wiederholen, Nutzer informieren. Umbau funktioniert auch ohne GitHub.

## 5. Architektur (ONNX — der einzige Weg)

Neu: `VoxtralEngine.kt` (3 Graphen):

1. **Audio-Frontend:** 16-kHz-Puffer → Mel-Spektrogramm exakt gemäß `preprocessor_config.json` (Referenz: `VoxtralProcessor` in transformers.js — nicht raten)
2. **Audio-Encoder** `audio_encoder_q4f16.onnx` → Audio-Embeddings (30-s-Patches)
3. **Chat-Template** (SentencePiece): Transkription → Audio-Platzhalter + `lang:<ISO> [TRANSCRIBE]`; Verstehen → Nutzerfrage. Eigentliche Falle: Audio-Platzhalter-Token-ID ermitteln, deren Embeddings durch Encoder-Ausgaben ersetzen
4. **embed_tokens_q4.onnx** (Token-ID → Vektor)
5. **Decoder-Loop** `decoder_model_merged_q4.onnx` mit KV-Cache (`inputs_embeds`, `attention_mask`, `position_ids`, `past_*` → `logits`, `present_*`). Transkription greedy, Verstehen `temperature=0.2`/`top_p=0.95`. Stop bei EOS, max. 256–512 Tokens
6. **UI:** Settings-Schalter „Audio-Verstehen“ + Frage-Feld; IME-Standard Transkription; Manifest unverändert

**Modell-Download:** `DownloadActivity` erweitern (Voxtral ~2,7 GB, chunked mit Resume; Whisper small als Ausweichmodell).

**RAM (6-GB-Gerät):** Decoder 2,3 + Encoder 0,4 + Embeddings 0,25 + Runtime ~0,4 = ~3,0–3,3 GB → knapp machbar. Arena klein, mmap, Lazy-Load, Entladen bei `onDestroy`; bei OOM → `audio_encoder_q4`.

**Latenz:** 3–8 Token/s → kurzes Diktat 5–15 s. Im UI ehrlich ankündigen.

## 6. Lizenzen (geklärt)

App GPLv3 (gilt für Fork, Source offen halten); Voxtral-Gewichte + ONNX-Export Apache-2.0 (GPLv3-kompatibel); ONNX Runtime MIT. Pflichten: LICENSE + NOTICE.md + Credits (© woheller69, Mistral AI) intakt, keine Endorsement-Vermutung. Name „VoxIME“ vermeidet Markenkonflikt. Nutzer hat GPLv3 + VoxIME bestätigt; beides ist im Repo umgesetzt.

## 7. Ausführungsreihenfolge

1. GitHub verbinden (§4) + Repo `voxime-android`, pushen (ohne build/keystore/apk)
2. **Spike (kritisch):** Python-Skript (onnxruntime), 3 ONNX-Dateien + Test-WAV → `lang:en [TRANSCRIBE]` → Soll: „I have a dream …“. Klären: Tensor-Namen, Platzhalter-Token-ID, Template. Erst wenn plausibler Text kommt: weiter
3. Android-Integration: `VoxtralEngine.kt`, `DownloadActivity`, Settings
4. Verstehen-Modus (nach stabiler Transkription)
5. RAM-/Latenz-Härtung (§5)
6. Build + Sign (alter Keystore) + APK nach `/workspace/artifacts/`, Zustellung klären
7. Doku: README, NOTICE.md

**Aufwand:** Schritt 2 = 1–2 Sessions (Platzhalter-Falle!); Schritte 3–5 = 1–2 Sessions. Fortschritt committen.

## 8. Offene Entscheidungen (vor Start bestätigen!)

- Repo `voxime-android` public oder privat?
- Decoder q4 + Encoder q4f16 — okay?
- APK-Zustellungsweg (Sandbox nicht einsehbar)?
- Start mit GitHub (Schritt 1) oder direkt Spike (Schritt 2)?

## 9. Kurzfassung für die ausführende Inferenz

Deutsch, technikaffin, ehrlich bei Rückschritten. Erst §8 bestätigen lassen, dann §7 strikt abarbeiten — Spike vor UI-Arbeit, jeder Schritt mit lauffähigem Test. §3 hat funktionierende Befehle + alle Fallen. Keine Fallback-Architekturen; Quantisierung ist die Stellschraube. GPLv3 + NOTICE.md intakt halten, Keystore weiterverwenden.
