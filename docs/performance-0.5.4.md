# Performance-Änderungen 0.5.3 -> 0.5.4 (Latenz nach Aufnahmeende)

Stand: statische Analyse + lokale Verifikation (Build, Unit-Tests, FFT/DFT-Parität).
**Kein Zielgerät verfügbar**: alle Laufzeitangaben müssen auf dem Gerät mit der
neuen StageTiming-Instrumentierung nachgemessen werden (eine JSON-Zeile pro
Diktat in voxime.log und logcat, Tag `StageTiming`). Es wird keine bestimmte
Gesamtlatenz versprochen.

## Implementierte Änderungen

### P0 — Lifecycle, Abbruch, Aufnahmeende

| Änderung | Vorher | Nachher |
|---|---|---|
| RecognitionService-Requests | Jeder `onStartListening()` erzeugte `new Voxime(this)` + `new Recorder(this)`, alte Instanzen wurden überschrieben (verlorene Worker-Threads, wiederholte Kaltstarts) | Engine und Recorder werden pro Service-Instanz wiederverwendet, Listener pro Request aktualisiert |
| Engine-Mehrfachbelegung | Activity, IME und RecognitionService luden je eine eigene Kopie aller drei ONNX-Sessions im selben Prozess | Prozessweite Engine (`Voxime.getShared/releaseShared`), ein residenter Satz Sessions; IME bleibt wie bisher resident |
| Abbruch | `Voxime.stop()` setzte nur ein Flag, der Decoder lief weiter; Resultat traf ggf. trotzdem ein | `stop()` setzt Abbruchflag; Worker prüft vor jeder Phase und jedem Decoder-Schritt; abgebrochene Aufgaben liefern kein Resultat (Task-IDs); `TranscriptionCancelledException` |
| unloadModel | `join(3000)` ohne Verifikation, Sessions konnten bei laufendem Zugriff geschlossen werden | Verifiziert Thread-Ende vor Session-Close; bei hängendem Worker bleiben Sessions offen und werden protokolliert; `unloadModel(force)` mit Interrupt |
| Recorder.stop() | `fileSavedLock.wait()` ohne Prädikat/Timeout auf dem Main-Thread | Nicht-blockierend; `stopAndWait(ms)` wartet auf Zustandsprädikat mit Timeout; Completion in `finally` signalisiert |
| Recorder-Shutdown | `recordLoop()` endete nur durch Interrupt, kein öffentlicher Pfad | `shutdown()` als Lifecycle-Pfad |
| VAD-Kopien (P2, miterledigt) | Pro 480-Sample-Schritt `outputBuffer.toByteArray()` — volle Pufferkopie, ~480 MB Kopiervolumen bei 30 s | VAD liest direkt den frischen Frame; ein vorab allozierter 30-s-Puffer |

### P1 — Rechen- und Speicheroptimierungen

| Änderung | Vorher | Nachher |
|---|---|---|
| Mel-Frontend | Direkte DFT: 201 Bins x 400 Samples = 241 Mio. innere Durchläufe pro Diktat (immer, auch für 1 s Audio), plus ~77 Mio. Mel-Produkte; Fenster/Filterbank pro Aufruf neu berechnet | Mixed-Radix-FFT (25x16, vorberechnete Twiddles): ~10.400 komplexe MACs pro Frame statt 160.800 reelle; Hann-Fenster + Filterbank statisch; Frames vollständig hinter dem realen Audio (inkl. Silence Margin) werden analytisch als Nullleistung behandelt. Parität mit der DFT-Referenz: maxAbs = 0.0 (Unit-Test), FFT-Power maxRel < 1e-8 |
| Logits | `getValue()` materialisierte [1,S,131072] als float[][][] auf dem Java-Heap (0,5 MiB pro Position; ~190 MiB bei 383er-Prefill) | Nur die letzte Position wird über den flachen FloatBuffer gelesen (`argmaxLastRow`); Heap wächst beim Prefill nicht mehr um die vollständige Logits-Kopie |
| Logging (P3) | `AppLog` öffnete pro Eintrag synchron einen FileWriter auf dem aufrufenden Thread (Decoder-Loop betraf dies pro Step) | Zeilen werden an einen Hintergrunds-Writer übergeben; Fehler bleiben synchron |

### Baseline-Instrumentierung (StageTiming)

Pro Diktat eine JSON-Zeile: Ladezeiten je Session/Vokabular, Stop-zu-Audio,
PCM-Konvertierung, Mel, Encoder-run/-copy, Prompt-Embeddings, Audio-Scatter,
Prefill-run/-Logits, Decoder-run je Schritt, Embedding-run je Schritt,
Tensor-Prep, Logits-Auswertung, Zeit bis zum ersten Token, Token-Zählungen,
Abbruchgrund, Gesamtlatenz. Damit sind alle geforderten Messungen auf dem
Gerät ohne erneuten Codeumbau möglich.

## Geräte-Messung 2026-10-15 (8 Kern-Android, 8-core SoC, 7 Inferenz-Threads)

Erste StageTiming-Messung nach Aufnahmeende (4 s Audio, 59 Audio-Tokens, lang=auto):

| Phase | Zeit |
|---|---|
| encoder_run | **26.100 ms (78 %)** |
| decode_total (11 Tokens) | 3.099 ms (~282 ms/Token) |
| mel (FFT) | 1.479 ms |
| prefill_run | 2.207 ms |
| logits_eval / tensor_prep / embed | < 0,7 s zusammen |
| Gesamt | 33.663 ms |

Fehler in der Instrumentierung behoben: `pcm_convert` war negativ (gemischte
Zeitquellen SystemClock vs nanoTime); jetzt einheitlich System.nanoTime().

**Ergebnis:** Der Encoder dominiert, weil er bei kurzen Aufnahmen weiterhin das
volle 30-s-Fenster verrechnet. Decoder (q4) ist mit ~0,28 s/Token zweitrangig.

## Variable Encoder-Länge: implementiert als Experiment (default aus)

Analyse des Original-Graphen (HF-Revision, SHA-256 s. Skript): 3000 ist hart
kodiert in (1) der Input-Deklaration, (2) 128 Reshape-Shape-Konstanten mit der
Conv-Frame-Zahl 1500, (3) dem Positions-Embedding-Add [1500,1280]. Gewichte und
externe Datendatei bleiben byte-identisch.

- `tools/patch_encoder_varlen.py`: reproduzierbarer Graph-Patch mit
  SHA-256-Prüfung des Quell-Graphen (Erzeugt `audio_encoder_var_q4f16.onnx`,
  nutzt die vorhandene `.onnx_data` unverändert weiter).
- Engine: lädt den gepatchten Encoder nur, wenn er + Data-Datei vorhanden sind
  UND die Einstellung „Fast encoder (experimental)" aktiv ist; sonst exakt das
  bisherige Verhalten. Mel wird auf ein 8er-Frame-Raster begrenzt (1 Audio-Token
  = 8 Frames); Audio-Token-Zahl wird aus der tatsächlichen Encoder-Ausgabe
  abgeleitet; Scatter-Konsistenzprüfung bleibt.
- Verifikation (onnxruntime 1.31, x86): Output-Shape korrekt (59 Tokens bei 4 s),
  Laufzeit 5,5 s -> 0,7 s für dieses Beispiel — auf dem Zielgerät nachmessen.
- **Bekannte semantische Differenz:** Die Encoder-Attention ist NICHT kausal.
  Im Kurzfenster sieht jedes Token nur das reale Audio statt des
  Null-gepaddeten 30-s-Fensters; Feature-Werte unterscheiden sich daher (max.
  Absolutdifferenz ~1,5 auf Zufalls-Signal). Transkriptionsqualität muss per
  A/B-Vergleich auf dem Zielgerät geprüft werden, bevor der Schalter dauerhaft
  aktiviert wird. Mel-Präfix-Eigenschaft per Unit-Test gesichert.

Weitere vorgeschlagene Punkte entfallen hier; für Threadzahl-Experimente und
Embedding-Cache gilt weiterhin: erst messen (README dokumentiert negative
Erfahrung mit XNNPACK/Thread-Cap auf dem Gerät).
- **Threadzahl-Experimente** (Encoder/Decoder getrennt, 1/2/4/aktuell): README
  dokumentiert, dass XNNPACK + Encoder-Thread-Cap auf dem Gerät langsamer waren
  (0.4.1/0.5.1). Keine neue Hypothese ohne Messung wiederholt.
- **Embedding-LRU-Cache / Decoderexport mit interner Embedding-Schicht**:
  erst nach Messung von `embed_run_per_step_nanos`.
- **KV-Cache-Layout / anderes Cacheformat**: benötigt Graphunterstützung, nicht
  blind umbauen.
- **ORT RunOptions-Terminierung**: in ORT 1.22 (Java/Android) nicht vorhanden;
  Abbruchlatenz bleibt vom längsten einzelnen run() begrenzt (dokumentiert im
  Code).

## Messprotokoll (auf dem Gerät nachholen)

Pro Fall: 16-kHz-Mono-PCM-Fixtures mit 1/3/10/30 s; Stille; Sprache mit langer
Schlussstille; Deutsch mit `lang=de` und `auto`; kalt vs. warm (>10
Wiederholungen, Median + Streuung; p95 bei kleinen n als unsicher kennzeichnen);
thermisch vorbelastete Nutzung. Dokumentieren: Gerät, SoC, Android-Version,
Build-Typ, Modellrevision, ORT-Version, Threadkonfiguration, Java-Heap, PSS.

Erwartung aus der Codeanalyse (nicht garantiert): Der Mel-Block dominierte bei
kurzen Aufnahmen die Java-seitige Zeit (DFT-Kosten sind unabhängig von der
Audiolänge, da immer 3000 Frames berechnet wurden); die FFT + Nullframe-Behandlung
reduziert diesen Anteil um Großenordnungen, und entfällt bei Null-Frames
vollständig. Die Decoder-Inferenz (q4, 3B) bleibt der dominierende Anteil und
wird nur durch das bereits vorhandene dynamische Prompt (kürzere Prefill) und
die hier nicht erweiterten Runtime-Experimente beeinflusst —
`decode_run_per_step_nanos` und `prefill_run` zeigen beides getrennt.

## Qualitäts- und Verhaltenserhalt

- Silence Margin, dynamischer Prompt, Repetitionsschutz, maximale 128 neue
  Tokens, Eingabewege (App/IME/RecognitionService), Offline-Betrieb: unverändert.
- Mel-Ausgabeform [128][3000] und Normalisierung: unverändert, Parität per
  Unit-Test (Referenz = alte DFT-Pipeline).
- Logits: identisches Argmax-Ergebnis (gleicher Tensor, nur gelesene Kopie
  kleiner). Tokenfolge sollte identisch sein; auf dem Gerät über
  Transkriptionsvergleich gegen die Baseline prüfen.
- VAD: gleiche Frame-Größe, gleiche Silence-Timeout-Logik; nur die Kopierarbeit
  entfällt. `silenceDurationMs`-Einstellung unverändert.

## Verbleibende Engpässe (nach Messung zu entscheiden)

1. q4-3B-Decoderzeit je Token (Architekturgrenze; Alternativmodell = separate
   Architekturentscheidung).
2. Encoder über volles 30-s-Fenster trotz variablem Prompt (Experiment offen).
3. Embedding-Session pro Ausgabetoken (Messung über `embed_run_per_step_nanos`).
