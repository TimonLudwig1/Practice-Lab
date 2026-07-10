# Projekt 01 (basic) — Ein N-Gramm-Sprachmodell

**Modul 08 — NLP 1** · Format: **Jupyter Notebook** (`sprachmodell.ipynb`)

## Warum dieses Format?

Ein Sprachmodell versteht man, wenn man es *füttert, misst und generieren lässt*.
Ein Notebook verbindet das Laden des Korpus, das Zählen der N-Gramme, die
Perplexitäts-Messung und die Textgenerierung Schritt für Schritt mit sichtbaren
Zwischenergebnissen — ideal für diesen geführten Einstieg mit viel Anleitung.

## Ziel

Du baust ein statistisches **N-Gramm-Sprachmodell** auf echtem Text und prüfst die
Kernaussagen aus Teil 1 des Skripts empirisch:

- N-Gramm-Zählungen (Unigramm/Bigramm/Trigramm) und **Add-$k$-Glättung**;
- **Perplexität** als Bewertungsmaß auf einem Held-out-Testset;
- **ein zentraler Aha-Moment:** Mit plumpem Add-1 ist das *Trigramm schlechter* als
  das Unigramm (Über-Glättung bei großem Vokabular) — erst **Interpolation** (Mischen
  der Ordnungen mit kleinem $k$) senkt die Perplexität deutlich;
- **Textgenerierung** durch Sampling — lokal plausibel, global Unsinn: die Grenze
  von N-Grammen (Ausblick auf neuronale Modelle in Modul 09).

## Vorwissen

- Skript Teil 1 (Tokenisierung, N-Gramm-Modelle, Add-$k$/Interpolation, Perplexität).
- Python: `collections.Counter`/`defaultdict`, `math.log`, `random.choices`.

## Setup

Nur Standardbibliothek. Die erste Zelle **lädt den Korpus** (*The Adventures of
Sherlock Holmes*, ~600 KB) von Project Gutenberg nach `daten/` und cached ihn (per
`.gitignore` nicht eingecheckt). In Jupyter/VS Code den Kernel des Repo-`.venv`
wählen und die Zellen von oben nach unten ausführen.

```bash
source ../../../../.venv/bin/activate
jupyter lab    # oder sprachmodell.ipynb in VS Code öffnen
```

## Aufgabenstellung (Schritt für Schritt)

Teil A (Daten, Tokenisierung, Vokabular mit `<unk>`, N-Gramm-Zählungen) ist
vorgegeben. Dann drei Aufgaben:

1. **Add-$k$ & Perplexität:** implementiere `p_unigram/p_bigram/p_trigram`
   (Add-$k$-Formel) und `perplexity`. Beobachte, dass Add-1 das Trigramm
   *verschlechtert*.
2. **Interpolation:** mische die drei Ordnungen ($\lambda_1,\lambda_2,\lambda_3$)
   mit kleinem $k$ — die Perplexität soll deutlich fallen.
3. **Textgenerierung:** sample Sätze aus dem Bigramm-Modell.

Zum Schluss ein kurzer schriftlicher Reflexionsteil.

## Was am Ende funktionieren soll

- Perplexitäten für Uni-/Bi-/Trigramm (Add-1) — mit der Beobachtung, dass höhere
  Ordnung hier *schlechter* wird (Referenz-Größenordnung: Unigramm ~350, Trigramm ~2000).
- **Interpolation** deutlich besser (Referenz: ~190).
- Generierte Sätze, die nach dem Korpus „klingen", aber grammatisch grob sind.

Die exakten Zahlen hängen leicht vom Zufalls-Split ab (Seed gesetzt), die Tendenzen
nicht.

## Musterlösung

In **`loesung/`** liegt `sprachmodell_loesung.ipynb` — vollständig implementiert und
**ausgeführt**, mit Reflexions-Antworten am Ende. Erst nach eigenem Versuch ansehen.
