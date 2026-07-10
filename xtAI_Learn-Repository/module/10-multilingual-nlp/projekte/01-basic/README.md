# Projekt 01 (basic) — Mehrsprachige Subword-Tokenisierung mit SentencePiece

**Modul 10 — Multilingual NLP** · Format: **Jupyter Notebook** (`tokenisierung.ipynb`)

## Warum dieses Format?

Tokenisierung versteht man durch *Ausprobieren und Messen*: einen Satz zerlegen sehen,
fertility ausrechnen, Vokabulare vergleichen. Ein Notebook verbindet Training, Zerlegung
und Kennzahlen mit sichtbaren Zwischenergebnissen — ideal für diesen geführten Einstieg.

## Ziel

Du trainierst mit **SentencePiece** ein **gemeinsames** Subword-Vokabular auf echten
Deutsch–Englisch-Daten (Tatoeba) und prüfst die Kernaussagen aus Skript-Abschnitt 2
empirisch:

- **BPE-Zerlegung** und das `▁`-Leerzeichen-Symbol; warum es **nie OOV** gibt.
- **fertility** (Ø Subword-Tokens pro Wort) für Deutsch vs. Englisch.
- **Vokabular-Teilung**: welchen Anteil der Tokens teilen sich beide Sprachen?
- **Vokabular-Bias**: ein englisch-dominiertes Vokabular **verdoppelt** die deutsche
  fertility — der Kosten-/Fairness-Nachteil ressourcenarmer Sprachen in großen LLMs.
- **Subword vs. Wort**: Wort-Vokabulargröße und OOV-Rate als Argument für Subwords.

## Vorwissen

- **Skript** Abschnitt 2 (BPE, WordPiece, Unigram-LM/SentencePiece, fertility, gemeinsames
  Vokabular).
- Python-Basics (`Counter`, Mengen, `str.split`).

## Setup

Benötigt `sentencepiece` (in der Repo-`requirements.txt`). Die erste Zelle lädt den
Tatoeba-Datensatz (~12 MB) automatisch nach `daten/` (Browser-Header nötig, ist im Code)
und cached ihn; nicht eingecheckt.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # oder tokenisierung.ipynb in VS Code öffnen, Kernel = Repo-.venv
```

Läuft in **unter einer Minute** (SentencePiece ist in C++).

## Datensatz

**Tatoeba** Deutsch–Englisch-Satzpaare (~331k, offen, CC-BY) über manythings.org/anki.
Echte, kurze Alltagssätze — klein, sauber, ideal, um Tokenisierung sichtbar zu machen und
später (Projekt 02/03) für Embeddings und Übersetzung weiterzuverwenden.

## Aufgabenstellung (Schritt für Schritt)

**Teil A** (Download, Parsing, Korpus schreiben) ist vorgegeben. Dann drei Aufgaben (an
`# TODO` markiert):

1. **BPE trainieren & encoden**: das gemeinsame Modell ist per Aufruf gegeben — zerlege
   zwei Beispielsätze und lies die Subword-Struktur.
2. **fertility** implementieren: Subword-Tokens pro Wort, für EN und DE getrennt.
3. **Vokabular-Teilung** (vorgegeben) messen und den **Vokabular-Bias** zeigen: ein
   EN-only-Vokabular trainieren und die deutsche fertility damit vergleichen.

Zum Schluss ein kurzer schriftlicher **Reflexionsteil** (4 Fragen).

## Was am Ende funktionieren soll

- Zerlegungen für EN & DE mit sichtbarer Subword-Struktur (`▁`, Wortstücke).
- fertility EN ≈ 1.5, DE ≈ 1.4 (gemeinsames Vokabular).
- **Bias-Ergebnis**: EN-only-Vokabular → deutsche fertility ≈ 2.9 (Faktor ~2.0).
- Wort-Vokabular ~20k Typen mit ~11 % OOV, während BPE nie OOV hat.

## Musterlösung

Voll ausgefülltes, **ausgeführtes** Notebook unter
[`loesung/tokenisierung_loesung.ipynb`](loesung/tokenisierung_loesung.ipynb). Erst selbst
probieren — die Stub-Zellen werfen `NotImplementedError`, bis du sie füllst.
