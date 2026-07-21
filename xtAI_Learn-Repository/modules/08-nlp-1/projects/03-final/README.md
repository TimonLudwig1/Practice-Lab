# Project 03 (final) — An HMM POS tagger with Viterbi

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 08 — NLP 1** · Format: **a Python project, built from scratch by you**

> **The final project of the module.** There is **no given code** — you design and
> implement everything yourself. The project consolidates part 3 (sequence
> labeling, HMM, Viterbi) and is the direct, practical application of the
> HMM/Viterbi theory from **module 07**. Level: a genuine master's examination
> piece.

## Why this format and this topic?

A POS tagger is the classical sequence labeling problem and the most beautiful
application of Viterbi to real language. If you write the **hidden Markov model**
and the **Viterbi algorithm** yourself and evaluate them on a real treebank, you
connect probability estimation (module 08), dynamic programming (module 07) and
the hard business of **unknown words** — which is what decides the accuracy in
practice. A modular code base is right here (data, model and evaluation kept
separate).

## Goal

Build an **HMM-based POS tagger** that assigns its word class (UPOS tag) to every
word of a sentence, and evaluate it on real **Universal Dependencies** data
(English-EWT). The core:

$$
\hat t_{1:n} = \arg\max_{t_{1:n}} \prod_i \underbrace{P(w_i\mid t_i)}_{\text{emission}}\;
\underbrace{P(t_i\mid t_{i-1})}_{\text{transition}},
$$

solved with **Viterbi** in $O(n\,|T|^2)$.

## Prior knowledge

- Part 3.1 of the script (the HMM tagger, Viterbi) and **module 07** (HMM,
  filtering, Viterbi — here the concrete application).
- Part 1 (MLE, smoothing) for the probability estimation.
- Python: `collections.Counter`/`defaultdict`, `math.log`, nested dicts.

## What you should build — the components

1. **Load and parse the data.** Read the `.conllu` format (Universal
   Dependencies): skip the comment lines (`#`) and the multi-word tokens (an ID
   with a `-`), take **FORM** (column 2) and **UPOS** (column 4) from every token
   line; split sentences at the blank lines. (Downloaded from GitHub, cached in
   `datasets/`.)

2. **Estimate the HMM (MLE + smoothing).**
   - The **transition probabilities** $P(t_i\mid t_{i-1})$ from tag bigrams (with
     a start symbol per sentence), add-$k$ smoothed over the tag set.
   - The **emission probabilities** $P(w\mid t)$ from (word, tag) counts, smoothed.
   - **Unknown words** are the crux: replace rare training words (and unknown
     words at test time) by a **signature** (e.g. `<NUM>`, `<CAP>`, suffix classes
     such as `<~ing>`, `<~ed>`, `<~ly>`). That way the model learns a sensible
     distribution for words it has never seen.

3. **Viterbi decoding.** Dynamic programming over a table
   $v_t(j)=\max_i v_{t-1}(i)\,P(t_j\mid t_i)\,P(w_t\mid t_j)$ (in log space!), with
   **back pointers** for reconstructing the best tag sequence. Mind the start of
   the sentence (the start context) and the backtracking.

4. **Evaluation.** Compute the **tag accuracy** on the test set, **separately for
   known and unknown words**, and list the most frequent confusions. Tag an
   example sentence for a visual check.

## Acceptance criteria

- [ ] the `.conllu` parser delivers sentences as lists of (word, tag) (17 UPOS tags);
- [ ] Viterbi tags a small, unambiguous toy corpus perfectly;
- [ ] the overall tag accuracy on the EWT test set is **> 0.88** (reference about
      **0.91**);
- [ ] the evaluation reports **unknown words separately** (clearly lower,
      reference about 0.65 — the bottleneck);
- [ ] a standard sentence ("The quick brown fox …") is tagged plausibly
      (`The`→DET, `dog`→NOUN).

## Self-check questions (answer them in writing)

1. **Why does Viterbi work in log space?** What would happen with a product of
   many small probabilities over a long sentence?
2. **Why is the first-order Markov assumption** usually not quite enough, and how
   would a trigram HMM (state = a *pair* of tags) help? What does that cost?
3. **Unknown words** make up about 9 % of the test set but a large share of the
   errors. Why? How does the signature idea improve the emission for them?
4. **NOUN↔PROPN** is a frequent confusion. What causes it (the `<CAP>` signature,
   sentence beginnings), and how could one fix it (a feature: the position in the
   sentence)?
5. **Why does this HMM only reach about 91 % on EWT** while textbooks quote about
   95–96 %? (Keywords: noisy web text vs. WSJ, unknown words, bigram vs. neural
   taggers — module 09 reaches about 97–98 %.)

## Extensions (optional, for going deeper)

- A **trigram HMM** (Viterbi over pairs of tags) with deleted-interpolation smoothing.
- **Position and context features** in a **CRF** (script 3.2) instead of an HMM —
  it eliminates the label bias and uses rich features.
- **Error analysis:** the confusion matrix as a heatmap; which pairs of tags are
  the hardest?
- A comparison with `sklearn`-free baselines (the most-frequent-tag baseline) —
  how much does the context buy beyond the plain word→most-frequent-tag mapping?

## Reference solution

**`solution/`** holds a complete, tested reference implementation:
- `data.py` — the `.conllu` download (with a retry) and the parser,
- `hmm_tagger.py` — the signatures, the HMM estimation, Viterbi,
- `evaluate.py` — training + accuracy (overall/unknown) + confusions + an example,
- `test_tagger.py` — the acceptance test.

Reference: about **0.91** overall accuracy, about 0.65 on unknown words.
**Look only after your own attempt.**

```bash
source ../../../../.venv/bin/activate    # only the standard library is needed
cd solution && python evaluate.py        # training + evaluation
python test_tagger.py                    # the acceptance test
```

> **Note:** the first call downloads UD English-EWT from GitHub (about 5 MB) into
> `datasets/` (not checked in, via `.gitignore`). GitHub raw can briefly answer
> with HTTP 429 ("too many requests") — `data.py` retries the download
> automatically.

---
---

# Projekt 03 (final) — Ein HMM-POS-Tagger mit Viterbi (deutsche Fassung)

**Modul 08 — NLP 1** · Format: **Python-Projekt, von Grund auf selbst gebaut**

> **Abschlussprojekt des Moduls.** Es gibt **keinen vorgegebenen Code** — du
> entwirfst und implementierst alles selbst. Das Projekt konsolidiert Teil 3
> (Sequenz-Labeling, HMM, Viterbi) und ist die direkte, praktische Anwendung der
> HMM/Viterbi-Theorie aus **Modul 07**. Niveau: echte Master-Prüfungsleistung.

## Warum dieses Format & dieses Thema?

Ein POS-Tagger ist das klassische Sequenz-Labeling-Problem und die schönste
Anwendung von Viterbi auf echte Sprache. Wenn du das **Hidden Markov Model** und
den **Viterbi-Algorithmus** selbst schreibst und auf einer echten Treebank
evaluierst, verbindest du Wahrscheinlichkeits­schätzung (Modul 08), dynamische
Programmierung (Modul 07) und den harten Umgang mit **unbekannten Wörtern** — der
in der Praxis über die Genauigkeit entscheidet. Eine modulare Codebasis ist hier
richtig (Daten, Modell, Evaluation getrennt).

## Ziel

Baue einen **HMM-basierten POS-Tagger**, der jedem Wort eines Satzes seine Wortart
(UPOS-Tag) zuweist, und evaluiere ihn auf echten **Universal-Dependencies**-Daten
(English-EWT). Kern:

$$
\hat t_{1:n} = \arg\max_{t_{1:n}} \prod_i \underbrace{P(w_i\mid t_i)}_{\text{Emission}}\;
\underbrace{P(t_i\mid t_{i-1})}_{\text{Transition}},
$$

gelöst mit **Viterbi** in $O(n\,|T|^2)$.

## Vorwissen

- Skript Teil 3.1 (HMM-Tagger, Viterbi) und **Modul 07** (HMM, Filtering, Viterbi —
  hier die konkrete Anwendung).
- Teil 1 (MLE, Glättung) für die Wahrscheinlichkeits­schätzung.
- Python: `collections.Counter`/`defaultdict`, `math.log`, verschachtelte Dicts.

## Was du bauen sollst — die Komponenten

1. **Daten laden & parsen.** Lies das `.conllu`-Format (Universal Dependencies):
   Kommentarzeilen (`#`) und Mehrwort-Tokens (ID mit `-`) überspringen, aus jeder
   Tokenzeile **FORM** (Spalte 2) und **UPOS** (Spalte 4) ziehen; Sätze an
   Leerzeilen trennen. (Download von GitHub, cachen in `datasets/`.)

2. **HMM schätzen (MLE + Glättung).**
   - **Übergangswahrscheinlichkeiten** $P(t_i\mid t_{i-1})$ aus Tag-Bigrammen
     (mit einem Start-Symbol pro Satz), Add-$k$-geglättet über das Tagset.
   - **Emissionswahrscheinlichkeiten** $P(w\mid t)$ aus (Wort, Tag)-Zählungen, geglättet.
   - **Unbekannte Wörter** sind der Knackpunkt: Ersetze seltene Trainingswörter
     (und zur Testzeit unbekannte Wörter) durch eine **Signatur** (z. B. `<NUM>`,
     `<CAP>`, Suffix-Klassen wie `<~ing>`, `<~ed>`, `<~ly>`). So lernt das Modell
     eine sinnvolle Verteilung für Wörter, die es nie gesehen hat.

3. **Viterbi-Dekodierung.** Dynamische Programmierung über eine Tabelle
   $v_t(j)=\max_i v_{t-1}(i)\,P(t_j\mid t_i)\,P(w_t\mid t_j)$ (im Log-Raum!), mit
   **Rückzeigern** zur Rekonstruktion der besten Tag-Folge. Achte auf den
   Satzanfang (Start-Kontext) und die Rückverfolgung.

4. **Evaluation.** Berechne die **Tag-Genauigkeit** auf dem Testset, **separat für
   bekannte und unbekannte Wörter**, und liste die häufigsten Verwechslungen
   (Konfusionen) auf. Tagge einen Beispielsatz zur Sichtprüfung.

## Akzeptanzkriterien (Abnahmetest)

- [ ] `.conllu`-Parser liefert Sätze als (Wort, Tag)-Listen (17 UPOS-Tags);
- [ ] Viterbi taggt ein kleines, eindeutiges Spielzeug-Korpus perfekt;
- [ ] Gesamt-Tag-Genauigkeit auf EWT-Test **> 0,88** (Referenz ~**0,91**);
- [ ] die Evaluation weist **unbekannte Wörter separat** aus (deutlich niedriger,
      Referenz ~0,65 — der Flaschenhals);
- [ ] ein Standardsatz („The quick brown fox …") wird plausibel getaggt
      (`The`→DET, `dog`→NOUN).

## Selbstcheck-Fragen (schriftlich beantworten)

1. **Warum arbeitet Viterbi im Log-Raum?** Was passierte bei einem Produkt vieler
   kleiner Wahrscheinlichkeiten über einen langen Satz?
2. **Warum genügt die Markov-Annahme erster Ordnung** meist nicht ganz, und wie
   würde ein Trigramm-HMM (Zustand = Tag-*Paar*) helfen? Was kostet das?
3. **Unbekannte Wörter** machen ~9 % des Tests aus, aber einen Großteil der Fehler.
   Warum? Wie verbessert die Signatur-Idee die Emission für sie?
4. **NOUN↔PROPN** ist eine häufige Verwechslung. Woran liegt das (Signatur `<CAP>`,
   Satzanfänge), und wie könnte man es beheben (Feature: Position im Satz)?
5. **Warum erreicht dieser HMM auf EWT nur ~91 %**, während Lehrbücher ~95–96 %
   nennen? (Stichworte: verrauschter Web-Text vs. WSJ, unbekannte Wörter,
   Bigramm vs. neuronale Tagger — Modul 09 erreicht ~97–98 %.)

## Erweiterungen (optional, für Vertiefung)

- **Trigramm-HMM** (Viterbi über Tag-Paare) mit deleted-interpolation-Glättung.
- **Positions- und Kontext-Features** in einem **CRF** (Skript 3.2) statt HMM —
  eliminiert den Label Bias und nutzt reiche Features.
- **Fehleranalyse:** Konfusionsmatrix als Heatmap; welche Tag-Paare sind am schwersten?
- Vergleich mit `sklearn`-freien Baselines (Most-Frequent-Tag-Baseline) — wie viel
  bringt der Kontext über die reine Wort→häufigstes-Tag-Zuordnung hinaus?

## Musterlösung

In **`solution/`** liegt eine vollständige, getestete Referenzimplementierung:
- `data.py` — `.conllu`-Download (mit Retry) und Parser,
- `hmm_tagger.py` — Signaturen, HMM-Schätzung, Viterbi,
- `evaluate.py` — Training + Genauigkeit (gesamt/unbekannt) + Konfusionen + Beispiel,
- `test_tagger.py` — Abnahmetest.

Referenz: **~0,91** Gesamt-Genauigkeit, ~0,65 auf unbekannten Wörtern.
**Erst nach eigenem Versuch ansehen.**

```bash
source ../../../../.venv/bin/activate    # nur Standardbibliothek nötig
cd solution && python evaluate.py        # Training + Evaluation
python test_tagger.py                    # Abnahmetest
```

> **Hinweis:** Der erste Aufruf lädt UD English-EWT von GitHub (~5 MB) nach `datasets/`
> (per `.gitignore` nicht eingecheckt). GitHub-raw kann kurzzeitig mit HTTP 429
> („too many requests") antworten — `data.py` wiederholt den Download automatisch.
