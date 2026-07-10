# Projekt 03 (final) — Ein HMM-POS-Tagger mit Viterbi

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
   Leerzeilen trennen. (Download von GitHub, cachen in `daten/`.)

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

In **`loesung/`** liegt eine vollständige, getestete Referenzimplementierung:
- `data.py` — `.conllu`-Download (mit Retry) und Parser,
- `hmm_tagger.py` — Signaturen, HMM-Schätzung, Viterbi,
- `evaluate.py` — Training + Genauigkeit (gesamt/unbekannt) + Konfusionen + Beispiel,
- `test_tagger.py` — Abnahmetest.

Referenz: **~0,91** Gesamt-Genauigkeit, ~0,65 auf unbekannten Wörtern.
**Erst nach eigenem Versuch ansehen.**

```bash
source ../../../../.venv/bin/activate    # nur Standardbibliothek nötig
cd loesung && python evaluate.py         # Training + Evaluation
python test_tagger.py                    # Abnahmetest
```

> **Hinweis:** Der erste Aufruf lädt UD English-EWT von GitHub (~5 MB) nach `daten/`
> (per `.gitignore` nicht eingecheckt). GitHub-raw kann kurzzeitig mit HTTP 429
> („too many requests") antworten — `data.py` wiederholt den Download automatisch.
