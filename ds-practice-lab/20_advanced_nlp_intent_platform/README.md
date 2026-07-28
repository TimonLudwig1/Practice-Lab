# 20 — Project Brief: Banking Intent Classification Service

Difficulty: ⚫ Advanced / Portfolio | Topic: NLP — Transformers (TensorFlow)

---

## Project Brief

**From:** VP Customer Operations, FinServe Bank
**To:** NLP Engineering (you)
**Re:** Intent router for the support inbox

Customer messages must be routed to the right team automatically. We license the BANKING77 taxonomy (77 fine-grained intents, e.g. `card_arrival`, `exchange_rate`, `lost_or_stolen_card`). Misrouting costs us ~4 minutes of agent time per ticket; messages the model is unsure about should go to a human triage queue instead of being misrouted.

## Business Context
- ~50,000 messages/month; triage queue capacity is **at most 15%** of volume
- Some intents are business-critical: misrouting `lost_or_stolen_card`, `compromised_card`, or `cancel_transfer` has compliance implications — these need near-perfect recall
- Leadership is skeptical of "AI hype": your report must compare the transformer against a strong classical baseline and quantify what the extra complexity buys

## Data
**BANKING77** — 13,083 customer queries, 77 intents (10,003 train / 3,080 test). Available via Hugging Face datasets (`banking77`) or the PolyAI GitHub repository. The official test set is the acceptance set — evaluate your final system on it once.

## Technical Constraints
- Deep learning in **TensorFlow/Keras** (fine-tune a pretrained transformer via Hugging Face `transformers` with TF weights, or train your own encoder)
- Strong classical baseline mandatory (e.g., TF-IDF + linear model) on identical splits
- Selective prediction: the service must expose a confidence-based abstention mechanism calibrated so that ≤15% of test messages are abstained
- Inference: batch of 32 messages in < 1 s on CPU, or document the GPU requirement honestly
- Model card documenting data, intended use, metrics per intent group, and known failure modes

## Deliverables
1. Training + evaluation code as modules; experiment configs in files, not hardcoded
2. Comparison report: classical baseline vs transformer — macro-F1, per-intent F1 distribution, and error overlap analysis (do they fail on the same messages?)
3. Calibration analysis (reliability diagram) and the abstention threshold derivation
4. Recall report for the three compliance-critical intents, with mitigation if below target
5. A `POST /classify` demo endpoint returning intent, confidence, and `route_to_human` flag
6. Model card + 1-page executive summary answering: what does the transformer buy us over the baseline, in agent-minutes per month?

## Evaluation Rubric
| Criterion | Bar |
|---|---|
| Macro-F1 (acceptance set, non-abstained) | ≥ 0.93 transformer; baseline reported |
| Compliance intents recall | ≥ 0.98 (after abstention routing) |
| Abstention budget respected | ≤ 15% of messages |
| Calibration | Reliability diagram + ECE reported; threshold justified |
| Honest comparison | Baseline given a fair fight (tuned, same data); verdict in business units |

---

# Deutsche Übersetzung

# 20 — Projektauftrag: Dienst zur Klassifikation von Bankanfragen

Schwierigkeit: ⚫ Fortgeschritten / Portfolio | Thema: NLP mit Transformern in TensorFlow

---

## Projektauftrag

**Von:** VP Customer Operations, FinServe Bank
**An:** NLP Engineering
**Betreff:** Intent-Router für das Support-Postfach

Kundennachrichten müssen automatisch an das richtige Team weitergeleitet werden. Wir verwenden die BANKING77-Taxonomie mit 77 detaillierten Absichten wie `card_arrival`, `exchange_rate` oder `lost_or_stolen_card`. Eine falsche Weiterleitung kostet ungefähr vier Minuten Bearbeitungszeit. Bei unsicheren Nachrichten soll das Modell an eine menschliche Triage weiterleiten.

## Geschäftlicher Kontext
- Etwa 50.000 Nachrichten pro Monat; die Triage kann **höchstens 15 %** des Volumens bearbeiten.
- Einige Absichten sind geschäftskritisch. Falsche Weiterleitungen bei `lost_or_stolen_card`, `compromised_card` oder `cancel_transfer` haben Compliance-Folgen und benötigen nahezu perfekten Recall.
- Die Leitung erwartet einen Vergleich mit einer starken klassischen Baseline und eine Quantifizierung des Nutzens der zusätzlichen Transformer-Komplexität.

## Daten
**BANKING77** enthält 13.083 Kundenanfragen in 77 Klassen, davon 10.003 für Training und 3.080 für Test. Der Datensatz ist über Hugging Face Datasets als `banking77` oder über das PolyAI-GitHub-Repository verfügbar. Der offizielle Testsatz dient als Abnahmesatz und darf einmal mit dem endgültigen System ausgewertet werden.

## Technische Vorgaben
- Deep Learning in **TensorFlow/Keras**, entweder durch Feinabstimmung eines vortrainierten Transformers mit TensorFlow-Gewichten aus Hugging Face `transformers` oder durch einen eigenen Encoder
- Verbindliche starke klassische Baseline, etwa TF-IDF plus lineares Modell, auf identischer Aufteilung
- Selektive Vorhersage mit vertrauensbasierter Enthaltung, so kalibriert, dass höchstens 15 % der Testnachrichten abgelehnt und an Menschen weitergeleitet werden
- CPU-Inferenz für einen Batch von 32 Nachrichten in unter einer Sekunde oder ehrliche Dokumentation des GPU-Bedarfs
- Modellkarte zu Daten, Einsatzzweck, Metriken je Intent-Gruppe und bekannten Fehlerarten

## Liefergegenstände
1. Trainings- und Bewertungscode als Module sowie Experimentkonfigurationen in Dateien statt fest im Code
2. Vergleich von klassischer Baseline und Transformer anhand von Macro-F1, Verteilung des F1 je Intent und Überschneidung ihrer Fehler
3. Kalibrierungsanalyse mit Reliability-Diagramm und Herleitung des Enthaltungsschwellenwerts
4. Recall-Bericht für die drei Compliance-kritischen Intents samt Gegenmaßnahmen bei Unterschreitung des Ziels
5. Demonstrationsendpunkt `POST /classify`, der Intent, Vertrauen und das Flag `route_to_human` zurückgibt
6. Modellkarte und einseitige Management-Zusammenfassung zum monatlichen Nutzen des Transformers gegenüber der Baseline in eingesparten Agentenminuten

## Bewertungskriterien
| Kriterium | Anforderung |
|---|---|
| Macro-F1 im Abnahmesatz ohne abgelehnte Fälle | ≥ 0,93 für den Transformer; Baseline dokumentiert |
| Recall der Compliance-Intents | ≥ 0,98 nach Weiterleitung unsicherer Fälle |
| Enthaltungsbudget | ≤ 15 % der Nachrichten |
| Kalibrierung | Reliability-Diagramm und ECE dokumentiert; Schwellenwert begründet |
| Ehrlicher Vergleich | Faire, abgestimmte Baseline auf denselben Daten; Urteil in Geschäftseinheiten |
