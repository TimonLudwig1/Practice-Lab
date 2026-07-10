# Projekt 03 (final) — Spamfilter mit Naive Bayes auf echten SMS-Daten

**Format:** Jupyter Notebook (`spamfilter.ipynb`) + Download-Skript.
**Warum dieses Format?** Das ist ein Daten-Projekt: laden, explorieren, modellieren, auswerten — genau der Workflow, für den Notebooks gemacht sind, und zugleich eine Vorschau auf die Data-Science- und ML-Module.

**Daten: echt.** Wir nutzen die **SMS Spam Collection** (UCI Machine Learning Repository, Almeida & Hidalgo 2011): 5.574 echte SMS, davon ~13 % Spam. Echte Daten sind hier bewusst gewählt — der Praxisbezug (Spamfilterung war *die* Erfolgsgeschichte von Naive Bayes) ist genau der Punkt dieses Abschlussprojekts. Die Datei (~470 KB) wird per Skript geladen und nicht ins Repo committet.

## Ziel

Du übersetzt die Bayes-Regel aus dem Skript eigenhändig in einen funktionierenden Spam-Klassifikator (inkl. Laplace-Glättung und Log-Wahrscheinlichkeiten), bewertest ihn mit den richtigen Metriken und vergleichst ihn mit scikit-learn. Referenzergebnis: **~98,6 % Accuracy** — identisch mit der scikit-learn-Implementierung.

## Vorwissen

- Modul-Skript Abschnitte 2.4 (Bayes, Naive Bayes) und 2.5 (Supervised Learning, Generalisierung)
- Projekte 01 und 02
- Als Final-Projekt schreibst du **allen Code selbst** — das Notebook enthält nur noch die Schritt-für-Schritt-Anleitung, Formeln und Selbstchecks (keine vorgeschriebenen Code-Zeilen). pandas lernst du systematisch erst in Modul 02; die wenigen nötigen pandas-Aufrufe (z. B. `read_csv`) sind darum als **Prosa-Hinweis** in den Aufgaben beschrieben, sodass du sie nachbauen kannst.

## Aufgaben

1. **Daten holen** (einmalig, im Ordner `03-final`, venv aktiv):
   ```bash
   python daten/download_daten.py
   ```
2. Notebook `spamfilter.ipynb` öffnen und von oben durcharbeiten. Jede Code-Zelle ist **leer** — du füllst sie selbst; die Markdown-Zelle darüber sagt jeweils, was zu tun ist (mit Formeln, Hinweisen und Selbstchecks). Der Weg: Daten laden & explorieren → Train/Test-Split → `tokenisiere(text)` → Naive Bayes von Hand (`trainiere`, `log_wort_wkt`, `log_posterior`, `klassifiziere`) → Auswertung → gelernte Wörter → scikit-learn-Vergleich.
3. Auswertung ausführen und **interpretieren**: Schau dir die Confusion Matrix genau an — welche Fehlerart ist bei einem Spamfilter schlimmer, und wie schlägt sich dein Modell da?
4. Abschnitt „Was hat das Modell gelernt?" und den scikit-learn-Vergleich ausführen.

## Was am Ende funktionieren soll

- Der Tokenizer-Mini-Test gibt `True` aus.
- Die Beispiel-SMS werden korrekt als `spam` / `ham` erkannt.
- Accuracy auf dem Test-Set **> 98 %** (Referenz: 0,9857; naive „immer ham"-Baseline: 0,8664).
- Dein Modell und `MultinomialNB` liegen praktisch gleichauf.

## Lösung

Vollständig ausgeführte Musterlösung: [`loesung/loesung.ipynb`](loesung/loesung.ipynb) — erst selbst probieren. Dort siehst du auch alle Referenz-Outputs (Confusion Matrix, Top-Spam-Wörter etc.).
