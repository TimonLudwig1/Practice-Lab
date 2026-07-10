# Modul 04 — Machine Learning 1

**Worum geht es?** Jetzt wird das Lernen aus Daten systematisch: Wie baut, bewertet und verbessert man Vorhersagemodelle — ehrlich, reproduzierbar und ohne sich selbst zu betrügen? Du lernst die wichtigsten Modellfamilien des klassischen Supervised Learning (kNN, lineare Modelle mit Regularisierung, Entscheidungsbäume, Ensembles), vor allem aber das **Handwerk drumherum**: Validierung, Metriken, Pipelines, Hyperparameter-Tuning. Das ist das Modul, dessen Inhalte in der Praxis am häufigsten gebraucht werden.

**Vorkenntnisse:** Module 02/03 (pandas, Train/Test-Gedanke, Regression, Leakage). Modul 01 (Naive Bayes) hilft.

**Vorher zu machen:** Modul 01–03.

---

## Lernziele

Nach diesem Modul kannst du:

- den Kern des Supervised Learning präzise formulieren (Hypothese, Verlustfunktion, Generalisierung) und **Overfitting/Underfitting** über den **Bias-Variance-Tradeoff** erklären,
- Modelle korrekt evaluieren: Train/Validation/Test, **k-fache Kreuzvalidierung**, und begründen, warum man den Testsatz nur einmal anfasst,
- die richtigen **Metriken** wählen und interpretieren (Confusion Matrix, Precision/Recall/F1, ROC-AUC; MAE/RMSE/$R^2$) — insbesondere bei **unausgewogenen Klassen**,
- **kNN, logistische Regression (mit Regularisierung), Entscheidungsbäume, Random Forests und Gradient Boosting** erklären (wie lernen sie? wann nutzt man was?) und mit scikit-learn anwenden,
- saubere **Pipelines** bauen (Skalierung, One-Hot-Encoding), die Leakage konstruktiv verhindern,
- **Hyperparameter** systematisch tunen (Grid/Random Search mit CV) und **Lernkurven** lesen,
- Modelle inspizieren: Feature Importances, Permutation Importance — und deren Fallstricke.

---

## 1. Grundlagen (Basics)

### 1.1 Was heißt „Lernen" formal?

Gegeben: Trainingsdaten $(x_1, y_1), \dots, (x_n, y_n)$ aus einer unbekannten Verteilung. Gesucht: eine Funktion $h$ (Hypothese) aus einer Modellfamilie $\mathcal{H}$, die den **erwarteten Fehler auf neuen Daten** minimiert — nicht den Trainingsfehler! Formal minimiert man ersatzweise den Trainingsverlust

$$\hat{h} = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_i L(h(x_i), y_i) \; (+ \text{Regularisierung})$$

mit einer **Verlustfunktion** $L$ (z. B. quadratischer Fehler für Regression, Log-Loss für Klassifikation). Die ganze Kunst des Machine Learning liegt in der Lücke zwischen Trainingsverlust und echtem Fehler — der **Generalisierungslücke**.

**Die drei Zutaten jedes ML-Verfahrens:** (1) Modellfamilie $\mathcal{H}$ (welche Funktionen sind überhaupt möglich?), (2) Verlustfunktion (was heißt „gut"?), (3) Optimierungsverfahren (wie findet man das beste $h$?). Wenn du ein neues Verfahren kennenlernst, stelle immer diese drei Fragen.

### 1.2 Das erste Modell: k-Nearest Neighbors (kNN)

**Idee:** Um einen neuen Punkt zu klassifizieren, schau dir die $k$ nächstgelegenen Trainingspunkte an und nimm deren Mehrheitsklasse (bzw. für Regression: deren Mittelwert). Kein „Training" im eigentlichen Sinn — das Modell *ist* der Datensatz.

kNN ist didaktisch Gold, weil man an ihm alles Wichtige sieht:

- **Hyperparameter $k$ steuert die Komplexität**: $k = 1$ → extrem flexible, zackige Entscheidungsgrenze (jeder Trainingspunkt bekommt recht — Overfitting-Gefahr). $k = n$ → immer die häufigste Klasse (maximal starr — Underfitting). Dazwischen liegt das Optimum.
- **Skalierung ist Pflicht**: Distanzen mischen Einheiten! Ohne Standardisierung dominiert das Merkmal mit den größten Zahlen (Gramm schlägt Millimeter — dieselbe Lektion wie bei PCA in Modul 03).
- **Fluch der Dimensionalität**: In hohen Dimensionen sind alle Punkte ungefähr gleich weit voneinander entfernt — „Nähe" verliert ihre Bedeutung. Deshalb skaliert kNN schlecht mit vielen Features.

### 1.3 Overfitting, Underfitting und der Bias-Variance-Tradeoff

Der erwartete Vorhersagefehler eines Modells lässt sich (für quadratischen Verlust) zerlegen:

$$\text{Fehler} = \underbrace{\text{Bias}^2}_{\text{systematisch daneben}} + \underbrace{\text{Varianz}}_{\text{instabil je nach Trainingsdaten}} + \underbrace{\sigma^2}_{\text{irreduzibles Rauschen}}$$

- **Hoher Bias (Underfitting):** Das Modell ist zu einfach für die Wahrheit (Gerade für eine Parabel). Symptom: Trainings- *und* Testfehler hoch, nah beieinander.
- **Hohe Varianz (Overfitting):** Das Modell ist so flexibel, dass es das Rauschen der konkreten Stichprobe mitlernt. Symptom: Trainingsfehler klein, Testfehler deutlich größer.

> **Intuition:** Stell dir vor, du trainierst dasselbe Modell auf 100 verschiedenen Stichproben. Bias = wie weit liegt der *Durchschnitt* der 100 Modelle von der Wahrheit weg. Varianz = wie stark *streuen* die 100 Modelle untereinander. Einfache Modelle: alle ähnlich, alle ähnlich falsch. Komplexe Modelle: im Mittel richtig, einzeln wild verschieden.

**Stellschrauben für die Komplexität**: $k$ bei kNN, Baumtiefe, Polynomgrad, Regularisierungsstärke, … Die richtige Einstellung findet man nie auf den Trainingsdaten (dort gewinnt immer das komplexeste Modell), sondern per **Validierung**.

### 1.4 Ehrliche Evaluation: Train / Validation / Test und Kreuzvalidierung

Drei Rollen, die man nicht vermischen darf:

1. **Training**: Parameter lernen.
2. **Validierung**: Hyperparameter wählen, Modelle vergleichen.
3. **Test**: EINMAL ganz am Ende die finale Leistung messen.

Warum reicht Train/Test nicht? Wer 50 Modellvarianten am Testsatz vergleicht, hat den Testsatz zum Validierungssatz gemacht — die beste der 50 Zahlen ist nach oben verzerrt (Multiple-Testing-Logik aus Modul 02!).

**k-fache Kreuzvalidierung (CV)** nutzt die Daten effizienter: Teile die Trainingsdaten in $k$ (üblich 5 oder 10) Blöcke; trainiere $k$-mal auf $k{-}1$ Blöcken und validiere auf dem übrigen; mittle die Ergebnisse. Vorteile: jede Beobachtung wird einmal validiert, und man bekommt zusätzlich die **Streuung** der Schätzung (eine Modellwahl, die auf ±0,5 % Unterschied bei ±2 % Streuung beruht, ist Kaffeesatzleserei). Bei Klassifikation **stratifiziert** (Klassenanteile in jedem Fold gleich); bei Zeitreihen zeitlich geordnet (Modul 03!); bei gruppierten Daten (mehrere Zeilen pro Patient!) gruppenweise splitten — sonst Leakage.

### 1.5 Metriken: Woran misst man „gut"?

**Klassifikation — die Confusion Matrix** ist die Basis von allem:

|  | vorhergesagt + | vorhergesagt − |
|--|--|--|
| **echt +** | True Positive (TP) | False Negative (FN) |
| **echt −** | False Positive (FP) | True Negative (TN) |

- **Accuracy** $= (TP{+}TN)/n$ — irreführend bei unausgewogenen Klassen (Modul 02: 87 % „ham" raten!)
- **Precision** $= TP/(TP{+}FP)$ — wenn Alarm, wie oft zurecht? (Kosten von Fehlalarmen)
- **Recall** $= TP/(TP{+}FN)$ — wie viel vom Echten gefunden? (Kosten des Übersehens)
- **F1** = harmonisches Mittel aus Precision und Recall — Kompromisszahl
- **ROC-Kurve & AUC**: Fast alle Klassifikatoren geben Scores/Wahrscheinlichkeiten aus; die Entscheidungsschwelle ist frei wählbar. Die ROC-Kurve zeigt True-Positive-Rate gegen False-Positive-Rate über alle Schwellen; die Fläche darunter (**AUC**) misst schwellenunabhängig, wie gut das Modell positiv von negativ trennt (0,5 = Münzwurf, 1,0 = perfekt). Bei starker Unausgewogenheit ist die **Precision-Recall-Kurve** aussagekräftiger.

**Welche Metrik?** Ist keine Statistikfrage, sondern eine **Kostenfrage der Anwendung**: Beim Spamfilter sind False Positives teuer (echte Mail weg → Precision hoch halten), beim Krebs-Screening False Negatives (Fall übersehen → Recall hoch halten). Diese Abwägung *vor* dem Modellieren festlegen.

**Regression:** MAE (robust, direkt interpretierbar), RMSE (bestraft große Fehler stärker), $R^2$ (erklärte Varianz, Vergleich gegen „immer Mittelwert raten").

---

## 2. Aufbau (Intermediate)

### 2.1 Lineare Modelle mit Regularisierung

Lineare/logistische Regression kennst du aus Modul 03. Neu ist die **Regularisierung**: ein Strafterm gegen zu große Koeffizienten,

$$\min_\beta \; \text{Verlust}(\beta) + \alpha \cdot \text{Strafe}(\beta)$$

- **Ridge (L2)**: Strafe $\sum \beta_j^2$ — schrumpft alle Koeffizienten gleichmäßig Richtung 0. Erste Wahl bei vielen korrelierten Features (stabilisiert die Multikollinearität aus Modul 03).
- **Lasso (L1)**: Strafe $\sum |\beta_j|$ — drückt unwichtige Koeffizienten auf *exakt* 0, macht also automatisch Feature-Selektion.
- $\alpha$ steuert den Bias-Variance-Tradeoff direkt: $\alpha = 0$ → OLS (volle Varianz), $\alpha \to \infty$ → Nullmodell (voller Bias). Gewählt wird $\alpha$ per CV.
- **Skalierung Pflicht** (die Strafe vergleicht Koeffizientengrößen — unfair bei verschiedenen Einheiten). In sklearn heißt der Regularisierungsparameter bei `LogisticRegression` übrigens `C` = $1/\alpha$ (groß = wenig Regularisierung) — beliebte Verwechslung.

### 2.2 Entscheidungsbäume

**Idee:** Stelle nacheinander Ja/Nein-Fragen an die Features („temp > 12 °C?"), bis eine Entscheidung fällt. Gelernt wird gierig: An jedem Knoten wird der Split gewählt, der die Klassen am besten trennt — gemessen an der Verunreinigung (**Gini-Index** $1 - \sum_c p_c^2$ oder Entropie).

- **Stärken:** interpretierbar (der Baum *ist* die Erklärung), keine Skalierung nötig, kategoriale und numerische Features gemischt, nichtlineare Zusammenhänge und Interaktionen automatisch.
- **Schwäche:** Einzelbäume overfitten brutal (voll ausgewachsen lernen sie jede Trainingszeile auswendig) und sind **instabil** — kleine Datenänderung, ganz anderer Baum (hohe Varianz!). Gegenmittel: Tiefe begrenzen (`max_depth`), Mindestblattgröße — oder besser: Ensembles.

### 2.3 Ensembles: Weisheit der vielen Modelle

**Random Forest (Bagging):** Trainiere viele (z. B. 300) tiefe Bäume, jeder auf einer Bootstrap-Stichprobe (Modul 03!) der Daten und mit zufälliger Feature-Auswahl pro Split; mittle die Vorhersagen. Die Zufälligkeit **dekorreliert** die Bäume, und der Durchschnitt vieler dekorrelierter Hochvarianz-Modelle hat **niedrige Varianz** bei fast unverändertem Bias. Ergebnis: robust, wenig tuningbedürftig, hervorragende Standardwahl für Tabellendaten.

**Gradient Boosting:** Baue flache Bäume **nacheinander**, jeder korrigiert die Fehler (genauer: den Gradienten des Verlusts) der bisherigen Summe:

$$F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x), \qquad h_m \approx -\nabla L(F_{m-1})$$

Mit klener Lernrate $\eta$ und genügend Bäumen entsteht ein sehr genaues Modell — Gradient-Boosting-Varianten (XGBoost, LightGBM, sklearns `HistGradientBoosting`) **gewinnen auf Tabellendaten fast alles** (auch gegen Deep Learning!). Preis: mehr Hyperparameter, overfittet ohne Sorgfalt (Lernrate, Baumzahl, Tiefe per CV/Early Stopping).

**Bagging vs. Boosting in einem Satz:** Bagging mittelt unabhängige komplexe Modelle (senkt Varianz), Boosting addiert abhängige einfache Modelle (senkt Bias).

### 2.4 Pipelines: Preprocessing ohne Selbstbetrug

Reale Daten brauchen Vorverarbeitung: numerische Features skalieren (`StandardScaler`), kategoriale one-hot-encoden (`OneHotEncoder`), fehlende Werte imputieren. Die Leakage-Regel aus Modul 03 („alles Lernbare nur auf Train lernen") gilt für **jeden** dieser Schritte — auch der Mittelwert fürs Imputieren ist gelernt!

Die sklearn-**`Pipeline`** (+ `ColumnTransformer` für „Spalte A skalieren, Spalte B encoden") erzwingt das automatisch: `fit` lernt alle Schritte nur auf den Trainingsdaten des jeweiligen (CV-)Splits, `transform` wendet sie auf Validierung/Test an. **Pipelines sind keine Kür, sondern der einzige Weg, CV + Preprocessing korrekt zu kombinieren.**

### 2.5 Hyperparameter-Tuning

- **Grid Search**: alle Kombinationen eines Rasters per CV durchprobieren (`GridSearchCV`). Vollständig, aber exponentiell teuer.
- **Random Search**: zufällige Kombinationen — bei gleichem Budget meist besser, weil wichtige Parameter feiner abgetastet werden (klassisches Ergebnis von Bergstra & Bengio).
- Danach: finale Leistung **auf dem unberührten Testsatz** messen — die beste CV-Zahl selbst ist leicht optimistisch (sie wurde ja ausgewählt).
- **Lernkurven** (Leistung vs. Trainingsmenge) beantworten die Frage „brauche ich mehr Daten oder ein anderes Modell?": Konvergieren Train- und Validierungskurve auf niedrigem Niveau → Bias-Problem (mehr Daten helfen nicht). Große Lücke → Varianz-Problem (mehr Daten oder mehr Regularisierung helfen).

### 2.6 Unausgewogene Klassen

Bei 1:99-Verhältnissen (Betrug, Krankheit, Kündigung) braucht es Sonderbehandlung:

1. **Richtige Metrik** (PR-AUC, Recall bei fixierter Precision — nie Accuracy),
2. **Schwelle verschieben**: Die 0,5-Schwelle ist Konvention, nicht Gesetz — wähle sie nach Kostenabwägung auf der Validierungsmenge,
3. **Klassengewichte** (`class_weight="balanced"`): Fehler auf der seltenen Klasse zählen mehr,
4. Resampling (Over-/Undersampling, SMOTE) — mit Vorsicht und nur auf den Trainingsdaten.

---

## 3. Advanced-Themen

### 3.1 Modelle verstehen: Importances und ihre Fallen

- **Impurity-based Feature Importance** (Random Forest Standard): misst, wie viel ein Feature die Verunreinigung reduziert. Fallen: bevorzugt Features mit vielen möglichen Splitwerten; auf Trainingsdaten berechnet (Overfitting färbt ab).
- **Permutation Importance**: Verwürfle ein Feature in den *Validierungsdaten* und miss den Leistungseinbruch — modellagnostisch und ehrlicher. Falle: Bei korrelierten Features teilt sich die Wichtigkeit unvorhersehbar auf („mein Feature ist unwichtig" kann heißen „sein Zwilling trägt schon dieselbe Information").
- **Partial Dependence Plots**: mittlerer Vorhersageverlauf, wenn man ein Feature durch seinen Wertebereich schiebt — zeigt die *Form* des Zusammenhangs (z. B. die Wohlfühltemperatur-Parabel aus Modul 03).
- Ausblick: SHAP-Werte (spieltheoretisch fundierte Beitragszerlegung pro Vorhersage) — Standard in der Praxis, Details sprengen dieses Modul.

**Warnung:** Alle diese Werkzeuge beschreiben, **was das Modell nutzt** — nicht, was in der Welt kausal wirkt.

### 3.2 Kalibrierung: Sind 80 % wirklich 80 %?

Viele Modelle geben „Wahrscheinlichkeiten" aus, die keine sind: Von allen Vorhersagen mit Score 0,8 sollten ~80 % tatsächlich positiv sein (**Kalibrierung**). Random Forests sind oft zu vorsichtig (Scores gestaucht Richtung 0,5), Boosting oft zu selbstsicher; logistische Regression ist von Haus aus meist gut kalibriert. Prüfen mit dem **Reliability Diagram** (vorhergesagte vs. beobachtete Häufigkeit in Score-Bins), reparieren mit Platt Scaling / Isotonic Regression (`CalibratedClassifierCV`). Wichtig überall dort, wo Scores in Entscheidungen mit Schwellen oder Erwartungswerten eingehen (Risiko-Scores, Preise).

### 3.3 Was dieses Modul bewusst auslässt

- **Neuronale Netze & Deep Learning** → Modul 05 (Machine Learning 2). Merksatz bis dahin: Auf strukturierten Tabellendaten sind Gradient-Boosting-Modelle meist mindestens ebenbürtig; Deep Learning dominiert bei Rohdaten (Bilder, Audio, Text).
- **Unsupervised Learning** (Clustering, Dichteschätzung) → Modul 05.
- **SVMs & Kernel-Methoden**: heute seltener erste Wahl, aber konzeptionell wichtig (Maximum-Margin, Kernel-Trick) — Kurzkontakt im Medium-Projekt, Theorie in Modul 06/07.
- **AutoML**: automatisiertes Tuning/Modellwahl — nützlich, ersetzt aber keine der ehrlichen Evaluationsregeln dieses Moduls.

---

## 4. Zusammenfassung / Cheat-Sheet

**Grundgerüst**
- Lernen = Verlust auf Train minimieren, Ziel = Fehler auf Neuem; die Lücke heißt Generalisierung
- Fehler = Bias² + Varianz + Rauschen; einfach = Bias, flexibel = Varianz
- Symptome: Train schlecht ≈ Test schlecht → Underfitting · Train gut, Test schlecht → Overfitting

**Evaluation**
- Train (Parameter) / Validation (Wahl) / Test (einmal!); k-fach-CV stratifiziert; Zeit → zeitlich, Gruppen → gruppenweise
- Klassifikation: Confusion Matrix; Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1; ROC-AUC (schwellenfrei); unausgewogen → PR-Kurve, nie Accuracy
- Metrik = Kostenfrage der Anwendung, vorab festlegen

**Modelle**
- kNN: k klein → Varianz, k groß → Bias; skalieren!; leidet in hohen Dimensionen
- Ridge (L2) schrumpft, Lasso (L1) selektiert; α per CV; skalieren; sklearn-`C` = 1/α
- Baum: gierige Splits nach Gini/Entropie; interpretierbar, aber instabil → Tiefe begrenzen
- Random Forest: Bagging + Feature-Zufall = Varianz runter; robuste Standardwahl
- Gradient Boosting: sequentielle Fehlerkorrektur, Lernrate klein; Tabellendaten-Champion
- Naive Bayes (Modul 01) und logistische Regression: schnelle, starke Baselines

**Workflow**
- Pipeline + ColumnTransformer: Preprocessing wird im CV mitgelernt → keine Leakage
- GridSearchCV/RandomizedSearchCV; danach EINMAL Test
- Lernkurve: Lücke = Varianz (Daten/Regularisierung), kein Fortschritt = Bias (Modell wechseln)
- Unausgewogen: Metrik, Schwelle, class_weight — in dieser Reihenfolge

**Interpretation**
- Permutation Importance > Impurity Importance; Korrelation verwässert beide
- PDP für Formen; Kalibrierung prüfen, wenn Scores Entscheidungen steuern
- Modell-Nutzung ≠ Kausalität

---

## 5. Selbsttest

<details><summary><b>1. Dein Modell: Trainingsfehler 2 %, Testfehler 19 %. Diagnose und drei mögliche Gegenmittel?</b></summary>

Klassisches **Overfitting** (hohe Varianz): Das Modell lernt Rauschen der Trainingsdaten. Gegenmittel: Komplexität senken (kleinerer Baum, größeres $k$, stärkere Regularisierung $\alpha$↑/`C`↓), mehr Trainingsdaten, Ensembles mit Bagging (Random Forest), Feature-Menge reduzieren. Und prüfen, ob nicht Leakage die Trainingsleistung schönt.
</details>

<details><summary><b>2. Warum darf man Hyperparameter nicht auf dem Testsatz wählen — und was ist die Lösung, wenn Daten knapp sind?</b></summary>

Wer viele Varianten am Testsatz vergleicht und die beste nimmt, wählt teilweise das Zufallsrauschen des Testsatzes aus — die berichtete Leistung ist nach oben verzerrt und der Testsatz „verbraucht". Lösung: Kreuzvalidierung auf den Trainingsdaten für alle Entscheidungen; der Testsatz wird genau einmal am Ende benutzt.
</details>

<details><summary><b>3. Betrugserkennung: 0,5 % Positive. Modell A: Accuracy 99,5 %. Modell B: Accuracy 97 %, Recall 80 %, Precision 12 %. Welches ist vermutlich nützlicher und warum?</b></summary>

Modell A erreicht 99,5 % vermutlich durch „nie Betrug sagen" (Recall ≈ 0) — nutzlos. Modell B findet 80 % der Betrugsfälle; die niedrige Precision (12 %) heißt viele Fehlalarme, was je nach Prüfkosten akzeptabel sein kann. Bewerten muss man mit PR-Metriken und den echten Kosten von FP (Prüfung) vs. FN (Schaden) — nicht mit Accuracy.
</details>

<details><summary><b>4. Warum braucht kNN Skalierung, ein Entscheidungsbaum aber nicht?</b></summary>

kNN vergleicht **Distanzen** über alle Features hinweg — ohne Skalierung dominiert das Feature mit der größten Zahlenskala die Distanz. Ein Baum betrachtet jedes Feature **einzeln** und fragt nur „Wert > Schwelle?" — jede monotone Transformation (also auch Skalierung) verändert die möglichen Splits nicht.
</details>

<details><summary><b>5. Erkläre in je einem Satz, warum Random Forest die Varianz senkt und Gradient Boosting den Bias.</b></summary>

Random Forest mittelt viele durch Bootstrap und Feature-Zufall **dekorrelierte**, einzeln overfittende Bäume — der Durchschnitt unabhängiger Schätzungen streut weniger als jede einzelne. Gradient Boosting addiert nacheinander flache (stark verzerrte) Bäume, von denen jeder die verbleibenden systematischen Fehler der Summe korrigiert — die Summe kann Strukturen abbilden, die jedes Einzelmodell verfehlt.
</details>

<details><summary><b>6. Du skalierst die Daten mit StandardScaler auf dem GANZEN Datensatz und machst danach CV. Was ist falsch, wie schlimm ist es, und was ist die saubere Lösung?</b></summary>

Der Scaler hat Mittel/Streuung auch aus den späteren Validierungs-Folds gelernt — Leakage. Bei purem Skalieren ist der Effekt meist klein, aber das Prinzip skaliert: Bei Imputation, Feature-Selektion oder Target-Encoding auf dem Gesamtdatensatz wird die CV-Schätzung massiv geschönt. Sauber: Preprocessing in eine `Pipeline`, die im CV pro Fold nur auf dem Trainingsteil gefittet wird.
</details>

<details><summary><b>7. Die Lernkurve zeigt: Trainings- und Validierungsleistung liegen eng beieinander, beide schlecht, und bewegen sich ab 10.000 Beispielen nicht mehr. Hilft Datensammeln?</b></summary>

Nein — das ist ein **Bias-Problem**: Das Modell ist zu einfach (oder die Features tragen die Information nicht). Mehr gleichartige Daten verschieben nichts. Abhilfe: ausdrucksstärkeres Modell, bessere Features (Feature Engineering!), weniger Regularisierung. Datensammeln hilft im umgekehrten Fall: große Lücke zwischen Train und Validierung (Varianz).
</details>

<details><summary><b>8. Random Forest: Feature „Kunden-ID" hat die höchste Impurity-Importance. Was ist passiert?</b></summary>

Eine (Quasi-)ID hat extrem viele eindeutige Werte — der Baum kann damit Trainingszeilen praktisch auswendig adressieren; Impurity-Importance belohnt genau solche Split-freudigen Features. Das ist Overfitting + Metrik-Artefakt, keine echte Wichtigkeit: ID entfernen (generell: keine Identifikatoren als Features) und Permutation Importance auf Validierungsdaten verwenden — dort fiele die ID auf ~0.
</details>

<details><summary><b>9. Warum ist die 0,5-Schwelle bei Klassifikationsscores oft die falsche — und wie wählt man besser?</b></summary>

0,5 ist nur bei symmetrischen Fehlkosten und kalibrierten Scores sinnvoll. Sind FP und FN unterschiedlich teuer (fast immer) oder die Klassen unausgewogen, gehört die Schwelle dorthin, wo die erwarteten Kosten minimal sind bzw. die geforderte Precision/Recall erreicht wird — bestimmt auf der Validierungsmenge (z. B. über die PR-Kurve), nicht auf dem Test.
</details>

<details><summary><b>10. Ein Kollege sagt: „Der Random Forest hat gezeigt, dass Rauchen das wichtigste Feature für Krankheit X ist — also verursacht Rauchen X." Zwei Einwände.</b></summary>

(1) Importance misst nur, was **das Modell zur Vorhersage nutzt** — ein Feature kann wichtig sein, weil es mit der wahren Ursache korreliert (Confounding, Modul 02). (2) Bei korrelierten Features verteilt sich Wichtigkeit willkürlich; ein kausal irrelevanter, aber gut gemessener Proxy kann die echte Ursache „verdrängen". Kausalaussagen brauchen Studiendesign (Randomisierung, Adjustierung), nicht Feature-Rankings.
</details>

---

## 6. Literatur & Quellen

**Lehrbücher**

- **James, Witten, Hastie, Tibshirani — „An Introduction to Statistical Learning" (ISLR), 2. Aufl.** — deckt fast das ganze Modul ab (Kap. 2 Bias/Variance, 4 Klassifikation, 5 CV/Bootstrap, 6 Regularisierung, 8 Bäume/Ensembles). **Kostenlos**: https://www.statlearning.com. Es gibt eine Python-Ausgabe (ISLP). *(die Hauptempfehlung)*
- **Géron — „Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow", 3. Aufl.** — der Praxis-Klassiker; Teil 1 ist exakt dieses Modul. *(einsteigerfreundlich, kostenpflichtig)*
- **Hastie, Tibshirani, Friedman — „The Elements of Statistical Learning"** — die vertiefte Theorie-Referenz, **kostenlos** als PDF. *(vertiefend, erst nach ISLR)*

**Onlinekurse (kostenlos)**

- **Andrew Ng — Machine Learning Specialization** (Coursera, Audit kostenlos) — die didaktisch geschliffenste Einführung *(einsteigerfreundlich)*
- **scikit-learn MOOC** (inria.github.io/scikit-learn-mooc) — von den sklearn-Entwicklern, mit exakt den Pipeline-/CV-Praktiken dieses Moduls *(einsteigerfreundlich, sehr empfohlen als Begleitung)*
- **StatQuest** (YouTube): Playlists zu Bias/Variance, ROC/AUC, Random Forest, Gradient Boosting *(einsteigerfreundlich)*

**Interaktiv / Blogposts (kostenlos)**

- *MLU-Explain* (mlu-explain.github.io): interaktive Visualisierungen zu Bias-Variance, ROC, Random Forest, Cross-Validation — Pflichtbesuch
- *A visual introduction to machine learning* (r2d3.us): Entscheidungsbäume visuell erklärt
- scikit-learn User Guide, Kapitel „Common pitfalls and recommended practices" — Leakage & Co. offiziell

**Vertiefend**

- Bergstra & Bengio (2012): *Random Search for Hyper-Parameter Optimization* (JMLR)
- Grinsztajn et al. (2022): *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS)

---

**Nächster Schritt:** `projekte/01-basic/` (kNN von Hand + Bias-Variance sichtbar machen) → `projekte/02-medium/` (das ehrliche Modellrennen: Pipelines, CV, Tuning) → `projekte/03-final/` (Einkommensvorhersage auf echten Zensusdaten: unausgewogen, gemischte Typen, Schwellenwahl).
