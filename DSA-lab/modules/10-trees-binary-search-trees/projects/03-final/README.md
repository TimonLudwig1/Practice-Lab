# Projekt 03-final: Decision Tree von Hand

## Ziel

In diesem Projekt wird aus einem allgemeinen Binärbaum ein lernendes Modell.
Der eigene `ScratchDecisionTreeClassifier` sucht numerische Schwellen, bewertet
sie mit Gini-Impurity und baut rekursiv einen Klassifikationsbaum. Anschließend
wird er unter identischen Tiefen- und Blattgrenzen mit
`sklearn.tree.DecisionTreeClassifier` verglichen.

Die Umsetzung besteht aus Python-Skripten statt aus einem Notebook. Dadurch
bleiben Datenaufbereitung, Modell, Experiment und Tests voneinander getrennt;
der erzeugte Baum ist importierbar und jeder Split kann reproduzierbar geprüft
werden.

## Dateien

- `../../data/generate_data.py`: reproduzierbarer Datengenerator mit festem Seed
- `../../data/decision_tree_train.csv`: 900 Trainingsbeispiele
- `../../data/decision_tree_test.csv`: 300 unberührte Testbeispiele
- `decision_tree.py`: Eigenimplementierung des Klassifikators
- `run_experiment.py`: fairer Vergleich mit scikit-learn
- `test_decision_tree.py`: Unit-, Struktur-, Daten- und Referenztests
- `output/`: Vergleichsmetriken, Feature Importances und lesbarer eigener Baum

## Intuition: Ein Baum stellt Ja/Nein-Fragen

Ein klassischer BST erhält eine feste Suchinvariante. Ein Decision Tree muss
seine Fragen erst aus Daten lernen. Ein innerer Knoten fragt beispielsweise:

```text
signal_x <= 0.44?
+-- ja:  nächste Frage oder Klasse 0
`-- nein: nächste Frage oder Klasse 1
```

Jede Frage zerlegt die am Knoten angekommenen Trainingszeilen in zwei Mengen.
Gute Fragen erzeugen möglichst reine Kinder, also Gruppen, in denen vorwiegend
nur noch eine Klasse vorkommt.

## Simulation: Vom Gini-Wert zum Split

Angenommen, ein Knoten enthält vier Labels `[0, 0, 1, 1]`. Beide Klassen haben
Anteil `1/2`:

```text
Gini(parent) = 1 - (1/2)^2 - (1/2)^2 = 0.5
```

Die Schwelle `x <= 1.5` teile sie in `[0, 0]` und `[1, 1]`. Beide Kinder sind
rein und haben Gini `0`:

```text
weighted_child_gini = 2/4 * 0 + 2/4 * 0 = 0
gain                = 0.5 - 0 = 0.5
```

Der Algorithmus sortiert für jedes Feature die Werte und prüft Schwellen nur
zwischen zwei verschiedenen Nachbarwerten. Die beste Kandidatin wird im Knoten
gespeichert. Dann startet derselbe Prozess rekursiv links und rechts.

## Rekursiver Aufbau

```text
build(rows, depth=0)
+-- best_split(rows)
+-- build(left_rows, depth=1)
|   +-- best_split(left_rows)
|   `-- ...
`-- build(right_rows, depth=1)
    +-- best_split(right_rows)
    `-- ...
```

Ein Aufruf wird zum Blatt, wenn mindestens eine Bedingung gilt:

- alle Labels sind gleich,
- `max_depth` ist erreicht,
- zu wenige Zeilen für Split oder Blätter sind vorhanden,
- kein Feature besitzt zwei verschiedene Werte,
- kein Split erreicht `min_impurity_decrease`.

Das Blatt sagt die häufigste Klasse voraus. Bei Gleichstand gewinnt
deterministisch das kleinere Label.

## Datensatz und fairer Vergleich

Der Generator erzeugt vier numerische Features. Drei davon bestimmen über eine
nichtlineare, aus mehreren Schwellen zusammengesetzte Regel das Label; `noise`
ist irrelevant. Sieben Prozent der Labels werden umgedreht, damit das Experiment
nicht künstlich perfekt ist. Der feste Seed `20260720` reproduziert Werte,
Labelrauschen und stratifizierten Split exakt.

Beide Modelle erhalten dieselben 900 Trainingszeilen und werden nur auf den 300
Testzeilen bewertet. Beide nutzen Gini, `max_depth=5` und
`min_samples_leaf=8`. `random_state` beeinflusst nur die Referenz und ist
ebenfalls fixiert.

Der vollständige Standardlauf ergibt:

| Modell | Train Accuracy | Test Accuracy | Tiefe | Blätter |
|---|---:|---:|---:|---:|
| Eigenbau | 92,67 % | 94,00 % | 5 | 27 |
| scikit-learn | 92,67 % | 94,00 % | 5 | 27 |

Beide Modelle treffen auf allen 300 Testzeilen dieselbe Vorhersage. Auch die
normalisierten Feature Importances stimmen in diesem Lauf bis auf sechs
Nachkommastellen überein: `signal_x=0,321048`, `signal_y=0,399505`,
`context=0,273048` und `noise=0,006400`. Das ist kein pauschales Versprechen für
andere Daten, sondern ein reproduzierbares Ergebnis der festgelegten Daten und
Hyperparameter.

## Komplexität

An jedem Knoten werden `m` Features für `n_node` Zeilen sortiert. Das kostet
`O(m * n_node log n_node)`. Bei einem ungefähr ausgeglichenen Baum ergibt sich
über alle Ebenen grob `O(m * n log² n)` für diese bewusst einfache
Implementierung. Produktionsbibliotheken reduzieren Kosten durch optimierte
Arrays, wiederverwendete Sortierungen und kompilierte Schleifen.

Vorhersage folgt genau einem Wurzel-Blatt-Pfad und kostet `O(h)` pro Zeile. Der
Baum selbst belegt `O(number_of_nodes)` Speicher; der rekursive Aufbau zusätzlich
`O(h)` Call Stack.

## Aufgabenstellung

1. Berechne Gini und Gain des Ein-Feature-Beispiels von Hand.
2. Verfolge `_best_split` für mindestens vier sortierte Werte. Markiere, warum
   gleiche Nachbarwerte keine zulässige Schwelle ergeben.
3. Führe den Generator und danach das Experiment aus. Vergleiche Testgenauigkeit,
   Tiefe, Blattzahl und Feature Importances beider Modelle.
4. Öffne `output/scratch_tree.txt` und verfolge drei Testzeilen manuell vom
   Wurzelknoten bis zur Vorhersage.
5. Variiere `max_depth` zwischen 1 und 8. Notiere Train-/Testgenauigkeit und
   erkläre Underfitting beziehungsweise Overfitting.
6. Erhöhe `min_samples_leaf` und beschreibe, wie sich Baumgröße und Robustheit
   verändern.

## Ausführen

Vom Projektordner aus:

```bash
python3 ../../data/generate_data.py
python3 run_experiment.py
python3 -m pytest -q
```

## Hinweise

- Der Testdatensatz darf niemals zur Split-Auswahl verwendet werden.
- Gini bewertet Klassenreinheit, nicht die numerische Streuung eines Features.
- `max_depth` zählt Kanten: Ein reines Wurzelblatt hat Tiefe `0`.
- Feature Importance ist die normalisierte Summe gewichteter Impurity-Decreases.
- Ähnliche Metriken bedeuten nicht zwingend identische Baumstrukturen; bei
  gleichwertigen Splits können Tie-Breaking-Regeln abweichen.

## Fertig, wenn …

- Gini-Impurity und gewichtete Split-Güte nachvollziehbar berechnet werden,
- der Eigenbaum Splits über alle Features sucht und rekursiv Kinder baut,
- Max-Tiefe, Mindestgrößen und reine beziehungsweise konstante Daten stoppen,
- Vorhersage, Accuracy, Baumtiefe, Blattzahl und Feature Importances stimmen,
- der Datensatz mit festem Seed reproduzierbar und stratifiziert erzeugt wird,
- Eigenbau und scikit-learn auf exakt denselben Train-/Testdaten laufen,
- Vergleichsdateien und lesbare Baumdarstellung erzeugt werden,
- Generator, Experiment, Syntaxprüfung und alle Tests fehlerfrei durchlaufen.
