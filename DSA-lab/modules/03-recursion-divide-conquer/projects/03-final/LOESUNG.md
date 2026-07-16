# Lösung und Auswertung

## Warum ein Verzeichnisbaum natürlich rekursiv ist

Ein Verzeichnis enthält Dateien und weitere Verzeichnisse. Jeder Unterordner ist
wieder ein vollständiges Problem derselben Art. Der Basisfall muss nicht als
besondere `if`-Zeile erscheinen: Ein Verzeichnis ohne Unterordner erzeugt keine
weiteren Aufrufe, und seine Schleife endet.

Die Fortschrittsgarantie beruht auf dem endlichen Baum und darauf, dass Symlinks
nicht verfolgt werden. Jeder rekursive Aufruf erhält einen echten Kindpfad. Ein
Zyklus kann im generierten Baum nicht entstehen.

## Rekursive Traversierung

`visit(directory, depth)` führt drei Schritte aus:

1. aktuelles Verzeichnis zählen,
2. reguläre Dateien im Verzeichnis aggregieren,
3. jeden echten Unterordner rekursiv besuchen.

Korrektheit lässt sich per Strukturinduktion zeigen:

- Ein Blattverzeichnis zählt sich und alle eigenen Dateien korrekt.
- Angenommen, alle Kindaufrufe analysieren ihre Teilbäume korrekt. Dann enthält
  der aktuelle Aufruf nach der Schleife eigene Dateien plus alle disjunkten
  Kindteilbäume und damit den gesamten aktuellen Teilbaum genau einmal.

Die maximale Zahl aktiver Frames ist `H + 1`, wenn die Wurzel Tiefe null hat.
Sehr tiefe fremde Bäume können deshalb Pythons Rekursionslimit überschreiten.

## Iterative Traversierung

Die Liste `stack` speichert genau die Verzeichnisse, die ein rekursiver Aufruf
später noch betreten würde. Ein Paar enthält Pfad und Tiefe. `pop()` wählt das
zuletzt abgelegte Verzeichnis und erzeugt Depth First Search.

Kinder werden in umgekehrter sortierter Reihenfolge abgelegt. Dadurch wird beim
späteren `pop()` das lexikografisch erste Kind zuerst verarbeitet. Für die
Korrektheit wäre die Reihenfolge egal, für reproduzierbare Beobachtung ist sie
hilfreich.

Die iterative Variante ist nicht „ohne Stack“. Sie macht denselben logischen
Zustand nur als normale Python-Liste explizit. Dadurch ist ihre Tiefe nicht an das
Rekursionslimit gebunden.

## Gemeinsame Aggregation

Beide Traversierungen verwenden dieselben Hilfsfunktionen für Verzeichnisse und
Dateien. Das reduziert das Risiko, versehentlich unterschiedliche Fachlogik zu
benchmarken.

Pro Datei werden konstant viele Aggregate aktualisiert:

- Gesamtbytes und Dateizahl,
- Bucket der aktuellen Tiefe,
- Bucket der kleingeschriebenen Extension,
- optional ein `FileRecord` bei Pattern-Treffer.

`_finish` sortiert Tiefe, Extension und Trefferpfade. Dadurch hängt
`TreeAnalysis` nicht von der Reihenfolge des Dateisystems oder der
Traversalstrategie ab und kann direkt auf Gleichheit geprüft werden.

## Pattern-Semantik

`fnmatchcase` wird sowohl auf den Basename als auch den relativen POSIX-Pfad
angewendet. Die Suche ist bewusst case-sensitive und plattformübergreifend
kanonisch. Ein Treffer wird nur einmal aufgenommen, selbst wenn beide Prüfungen
wahr sind.

Symlinks werden vor `is_dir()` und `is_file()` aussortiert. Das verhindert:

- Zyklen durch Links auf Elternverzeichnisse,
- Doppelzählung derselben Zieldatei,
- unbeabsichtigtes Verlassen des Analysebaums.

## Generator und Reproduzierbarkeit

Der Generator besitzt eine lokale Zufallsquelle. Reihenfolge und Zahl der
Zufallsaufrufe sind festgelegt, Namen enthalten monotone IDs, und das Manifest
wird abschließend nach relativen Pfaden sortiert. Deshalb erzeugen gleiche
Parameter unabhängig vom Zielpfad dasselbe `GenerationSummary`.

Dateien werden mit `truncate` auf ihre synthetische Größe gesetzt. Es müssen
keine zufälligen Bytes erzeugt oder gespeichert werden; für `stat().st_size` ist
das Ergebnis dennoch real.

`overwrite=False` schützt nichtleere Ziele. Der CLI-Lernlauf ersetzt dagegen
seinen kontrollierten Datenordner standardmäßig, damit wiederholte Benchmarks
reproduzierbar bleiben. Ein Dateisystem-Root wird grundsätzlich abgewiesen.

## Laufzeitvergleich korrekt interpretieren

Beide Varianten haben O(D + F) Zeit, weil sie jedes Verzeichnis und jede Datei
einmal besuchen. Ein gemessener Unterschied stammt aus konstanten Faktoren:

- Erzeugen und Auflösen von Python-Frames,
- Operationen auf der expliziten Liste,
- Dateisystem- und Betriebssystem-Caches,
- konkrete Verzweigung des Baums.

Der Benchmark validiert vollständige Gleichheit vor der Messung. Die
Ergebnisprüfung selbst liegt außerhalb der gemessenen Funktionen. Der Median
mehrerer Wiederholungen dämpft einzelne Ausreißer, macht die Messung aber nicht
plattformunabhängig.

## Speichervergleich

Die rekursive Variante benötigt O(H) Frames. Der explizite DFS-Stack kann bei
einem Baum mit Verzweigungsfaktor `b` ungefähr O(b · H) noch offene Geschwister
halten, in Sonderformen aber anders wachsen. Beide brauchen außerdem Speicher
für:

- O(H) Tiefen-Buckets,
- O(E) Extension-Buckets,
- O(M) Pattern-Treffer.

Diese Ergebnisdaten sind fachlich gefordert und keine reine Traversal-Overhead.

## Wann welche Variante?

Rekursion ist kompakt und bildet die Baumdefinition direkt ab. Sie eignet sich
für kontrolliert flache Bäume und Lehrzwecke. Der explizite Stack ist robuster,
wenn Tiefe unbekannt ist, Traversierung pausierbar sein soll oder Stackzustände
priorisiert, begrenzt und protokolliert werden müssen.

Die Wahl verändert nicht die asymptotische Zeit. Sie verändert, wo Zustand
gespeichert wird und welche praktischen Grenzen gelten.
