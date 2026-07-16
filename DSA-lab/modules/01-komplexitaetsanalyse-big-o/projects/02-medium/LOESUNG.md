# Lösung — Komplexitäts-Detektiv

Nutze diese Datei erst nach deiner eigenen Code-Analyse und Messung.

## Zuordnung

| Fall | Zeitkomplexität | Auxiliary Space | entscheidende Beobachtung |
|---|---|---|---|
| 01 | \(\Theta(1)\) | \(\Theta(1)\) | Tupelgröße und Index sind unabhängig von \(n\). |
| 02 | \(\Theta(\log n)\) | \(\Theta(1)\) | **remaining** wird in jedem Schritt halbiert. |
| 03 | \(\Theta(n)\) | \(\Theta(1)\) | Zwei aufeinanderfolgende Durchläufe ergeben \(n+n\). |
| 04 | \(\Theta(n)\) | \(\Theta(1)\) | Die innere Arbeit summiert sich zu \(n+n/2+n/4+\dots<2n\). |
| 05 | erwartete \(\Theta(n)\) | \(\Theta(n)\) | Set-Aufbau ist linear, jeder Membership-Test erwartet konstant. |
| 06 | \(\Theta(n^2)\) | \(\Theta(n)\) | Jeder der \(n\) erfolglosen Listentests durchsucht \(n\) Elemente. |
| 07 | \(\Theta(n^2)\) | \(\Theta(n)\) gleichzeitig | Die Slices kopieren \(n-1,n-2,\dots,1\) Referenzen. |
| 08 | \(\Theta(n^2)\) | \(\Theta(n)\) | Jeder Front-Insert verschiebt die bisherige Liste. |
| 09 | \(\Theta(n^2)\) | \(\Theta(n^2)\) | Alle bisherigen Strings bleiben referenziert; jede Konkatenation kopiert. |
| 10 | typischerweise \(\Theta(n\log n)\) | \(\Theta(n)\) | Lineare Erzeugung plus Sortieren unregelmäßiger Werte dominiert. |

## Die versteckten Kosten

### Fall 04: Verschachtelt, aber linear

Die äußere Schleife halbiert **remaining**. Die innere Schleife läuft daher
nicht auf jeder Ebene \(n\)-mal, sondern in den Längen

\[
n,\ n/2,\ n/4,\ \dots
\]

Diese geometrische Reihe bleibt unter \(2n\). Das Beispiel zeigt, warum die
Anzahl sichtbarer Schleifen allein keine Laufzeitklasse bestimmt.

### Fall 05 und 06: Derselbe Operator, anderer Container

Bei einem Set kostet **value in allowed** unter üblichen Hashing-Annahmen
erwartet \(\Theta(1)\). Der einmalige Set-Aufbau benötigt \(\Theta(n)\), danach
folgen \(n\) erwartbar konstante Tests.

Eine Liste muss für einen fehlenden Wert vollständig durchsucht werden. Fall 06
erzwingt mit -1 einen solchen Miss und wiederholt ihn \(n\)-mal. Dadurch entstehen
\(n\cdot n\) Vergleiche.

### Fall 07: Slicing in der Schleife

Ein List Slice erzeugt eine neue Liste und kopiert Referenzen. Die gleichzeitig
belegte Zusatzspeichermenge bleibt linear, weil der vorherige Slice nach der
Zuweisung freigegeben werden kann. Über den gesamten Lauf werden dennoch
quadratisch viele Referenzen kopiert und temporär allokiert.

### Fall 08: Front-Insert

**values.insert(0, value)** muss vorhandene Referenzen um eine Position
verschieben. Die Verschiebungen summieren sich zu

\[
0+1+2+\dots+(n-1)=\Theta(n^2).
\]

Dass die Verschiebung in optimiertem C implementiert ist, ändert ihre
asymptotische Anzahl nicht.

### Fall 09: String-Konkatenation und gehaltene Präfixe

Strings sind immutable. **history** hält jedes bisherige Präfix fest. Dadurch
kann der Speicher des alten Strings nicht für die nächste Konkatenation
wiederverwendet werden. Es bleiben Strings der Längen

\[
0,1,2,\dots,n-1
\]

gleichzeitig erreichbar. Zeit und Zusatzspeicher summieren sich quadratisch.
Ohne die gehaltenen Referenzen kann CPython bestimmte Konkatenationsmuster
optimieren; auf eine solche Implementierungsoptimierung sollte eine allgemeine
Algorithmusanalyse nicht stillschweigend vertrauen.

### Fall 10: Sortieren nach linearer Erzeugung

Die deterministische Generatorphase ist linear. Die erzeugten Werte sind
unregelmäßig, damit Timsort nicht einfach einen bereits sortierten Run ausnutzt.
Die Sortierphase ist im typischen und im Worst-Case-Wachstum
\(O(n\log n)\) und dominiert die lineare Vorbereitung für große \(n\).

## Messabweichungen erklären

Die empirischen Steigungen müssen nicht exakt 0, 1 oder 2 sein:

- Funktionsaufruf, Schleifenverwaltung und Timer besitzen konstante Kosten.
- Kleine Eingaben zeigen den asymptotischen dominanten Term noch nicht klar.
- **in**, Slicing, **insert** und **sorted** führen viel Arbeit in kompiliertem
  C aus und haben andere Konstanten als Python-Schleifen.
- Speicherallokator, Cache, CPU-Takt und Betriebssystem erzeugen Rauschen.
- \(n\log n\) verhält sich im Log-Log-Plot nicht wie eine reine Potenz. Seine
  lokale Steigung liegt nur etwas über 1.
- Der Set-Fall beruht auf erwarteten Hash-Kosten, nicht auf einer absoluten
  Worst-Case-Garantie.

Die korrekte Formulierung lautet daher: Die Messung ist mit der hergeleiteten
Klasse vereinbar. Die Klasse selbst folgt aus der Code- und Kostenanalyse.
