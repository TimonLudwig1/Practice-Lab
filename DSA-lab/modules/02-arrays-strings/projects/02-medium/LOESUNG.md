# Lösung und Auswertung

Lies diesen Abschnitt idealerweise erst, nachdem du eigene Lösungen und Tests
formuliert hast. Der lauffähige Referenzcode steht in `pattern_catalog.py`.

## Pattern-Auswahl statt Aufgabenauswendiglernen

Die Aufgaben wirken unterschiedlich, reduzieren sich aber auf wenige Fragen:

1. Muss dasselbe Array verändert werden?
2. Ist die Eingabe sortiert oder gibt es freien Pufferplatz?
3. Werden viele Abfragen auf unveränderten Daten gestellt?
4. Ist ein zusammenhängendes Fenster gesucht?
5. Genügen Häufigkeiten statt Positionen?

Diese Eigenschaften führen meist direkt zum passenden Pattern.

## 1. Rotation durch drei Umkehrungen

Für `[1, 2, 3, 4, 5]` und `k = 2`:

```text
alles umkehren:       [5, 4, 3, 2, 1]
erste k umkehren:     [4, 5, 3, 2, 1]
Rest umkehren:        [4, 5, 1, 2, 3]
```

Nach der ersten Umkehrung stehen beide Blöcke an der richtigen Seite, aber intern
rückwärts. Die beiden weiteren Umkehrungen reparieren ihre interne Reihenfolge.
Jedes Element wird konstant oft bewegt: O(n) Zeit und O(1) Extra-Speicher.

## 2. Merge vom freien Ende

Beim Merge von links könnte ein großer Wert aus `target` überschrieben werden,
bevor er gelesen wurde. Von rechts ist der Zielplatz dagegen frei. Die Invariante
lautet: Hinter `write` steht bereits der korrekte sortierte Suffix. Der größere
der beiden aktuellen Kandidaten gehört auf `write`.

Sobald `other` leer ist, sind verbleibende `target`-Werte schon an ihrer richtigen
Position. Daher muss die Schleife nur laufen, solange der rechte Zeiger gültig ist.
Zeit: O(n + m), Extra-Speicher: O(1).

## 3. Prefix Sums

Der Index speichert eine führende Null:

```text
Werte:   [ 4, -1, 7, 3]
Prefix:  [ 0,  4, 3, 10, 13]
```

Die Summe `[start, end)` ist `prefix[end] - prefix[start]`. Alle Werte vor
`start` kommen in beiden Summen vor und heben sich auf. Der O(n)-Aufbau lohnt
sich, sobald viele O(1)-Abfragen folgen. Der Index benötigt O(n) Speicher und ist
durch ein Tuple von späteren Änderungen der Quelle getrennt.

## 4. Frequency Map für Anagramme

Die erste Zeichenfolge erhöht Zähler, die zweite senkt sie. Ein negativer Zähler
beweist sofort, dass rechts ein Zeichen zu oft vorkommt. Am Ende muss die Map leer
sein. Das vermeidet die O(n log n)-Kosten einer Sortierlösung. Zeit: O(n),
Speicher: O(k) für `k` verschiedene Zeichen.

## 5. Deduplizierung mit Read-/Write-Pointer

Der Bereich vor `write` enthält stets genau die bisher gefundenen eindeutigen
Werte. `read` untersucht das nächste Element. Nur wenn es vom letzten eindeutigen
Wert abweicht, wird es geschrieben. Die Sortierung macht gleiche Werte
benachbart. Zeit: O(n), zusätzlicher Arbeitspeicher: O(1).

## 6. Stabiles Zero-Move

`write` zeigt auf den nächsten Platz für einen Nichtnullwert. `read` läuft durch
alle Elemente. Jeder gefundene Nichtnullwert wird nach vorne getauscht; weil sie
in Lesereihenfolge geschrieben werden, bleibt ihre relative Reihenfolge stabil.
Nach `write` sammeln sich die verdrängten Nullen. Zeit: O(n), Extra-Speicher:
O(1).

## 7. Prefix- und Suffix-Produkte

Im Vorwärtslauf erhält jede Position das Produkt links von ihr. Im Rückwärtslauf
wird das Produkt rechts von ihr hinzumultipliziert:

```text
Eingabe:     [1, 2, 3, 4]
links:       [1, 1, 2, 6]
rechts:      [24, 12, 4, 1]
Ergebnis:    [24, 12, 8, 6]
```

Da nicht dividiert wird, funktionieren auch eine oder mehrere Nullen. Die
Ergebnisliste zählt üblicherweise nicht als Extra-Speicher; darüber hinaus werden
nur zwei Skalare verwendet. Zeit: O(n), zusätzlicher Speicher: O(1).

## 8. Sliding Window

Das Fenster `[window_start, index]` enthält stets keine doppelten Zeichen. Die Map
merkt die letzte Position jedes Zeichens. Taucht ein Zeichen im aktuellen Fenster
erneut auf, springt der linke Rand direkt hinter dessen alte Position. Wichtig:
Der Rand darf niemals rückwärts wandern. Jeder Index wird konstant oft
verarbeitet: O(n) Zeit und O(k) Speicher.

## 9. Schrumpfende Matrixgrenzen

Vier Grenzen beschreiben den noch ungelesenen Teil: `top`, `right`, `bottom` und
`left`. Nach dem Lesen einer Seite rückt ihre Grenze nach innen. Vor unterer und
linker Seite wird erneut geprüft, ob noch eine Zeile beziehungsweise Spalte übrig
ist; das verhindert doppelte Besuche bei dünnen Matrizen. Jedes Element wird
genau einmal gelesen: O(r · c) Zeit, abgesehen vom Ergebnis O(1) Speicher.

## 10. In-place-Kompression

`read` bestimmt die Länge eines vollständigen Laufs, `write` schreibt dessen
Zeichen und gegebenenfalls die Ziffern der Länge. Der Schreibzeiger kann den
Lesebereich nicht gefährlich überholen: Eine kodierte Gruppe braucht nie mehr
Platz als der ursprüngliche Lauf. Am Ende wird der ungenutzte Suffix entfernt.
Zeit: O(n), Extra-Speicher: O(1), wenn die kurze Dezimaldarstellung des Zählers als
konstant betrachtet wird.

## Komplexitätsübersicht

| Pattern | Zeit | Extra-Speicher | Entscheidende Invariante |
|---|---:|---:|---|
| Rotation | O(n) | O(1) | Blöcke liegen nach drei Reversals korrekt |
| Merge | O(n + m) | O(1) | Sortierter Suffix hinter `write` |
| Prefix Sum | O(n) + O(1)/Query | O(n) | `prefix[i]` summiert `[0, i)` |
| Anagramm | O(n) | O(k) | Map enthält noch nicht ausgeglichene Zeichen |
| Deduplizierung | O(n) | O(1) | Eindeutiger Prefix vor `write` |
| Zero-Move | O(n) | O(1) | Nichtnull-Prefix ist stabil |
| Product Except Self | O(n) | O(1)* | Linkes mal rechtes Teilprodukt |
| Sliding Window | O(n) | O(k) | Aktuelles Fenster ist duplikatfrei |
| Spirale | O(r · c) | O(1)* | Außerhalb der Grenzen ist alles besucht |
| Kompression | O(n) | O(1) | Vollständige Läufe vor `read` sind kodiert |

`*` ohne den Speicher für die geforderte Ergebnisliste.

## Typische Fehlentscheidungen

- Eine Rotation per wiederholtem `pop(0)` wird bei vielen Schritten quadratisch.
- Merge von links zerstört ohne Zusatzpuffer noch ungelesene Zielwerte.
- Bereichssummen immer neu zu berechnen verschenkt die Wiederverwendung.
- Zwei verschachtelt wirkende Zeiger sind nicht automatisch O(n²): Wenn jeder
  Zeiger insgesamt nur vorwärts läuft, bleibt die Arbeit linear.
- Sliding-Window-Grenzen dürfen nicht zurückspringen.
- Bei Matrixgrenzen müssen einzelne Restzeilen und Restspalten separat geprüft
  werden.
