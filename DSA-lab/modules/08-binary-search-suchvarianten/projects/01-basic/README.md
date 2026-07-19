# 01-basic — Binary Search sauber bauen

## Ziel

Dieses Projekt baut Binary Search ohne Bibliotheksabkürzung auf. Im Mittelpunkt
stehen nicht möglichst wenige Codezeilen, sondern konsistente Intervallverträge
und explizite Invarianten. Neben der exakten Suche entstehen First/Last
Occurrence, Lower/Upper Bound, Insert Position und eine Bereichsabfrage für
Duplikate.

Python-Skripte sind hier das passende Format, weil sich dieselben Funktionen
sowohl mit kleinen Hand-Traces als auch mit vielen automatisierten Rand- und
Property-Fällen prüfen lassen.

## Aufgabenstellung

Implementiere in `binary_search.py`:

1. eine exakte Binary Search auf einem geschlossenen Intervall `[left, right]`,
2. einen Lower Bound auf `[left, right)`,
3. einen Upper Bound auf `[left, right)`,
4. First und Last Occurrence,
5. die stabile linke Insert Position,
6. den halb offenen Bereich aller Vorkommen und deren Anzahl.

Jede Kernsuche soll optional strukturierte Trace-Schritte erfassen. Ein Schritt
dokumentiert das aktuelle Intervall, Mitte und Wert, die Entscheidung sowie das
nächste Intervall.

## Dateien

- `binary_search.py`: Eigenimplementierungen und unveränderliche Trace-Datensätze
- `demo.py`: lesbare Hand-Traces für exakte Suche, Lower Bound und Upper Bound
- `test_binary_search.py`: Randfall-, Invarianten- und Property-Tests

## Invarianten

### Exakte Suche

Falls das Ziel vorkommt und noch nicht zurückgegeben wurde, liegt mindestens ein
Vorkommen im geschlossenen Intervall `[left, right]`. Eine erfolglose Runde
verwirft die bereits geprüfte Mitte mit `middle + 1` oder `middle - 1`.

### Lower Bound

Alle Indizes vor `left` enthalten Werte `< target`. Alle Indizes ab `right`
enthalten Werte `>= target`. Nur `[left, right)` ist noch unentschieden.

### Upper Bound

Alle Indizes vor `left` enthalten Werte `<= target`. Alle Indizes ab `right`
enthalten Werte `> target`. Wieder ist `[left, right)` der unentschiedene Bereich.

## Hinweise

Die Funktionen prüfen absichtlich nicht, ob die Eingabe sortiert ist. Eine
lineare Vorprüfung würde die logarithmische Laufzeit jeder Suche zerstören. Die
Sortierung ist deshalb eine dokumentierte Vorbedingung und wird von den
Aufrufenden garantiert.

`lower_bound` liefert auch dann eine gültige Einfügeposition, wenn das Ziel nicht
vorkommt. First Occurrence muss anschließend zusätzlich prüfen, ob der Index noch
in der Sequenz liegt und dort wirklich das Ziel steht.

Ein Trace ist leer, wenn der ursprüngliche Suchraum leer ist. Bei jeder
erfolglosen Iteration muss seine Kandidatenzahl strikt sinken.

## Ausführen

```bash
python3 -m pytest -q
python3 demo.py
```

## Fertig, wenn …

- Standard-Binary-Search Treffer am Anfang, in der Mitte und am Ende findet,
- fehlende Werte vor, zwischen und nach vorhandenen Werten korrekt behandelt
  werden,
- First/Last Occurrence bei Duplikaten die tatsächlichen Außengrenzen liefern,
- Insert Position für leere Listen und beide Außenränder korrekt ist,
- Lower und Upper Bound für reproduzierbare Zufallsfälle mit `bisect` übereinstimmen,
- die Trace-Tests beide Invarianten und strikten Fortschritt bestätigen,
- keine Funktion ihre Eingabesequenz verändert,
- alle Tests und die vollständige Trace-Demo erfolgreich laufen.
