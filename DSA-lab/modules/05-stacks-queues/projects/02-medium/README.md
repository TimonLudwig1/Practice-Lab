# Projekt 02-medium: Ausdrucks-Rechner

## Ziel

Dieses Projekt übersetzt arithmetische Ausdrücke aus der gewohnten Infix-Notation
in Postfix-Notation und wertet sie anschließend aus. Beide Phasen verwenden einen
Stack: Beim Shunting-Yard-Algorithmus speichert er noch nicht ausgegebene
Operatoren, bei der Auswertung speichert er Zwischenergebnisse.

Die Lösung besteht aus Python-Skripten und einer pytest-Suite. Ein Notebook wäre
für einzelne Beispiele bequem, die Skriptform macht Parser, Algorithmus und Tests
jedoch besser wiederverwendbar. `demo.py` übernimmt die visuelle Schritt-für-
Schritt-Simulation.

## Aufgabenstellung

1. Zerlege einen Eingabestring in Zahlen, Operatoren und Klammern. Unterstütze
   Ganzzahlen, Dezimalzahlen und wissenschaftliche Schreibweise.
2. Prüfe, ob öffnende und schließende Klammern ausgeglichen und korrekt geordnet
   sind.
3. Übersetze Infix mit dem Shunting-Yard-Algorithmus nach Postfix. Beachte
   Präzedenz, Assoziativität, Klammern und unäre Vorzeichen.
4. Werte die Postfix-Folge mit einem zweiten Stack aus. Jeder binäre Operator
   entnimmt zuerst den rechten, dann den linken Operanden.
5. Melde Syntaxfehler verständlich und unterscheide sie von Rechenfehlern wie
   Division durch null.

Unterstützte Operatoren:

| Operator | Bedeutung | Assoziativität |
|---|---|---|
| `+`, `-` | Addition, Subtraktion | links |
| `*`, `/` | Multiplikation, Division | links |
| `^` | Potenz | rechts |
| unär `+`, `-` | Vorzeichen | rechts |

Wie in der üblichen mathematischen Konvention bindet die Potenz stärker als ein
unäres Vorzeichen: `-2^2` ergibt `-(2^2) = -4`, während `(-2)^2 = 4` ergibt.

## Simulation: Infix zu Postfix

Für `3 + 4 * 2` entstehen diese Zustände:

| gelesenes Token | Ausgabewarteschlange | Operator-Stack |
|---|---|---|
| `3` | `3` | leer |
| `+` | `3` | `+` |
| `4` | `3 4` | `+` |
| `*` | `3 4` | `+ *` |
| `2` | `3 4 2` | `+ *` |
| Ende | `3 4 2 * +` | leer |

`*` bleibt oberhalb von `+`, weil es höhere Präzedenz hat. Am Ende leert der
Algorithmus den Operator-Stack in die Ausgabe.

Die Postfix-Auswertung von `3 4 2 * +` benötigt keine Präzedenzregeln mehr:

1. `3`, `4`, `2` werden nacheinander auf den Wertestack gelegt.
2. `*` entnimmt `2` und `4` und legt `8` zurück.
3. `+` entnimmt `8` und `3` und legt `11` zurück.
4. Genau ein Wert bleibt übrig; er ist das Ergebnis.

## Invarianten und Komplexität

Beim Konvertieren enthält die Ausgabe nur Zahlen und bereits endgültig
platzierte Operatoren. Eine öffnende Klammer bildet auf dem Operator-Stack eine
Grenze: Kein Operator links davon wird vor der passenden schließenden Klammer
entnommen.

Bei der Postfix-Auswertung enthält der Wertestack nach jedem Token genau die noch
nicht von einem späteren Operator verbrauchten Teilergebnisse. Am Ende muss seine
Größe exakt eins sein; sonst war die Eingabe unvollständig oder mehrdeutig.

Jedes Token wird höchstens einmal auf einen Stack gelegt und einmal entnommen.
Bei `n` Tokens benötigen Konvertierung und Auswertung daher jeweils O(n) Zeit und
im Worst Case O(n) zusätzlichen Speicher.

## Ausführen

Im Projektordner:

```bash
python demo.py
python -m pytest -q
```

Die Demo zeigt für einen Ausdruck nach jedem Token den Postfix-Ausgabestand und
den Operator-Stack.

## Hinweise

- Tokenisierung und Syntaxanalyse sind verschiedene Aufgaben. Ein gültiges Token
  kann an einer ungültigen Stelle stehen, etwa die zweite Zahl in `2 3`.
- Wenn ein Operator ausgewertet wird, ist der erste entnommene Wert der rechte
  Operand. Das ist bei Subtraktion und Division entscheidend.
- Potenzen sind rechtsassoziativ: `2^3^2` bedeutet `2^(3^2)`, nicht `(2^3)^2`.
- Ein unäres Minus wird intern als `u-` markiert. So kann die Postfix-Auswertung
  zwischen `5 - 2` und `-2` unterscheiden.
- Implizite Multiplikation wie `2(3 + 4)` gehört bewusst nicht zur Grammatik.

## Fertig, wenn …

- der Tokenizer alle unterstützten Zahlenformate und Operatoren erkennt,
- unausgeglichene, leere und falsch platzierte Klammern abgewiesen werden,
- Präzedenz und Links-/Rechtsassoziativität korrekt umgesetzt sind,
- unäre Vorzeichen vor Zahlen und Klammerausdrücken funktionieren,
- eine ungültige Tokenfolge eine `ExpressionSyntaxError` auslöst,
- Division durch null und komplexe Ergebnisse klar gemeldet werden,
- die Trace-Demo ohne Fehler läuft und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
