# Projekt 02 (medium) — CSP-Solver & DPLL-SAT-Solver: Sudoku auf zwei Wegen

**Modul 06 — Theorie der KI 1** · Format: **Python-Projekt** (mehrere Module + Tests)

## Warum dieses Format?

Hier geht es um *echte Algorithmen mit Struktur* — ein generisches
CSP-Framework, ein SAT-Solver und die Kodierung eines Problems von einem
Formalismus in den anderen. Das ist kein exploratives Notebook mehr, sondern
eine kleine **Codebasis** mit sauber getrennten Modulen und einer Testsuite,
gegen die du inkrementell entwickelst. Genau so arbeitet man in der Praxis mit
solchen Solvern.

## Ziel

Du implementierst die beiden zentralen Inferenz-/Suchmaschinen aus Teil 2 & 3
des Skripts und wendest **beide auf dasselbe Problem** an — Sudoku:

1. **CSP-Weg:** Backtracking-Suche mit **MRV**-Heuristik und **AC-3**
   (Kantenkonsistenz) als Inferenz (`csp.py`).
2. **SAT-Weg:** ein **DPLL**-Solver mit Unit Propagation und Pure-Literal-Regel
   (`dpll.py`), plus die **Kodierung von Sudoku als aussagenlogische
   KNF-Formel** (`sudoku.py`).

Die Pointe: Beide Wege müssen **dieselbe** Lösung liefern. Du erlebst, dass
Sudoku sowohl ein Constraint- als auch ein Erfüllbarkeitsproblem *ist* — zwei
Sichten auf dieselbe kombinatorische Struktur.

## Vorwissen

- Teil 2 (CSP: AC-3, MRV, Backtracking) und Teil 3 (Aussagenlogik, KNF,
  DPLL, Unit Propagation, Pure Literal) des Skripts.
- Python: Module/Imports, Mengen (`set`, `frozenset`), Rekursion, `deque`.

## Projektstruktur

```
02-medium/
  csp.py           # generisches CSP-Framework   <- DU: revise, ac3, backtracking_search
  dpll.py          # DPLL-SAT-Solver             <- DU: clause_value, unit/pure, _dpll
  sudoku.py        # Sudoku als CSP + als SAT     <- DU: at_most_one, encode_sudoku
  test_solver.py   # Testsuite (vorgegeben)       -> zum Verifizieren
  solution/         # vollständige Musterlösung (erst nach eigenem Versuch ansehen)
```

Vorgegeben ist jeweils das Gerüst (Datenstrukturen, Signaturen, ausführliche
Docstrings mit Pseudocode) sowie die nicht-algorithmischen Teile (Sudoku-Parsing,
das CSP-Nachbarschaftsmodell, die Variablen-Nummerierung fürs SAT-Encoding). Die
**algorithmischen Kerne** sind mit `TODO` / `raise NotImplementedError` markiert.

## Aufgabenstellung (Schritt für Schritt)

Arbeite von unten nach oben und teste nach jedem Baustein mit `test_solver.py`:

1. **`dpll.py`** — implementiere `clause_value` (Wahrheitswert einer Klausel
   unter partiellem Modell), dann `find_unit_clause` und `find_pure_symbol`,
   schließlich die Rekursion `_dpll` mit den vier Schritten (Frühterminierung →
   Unit Propagation → Pure Literal → Verzweigen). *Test:* `test_dpll_*`.
2. **`csp.py`** — implementiere `revise` (streiche partnerlose Werte), `ac3`
   (Propagationsschleife), dann `backtracking_search` (MRV + MAC mit
   Snapshot/Rollback). *Test:* `test_csp_map_coloring`.
3. **`sudoku.py`** — implementiere `at_most_one` und `encode_sudoku`
   (die sechs Klauselgruppen). *Test:* `test_sudoku_both_ways`.

Führe abschließend `python sudoku.py` aus: Es löst das Beispielrätsel beide Wege,
misst die Zeit und prüft per `assert`, dass die Lösungen identisch und gültig sind.

## Was am Ende funktionieren soll

```bash
source ../../../../.venv/bin/activate   # Repo-venv (nur Standardbibliothek nötig)
python test_solver.py     # -> "Alle Tests bestanden."
python sudoku.py          # -> gültiges, identisches Gitter aus CSP und SAT
```

Referenz (Musterlösung): CSP ~**10 ms**, SAT-Kodierung ~**11 800 Klauseln**,
DPLL ~**1–2 s**. Der Zeitunterschied ist lehrreich: Der CSP-Solver nutzt
problemspezifische Struktur (AC-3 auf dem Constraint-Graphen) direkt, während
DPLL die generische, aber blindere aussagenlogische Sicht bearbeitet — beide
korrekt, unterschiedlich effizient.

## Reflexion (schriftlich, kurz)

1. Warum ist die **Unit Propagation** bei der Sudoku-SAT-Instanz so wirksam?
   (Denke an die Unit-Klauseln der Vorgaben und ihre Kettenreaktion.)
2. AC-3 macht Sudoku vor dem Backtracking oft schon fast eindeutig. Welche
   Constraint-Struktur begünstigt das? (Skript: Kantenkonsistenz.)
3. Die SAT-Kodierung braucht die „at-most-one"-Klauseln pro Zeile/Spalte/Block.
   Was passiert, wenn du sie weglässt — bleibt die Formel korrekt? (Tipp: dann
   wären mehrere Ziffern pro Feld/Einheit erlaubt.)
4. Beide Verfahren sind für Sudoku vollständig und korrekt. Wann würdest du
   trotzdem den SAT-Weg bevorzugen? (Stichwort: ein hochoptimierter CDCL-Solver
   als Blackbox, in den du beliebige logische Constraints kodieren kannst.)

## Musterlösung

Vollständig in **`solution/`** — inklusive `test_solver.py`, alle Tests bestehen.
Erst nach eigenem Versuch ansehen.
