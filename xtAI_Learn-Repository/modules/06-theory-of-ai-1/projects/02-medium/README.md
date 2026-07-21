# Project 02 (medium) — A CSP solver and a DPLL SAT solver: Sudoku by two routes

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 06 — Theory of AI 1** · Format: **Python project** (several modules + tests)

## Why this format?

This is about *real algorithms with structure* — a generic CSP framework, a
SAT solver and the encoding of a problem from one formalism into the other.
This is no longer an exploratory notebook but a small **code base** with
cleanly separated modules and a test suite that you develop against
incrementally. That is exactly how one works with such solvers in practice.

## Goal

You implement the two central inference/search engines from parts 2 and 3 of
the script and apply **both to the same problem** — Sudoku:

1. **The CSP route:** backtracking search with the **MRV** heuristic and
   **AC-3** (arc consistency) as inference (`csp.py`).
2. **The SAT route:** a **DPLL** solver with unit propagation and the pure
   literal rule (`dpll.py`), plus the **encoding of Sudoku as a propositional
   CNF formula** (`sudoku.py`).

The point: both routes must deliver **the same** solution. You experience that
Sudoku *is* both a constraint problem and a satisfiability problem — two views
of the same combinatorial structure.

## Prior knowledge

- Part 2 (CSP: AC-3, MRV, backtracking) and part 3 (propositional logic, CNF,
  DPLL, unit propagation, pure literal) of the script.
- Python: modules/imports, sets (`set`, `frozenset`), recursion, `deque`.

## Project structure

```
02-medium/
  csp.py           # generic CSP framework      <- YOU: revise, ac3, backtracking_search
  dpll.py          # DPLL SAT solver            <- YOU: clause_value, unit/pure, _dpll
  sudoku.py        # Sudoku as a CSP and as SAT <- YOU: at_most_one, encode_sudoku
  test_solver.py   # test suite (given)         -> for verification
  solution/        # the complete reference solution (look only after your own attempt)
```

What is given in each case is the scaffold (data structures, signatures,
detailed docstrings with pseudocode) as well as the non-algorithmic parts
(Sudoku parsing, the CSP neighbourhood model, the variable numbering for the
SAT encoding). The **algorithmic cores** are marked with `TODO` /
`raise NotImplementedError`.

## Tasks (step by step)

Work from the bottom up and test after every building block with
`test_solver.py`:

1. **`dpll.py`** — implement `clause_value` (the truth value of a clause under
   a partial model), then `find_unit_clause` and `find_pure_symbol`, and
   finally the recursion `_dpll` with its four steps (early termination →
   unit propagation → pure literal → branch). *Test:* `test_dpll_*`.
2. **`csp.py`** — implement `revise` (delete values without a partner), `ac3`
   (the propagation loop), then `backtracking_search` (MRV + MAC with
   snapshot/rollback). *Test:* `test_csp_map_coloring`.
3. **`sudoku.py`** — implement `at_most_one` and `encode_sudoku` (the six
   groups of clauses). *Test:* `test_sudoku_both_ways`.

Finally run `python sudoku.py`: it solves the example puzzle by both routes,
measures the time and checks with an `assert` that the solutions are identical
and valid.

## What should work in the end

```bash
source ../../../../.venv/bin/activate   # the repository venv (only the standard library is needed)
python test_solver.py     # -> "All tests passed."
python sudoku.py          # -> a valid, identical grid from the CSP and from SAT
```

Reference (the reference solution): CSP about **10 ms**, the SAT encoding about
**11,800 clauses**, DPLL about **1–2 s**. The difference in time is
instructive: the CSP solver uses problem-specific structure (AC-3 on the
constraint graph) directly, while DPLL works on the generic but blinder
propositional view — both correct, differently efficient.

## Reflection (in writing, briefly)

1. Why is **unit propagation** so effective on the Sudoku SAT instance?
   (Think of the unit clauses of the givens and their chain reaction.)
2. AC-3 often makes Sudoku almost unique before the backtracking even starts.
   Which constraint structure favours that? (Script: arc consistency.)
3. The SAT encoding needs the "at-most-one" clauses per row/column/block.
   What happens if you leave them out — does the formula stay correct? (Hint:
   several digits per cell/unit would then be permitted.)
4. Both procedures are complete and correct for Sudoku. When would you
   nevertheless prefer the SAT route? (Keyword: a highly optimized CDCL solver
   as a black box into which you can encode arbitrary logical constraints.)

## Reference solution

Complete in **`solution/`** — including `test_solver.py`, and all tests pass.
Look only after your own attempt.

---
---

# Projekt 02 (medium) — CSP-Solver & DPLL-SAT-Solver: Sudoku auf zwei Wegen (deutsche Fassung)

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
