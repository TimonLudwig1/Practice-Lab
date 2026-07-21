# Project 01 (basic) — A STRIPS forward planner

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 07 — Theory of AI 2** · Format: **Jupyter notebook** (`planning.ipynb`)

## Why this format?

Planning becomes tangible once you see a planner *run*: the states, the
applicable actions, the plan that emerges, the number of expanded states. A
notebook combines the STRIPS representation, the search and the output in one
place and ties in seamlessly with the search procedures of module 06 — hence the
guided, exploratory start with plenty of guidance.

## Goal

You implement a **classical forward planner (progression)** and experience how
planning is formulated as a search problem (module 06) and how a **heuristic
obtained automatically from the action description** guides the search:

- STRIPS states (a `frozenset` of fluents) and actions `⟨PRE, ADD, DEL⟩`,
  progression $(s\setminus\mathrm{DEL})\cup\mathrm{ADD}$ (given);
- forward search with **BFS** (as a model, given);
- the **$h_{\text{add}}$ heuristic** from the **delete relaxation** (task 1);
- **A\*** with $h_{\text{add}}$ (task 2) and a comparison (task 3).

The test case is the **Sussman anomaly** of the blocks world — a famous small
problem on which naive planners that decompose the goal fail, while state search
cleanly finds an optimal 6-step plan.

## Prior knowledge

- Part 1 of the script (STRIPS/PDDL, progression/regression, relaxation heuristics).
- Module 06: BFS, A\*, heuristics (the search infrastructure is the same).
- Python: `frozenset`, `dataclass`, `heapq`.

## Setup

Only the standard library. In Jupyter/VS Code select the kernel of the
repository `.venv` (see `SETUP.md` in the root directory) and run the cells from
top to bottom; then solve tasks 1–3.

```bash
source ../../../../.venv/bin/activate
jupyter lab    # or open planning.ipynb in VS Code
```

## Tasks (step by step)

Open **`planning.ipynb`**. Part A (STRIPS + blocks world) and part B (search
nodes, BFS, generic best-first search) are given — read them and run them. Then:

1. **`h_add` (task 1):** implement the delete-relaxation heuristic by fixed-point
   iteration: $\Delta(p)=0$ for $p\in s$, otherwise
   $\min_{a:\,p\in\mathrm{ADD}(a)}\big(1+\sum_{q\in\mathrm{PRE}(a)}\Delta(q)\big)$;
   $h_{\text{add}}(s)=\sum_{g\in\text{goal}}\Delta(g)$. (Reference: $h_{\text{add}}(s_0)=5$.)
2. **`astar_plan` (task 2):** wire the given `best_first_plan` up with
   $f(s,g)=g+h_{\text{add}}(s)$.
3. **The comparison (task 3):** BFS vs. A\*(h_add) — the plan length and the
   expanded states.

Finally part C verifies that the plan is executable and reaches the goal.

## What should work in the end

- Both procedures find an **optimal 6-step plan**
  (`Unstack(C,A) → PutDown(C) → PickUp(B) → Stack(B,C) → PickUp(A) → Stack(A,B)`).
- A\*(h_add) expands **fewer** states than BFS (reference: **about 11 vs. 18**).
- The verification confirms: the plan is applicable step by step and reaches the goal.

## Reference solution

The folder **`solution/`** holds `planning_solution.ipynb` — fully implemented
and **executed**, with the answers to the reflection questions at the end. Look
only after your own attempt.

---
---

# Projekt 01 (basic) — Ein STRIPS-Vorwärtsplaner (deutsche Fassung)

**Modul 07 — Theorie der KI 2** · Format: **Jupyter Notebook** (`planning.ipynb`)

## Warum dieses Format?

Planung wird greifbar, wenn man einen Planer *laufen* sieht: Zustände, anwendbare
Aktionen, der entstehende Plan, die Zahl expandierter Zustände. Ein Notebook
verbindet die STRIPS-Repräsentation, die Suche und die Ausgabe an einer Stelle und
knüpft nahtlos an die Suchverfahren aus Modul 06 an — daher der geführte,
explorative Einstieg mit viel Anleitung.

## Ziel

Du implementierst einen **klassischen Vorwärtsplaner (Progression)** und erlebst,
wie Planung als Suchproblem (Modul 06) formuliert wird und wie eine **automatisch
aus der Aktionsbeschreibung gewonnene Heuristik** die Suche lenkt:

- STRIPS-Zustände (`frozenset` von Fluenten) und Aktionen `⟨PRE, ADD, DEL⟩`,
  Progression $(s\setminus\mathrm{DEL})\cup\mathrm{ADD}$ (vorgegeben);
- Vorwärtssuche mit **BFS** (Vorbild, vorgegeben);
- die **$h_{\text{add}}$-Heuristik** aus der **Delete-Relaxation** (Aufgabe 1);
- **A\*** mit $h_{\text{add}}$ (Aufgabe 2) und ein Vergleich (Aufgabe 3).

Testfall ist die **Sussman-Anomalie** der Blocksworld — ein berühmtes kleines
Problem, an dem naive teilziel-zerlegende Planer scheitern, die Zustandssuche aber
sauber einen optimalen 6-Schritte-Plan findet.

## Vorwissen

- Skript Teil 1 (STRIPS/PDDL, Progression/Regression, Relaxationsheuristiken).
- Modul 06: BFS, A\*, Heuristiken (die Suchinfrastruktur ist dieselbe).
- Python: `frozenset`, `dataclass`, `heapq`.

## Setup

Nur Standardbibliothek. In Jupyter/VS Code den Kernel des Repo-`.venv` wählen
(siehe `SETUP.md` im Wurzelverzeichnis) und die Zellen von oben nach unten
ausführen; dann Aufgabe 1–3 lösen.

```bash
source ../../../../.venv/bin/activate
jupyter lab    # oder planning.ipynb in VS Code öffnen
```

## Aufgabenstellung (Schritt für Schritt)

Öffne **`planning.ipynb`**. Teil A (STRIPS + Blocksworld) und Teil B (Suchknoten,
BFS, generische Bestensuche) sind vorgegeben — lesen und ausführen. Dann:

1. **`h_add` (Aufgabe 1):** Implementiere die Delete-Relaxations-Heuristik per
   Fixpunkt-Iteration: $\Delta(p)=0$ für $p\in s$, sonst
   $\min_{a:\,p\in\mathrm{ADD}(a)}\big(1+\sum_{q\in\mathrm{PRE}(a)}\Delta(q)\big)$;
   $h_{\text{add}}(s)=\sum_{g\in\text{goal}}\Delta(g)$. (Referenz: $h_{\text{add}}(s_0)=5$.)
2. **`astar_plan` (Aufgabe 2):** Verdrahte die vorgegebene `best_first_plan` mit
   $f(s,g)=g+h_{\text{add}}(s)$.
3. **Vergleich (Aufgabe 3):** BFS vs. A\*(h_add) — Planlänge und expandierte Zustände.

Zum Schluss verifiziert Teil C, dass der Plan ausführbar ist und das Ziel erreicht.

## Was am Ende funktionieren soll

- Beide Verfahren finden einen **optimalen 6-Schritte-Plan**
  (`Unstack(C,A) → PutDown(C) → PickUp(B) → Stack(B,C) → PickUp(A) → Stack(A,B)`).
- A\*(h_add) expandiert **weniger** Zustände als BFS (Referenz: **~11 vs. 18**).
- Die Verifikation bestätigt: Plan Schritt für Schritt anwendbar, Ziel erreicht.

## Musterlösung

In **`solution/`** liegt `planning_solution.ipynb` — vollständig implementiert und
**ausgeführt**, mit Reflexions-Antworten am Ende. Erst nach eigenem Versuch ansehen.
