# Project 01 (basic) — Search algorithms: BFS, UCS, IDDFS, A\*

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 06 — Theory of AI 1** · Format: **Jupyter notebook** (`search.ipynb`)

## Why this format?

You only understand search procedures once you see them *run*: the same
algorithm, two problems, and you observe directly how many nodes each
procedure expands and whether it finds the *optimal* solution. A notebook
combines code, explanation and output in one place — ideal for this
step-by-step, exploratory start with plenty of guidance.

## Goal

After this project you have **implemented yourself** the uninformed and
informed search procedures from parts 1 and 2 of the script and **checked
their core statements empirically**:

- **Optimality:** UCS and A\* find the *cheapest* route (Arad → Bucharest =
  418 km), BFS and greedy only *a* route (450 km).
- **Heuristic dominance:** the stronger admissible heuristic $h_2$ (Manhattan)
  expands considerably fewer nodes than $h_1$ (Hamming), which in turn expands
  fewer than UCS ($h\equiv 0$) — made visible on the 8-puzzle.
- **UCS and A\* differ in only one line** (the evaluation function $f$) — the
  central unification of the script.

## Prior knowledge

- Parts 1 and 2 of the script (`../../README.md`): the search problem
  formalism, BFS/UCS/DFS/IDDFS, heuristics, A\*, admissibility, dominance.
- Python basics: classes, `dataclass`, `heapq` (priority queue),
  `collections.deque`, `lambda`.

## Setup

Only the standard library + `matplotlib` (for a bar chart) — both are
installed in the repository `venv` (see `SETUP.md` in the root directory). In
Jupyter or VS Code, select the kernel of the `.venv` and run the cells from
top to bottom.

```bash
# from the repository root, if you have not done it yet:
source .venv/bin/activate
jupyter lab   # or: open the file search.ipynb in VS Code
```

## Tasks (step by step)

Open **`search.ipynb`**. The infrastructure (nodes, `expand`, the two
problems, BFS as a model and the generic `best_first_search`) is given — read
it and run it. Then you solve three tasks:

1. **UCS and A\* (task 1):** build `uniform_cost_search`, `astar_search` and
   `greedy_search` by passing the right $f$ to the given `best_first_search`
   ($g$ / $g+h$ / $h$). The goal: both return **418 km**.
2. **IDDFS (task 2):** implement the recursive depth-limited search and the
   outer deepening loop — with a cycle check along the path.
3. **The dominance experiment (task 3):** solve the same 8-puzzle with UCS,
   A\*($h_1$) and A\*($h_2$), count the expanded nodes and draw a comparison
   bar chart. Check with an `assert` that all of them find the same optimal
   length.

At the end there is a short **reflection part** (in writing, bullet points).

## What should work in the end

- All four procedures solve Arad → Bucharest; **UCS and A\* return 418 km**,
  BFS and greedy 450 km (the fewest steps, not the cheapest route).
- The 8-puzzle (a starting position 15 moves deep) is solved optimally, and
  the node counts show clearly **UCS ≫ A\*($h_1$) ≫ A\*($h_2$)**
  (reference values: about **5257 ≫ 457 ≫ 147**).

## Reference solution

The folder **`solution/`** holds `search_solution.ipynb` — fully implemented
and **executed** (with outputs and the bar chart). Only use it once you have
tried the tasks yourself. The reflection questions are deliberately left
unanswered — they are yours to work through.

> **Tip:** if your A\* does *not* return 418 km, check two things: (a) does
> `best_first_search` do the goal test at *expansion* time (not at
> generation)? — yes, that is given. (b) Did you really choose $g+h$ for A\*
> and not just $h$? Just $h$ is *greedy* and not optimal.

---
---

# Projekt 01 (basic) — Suchalgorithmen: BFS, UCS, IDDFS, A\* (deutsche Fassung)

**Modul 06 — Theorie der KI 1** · Format: **Jupyter Notebook** (`search.ipynb`)

## Warum dieses Format?

Suchverfahren versteht man erst, wenn man sie *laufen* sieht: gleicher
Algorithmus, zwei Probleme, und man beobachtet direkt, wie viele Knoten
jedes Verfahren expandiert und ob es die *optimale* Lösung findet. Ein
Notebook verbindet Code, Erklärung und Ausgabe an einer Stelle — ideal für
diesen schrittweisen, explorativen Einstieg mit viel Anleitung.

## Ziel

Nach diesem Projekt hast du die uninformierten und informierten Suchverfahren
aus Teil 1 & 2 des Skripts **selbst implementiert** und ihre Kernaussagen
**empirisch nachgeprüft**:

- **Optimalität:** UCS und A\* finden den *billigsten* Weg (Arad → Bukarest =
  418 km), BFS und Greedy nur *einen* Weg (450 km).
- **Heuristik-Dominanz:** Die stärkere zulässige Heuristik $h_2$ (Manhattan)
  expandiert deutlich weniger Knoten als $h_1$ (Hamming), diese wiederum
  weniger als UCS ($h\equiv 0$) — am 8-Puzzle sichtbar gemacht.
- **UCS und A\* unterscheiden sich nur in einer Zeile** (der Bewertungsfunktion
  $f$) — die zentrale Vereinheitlichung des Skripts.

## Vorwissen

- Teil 1 & 2 des Skripts (`../../README.md`): Suchproblem-Formalismus,
  BFS/UCS/DFS/IDDFS, Heuristiken, A\*, Zulässigkeit, Dominanz.
- Python-Grundlagen: Klassen, `dataclass`, `heapq` (Prioritätsschlange),
  `collections.deque`, `lambda`.

## Setup

Nur Standardbibliothek + `matplotlib` (für ein Balkendiagramm) — beides ist
im Repo-`venv` installiert (siehe `SETUP.md` im Wurzelverzeichnis). In
Jupyter oder VS Code den Kernel des `.venv` wählen und die Zellen von oben
nach unten ausführen.

```bash
# aus dem Repo-Wurzelverzeichnis, falls noch nicht geschehen:
source .venv/bin/activate
jupyter lab   # oder: in VS Code die Datei search.ipynb öffnen
```

## Aufgabenstellung (Schritt für Schritt)

Öffne **`search.ipynb`**. Die Infrastruktur (Knoten, `expand`, die zwei
Probleme, BFS als Vorbild und die generische `best_first_search`) ist
vorgegeben — lies und führe sie aus. Dann löst du drei Aufgaben:

1. **UCS & A\* (Aufgabe 1):** Baue `uniform_cost_search`, `astar_search` und
   `greedy_search`, indem du der vorgegebenen `best_first_search` das
   richtige $f$ übergibst ($g$ / $g+h$ / $h$). Ziel: beide liefern **418 km**.
2. **IDDFS (Aufgabe 2):** Implementiere die rekursive tiefenbeschränkte Suche
   und die äußere Vertiefungsschleife — mit Zyklencheck entlang des Pfades.
3. **Dominanz-Experiment (Aufgabe 3):** Löse dasselbe 8-Puzzle mit UCS,
   A\*($h_1$) und A\*($h_2$), zähle die expandierten Knoten und zeichne einen
   Vergleichsbalken. Prüfe per `assert`, dass alle dieselbe optimale Länge
   finden.

Zum Schluss ein kurzer **Reflexionsteil** (schriftlich, Stichpunkte).

## Was am Ende funktionieren soll

- Alle vier Verfahren lösen Arad → Bukarest; **UCS und A\* liefern 418 km**,
  BFS und Greedy 450 km (nur wenigste Schritte, nicht billigster Weg).
- Das 8-Puzzle (Startstellung 15 Züge tief) wird optimal gelöst, und die
  Knotenzahlen zeigen klar **UCS ≫ A\*($h_1$) ≫ A\*($h_2$)**
  (Referenzwerte: ca. **5257 ≫ 457 ≫ 147**).

## Musterlösung

Im Ordner **`solution/`** liegt `search_solution.ipynb` — vollständig
implementiert und **ausgeführt** (mit Ausgaben und Balkendiagramm). Nutze es
erst, wenn du die Aufgaben selbst versucht hast. Die Reflexionsfragen bleiben
bewusst unbeantwortet — die sind zum Selbstdurcharbeiten da.

> **Tipp:** Wenn dein A\* *nicht* 418 km liefert, prüfe zwei Dinge: (a) Nutzt
> `best_first_search` den Zieltest beim *Expandieren* (nicht beim Erzeugen)?
> — ja, ist vorgegeben. (b) Hast du bei A\* wirklich $g+h$ und nicht nur $h$
> gewählt? Nur $h$ ist *greedy* und nicht optimal.
