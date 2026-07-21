# Projekt 01 (basic) — Ein STRIPS-Vorwärtsplaner

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
