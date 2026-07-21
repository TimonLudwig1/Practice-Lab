# Projekt 01 (basic) — Suchalgorithmen: BFS, UCS, IDDFS, A\*

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
erst, wenn du die Aufgaben selbst versucht hast. Am Ende der Lösung stehen
auch die Antworten auf die Reflexionsfragen.

> **Tipp:** Wenn dein A\* *nicht* 418 km liefert, prüfe zwei Dinge: (a) Nutzt
> `best_first_search` den Zieltest beim *Expandieren* (nicht beim Erzeugen)?
> — ja, ist vorgegeben. (b) Hast du bei A\* wirklich $g+h$ und nicht nur $h$
> gewählt? Nur $h$ ist *greedy* und nicht optimal.
