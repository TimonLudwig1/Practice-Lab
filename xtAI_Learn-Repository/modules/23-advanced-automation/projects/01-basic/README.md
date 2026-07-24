# P01 (basic) — discrete-event systems: finite automata, reachability & composition

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 23 — Advanced Automation** · Format: **Jupyter notebook**

## Goal

The discrete layer of automation asks: *what state is the plant in, and which event triggers which action?* The tool is the **finite automaton**. You model an automated **bottle-filling station** as two automata (a fill valve and a bottle position), **simulate** event sequences, compute the **reachable** states, and **compose** the two components into one model. The punchline is a **safety** finding: the uncontrolled system can reach a forbidden "valve open with no bottle" state — exactly what the supervisor of P02 will prevent.

1. represent a **deterministic finite automaton (DFA)** $G=(Q,\Sigma,\delta,q_0,Q_m)$ (script ch. 3),
2. **simulate** it and compute the **reachable set** by breadth-first search,
3. build the **parallel composition** (synchronous product), with shared events synchronising,
4. verify a **safety property** by checking whether a forbidden state is reachable.

## Why this format?

A **notebook** — automata are best understood by defining one, running it, and inspecting the reachable state set next to the model.

## Why synthetic data?

Automation logic is a **model**, not a dataset: a DFA *is* the specification. The bottle-filling station is a minimal but faithful example that exhibits every concept (events, transitions, reachability, synchronisation, a real safety violation) and motivates P02 directly.

## Prior knowledge

The module 23 script ch. 1–3, graph search / BFS (module 06), basic set operations.

## Task (step by step)

Open `automata.ipynb`. Much is given; at the `# TODO` spots you build the cores:

- **Part A** (given) — the two component automata (valve, bottle) and their shared event `depart`.
- **Part B** — `run(G, string)`: follow $\delta$ along an event string; return the final state or `None`.
- **Part C** — `reachable(G)`: BFS from $q_0$ returning the reachable state set.
- **Part D** — the transition rule inside `compose(G1, G2)`: shared events synchronise, private events interleave.
- **Part E** (given) — the safety check: is the forbidden state reachable?

## What should come out (expected values)

- **Part B**: `[open_valve, close_valve]` → `closed`; `[close_valve]` from the start → `None` (undefined); `[arrive, depart]` → `no_bottle`.
- **Part C**: both components have all states reachable.
- **Part D**: the product has **4 states**, all reachable; from `('open','bottle')` the shared event `depart` is **not** enabled (a bottle can never leave with the valve open — the synchronisation works).
- **Part E**: the forbidden state `('open','no_bottle')` **is reachable**, via a single `open_valve` — the uncontrolled plant is **unsafe**.

> **The lesson.** A finite automaton is the specification of an automated system's legal behaviour, and **reachability** (plain BFS) is the engine of every safety proof: a bad state is safe iff it is *not* reachable. **Parallel composition** builds a system model from components, with shared events forcing synchronisation. And the key finding sets up the whole field: the uncontrolled plant *can* reach a forbidden state, so it needs a **supervisor** — but because some events (`arrive`) are **uncontrollable**, the supervisor can only act on the controllable ones. That asymmetry is the subject of P02.

## Setup

```bash
cd modules/23-advanced-automation/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # automata.ipynb
```

Plain Python — no numerical libraries. Runtime instant.

## Solution

The complete, executed solution is in [`solution/automata_solution.ipynb`](solution/automata_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: **Petri nets + supervisory control** — model a resource-sharing cell, find its **deadlocks**, and **synthesise a supervisor** that provably prevents them.
- **P03 (final)**: **Model Predictive Control** — the constrained optimising controller of the continuous layer.

---
---

# P01 (basic) — diskrete-Ereignis-Systeme: endliche Automaten, Erreichbarkeit & Komposition (deutsche Fassung)

**Modul 23 — Advanced Automation** · Format: **Jupyter-Notebook**

## Ziel

Die diskrete Ebene der Automatisierung fragt: *In welchem Zustand ist die Anlage, und welches Ereignis löst welche Aktion aus?* Das Werkzeug ist der **endliche Automat**. Du modellierst eine automatisierte **Abfüllstation** als zwei Automaten (ein Füllventil und eine Flaschenposition), **simulierst** Ereignisfolgen, berechnest die **erreichbaren** Zustände und **komponierst** die zwei Komponenten zu einem Modell. Die Pointe ist ein **Sicherheits**-Befund: das ungeregelte System kann einen verbotenen Zustand „Ventil offen ohne Flasche" erreichen — genau das, was der Supervisor aus P02 verhindern wird.

1. einen **deterministischen endlichen Automaten (DFA)** $G=(Q,\Sigma,\delta,q_0,Q_m)$ darstellen (Skript Kap. 3),
2. ihn **simulieren** und die **erreichbare Menge** per Breitensuche berechnen,
3. die **Parallelkomposition** (synchrones Produkt) bauen, mit synchronisierenden geteilten Ereignissen,
4. eine **Sicherheits-Property** verifizieren, indem man prüft, ob ein verbotener Zustand erreichbar ist.

## Warum dieses Format?

Ein **Notebook** — Automaten versteht man am besten, indem man einen definiert, ihn laufen lässt und die erreichbare Zustandsmenge neben dem Modell betrachtet.

## Warum synthetische Daten?

Automatisierungslogik ist ein **Modell**, kein Datensatz: Ein DFA *ist* die Spezifikation. Die Abfüllstation ist ein minimales, aber treues Beispiel, das jedes Konzept zeigt (Ereignisse, Transitionen, Erreichbarkeit, Synchronisation, eine echte Sicherheitsverletzung) und P02 direkt motiviert.

## Vorwissen

Das Modul-23-Skript Kap. 1–3, Graphsuche / BFS (Modul 06), grundlegende Mengenoperationen.

## Aufgabenstellung (Schritt für Schritt)

Öffne `automata.ipynb`. Vieles ist vorgegeben; an den `# TODO`-Stellen baust du die Kerne:

- **Teil A** (vorgegeben) — die zwei Komponenten-Automaten (Ventil, Flasche) und ihr geteiltes Ereignis `depart`.
- **Teil B** — `run(G, string)`: $\delta$ entlang einer Ereigniskette folgen; Endzustand oder `None` zurückgeben.
- **Teil C** — `reachable(G)`: BFS von $q_0$, gibt die erreichbare Zustandsmenge zurück.
- **Teil D** — die Transitionsregel in `compose(G1, G2)`: geteilte Ereignisse synchronisieren, private verschränken.
- **Teil E** (vorgegeben) — der Sicherheits-Check: ist der verbotene Zustand erreichbar?

## Was am Ende herauskommt (Erwartungswerte)

- **Teil B**: `[open_valve, close_valve]` → `closed`; `[close_valve]` vom Start → `None` (undefiniert); `[arrive, depart]` → `no_bottle`.
- **Teil C**: beide Komponenten haben alle Zustände erreichbar.
- **Teil D**: das Produkt hat **4 Zustände**, alle erreichbar; von `('open','bottle')` ist das geteilte Ereignis `depart` **nicht** aktiviert (eine Flasche kann nie bei offenem Ventil weg — die Synchronisation funktioniert).
- **Teil E**: der verbotene Zustand `('open','no_bottle')` **ist erreichbar**, über ein einziges `open_valve` — die ungeregelte Anlage ist **unsicher**.

> **Die Lehre.** Ein endlicher Automat ist die Spezifikation des legalen Verhaltens eines automatisierten Systems, und **Erreichbarkeit** (schlichtes BFS) ist der Motor jedes Sicherheitsbeweises: Ein schlechter Zustand ist sicher, genau dann wenn er *nicht* erreichbar ist. **Parallelkomposition** baut ein Systemmodell aus Komponenten, wobei geteilte Ereignisse Synchronisation erzwingen. Und der zentrale Befund richtet das ganze Feld ein: Die ungeregelte Anlage *kann* einen verbotenen Zustand erreichen, braucht also einen **Supervisor** — aber weil manche Ereignisse (`arrive`) **nicht-steuerbar** sind, kann der Supervisor nur auf die steuerbaren wirken. Diese Asymmetrie ist das Thema von P02.

## Setup

```bash
cd modules/23-advanced-automation/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # automata.ipynb
```

Reines Python — keine numerischen Bibliotheken. Laufzeit sofort.

## Lösung

Die vollständige, ausgeführte Lösung liegt in [`solution/automata_solution.ipynb`](solution/automata_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: **Petri-Netze + Supervisory Control** — eine ressourcenteilende Zelle modellieren, ihre **Deadlocks** finden und einen **Supervisor synthetisieren**, der sie beweisbar verhindert.
- **P03 (final)**: **Model Predictive Control** — der beschränkte optimierende Regler der kontinuierlichen Ebene.
