# Projekt 03 (final) — Ein entscheidungstheoretischer MDP-Agent

**Modul 07 — Theorie der KI 2** · Format: **Python-Projekt, von Grund auf selbst gebaut**

> **Abschlussprojekt des Moduls.** Es gibt **keinen vorgegebenen Code** — du
> entwirfst und implementierst alles selbst. Das Projekt konsolidiert Teil 3
> (Nutzentheorie, MDPs, Bellman-Gleichungen, Value/Policy Iteration) und schlägt
> die Brücke zum Reinforcement Learning (Modul 13). Niveau: echte
> Master-Prüfungsleistung.

## Warum dieses Format & dieses Thema?

Ein MDP-Agent ist die direkte, ausführbare Umsetzung der Bellman-Optimalitäts­gleichung
und des MEU-Prinzips. Wenn du **Value Iteration** und **Policy Iteration** selbst
schreibst und ihre Konvergenz *beobachtest*, verstehst du den Kern der
sequenziellen Entscheidungsfindung — und siehst genau die Bausteine wieder, auf
denen RL aufbaut (nur dass dort $P$ und $R$ unbekannt sind). Eine echte,
modularisierte Codebasis ist hier richtig; das reine Verfahren ist zu strukturiert
für ein Wegwerf-Notebook.

## Ziel

Baue einen Agenten, der für einen **Markov-Entscheidungsprozess** $(S, A, P, R,
\gamma)$ die optimale Wertfunktion $V^\ast$ und die optimale Policy $\pi^\ast$
berechnet — auf **zwei** Wegen, deren Übereinstimmung du prüfst:

- **Value Iteration** — iteriere den Bellman-Optimalitäts-Operator bis Konvergenz;
- **Policy Iteration** — alterniere Policy Evaluation und Policy Improvement.

Als Umgebung dient die klassische **stochastische 4×3-Gridworld** (Russell &
Norvig): Bewegung gelingt zu 80 % in die gewünschte Richtung, je 10 % senkrecht
daneben; ein Wandfeld; ein +1- und ein −1-Terminal; Living Reward $-0{,}04$ pro
Schritt. Du sollst die optimale Policy **visualisieren** (Pfeilgitter), die
**Konvergenz** empirisch prüfen und untersuchen, **wie $\gamma$ und der Living
Reward die optimale Policy verändern**.

## Vorwissen

Skript Teil 3, insbesondere:
- MDP-Definition, Policy, Wertfunktion, **Bellman-(Optimalitäts-)Gleichung**;
- **Value Iteration** und der **Kontraktionsbeweis** (warum $B$ konvergiert);
- **Policy Iteration** (Policy Evaluation als lineares System, Policy Improvement).

Python: `dict`, Iteration, optional `matplotlib` für eine schönere Visualisierung.

## Was du bauen sollst — die Komponenten

1. **Die MDP-Umgebung.** Repräsentiere Zustände (Gitterzellen), Aktionen
   (N/S/E/W), das **verrauschte Übergangsmodell** $P(s'\mid s,a)$ (0.8/0.1/0.1,
   gegen Wand/Rand → stehen bleiben), die Belohnung $R(s)$ (Living Reward,
   ±1-Terminals) und $\gamma$. Terminalzustände sind absorbierend.

2. **Value Iteration.** Implementiere den Bellman-Optimalitäts-Update
   $$V_{k+1}(s) \leftarrow R(s) + \gamma \max_a \sum_{s'} P(s'\mid s,a)\,V_k(s'),$$
   iteriere bis das **maximale Bellman-Residuum** $\lVert V_{k+1}-V_k\rVert_\infty$
   unter eine Schranke fällt, und zähle die Iterationen. Lies die **greedy Policy**
   $\pi^\ast(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)V^\ast(s')$ ab.

3. **Policy Iteration.** Implementiere **Policy Evaluation** (löse bzw. iteriere
   $V^\pi(s)=R(s)+\gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$) und **Policy
   Improvement** (greedy bzgl. $V^\pi$); alterniere bis die Policy stabil ist.

4. **Analyse & Visualisierung.**
   - Gib $V^\ast$ als Zahlengitter und $\pi^\ast$ als **Pfeilgitter** aus.
   - Zeige, dass VI und PI **dieselbe** Policy und (bis auf Toleranz) dieselben
     Werte liefern, und vergleiche die **Iterationszahlen**.
   - **$\gamma$-Studie:** Berechne $\pi^\ast$ für mehrere $\gamma$ (z. B. 1.0, 0.9,
     0.5, 0.2) und beschreibe, wie sich die Policy ändert.
   - **Living-Reward-Studie:** Variiere $R$ (z. B. $-0{,}04, -0{,}5, -2{,}0, 0{,}0$)
     und erkläre das qualitative Verhalten (bei stark negativem $R$ nimmt der Agent
     riskante Abkürzungen sogar Richtung $-1$).

## Akzeptanzkriterien (Abnahmetest)

- [ ] Value Iteration reproduziert die bekannten AIMA-Utilities der 4×3-Welt
      (u. a. $V(0,0)\approx0{,}705$, $V(2,2)\approx0{,}918$, $V(3,0)\approx0{,}388$;
      Toleranz $<0{,}005$);
- [ ] die daraus abgelesene optimale Policy stimmt mit der klassischen Lösung
      überein (obere Reihe → → →, unten ↑ ← ← ←, mit ↑ neben der Wand);
- [ ] Policy Iteration liefert **dieselbe** Policy und Werte wie Value Iteration
      (und braucht dafür **deutlich weniger** äußere Iterationen);
- [ ] die $\gamma$- und Living-Reward-Studien zeigen nachvollziehbare
      Policy-Änderungen;
- [ ] VI meldet die Zahl der Iterationen bis zur Konvergenz (Referenz: ~30 bei
      $\gamma=1$, Residuum $<10^{-8}$).

## Selbstcheck-Fragen (schriftlich beantworten)

1. **Warum konvergiert Value Iteration?** Skizziere das Kontraktionsargument
   (Bellman-Operator ist $\gamma$-Kontraktion in $\lVert\cdot\rVert_\infty$,
   Banachscher Fixpunktsatz). Warum konvergiert es in dieser Welt **auch bei
   $\gamma=1$** (Stichwort: absorbierende Terminals / proper policy)?
2. **Warum braucht Policy Iteration so viel weniger Iterationen** als Value
   Iteration, obwohl jede Iteration teurer ist?
3. **Wieso ändert kleineres $\gamma$ die Policy** in Richtung kürzerer/riskanterer
   Wege? Argumentiere über den diskontierten Return.
4. **Warum nimmt der Agent bei stark negativem Living Reward** Abkürzungen sogar in
   Richtung des $-1$-Terminals? Verbinde das mit dem Vorzeichen und der Größe von $R$.
5. **Wo genau ist die Brücke zum Reinforcement Learning (Modul 13)?** Was kennt ein
   RL-Agent *nicht*, das dein MDP-Agent hier voraussetzt, und wie ändert das das
   Vorgehen (Stichwort: Sampling / Temporal-Difference statt exaktem $P,R$)?

## Erweiterungen (optional, für Vertiefung)

- **Q-Werte & modifizierte Policy Iteration** (Policy Evaluation nur wenige Sweeps).
- **matplotlib-Heatmap** der Utilities mit überlagerten Policy-Pfeilen.
- **Konvergenzkurve** ($\lVert V_{k+1}-V_k\rVert_\infty$ über $k$, log-Skala) —
  bestätige die *geometrische* Rate $\gamma^k$.
- **Asynchrones/priorisiertes Value Iteration** oder eine größere, selbst
  entworfene Gridworld mit Fallen und mehreren Zielen.

## Musterlösung

In **`solution/`** liegt eine vollständige, getestete Referenzimplementierung:
- `gridworld.py` — die 4×3-MDP-Umgebung + ASCII-Visualisierung (Pfeile/Werte),
- `mdp.py` — Value Iteration, greedy Policy, Policy Iteration (mit Policy Evaluation),
- `demo.py` — Utilities/Policy, VI-vs-PI-Vergleich, $\gamma$- und Living-Reward-Studie,
- `test_mdp.py` — Abnahmetest gegen die AIMA-Referenzwerte.

Referenz: VI ~34 Iterationen ($\gamma=1$), PI ~5 Iterationen, **identische Policy**.
**Erst nach eigenem Versuch ansehen.**

```bash
source ../../../../.venv/bin/activate    # nur Standardbibliothek nötig
cd loesung && python demo.py             # Utilities, Policy, gamma-/Reward-Studie
python test_mdp.py                       # Abnahmetest
```
