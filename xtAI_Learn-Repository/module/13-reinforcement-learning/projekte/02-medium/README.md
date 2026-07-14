# Projekt 02 (medium) — SARSA vs. Q-Learning auf Cliff Walking

**Format: Python-Projekt** (`.py` + Tests). **Warum?** Hier geht es um eine echte
Umgebung, wiederverwendbare Agenten und eine Test-Suite, die das Verhalten *absichert* — das
ist strukturierter Code, kein Explorations-Notebook. Genau die Trennung
Umgebung ↔ Agent ↔ Trainingsschleife, die jede RL-Codebasis (und `gymnasium`) hat, baust du
hier selbst — **ohne** externe RL-Bibliothek.

---

## Ziel

Du implementierst die beiden zentralen **modellfreien Kontroll-Algorithmen** und stellst ihren
Unterschied am kanonischen Beispiel **Cliff Walking** (Sutton & Barto, Beispiel 6.6) heraus:

- **SARSA** — *on-policy*: lernt den Wert der Policy, die es **ausführt** (inkl. Exploration).
- **Q-Learning** — *off-policy*: lernt direkt die **optimale** Policy $Q_*$, während es einer
  explorierenden Policy folgt.

Der **eine** Unterschied steckt im Bootstrapping-Ziel des TD-Updates — mehr nicht. Genau das
sollst du sehen und verstehen.

## Vorwissen

Skript **Modul 13, Abschnitt 2.4–2.5** (SARSA, Q-Learning, GPI, ε-greedy). Projekt 01 (ε-greedy,
inkrementelles Update). Grundlegendes Python (Klassen, NumPy).

## Dateien

| Datei | Rolle |
|---|---|
| `cliff_walking.py` | Die Umgebung (4×12-Gitter mit Klippe). **Vorgegeben** — Infrastruktur. |
| `td_control.py` | Der Agent + Trainingsschleife. **Hier ist deine Arbeit** (3 TODOs). |
| `run.py` | Experiment: mittelt über viele Läufe, druckt Ergebnisse, zeichnet den Plot. |
| `test_td.py` | Test-Suite (10 Tests) — sichert Umgebung *und* Agenten ab. |

## Aufgabe (Schritt für Schritt)

In `td_control.py` sind **drei** Stellen mit `# TODO` markiert:

1. **`TDAgent.select_action`** — ε-greedy: mit Wkt. ε zufällig, sonst $\arg\max_a Q(s,a)$
   (mit fairem Tie-Breaking).
2. **`TDAgent.update`** — das TD-Update. Fallunterscheidung:
   - `done` → Ziel $=r$;
   - `sarsa` → Ziel $=r+\gamma\,Q(s',a')$;
   - `qlearning` → Ziel $=r+\gamma\max_a Q(s',a)$;
   - dann $Q(s,a)\mathrel{+}=\alpha\,(\text{Ziel}-Q(s,a))$.
3. **`train`** — die Episoden-Schleife. **Achtung SARSA:** die nächste Aktion $a'$ muss *vor*
   dem Update aus $s'$ gezogen und im nächsten Schritt tatsächlich ausgeführt werden (der
   $(s,a)\!\to\!(s',a')$-Übergang). Der Docstring skizziert den Ablauf.

Danach: `python run.py` ausführen und die zwei greedy-Policies + die Lernkurve interpretieren.

## Was am Ende funktionieren soll

`python test_td.py` → **alle 10 Tests grün**. `python run.py` reproduziert die Lehrbuch-Figur:

```
Algorithmus   Online-Ertrag (letzte 100 Ep.)   greedy-Ertrag
sarsa                 ~ -27                          -17
qlearning             ~ -52                          -13
```

- **Q-Learning** findet die **optimale, riskante** Route direkt an der Klippenkante
  (greedy-Ertrag **−13**) — verliert aber während des Trainings mehr Belohnung, weil
  ε-Exploration es gelegentlich abstürzen lässt (Online ~ −52).
- **SARSA** lernt eine **sicherere** Route mit Abstand zur Klippe (greedy-Ertrag **−17**) und
  erzielt dafür **online** deutlich mehr (~ −27), weil es die Explorationskosten *einpreist*.

Die greedy-Policies (ASCII-Karte) zeigen es direkt: Q-Learning geht die Reihe *direkt über* der
Klippe entlang, SARSA weicht nach oben aus.

> **Das ist die Kernbotschaft des Moduls:** „optimal im Grenzwert" (Q-Learning) ≠ „gut, während
> man noch exploriert und handelt" (SARSA). *Off-policy* lernt das Beste; *on-policy* lernt das
> Beste **unter Berücksichtigung des eigenen Explorationsverhaltens**.

## Ausführen / Setup

Repo-`venv`, nur `numpy` (+ `matplotlib` optional für den Plot). Aus dem Projektordner:

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_td.py   # Tests
/.../xtAI_Learn-Repository/.venv/bin/python run.py        # Vergleich + Plot (cliff_comparison.png)
```

Läuft in **Sekunden** auf der CPU (tabellarisch). `pytest` ist optional — die Tests haben einen
eigenen `__main__`-Runner.

## Lösung

Vollständig in [`loesung/`](loesung/) (identische Umgebung/Tests, gelöste `td_control.py`).
Erst selbst versuchen! Wenn ein Test hakt, sagt seine Meldung meist genau, welcher Fall
(Ziel, Terminalfall, Tie-Breaking, SARSA-vs-Q) noch nicht stimmt.

## Weiterdenken

- Setze `epsilon=0` in `run.py` — beide Algorithmen kollabieren zur gleichen greedy-Route.
  Warum? (Ohne Exploration verschwindet der Online-Unterschied — SARSA *ist* dann Q-Learning.)
- Ergänze **Expected SARSA** (Ziel $=r+\gamma\sum_{a'}\pi(a'|s')Q(s',a')$) und zeige, dass es
  bei greedy $\pi$ zu Q-Learning wird (Skript 2.4).
- Lass $\varepsilon$ über die Episoden abfallen ($\varepsilon_k\propto1/k$, GLIE) — konvergiert
  SARSA dann auch zur optimalen Route?
