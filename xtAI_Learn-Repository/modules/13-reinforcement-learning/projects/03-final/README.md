# Project 03 (final) — Inventory management with reinforcement learning

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project, _no code given_.** This final project gets **no scaffold** — you
build everything yourself: environment, reference solution, learners, evaluation. This is the
master's exam performance of the module: formulate a realistic decision problem as an MDP, solve
it model-free **and** validate the solution against an exact optimum.

**Why a `.py` project (no notebook)?** Because here several components (environment, DP
reference, four learning algorithms, evaluation) interact cleanly separated and *testable* —
exactly the architecture of real RL software. A notebook would blur that.

---

## The scenario (practical relevance)

A retailer controls the **warehouse** of a product — a classical operations-research problem
with direct practical relevance (supply chain, retail, spare-parts logistics). Each period:

1. Observe the inventory $i$ (the state).
2. **Order** a quantity $a$ (the action) — arrives immediately, inventory $x=i+a$ (capped by the
   warehouse capacity $M$).
3. **Stochastic demand** $D\sim\text{Poisson}(\lambda)$ occurs.
4. Sales $=\min(x,D)$; unsatisfied demand is **lost** (*lost sales*); the new inventory is
   $i'=\max(x-D,0)$.

**Cost per period** (negative reward):
$$r = -\big(\underbrace{K\,\mathbb 1[a>0]}_{\text{fixed ordering cost}} + \underbrace{c\,a}_{\text{unit cost}} + \underbrace{h\,\max(x-D,0)}_{\text{holding cost}} + \underbrace{p\,\max(D-x,0)}_{\text{shortage penalty}}\big).$$

The **tension**: order a lot → high holding costs; order little → expensive shortages; order
often → many fixed costs. The optimum balances everything.

> **Theory anchor (which you should confirm):** with **fixed costs** $K>0$ the optimal policy is
> provably an **(s, S) policy**: if the inventory falls to or below an **order point $s$**, fill
> up to a **target level $S$**; otherwise order nothing. Without fixed costs this becomes a
> **base-stock policy** ($s=S-1$). Your RL agent should *discover this structure itself*, without
> having it programmed in.

## Learning objectives of this project

- Model a real decision problem **precisely as an MDP** (states, actions with admissibility
  constraints, stochastic dynamics, a cost function).
- Recognize that the **model is known** here → **value iteration** (module 07!) yields the
  *exact* optimal policy as a **reference benchmark**.
- Implement **four model-free** methods that must **not** use the model — MC control, SARSA,
  Q-learning, expected SARSA — and evaluate their learned policies **quantitatively** against the
  DP optimum.
- Carry out and interpret a **hyperparameter study**.

## Assignment (step by step)

Build the following building blocks (the file names are a suggestion):

1. **`inventory_env.py` — the environment.** States $0..M$, actions $0..M$ with a **mask** (in
   state $i$ only $a\le M-i$ admissible). A Gym-like API `reset()`/`step(a) → (i',r,False)` (a
   continuing task, no terminal state). Demand via the Poisson PMF (bundle the tail at $D_{\max}$
   so that it sums to 1). **In addition** — only for the reference — an *explicit model*:
   `expected_reward(i,a)` and `transition_probs(i,a)`.
2. **`dp_reference.py` — value iteration** on the known model → the optimal policy, $V^*$, $Q^*$.
   Plus an **exact policy evaluation** `policy_value(π)` (solve $(I-\gamma P_\pi)V=R_\pi$) to grade
   arbitrary policies.
3. **`agents.py` — the four learners** (all ε-greedy, with action masks): MC control (every-visit,
   episodes of fixed length), **SARSA**, **Q-learning**, **expected SARSA**. *None* may use
   `expected_reward`/`transition_probs` — only $(s,a,r,s')$ experience.
4. **`run.py` — the experiment.** Train all of them, extract the greedy policy, evaluate it
   exactly and compare: the **optimality gap** in % relative to $V^*$ and the action agreement.
   Plot learning curves and the **target inventory-level curves** ($x=i+a$) of the learned vs.
   the optimal policy. Carry out the **α study** for Q-learning.
5. **`test_*.py` — tests** (the PMF is normalized, the cost formula, the masks, the transitions
   sum to 1, VI shows the (s,S) structure, `policy_value` reproduces $V^*$, Q-learning comes close
   to the optimum).

**Parameter suggestion:** $M=20$, $\lambda=8$, $c=2$, $K=10$, $h=1$, $p=6$, $\gamma=0.95$.
(These values produce a clearly visible (s,S) structure; feel free to experiment.)

## What should work / come out in the end

- **Value iteration** yields a clean **(s,S) policy** — e.g. with the suggested values: order
  point $s\approx4$, target level $S\approx16$ (below 5 units → fill up to 16, otherwise order
  nothing).
- The **model-free** learned policies lie **close to the optimum**: the optimality gap typically
  **1–7 %** (at $\varepsilon=0.1$, $\alpha=0.1$, ~300,000 steps). Order of magnitude e.g.:

  | Method | opt. gap |
  |---|---|
  | MC control | ~1–2 % |
  | Q-learning | ~3 % |
  | expected SARSA | ~4 % |
  | SARSA | ~7 % |

- **Interpretation you should deliver:**
  - Why is **SARSA** (on-policy) tendentially a bit further from the optimum than **Q-learning**
    (off-policy)? → the same logic as with cliff walking in project 02: SARSA prices the
    ε-exploration behavior into its values and learns a slightly *more conservative* policy.
  - Why is the **action agreement** often only ~50–70 %, even though the **value** is almost
    optimal? → Because the value landscape around the optimum is **flat** (several order
    quantities are almost equally good) and rarely visited states (high inventory) have noisy Q
    values but barely contribute to the return. **An important RL insight: near-optimal
    *performance* ≠ exactly-optimal *policy*.**
  - **α study:** too small an $\alpha$ does not finish learning within the fixed budget, too large
    an $\alpha$ is too noisy → there is a **sweet spot** (at the suggested values around
    $\alpha\approx0.05$).

## Reference solution

A complete, tested reference solution is in [`solution/`](solution/) (environment, DP reference,
four learners, `run.py`, 10 tests). **Only look inside once you have tried it yourself.**

Running:
```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_inventory.py   # 10 tests, ~1-2 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py              # experiment + plot, ~30 s
```
Pure `numpy` (+ `matplotlib` optional). Everything **tabular, CPU, seconds** — no training of
heavy models.

## Extensions (for the especially motivated)

- **Backorders** instead of lost sales (unsatisfied demand is delivered later → negative
  inventory as a state): how does the optimal policy change?
- **Lead time** $L>0$: orders arrive only after $L$ periods → the state has to contain the
  *pipeline* of open orders (a larger state space).
- **Non-stationary demand** (a seasonal $\lambda_t$): a constant $\alpha$ instead of the
  sample-average (cf. the bandit project) — does it now beat the DP policy for a *fixed* $\lambda$?
- **Base-stock without fixed costs** ($K=0$): confirm that $s=S-1$ results (fill up every period).
- Compare against the **newsvendor approximation**: $S \approx$ the $\frac{p}{p+h}$ quantile of
  the demand distribution — how close is the exact (s,S) solution to it?

---

# Projekt 03 (final) — Bestandsmanagement mit Reinforcement Learning (deutsche Fassung)

**Format: Python-Projekt, _ohne Code-Vorgabe_.** Dieses Abschlussprojekt bekommt **kein
Gerüst** — du baust alles selbst: Umgebung, Referenzlösung, Lerner, Auswertung. Das ist die
Master-Prüfungsleistung des Moduls: ein realistisches Entscheidungsproblem als MDP formulieren,
es modellfrei lösen **und** die Lösung gegen ein exaktes Optimum validieren.

**Warum ein `.py`-Projekt (kein Notebook)?** Weil hier mehrere Komponenten (Umgebung, DP-Referenz,
vier Lern-Algorithmen, Auswertung) sauber getrennt und *testbar* zusammenspielen — genau die
Architektur echter RL-Software. Ein Notebook würde das verwischen.

---

## Das Szenario (Praxisbezug)

Ein Händler steuert das **Lager** eines Produkts — ein klassisches Operations-Research-Problem
mit direktem Praxisbezug (Supply-Chain, Retail, Ersatzteillogistik). Jede Periode:

1. Bestand $i$ beobachten (Zustand).
2. Menge $a$ **bestellen** (Aktion) — trifft sofort ein, Bestand $x=i+a$ (gedeckelt durch
   Lagerkapazität $M$).
3. **Stochastische Nachfrage** $D\sim\text{Poisson}(\lambda)$ tritt ein.
4. Verkauf $=\min(x,D)$; nicht befriedigte Nachfrage geht **verloren** (*lost sales*);
   neuer Bestand $i'=\max(x-D,0)$.

**Kosten je Periode** (negative Belohnung):
$$r = -\big(\underbrace{K\,\mathbb 1[a>0]}_{\text{Bestellfixkosten}} + \underbrace{c\,a}_{\text{Stückkosten}} + \underbrace{h\,\max(x-D,0)}_{\text{Lagerkosten}} + \underbrace{p\,\max(D-x,0)}_{\text{Fehlmengenstrafe}}\big).$$

Die **Spannung**: viel bestellen → hohe Lagerkosten; wenig bestellen → teure Fehlmengen; oft
bestellen → viele Fixkosten. Das Optimum balanciert alles.

> **Theorie-Anker (den du bestätigen sollst):** Bei **Fixkosten** $K>0$ ist die optimale Politik
> nachweislich eine **(s, S)-Politik**: fällt der Bestand auf oder unter einen **Bestellpunkt
> $s$**, fülle auf ein **Zielniveau $S$** auf; sonst bestelle nichts. Ohne Fixkosten wird daraus
> eine **Base-Stock-Politik** ($s=S-1$). Dein RL-Agent soll diese Struktur *selbst entdecken*,
> ohne sie einprogrammiert zu bekommen.

## Lernziele dieses Projekts

- Ein reales Entscheidungsproblem **präzise als MDP** modellieren (Zustände, Aktionen mit
  Zulässigkeits-Constraints, stochastische Dynamik, Kostenfunktion).
- Erkennen, dass hier das **Modell bekannt** ist → **Value Iteration** (Modul 07!) liefert die
  *exakte* Optimalpolitik als **Referenz-Messlatte**.
- **Vier modellfreie** Verfahren implementieren, die das Modell **nicht** benutzen dürfen —
  MC-Control, SARSA, Q-Learning, Expected SARSA — und ihre gelernten Politiken **quantitativ**
  gegen das DP-Optimum bewerten.
- Eine **Hyperparameter-Studie** durchführen und interpretieren.

## Aufgabenstellung (Schritt für Schritt)

Baue die folgenden Bausteine (Dateinamen als Vorschlag):

1. **`inventory_env.py` — die Umgebung.** Zustände $0..M$, Aktionen $0..M$ mit **Maske**
   (in Zustand $i$ nur $a\le M-i$ zulässig). Eine Gym-artige API `reset()`/`step(a) → (i',r,False)`
   (fortlaufende Aufgabe, kein Terminalzustand). Nachfrage per Poisson-PMF (Tail auf $D_{\max}$
   bündeln, damit sie sich zu 1 summiert). **Zusätzlich** — nur für die Referenz — ein
   *explizites Modell*: `expected_reward(i,a)` und `transition_probs(i,a)`.
2. **`dp_reference.py` — Value Iteration** auf dem bekannten Modell → optimale Politik, $V^*$,
   $Q^*$. Plus eine **exakte Policy-Bewertung** `policy_value(π)` (löse
   $(I-\gamma P_\pi)V=R_\pi$), um beliebige Politiken zu benoten.
3. **`agents.py` — die vier Lerner** (alle ε-greedy, mit Aktionsmasken):
   MC-Control (every-visit, Episoden fester Länge), **SARSA**, **Q-Learning**, **Expected SARSA**.
   *Keiner* darf `expected_reward`/`transition_probs` benutzen — nur $(s,a,r,s')$-Erfahrung.
4. **`run.py` — das Experiment.** Trainiere alle, extrahiere die greedy-Politik, bewerte sie
   exakt und vergleiche: **Optimalitätslücke** in % gegenüber $V^*$ und Aktions-Übereinstimmung.
   Plotte Lernkurven und die **Ziel-Lagerstand-Kurven** ($x=i+a$) von gelernter vs. optimaler
   Politik. Führe die **α-Studie** für Q-Learning durch.
5. **`test_*.py` — Tests** (PMF normiert, Kostenformel, Masken, Transitions summieren zu 1,
   VI zeigt (s,S)-Struktur, `policy_value` reproduziert $V^*$, Q-Learning kommt nah ans Optimum).

**Parameter-Vorschlag:** $M=20$, $\lambda=8$, $c=2$, $K=10$, $h=1$, $p=6$, $\gamma=0.95$.
(Diese Werte erzeugen eine gut sichtbare (s,S)-Struktur; experimentiere ruhig.)

## Was am Ende funktionieren / herauskommen soll

- **Value Iteration** liefert eine saubere **(s,S)-Politik** — z. B. mit den Vorschlagswerten:
  Bestellpunkt $s\approx4$, Zielniveau $S\approx16$ (unter 5 Stück → auffüllen auf 16, sonst
  nichts bestellen).
- Die **modellfrei** gelernten Politiken liegen **nahe am Optimum**: Optimalitätslücke typisch
  **1–7 %** (bei $\varepsilon=0.1$, $\alpha=0.1$, ~300 000 Schritten). Größenordnung z. B.:

  | Verfahren | Opt.-Lücke |
  |---|---|
  | MC-Control | ~1–2 % |
  | Q-Learning | ~3 % |
  | Expected SARSA | ~4 % |
  | SARSA | ~7 % |

- **Interpretation, die du liefern sollst:**
  - Warum ist **SARSA** (on-policy) tendenziell etwas weiter vom Optimum als **Q-Learning**
    (off-policy)? → gleiche Logik wie beim Cliff Walking in Projekt 02: SARSA preist das
    ε-Explorationsverhalten in seine Werte ein und lernt eine leicht *konservativere* Politik.
  - Warum ist die **Aktions-Übereinstimmung** oft nur ~50–70 %, obwohl der **Wert** fast optimal
    ist? → Weil die Wertlandschaft um das Optimum **flach** ist (mehrere Bestellmengen sind fast
    gleich gut) und selten besuchte Zustände (hoher Bestand) verrauschte Q-Werte haben, aber
    kaum zum Ertrag beitragen. **Wichtige RL-Erkenntnis: near-optimale *Performance* ≠
    exakt-optimale *Politik*.**
  - **α-Studie:** zu kleines $\alpha$ lernt im festen Budget nicht fertig, zu großes $\alpha$
    ist zu verrauscht → es gibt ein **Sweet Spot** (bei den Vorschlagswerten um $\alpha\approx0.05$).

## Referenzlösung

Eine vollständige, getestete Musterlösung liegt in [`solution/`](solution/) (Umgebung, DP-Referenz,
vier Lerner, `run.py`, 10 Tests). **Sieh erst hinein, wenn du es selbst versucht hast.**

Ausführen:
```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_inventory.py   # 10 Tests, ~1-2 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py              # Experiment + Plot, ~30 s
```
Reines `numpy` (+ `matplotlib` optional). Alles **tabellarisch, CPU, Sekunden** — kein Training
schwerer Modelle.

## Erweiterungen (für die besonders Motivierten)

- **Backorders** statt lost sales (unbefriedigte Nachfrage wird nachgeliefert → negativer
  Bestand als Zustand): wie ändert sich die optimale Politik?
- **Lieferzeit (lead time)** $L>0$: Bestellungen treffen erst nach $L$ Perioden ein → der
  Zustand muss die *Pipeline* offener Bestellungen enthalten (größerer Zustandsraum).
- **Nichtstationäre Nachfrage** (saisonaler $\lambda_t$): konstantes $\alpha$ statt sample-average
  (vgl. Bandit-Projekt) — schlägt es jetzt die DP-Politik für *festes* $\lambda$?
- **Base-Stock ohne Fixkosten** ($K=0$): bestätige, dass $s=S-1$ wird (jede Periode auffüllen).
- Vergleiche gegen die **Newsvendor-Näherung**: $S \approx$ das $\frac{p}{p+h}$-Quantil der
  Nachfrageverteilung — wie nah liegt die exakte (s,S)-Lösung daran?
