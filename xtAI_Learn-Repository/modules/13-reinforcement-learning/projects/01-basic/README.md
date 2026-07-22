# Project 01 (basic) — Multi-armed bandits & the explore-exploit dilemma

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format: Jupyter notebook** (`bandits.ipynb`). **Why?** The bandit is the simplest RL case
(a single state) and isolates the *one* question that separates RL from everything else:
**explore or exploit?** A notebook fits here, because the code, short experiments and above all
the **learning-curve plots** belong together — you *see* the difference between the strategies.

---

## Goal

You build the **k-armed Gaussian testbed** (Sutton & Barto, ch. 2) and four action-selection
strategies, and compare them empirically over many random bandit problems:

- **greedy** — always the best action so far (pure exploit),
- **ε-greedy** — with probability ε a random action,
- **optimistic initialization** — high initial values force early exploration,
- **UCB** — upper confidence bound, "optimism in the face of uncertainty".

## Prior knowledge

Script **module 13, section 3.1** (explore-exploit, bandits, regret). Python/NumPy basics. No
further RL needed — this is the entry point.

## Assignment (step by step)

In the notebook the environment (`GaussianBandit`) and the experiment harness
(`run_experiment`, plots) are **given**. Your task is the learning core in the class
`BanditAgent`:

1. **`select_action`** — implement the selection depending on the strategy:
   - `epsilon`/`greedy`: with probability ε random, otherwise $\arg\max_a Q(a)$ (greedy = ε 0);
   - `ucb`: first the unpulled arms ($N=0$), otherwise $\arg\max_a[Q(a)+c\sqrt{\ln t/N(a)}]$.
2. **`update`** — the incremental sample-average update $Q(a)\mathrel{+}=\frac1{N(a)}(R-Q(a))$.
3. Run the big experiment and **interpret** the two plots + the regret curve.

## What should work in the end

Two plots like in the textbook (**mean reward** and **% optimal action** over time, averaged
over 2000 bandit problems) plus a **regret curve**. Expected order of magnitude after 1000
steps (10 arms): greedy ~34 % optimal action (gets stuck), ε=0.1 ~80 %, ε=0.01 slower,
optimistic ~70 % (with an early hump), **UCB ~87 % (the winner)**.

> **Setup / running.** Use the repo `venv`. Open the notebook in Jupyter:
> `/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (or select the `.venv` kernel in
> VS Code). The full experiment takes ~1 min on the CPU; for a fast test reduce
> `N_RUNS`/`N_STEPS`. Only `numpy` + `matplotlib` needed (already installed).

## Solution

Fully solved and **executed** in [`solution/bandits_solution.ipynb`](solution/bandits_solution.ipynb)
— try it yourself first! It contains a written conclusion at the end and three extension tasks
(vary UCB's `c`, a **non-stationary** variant with a constant α, add softmax/Boltzmann).

## What you learn here (transfer to the next projects)

The explore-exploit dilemma is **not** a bandit special case — it returns in full RL (projects
02 & 03), only with **many states**. The ε-greedy strategy from here is exactly the one SARSA
and Q-learning use for exploration. The incremental update
$\text{old}\leftarrow\text{old}+\alpha(\text{target}-\text{old})$ is the **basic pattern of
every RL update**.

---

# Projekt 01 (basic) — Multi-armed Bandits & das Explore-Exploit-Dilemma (deutsche Fassung)

**Format: Jupyter Notebook** (`bandits.ipynb`). **Warum?** Der Bandit ist der einfachste
RL-Fall (ein einziger Zustand) und isoliert die *eine* Frage, die RL von allem anderen trennt:
**explorieren oder exploitieren?** Ein Notebook passt hier, weil Code, kurze Experimente und
vor allem die **Lernkurven-Plots** zusammengehören — man *sieht* den Unterschied der Strategien.

---

## Ziel

Du baust das **k-armed Gaussian Testbed** (Sutton & Barto, Kap. 2) und vier
Aktions-Auswahl-Strategien, und vergleichst sie empirisch über viele zufällige Bandit-Probleme:

- **greedy** — immer die bisher beste Aktion (reines Exploit),
- **ε-greedy** — mit Wahrscheinlichkeit ε eine zufällige Aktion,
- **optimistische Initialisierung** — hohe Startwerte erzwingen frühe Exploration,
- **UCB** — Upper Confidence Bound, „Optimismus angesichts von Unsicherheit".

## Vorwissen

Skript **Modul 13, Abschnitt 3.1** (Explore-Exploit, Bandits, Regret). Python/NumPy-Grundlagen.
Kein weiteres RL nötig — das ist der Einstieg.

## Aufgabe (Schritt für Schritt)

Im Notebook sind Umgebung (`GaussianBandit`) und Experiment-Harness (`run_experiment`, Plots)
**vorgegeben**. Deine Aufgabe ist der Lernkern in der Klasse `BanditAgent`:

1. **`select_action`** — implementiere die Auswahl je nach Strategie:
   - `epsilon`/`greedy`: mit Wkt. ε zufällig, sonst $\arg\max_a Q(a)$ (greedy = ε 0);
   - `ucb`: zuerst ungezogene Arme ($N=0$), sonst $\arg\max_a[Q(a)+c\sqrt{\ln t/N(a)}]$.
2. **`update`** — inkrementelles sample-average-Update $Q(a)\mathrel{+}=\frac1{N(a)}(R-Q(a))$.
3. Führe das große Experiment aus und **interpretiere** die zwei Plots + die Regret-Kurve.

## Was am Ende funktionieren soll

Zwei Plots wie im Lehrbuch (**mittlere Belohnung** und **% optimale Aktion** über die Zeit,
gemittelt über 2000 Bandit-Probleme) plus eine **Regret-Kurve**. Erwartete Größenordnung nach
1000 Schritten (10 Arme): greedy ~34 % optimale Aktion (bleibt stecken), ε=0.1 ~80 %, ε=0.01
langsamer, optimistisch ~70 % (mit frühem Buckel), **UCB ~87 % (Sieger)**.

> **Setup / Ausführen.** Repo-`venv` nutzen. Notebook öffnen in Jupyter:
> `/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (oder in VS Code den
> `.venv`-Kernel wählen). Das volle Experiment dauert ~1 min auf der CPU; zum schnellen Testen
> `N_RUNS`/`N_STEPS` reduzieren. Nur `numpy` + `matplotlib` nötig (bereits installiert).

## Lösung

Vollständig gelöst und **ausgeführt** in [`solution/bandits_solution.ipynb`](solution/bandits_solution.ipynb)
— erst selbst probieren! Enthält am Ende ein ausformuliertes Fazit und drei Erweiterungs-Aufgaben
(UCB-`c` variieren, **nichtstationäre** Variante mit konstantem α, Softmax/Boltzmann ergänzen).

## Was du hier lernst (Transfer zu den nächsten Projekten)

Das Explore-Exploit-Dilemma ist **kein** Bandit-Spezialfall — es kehrt im vollen RL (Projekte 02
& 03) wieder, nur mit **vielen Zuständen**. Die ε-greedy-Strategie von hier ist exakt die, die
SARSA und Q-Learning zur Exploration verwenden. Das inkrementelle Update
$\text{alt}\leftarrow\text{alt}+\alpha(\text{Ziel}-\text{alt})$ ist das **Grundmuster jedes
RL-Updates**.
