# Projekt 01 (basic) — Multi-armed Bandits & das Explore-Exploit-Dilemma

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
