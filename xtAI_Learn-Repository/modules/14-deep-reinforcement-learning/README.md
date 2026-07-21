# Modul 14 — Deep Reinforcement Learning for Optimal Control

> **Worum geht es?** In Modul 13 war die Wertfunktion eine **Tabelle** $Q(s,a)$ — das
> funktioniert nur bei kleinen, diskreten Zustandsräumen. Sobald der Zustand hochdimensional
> oder **kontinuierlich** wird (Kamerabild, Gelenkwinkel eines Roboters, Lage einer Drohne),
> ist keine Tabelle mehr speicherbar. **Deep RL** ersetzt die Tabelle durch ein **neuronales
> Netz**, das über ähnliche Zustände *verallgemeinert*. Dieses Modul führt die Kernfamilien
> ein: **wertbasiert** (DQN und Erweiterungen), **policy-basiert** (REINFORCE, Actor-Critic,
> PPO) und **kontinuierliche Steuerung** (DDPG/TD3/SAC) — und schlägt die Brücke zur
> **klassischen Optimalsteuerung** (LQR, Bellman/HJB, Pontryagin), von der der Modulname kommt.

**Hilfreiche Vorkenntnisse:** Modul 13 (MDP, Bellman, Q-Learning, Policy Gradient-Idee, die
„deadly triad"), Modul 05/09 (neuronale Netze & PyTorch, Backprop, Optimierer), Grundlagen
Analysis/lineare Algebra (Gradient, Eigenwerte, quadratische Formen).

**Diese Module solltest du vorher gemacht haben:**
- **Modul 13 (RL)** — *zwingend*. Wir bauen direkt auf Bellman-Gleichungen, Q-Learning, SARSA,
  ε-greedy, dem Policy-Gradient-Theorem und der **deadly triad** auf. Jeder Algorithmus hier ist
  die „tiefe" Version eines dortigen tabellarischen Verfahrens.
- **Modul 05 (ML 2)** — MLPs, Backpropagation, SGD/Adam, Regularisierung, PyTorch-Praxis.

> **⚠️ Hardware-Hinweis.** Echtes Deep RL (Atari aus Pixeln, MuJoCo-Roboter, großes PPO) braucht
> GPU-Stunden bis -Tage und ist auf einem Laptop **nicht** sinnvoll trainierbar. Deshalb: die
> **teuren** Verfahren erklären wir vollständig **theoretisch** in formaler Notation, und die
> Projekte nutzen bewusst **kleine** Aufgaben (selbstgebautes CartPole, lineare Systeme),
> **kleine Netze** und **wenige Episoden**, sodass alles in Minuten auf CPU/MPS läuft. Das
> Verständnis ist identisch — nur der Maßstab ist reduziert.

---

## Lernziele

Nach diesem Modul kannst du …

- erklären, **warum** und **wie** ein neuronales Netz die Q-Tabelle ersetzt, und die
  **deadly triad** (Funktionsapproximation + Bootstrapping + off-policy) als zentrale
  Instabilitätsquelle benennen;
- **DQN** vollständig herleiten — inkl. **Experience Replay** und **Target Network** — und
  begründen, wie beide Tricks die Triade zähmen; **Double DQN**, **Dueling** und
  **Prioritized Replay** einordnen;
- den Gegensatz **wertbasiert vs. policy-basiert** erklären und das **Policy-Gradient-Theorem**,
  **REINFORCE**, die **Baseline/Advantage** und **Actor-Critic** (A2C, GAE) formal aufschreiben;
- **PPO** und sein **clipped surrogate objective** verstehen (warum „proximal");
- **kontinuierliche Steuerung** (deterministischer Policy-Gradient → **DDPG/TD3**, Entropie →
  **SAC**) einordnen;
- die Verbindung zur **klassischen Optimalsteuerung** herstellen: **Bellman ↔ HJB-Gleichung**,
  **LQR/Riccati** als exakt lösbaren Spezialfall, **Pontryagins Maximumprinzip**, **MPC**;
- ein kleines Deep-RL-System in **PyTorch** selbst bauen und stabilisieren.

---

## 1 · Grundlagen — Von der Tabelle zum Netz

### 1.1 Warum Funktionsapproximation (Rückblick & Vertiefung)

Tabellarisches Q-Learning speichert einen Wert pro $(s,a)$. Das scheitert doppelt: **Speicher**
(Go hat $\sim10^{170}$ Zustände) und **Erfahrung** (man kann nie jeden Zustand besuchen).
Kontinuierliche Zustände ($s\in\mathbb R^n$) haben sogar *überabzählbar* viele Einträge. Lösung:
approximiere
$$\hat q(s,a;\mathbf w)\approx q_*(s,a),\qquad \hat v(s;\mathbf w)\approx v_*(s),$$
mit einem parametrisierten Funktionsapproximator (Gewichte $\mathbf w$). Ein **neuronales Netz**
ist die ausdrucksstärkste Wahl und lernt die **Features** selbst (statt sie von Hand zu bauen).
Der Gewinn ist **Generalisierung**: ein Update in einem Zustand verbessert die Schätzung in
*allen ähnlichen* Zuständen. Das ist der Grund, warum Deep RL überhaupt in riesigen Räumen
funktioniert — und zugleich die Quelle aller Instabilität.

### 1.2 Das Lernziel als Regression — und warum es tückisch ist

In Modul 13 war das **semi-gradient**-TD-Update
$$\mathbf w \leftarrow \mathbf w + \alpha\big[\underbrace{R+\gamma \hat q(S',A';\mathbf w)}_{\text{Ziel (bootstrapped)}} - \hat q(S,A;\mathbf w)\big]\nabla_{\mathbf w}\hat q(S,A;\mathbf w).$$
„Semi", weil man so tut, als sei das Ziel **fix**, obwohl es selbst von $\mathbf w$ abhängt. Bei
tabellarischer Darstellung harmlos; bei neuronaler Approximation entstehen drei Probleme, die
zusammen die **deadly triad** bilden:

1. **Funktionsapproximation** — ein Update „leckt" auf andere Zustände (kann falsche
   verschieben).
2. **Bootstrapping** — das Ziel enthält die eigene (fehlerhafte) Schätzung → Fehler können sich
   selbst verstärken.
3. **off-policy** — man lernt über Daten einer anderen Verteilung als die Zielpolitik.

Alle drei zusammen können **Divergenz** verursachen (Werte laufen nach $\pm\infty$). Zwei
weitere Verletzungen der üblichen Supervised-Learning-Annahmen kommen hinzu: die Daten sind
**stark korreliert** (aufeinanderfolgende Transitionen ähneln sich) und die **Zielverteilung
bewegt sich** (die Policy ändert sich beim Lernen → *non-stationary target*). Die Deep-RL-
Algorithmen sind im Kern **Tricks, um trotzdem stabil zu bleiben.**

---

## 2 · Wertbasiertes Deep RL: DQN

### 2.1 Deep Q-Network (DQN)

**DQN** (Mnih et al. 2013/2015, das „Atari-Paper") ist Q-Learning mit einem neuronalen Netz
$Q(s,a;\theta)$ (meist: Eingang $s$, ein Ausgang je Aktion). Man minimiert den erwarteten
quadratischen **TD-Fehler**:
$$L(\theta)=\mathbb E_{(s,a,r,s')\sim \mathcal D}\Big[\big(\underbrace{r+\gamma\max_{a'}Q(s',a';\theta^-)}_{\text{Ziel }y} - Q(s,a;\theta)\big)^2\Big].$$

Zwei Zutaten machen daraus ein *stabil trainierbares* Verfahren:

**(a) Experience Replay.** Speichere Transitionen $(s,a,r,s')$ in einem **Puffer** $\mathcal D$
(z. B. letzte $10^6$) und trainiere auf **zufälligen Minibatches** daraus. Das (i) **entkorreliert**
die Daten (bricht die zeitliche Abhängigkeit) und (ii) nutzt jede Erfahrung **mehrfach**
(Dateneffizienz). — Adressiert Triaden-Komponente „korrelierte Daten".

**(b) Target Network.** Das Ziel $y$ verwendet ein **eingefrorenes** Netz $\theta^-$, das nur
alle $C$ Schritte (oder per Polyak-Mittel $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$) auf
$\theta$ nachgezogen wird. Ohne das würde man „auf ein sich bewegendes Ziel schießen" — das Ziel
$y$ hinge sofort von jedem Gewichtsupdate ab und könnte oszillieren/divergieren. Das eingefrorene
Ziel macht das Lernen wieder einem stabilen **Regressionsproblem** ähnlich. — Adressiert
„non-stationary target".

**Trainingsschleife (Pseudocode):**
```
initialisiere Q(θ), Target Q(θ⁻)=θ, leeren Replay-Puffer D
für jede Episode:
    s = env.reset()
    wiederhole:
        a = ε-greedy(Q(s,·;θ))                 # Exploration wie in Modul 13
        s', r, done = env.step(a);  D.push(s,a,r,s',done)
        Minibatch B ~ D
        y = r + γ·(1-done)·max_a' Q(s',a';θ⁻)  # Ziel mit Target-Netz
        θ ← θ - lr·∇θ  mean_B (Q(s,a;θ) - y)²   # ein SGD/Adam-Schritt
        alle C Schritte:  θ⁻ ← θ                # Target-Netz nachziehen
        s = s'
    reduziere ε
```
Projekt 01 baut genau das auf einem selbstgebauten CartPole.

### 2.2 DQN-Erweiterungen (kurz, aber vollständig)

- **Double DQN** — der $\max$-Operator im Ziel **überschätzt** systematisch (max über
  verrauschte Schätzer ⇒ positiver Bias). Lösung: **entkopple Auswahl und Bewertung** —
  wähle die Aktion mit dem *Online*-Netz, bewerte sie mit dem *Target*-Netz:
  $$y = r + \gamma\,Q\big(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\big).$$
- **Dueling DQN** — zerlege $Q(s,a)=V(s)+A(s,a)$ in **Zustandswert** und **Vorteil**
  (mit $A$ um seinen Mittelwert zentriert). Nützlich, wenn der Zustandswert dominiert und die
  Aktionswahl kaum zählt.
- **Prioritized Experience Replay** — ziehe Transitionen mit großem TD-Fehler **häufiger**
  (dort ist am meisten zu lernen), korrigiert per Importance-Sampling-Gewichten.
- **Rainbow** kombiniert diese und weitere (n-step, verteilungsbasiertes RL, Noisy Nets).

**Grenze von DQN:** braucht ein diskretes $\arg\max_a$ → **keine kontinuierlichen** Aktionen
(dafür Abschnitt 4). Und: rein wertbasiert, lernt eine deterministische greedy-Policy.

---

## 3 · Policy-basiertes Deep RL

### 3.1 Warum die Policy direkt lernen?

Wertbasierte Methoden lernen $Q$ und leiten die Policy *indirekt* (greedy) ab. **Policy-
Gradient**-Methoden parametrisieren die Policy **direkt**, $\pi_\theta(a\mid s)$, und optimieren
sie per Gradientenaufstieg. Vorteile: (i) **kontinuierliche** Aktionsräume natürlich (Gaußsche
Policy), (ii) **stochastische** Optimalpolicies möglich (wichtig bei partieller
Beobachtbarkeit/Spielen), (iii) glatte Verbesserung statt sprunghaftem $\arg\max$. Nachteil:
höhere **Varianz**, oft weniger dateneffizient.

### 3.2 Das Policy-Gradient-Theorem & REINFORCE

Ziel ist die erwartete Rendite $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0]$. Das
**Policy-Gradient-Theorem** liefert (bemerkenswert: **ohne** Ableitung der Umgebungsdynamik):
$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\!\Big[\sum_{t} \nabla_\theta\log\pi_\theta(A_t\mid S_t)\,\Psi_t\Big],$$
wobei $\Psi_t$ ein **Kredit-Signal** ist. Verschiedene Wahlen von $\Psi_t$ ergeben verschiedene
Algorithmen:

| $\Psi_t$ | Verfahren |
|---|---|
| $G_t$ (voller Return) | **REINFORCE** (Monte Carlo) |
| $G_t - b(S_t)$ (Baseline) | REINFORCE **mit Baseline** (Varianzreduktion) |
| $Q^\pi(S_t,A_t)$ | Actor-Critic (Q-Form) |
| $A^\pi(S_t,A_t)=Q^\pi-V^\pi$ | **Advantage** Actor-Critic (A2C) |
| $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ | TD-Actor-Critic |

**REINFORCE**-Update (nach ganzer Episode): $\theta\leftarrow\theta+\alpha\sum_t \nabla_\theta\log\pi_\theta(A_t|S_t)\,G_t$.
Intuition: **erhöhe** die Wahrscheinlichkeit von Aktionen, die zu hohem Return führten. Die
$\log$-Ableitung heißt **score function** (der „REINFORCE-Trick").

**Baseline.** Man darf von $\Psi_t$ eine **beliebige zustandsabhängige** Funktion $b(S_t)$
abziehen, ohne den Erwartungswert (und damit die Unverzerrtheit) zu ändert — denn
$\mathbb E_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s)]=0$. Eine gute Baseline (typisch
$b=\hat V(s)$) senkt die **Varianz** drastisch. $G_t-\hat V(S_t)$ schätzt den **Advantage**:
„war diese Aktion besser als der Durchschnitt in diesem Zustand?".

### 3.3 Actor-Critic & GAE

**Actor-Critic** kombiniert beide Welten: ein **Actor** $\pi_\theta$ (wählt Aktionen) und ein
**Critic** $\hat V_\phi$ (bewertet Zustände, bootstrappt à la TD). Der Critic liefert die
Baseline/den Advantage, der Actor macht den Policy-Gradient-Schritt. **A2C** ist die synchrone,
**A3C** die asynchrone (parallele Worker) Variante.

Der **Advantage** kann über verschieden viele Schritte geschätzt werden — ein Bias/Varianz-
Trade-off wie bei n-step in Modul 13. **Generalized Advantage Estimation (GAE)** mittelt sie
geometrisch mit einem Parameter $\lambda$:
$$\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t=R_{t+1}+\gamma \hat V(S_{t+1})-\hat V(S_t).$$
$\lambda=0$ → reines TD (niedrige Varianz, mehr Bias), $\lambda=1$ → Monte-Carlo-Advantage.

### 3.4 PPO — Proximal Policy Optimization

Naives Policy-Gradient ist **empfindlich gegenüber der Schrittweite**: ein zu großer Update
kann die Policy „zerstören" (sie besucht dann nur noch schlechte Zustände, aus denen sie sich
kaum erholt). **TRPO** löste das mit einer harten KL-Nebenbedingung; **PPO** (Schulman et al.
2017) ist die einfachere, heute meistgenutzte Variante. Mit dem
**Wahrscheinlichkeitsverhältnis** $r_t(\theta)=\dfrac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{\text{old}}}(A_t|S_t)}$
maximiert PPO das **geklippte** Ziel
$$L^{\text{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big].$$
Das **Clipping** entfernt den Anreiz, $r_t$ weit über $1\pm\epsilon$ zu treiben — die Policy
bleibt in der **Nähe** (*proximal*) der alten, was den Update stabilisiert und mehrere
Epochen pro Datensatz erlaubt (dateneffizienter als Vanilla-PG). PPO ist der De-facto-Standard
für viele kontinuierliche und diskrete Aufgaben (Robotik, RLHF für LLMs).

---

## 4 · Kontinuierliche Steuerung & Optimal Control

### 4.1 Deterministischer Policy-Gradient: DDPG, TD3

Bei **kontinuierlichen** Aktionen ($a\in\mathbb R^m$) ist $\max_a Q(s,a)$ nicht mehr trivial.
**DDPG** (Deep Deterministic Policy Gradient) lernt eine **deterministische** Actor-Policy
$\mu_\theta(s)$ und einen Critic $Q_\phi(s,a)$; der Actor wird in Richtung höheren Q-Werts
bewegt: $\nabla_\theta J\approx\mathbb E[\nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)]$
(Kettenregel — „bewege die Aktion bergauf im Q-Gebirge"). Es ist im Kern **DQN für
kontinuierliche Aktionen** (mit Replay & Target-Netzen). Exploration per additivem Rauschen.

**TD3** (Twin Delayed DDPG) fixt DDPGs Überschätzung mit drei Tricks: **zwei** Critics (nimm das
Minimum → weniger Überschätzung), **verzögerte** Actor-Updates, und **Target-Policy-Smoothing**
(Rauschen im Ziel).

### 4.2 Maximum-Entropy-RL: SAC

**Soft Actor-Critic (SAC)** maximiert Rendite **plus Entropie** der Policy:
$$J(\pi)=\sum_t\mathbb E\big[R_{t+1}+\alpha\,\mathcal H(\pi(\cdot\mid S_t))\big].$$
Der **Entropie-Bonus** $\alpha\mathcal H$ belohnt „so zufällig wie möglich, solange die Aufgabe
gelöst wird" → bessere Exploration, robustere Policies, stabileres Training. SAC ist off-policy,
sample-effizient und einer der stärksten Algorithmen für kontinuierliche Steuerung.

### 4.3 Die Brücke zur klassischen Optimalsteuerung

Der Modulname ist **„for Optimal Control"** — RL ist die **daten-/modellfreie** Schwester der
klassischen **optimalen Regelung**. Die zentralen Verbindungen:

- **Bellman ↔ HJB.** Die Bellman-Optimalitätsgleichung ist die **zeitdiskrete** Form der
  **Hamilton-Jacobi-Bellman (HJB)**-Gleichung der kontinuierlichen Optimalsteuerung. Beide sagen:
  „optimaler Wert jetzt = sofortiger Nutzen + optimaler Wert danach". RL löst sie durch **Lernen
  aus Stichproben**, wenn das Modell unbekannt ist.
- **LQR — der exakt lösbare Fall.** Für ein **lineares** System $x_{t+1}=Ax_t+Bu_t$ mit
  **quadratischen** Kosten $\sum_t (x_t^\top Q x_t + u_t^\top R u_t)$ ist die optimale Steuerung
  eine **lineare Rückführung** $u_t=-Kx_t$, wobei $K$ aus der **algebraischen Riccati-Gleichung**
  folgt. Das ist die „Value Iteration mit geschlossener Lösung" — und dient (in Projekt 03) als
  **exakte Referenz**, an der sich ein gelernter RL-Regler messen muss (genau wie Value Iteration
  in Modul 13 die Referenz war).
- **Pontryagins Maximumprinzip** — der zweite große Zugang der Optimalsteuerung (notwendige
  Bedingungen über den *adjungierten Zustand/das Kostenfunktional*), komplementär zum
  HJB/Dynamic-Programming-Zugang.
- **MPC (Model Predictive Control)** — wenn ein (gelerntes oder bekanntes) Modell vorliegt:
  optimiere in jedem Schritt über einen **endlichen Horizont**, führe nur den ersten Schritt aus,
  wiederhole. Enge Verwandtschaft zu **modellbasiertem RL**.

### 4.4 Modellbasiertes RL (Ausblick)

Statt nur Werte/Policy zu lernen, kann man ein **Modell** der Dynamik $\hat p(s'|s,a)$ lernen und
darin *planen* (Dyna, PILCO, world models, MuZero). Vorteil: **Dateneffizienz** (Planung im
Modell ist billig). Nachteil: Modellfehler pflanzen sich fort. Hybridverfahren (Dyna:
kombiniere echte + modell-generierte Erfahrung) sind ein aktives Feld.

---

## 5 · Praxis-Stolpersteine (Deep RL ist berüchtigt fragil)

- **Reproduzierbarkeit** — Ergebnisse schwanken stark über Random-Seeds; immer über mehrere
  Seeds mitteln.
- **Reward Shaping** — schlecht gewählte Belohnungen führen zu *reward hacking* (Agent
  optimiert das Falsche).
- **Hyperparameter** — Lernraten, Netzgröße, Replay-Größe, Target-Update-Frequenz, $\gamma$ sind
  sensibel; kleine Änderungen kippen das Training.
- **Exploration** — in Umgebungen mit spärlicher Belohnung reicht ε-greedy/Gauß-Rauschen oft
  nicht (→ intrinsische Motivation, Curiosity).
- **Sim-to-Real** — in Simulation gelernte Policies übertragen sich schlecht auf echte Hardware
  (*reality gap*); Gegenmittel: Domain Randomization.
- **Debugging** — erst auf einer *winzigen*, lösbaren Aufgabe verifizieren (genau unser Ansatz),
  Lernkurven & Q-Werte beobachten (divergieren sie?).

---

## 6 · Zusammenfassung / Cheat-Sheet

**Landkarte.**
```
                         Deep RL
        ┌───────────────────┼─────────────────────┐
   wertbasiert         policy-basiert        kontinuierl. Control
   DQN                 REINFORCE             DDPG  (det. PG)
   +Double/Dueling     +Baseline             TD3   (2 Critics)
   +Prioritized Replay Actor-Critic (A2C)    SAC   (max-entropy)
   (diskrete Aktionen) PPO (clipped)         └── Optimal Control: LQR/HJB/MPC
```

**DQN-Ziel:** $y=r+\gamma\max_{a'}Q(s',a';\theta^-)$ · Loss $=(y-Q(s,a;\theta))^2$ · **Replay**
+ **Target-Netz** = Zähmung der deadly triad.

**Double-DQN-Ziel:** $y=r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.

**Policy-Gradient:** $\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)\,\Psi_t]$,
$\Psi_t\in\{G_t,\ G_t-b(s),\ A^\pi(s,a),\ \delta_t\}$.

**Advantage:** $A(s,a)=Q(s,a)-V(s)$ · **GAE**: $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$.

**PPO:** maximiere $\mathbb E[\min(r_t\hat A_t,\ \text{clip}(r_t,1{-}\epsilon,1{+}\epsilon)\hat A_t)]$,
$r_t=\pi_\theta/\pi_{\theta_{old}}$.

**Optimal Control:** Bellman ↔ **HJB**; linear+quadratisch ⇒ **LQR** $u=-Kx$ (Riccati) = exakte
Referenz; **MPC** = planen im Modell über Horizont.

---

## 7 · Selbsttest

<details>
<summary><b>1.</b> Warum ersetzt Deep RL die Q-Tabelle durch ein Netz, und was gewinnt/riskiert man?</summary>

Weil Tabellen bei großen/kontinuierlichen Zustandsräumen weder speicherbar noch besuchbar sind.
Ein Netz **generalisiert** über ähnliche Zustände (ein Update hilft vielen). Risiko: es entsteht
die **deadly triad** (Funktionsapproximation + Bootstrapping + off-policy) → mögliche Divergenz,
plus korrelierte Daten und bewegtes Ziel.
</details>

<details>
<summary><b>2.</b> Wofür dienen Experience Replay und Target Network in DQN — je genau ein Problem?</summary>

**Experience Replay**: bricht die **zeitliche Korrelation** der Daten (und nutzt Erfahrung
mehrfach → Dateneffizienz). **Target Network**: friert das Bootstrapping-**Ziel** ein →
verhindert das „Schießen auf ein bewegtes Ziel" (non-stationary target) und macht das Lernen
einem stabilen Regressionsproblem ähnlich.
</details>

<details>
<summary><b>3.</b> Was ist der Überschätzungs-Bias in DQN und wie behebt ihn Double DQN?</summary>

$\max_{a'}Q(s',a')$ nimmt das Maximum über *verrauschte* Schätzer → systematisch **zu hoch**
(positiver Bias). **Double DQN** entkoppelt **Auswahl** (Online-Netz $\theta$) und **Bewertung**
(Target-Netz $\theta^-$): $y=r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.
</details>

<details>
<summary><b>4.</b> Formuliere das Policy-Gradient-Theorem und den Nutzen einer Baseline.</summary>

$\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)\,\Psi_t]$. Eine
zustandsabhängige **Baseline** $b(s)$ (typisch $\hat V(s)$) darf von $\Psi_t$ abgezogen werden,
**ohne** den Erwartungswert zu ändern (unverzerrt), reduziert aber die **Varianz** stark;
$G_t-\hat V(s)$ schätzt den **Advantage**.
</details>

<details>
<summary><b>5.</b> Was unterscheidet Actor-Critic von REINFORCE?</summary>

REINFORCE nutzt den **vollen Monte-Carlo-Return** $G_t$ (unverzerrt, hohe Varianz, braucht
Episodenende). **Actor-Critic** hat zusätzlich einen **Critic** $\hat V_\phi$, der **bootstrappt**
(TD) und als Baseline/Advantage dient → niedrigere Varianz, Online-Updates. Actor = Policy,
Critic = Wertschätzer.
</details>

<details>
<summary><b>6.</b> Warum „proximal"? Was bewirkt das Clipping in PPO?</summary>

Zu große Policy-Updates können die Policy zerstören. PPO hält die neue Policy **nahe**
(*proximal*) an der alten, indem es das Verhältnis $r_t=\pi_\theta/\pi_{\theta_{old}}$ auf
$[1-\epsilon,1+\epsilon]$ **klippt** — das entfernt den Anreiz, $r_t$ weit zu treiben, und erlaubt
mehrere Update-Epochen pro Datenbatch (stabil + dateneffizient).
</details>

<details>
<summary><b>7.</b> Warum kann DQN keine kontinuierlichen Aktionen, und was tut DDPG dagegen?</summary>

DQN braucht $\arg\max_a Q(s,a)$ — bei kontinuierlichem $a$ nicht in geschlossener Form lösbar.
**DDPG** lernt einen **deterministischen Actor** $\mu_\theta(s)$ und bewegt ihn per Kettenregel
**bergauf** im Critic $Q_\phi$: $\nabla_\theta J\approx\mathbb E[\nabla_aQ_\phi\,\nabla_\theta\mu_\theta]$.
Es ist „DQN für kontinuierliche Aktionen".
</details>

<details>
<summary><b>8.</b> Was ist der Zusammenhang zwischen Bellman-Gleichung und HJB, und was ist LQR?</summary>

Die Bellman-Optimalitätsgleichung ist die **zeitdiskrete** Form der **Hamilton-Jacobi-Bellman**-
Gleichung der kontinuierlichen Optimalsteuerung. **LQR** ist der Spezialfall *linearer Dynamik +
quadratischer Kosten*: die optimale Steuerung ist die lineare Rückführung $u=-Kx$ mit $K$ aus der
**Riccati-Gleichung** — eine exakt lösbare Referenz.
</details>

<details>
<summary><b>9.</b> Was macht das Entropie-Ziel in SAC?</summary>

SAC maximiert Rendite **+** Policy-**Entropie** ($\alpha\mathcal H(\pi)$). Der Bonus belohnt
„so zufällig wie möglich, solange die Aufgabe gelöst wird" → bessere **Exploration**, robustere
Policies, stabileres Training.
</details>

<details>
<summary><b>10.</b> Nenne drei Gründe, warum Deep RL notorisch instabil/schwer reproduzierbar ist.</summary>

Beliebige drei: deadly triad (Divergenz), hohe **Seed-Varianz**, sensible **Hyperparameter**,
**korrelierte/nicht-stationäre** Daten, spärliche Belohnung/Exploration, **reward hacking**,
Sim-to-Real-Gap. Deshalb: erst auf winzigen Aufgaben verifizieren, über Seeds mitteln.
</details>

---

## 8 · Literatur & Quellen

**Bücher & Kurse (kostenlos):**
- 📗 **Sutton & Barto — *Reinforcement Learning: An Introduction* (2018)**, Kap. 9–13
  (Funktionsapproximation, Policy Gradient). Frei als PDF. *Fundament.*
- 🌐 **OpenAI Spinning Up in Deep RL** (spinningup.openai.com) — **die** praktische Einführung:
  saubere Herleitungen (VPG→TRPO→PPO→DDPG→TD3→SAC) **plus** referenzierbarer Code. Frei.
  *Einsteiger→vertiefend, sehr empfohlen.*
- 🎥 **UC Berkeley CS285 *Deep Reinforcement Learning* (Sergey Levine)** — Videos + Folien frei.
  *Vertiefend, umfassend.*
- 🎥 **DeepMind × UCL RL Lecture Series** — Nachfolger von Silvers Kurs. *Einsteiger→vertiefend.*

**Schlüssel-Paper (vertiefend):**
- Mnih et al. (2015), *Human-level control through deep RL* (**DQN/Nature**).
- van Hasselt et al. (2016), *Deep RL with Double Q-learning*.
- Wang et al. (2016), *Dueling Network Architectures*.
- Schulman et al. (2015), *High-Dimensional Continuous Control Using GAE*.
- Schulman et al. (2017), *Proximal Policy Optimization* (**PPO**).
- Lillicrap et al. (2016), *Continuous control with deep RL* (**DDPG**); Fujimoto et al. (2018),
  **TD3**; Haarnoja et al. (2018), **SAC**.

**Optimal Control (Brücke):**
- 📘 **Bertsekas — *Dynamic Programming and Optimal Control*** / *Reinforcement Learning and
  Optimal Control* (2019) — verbindet beide Welten formal. *Vertiefend.*
- 🌐 **Steven Brunton — *Control Bootcamp* (YouTube)** — LQR, Riccati, HJB anschaulich. *Einsteiger.*

**Praxis/Tooling:**
- 🌐 **Gymnasium** (gymnasium.farama.org) — Standard-Umgebungs-API (CartPole, Pendulum, MuJoCo).
  Wir bauen im Modul *ohne* Gym (didaktisch), aber die API solltest du kennen.
- 🌐 **Stable-Baselines3** — gepflegte, getestete Implementierungen (DQN/PPO/SAC/TD3) für die
  Praxis (nicht zum Lernen der Interna).

---

## Nächstes Modul

Damit ist der RL-Block (Module 13–14) abgeschlossen. Es folgt **Modul 15 — Machine Learning for
Networks 1**. Das in diesem Modul gebaute Fundament (wert- vs. policy-basiert, Actor-Critic,
Optimalsteuerung) ist Grundlage für RL-Anwendungen in Robotik (Module 21/22), Advanced
Automation (23) und überall dort, wo **sequenzielle Entscheidungen unter Unsicherheit** getroffen
werden.
