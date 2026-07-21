# Modul 13 — Reinforcement Learning and Computational Decision-Making

> **Worum geht es?** Ein **Agent** steht in einer **Umgebung**, wählt **Aktionen**, erhält
> dafür **Belohnungen** und sieht **Zustände** — und soll durch *Ausprobieren* lernen, so zu
> handeln, dass die **langfristige** Belohnung maximal wird. Niemand sagt ihm die richtige
> Aktion (kein Supervisor mit Labels); er bekommt nur ein *Bewertungssignal*, oft
> **verzögert**. Das ist **Reinforcement Learning (RL)**. In diesem Modul bauen wir die
> klassische, **tabellarische** RL-Theorie von Grund auf: vom **Markov-Entscheidungsprozess**
> über die **Bellman-Gleichungen** zu **Monte-Carlo**, **Temporal-Difference-Lernen**,
> **SARSA** und **Q-Learning**, dem **Explore-Exploit-Dilemma** und einem Ausblick auf
> **Funktionsapproximation** (die Brücke zu Deep RL in Modul 14). Alles läuft mit reinem
> `numpy` auf der CPU — tabellarisches RL ist rechnerisch billig.

**Hilfreiche Vorkenntnisse:** Wahrscheinlichkeitsrechnung (Erwartungswert, bedingte
Wahrscheinlichkeit), etwas lineare Algebra, Grundidee der dynamischen Programmierung, NumPy.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 07 (Theorie der KI 2)** — dort hast du **MDPs**, **Value Iteration** und **Policy
  Iteration** auf der 4×3-Gridworld schon von Hand gebaut. Das ist der Fall, in dem das Modell
  (Übergänge $P$ und Belohnungen $R$) **bekannt** ist. RL beginnt genau da, wo dieses Wissen
  **fehlt** — der Agent muss $P$ und $R$ nicht kennen, sondern lernt aus *Erfahrung*. Wir
  knüpfen direkt an und referenzieren die dortigen Ergebnisse.
- **Modul 04/05 (Machine Learning 1/2)** — Begriffe wie *Lernrate*, *Stichprobenschätzer*,
  *Bias/Varianz*, *Funktionsapproximation* kehren hier wieder.

---

## Lernziele

Nach diesem Modul kannst du …

- das **RL-Problem** formal als **Markov-Entscheidungsprozess** $(\mathcal S,\mathcal A,P,R,\gamma)$
  fassen und von *supervised/unsupervised learning* abgrenzen;
- **Return**, **Discount**, **Zustands-** und **Aktionswertfunktionen** ($V^\pi$, $Q^\pi$) sowie
  die **Bellman-Erwartungs-** und **Bellman-Optimalitätsgleichungen** herleiten und deuten;
- den Unterschied zwischen **modellbasiert** (Planung: Value/Policy Iteration) und
  **modellfrei** (Lernen: MC, TD) präzise erklären;
- **Monte-Carlo-Prädiktion/-Kontrolle**, **TD(0)**, **n-step TD** und **TD(λ)** mit
  Eligibility Traces gegenüberstellen (Bias/Varianz, Bootstrapping);
- **SARSA** (on-policy) und **Q-Learning** (off-policy) implementieren, ihren Unterschied am
  **Cliff-Walking**-Beispiel begründen und **Expected SARSA** einordnen;
- das **Explore-Exploit-Dilemma** verstehen und **ε-greedy**, **optimistische Initialisierung**,
  **UCB** und **Boltzmann/Softmax** vergleichen — zuerst am **Multi-armed Bandit**;
- **Konvergenzbedingungen** (Robbins-Monro, GLIE) benennen;
- erklären, warum **tabellarisches** RL bei großen Zustandsräumen scheitert und wie
  **Funktionsapproximation** (linear, semi-gradient) und die **„deadly triad"** das Bild
  verändern — als Übergang zu **Deep RL (Modul 14)**.

---

## 1 · Grundlagen — Das RL-Problem und der MDP

### 1.1 Der Regelkreis Agent ↔ Umgebung

RL formalisiert **sequenzielle Entscheidungsfindung**. Zu diskreten Zeitschritten
$t=0,1,2,\dots$ läuft folgende Schleife:

```
        Aktion A_t
   ┌──────────────────────►┌───────────────┐
   │                        │   Umgebung    │
┌──┴────┐                   │  (Environment)│
│ Agent │                   └───────┬───────┘
└──▲────┘   Zustand S_{t+1}         │
   │        Belohnung R_{t+1}       │
   └────────────────────────────────┘
```

Der Agent beobachtet den **Zustand** $S_t\in\mathcal S$, wählt eine **Aktion**
$A_t\in\mathcal A$, und die Umgebung antwortet mit einer **Belohnung** $R_{t+1}\in\mathbb R$
und einem **Folgezustand** $S_{t+1}$. Das erzeugt eine **Trajektorie**
$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots$$

Der entscheidende Unterschied zu **überwachtem Lernen**: es gibt **kein Label** „richtige
Aktion". Das Feedback ist nur **evaluativ** (wie gut war es?), nicht **instruktiv** (was wäre
richtig gewesen?), und oft **verzögert** — eine schlechte Belohnung jetzt kann Folge einer
Aktion vor 20 Schritten sein (**credit assignment problem**). Zusätzlich beeinflusst der Agent
durch sein Handeln, **welche Daten** er als Nächstes sieht (nicht i.i.d.!) — er muss selbst
**explorieren**.

### 1.2 Der Markov-Entscheidungsprozess (MDP)

Ein (endlicher) **MDP** ist ein Tupel $(\mathcal S,\mathcal A,P,R,\gamma)$:

- $\mathcal S$ — endliche **Zustandsmenge**;
- $\mathcal A$ — endliche **Aktionsmenge** (ggf. zustandsabhängig $\mathcal A(s)$);
- $P(s'\mid s,a)=\Pr[S_{t+1}=s'\mid S_t=s,A_t=a]$ — **Übergangsdynamik**;
- $R(s,a)$ bzw. $R(s,a,s')$ — erwartete **Belohnung**;
- $\gamma\in[0,1]$ — **Discount-Faktor**.

Die **Markov-Eigenschaft** ist die zentrale Annahme: die Zukunft hängt nur vom **aktuellen**
Zustand ab, nicht von der ganzen Vergangenheit —
$$\Pr[S_{t+1}\mid S_t,A_t] = \Pr[S_{t+1}\mid S_0,A_0,\dots,S_t,A_t].$$
„Der Zustand fasst alles Relevante zusammen." Ist das verletzt, hat man ein **POMDP**
(partiell beobachtbar) — dazu am Ende mehr.

> **Vollständige Dynamik.** Am kompaktesten schreibt man alles in eine Funktion
> $p(s',r\mid s,a)=\Pr[S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a]$, aus der sich
> $P$ und $R$ durch Marginalisierung ergeben:
> $P(s'\mid s,a)=\sum_r p(s',r\mid s,a)$ und $R(s,a)=\sum_{s',r} r\,p(s',r\mid s,a)$.

### 1.3 Policy, Return, Discount

Eine **Policy** $\pi$ ist die Strategie des Agenten — eine Abbildung von Zuständen auf
(Verteilungen über) Aktionen:
$$\pi(a\mid s)=\Pr[A_t=a\mid S_t=s]\quad(\text{stochastisch}),\qquad a=\pi(s)\ (\text{deterministisch}).$$

Ziel ist **nicht** die nächste Belohnung, sondern der **Return** — die kumulierte
(diskontierte) zukünftige Belohnung ab $t$:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.$$

Warum **diskontieren** (Faktor $\gamma^k$)?
- **Mathematisch:** bei $\gamma<1$ und beschränkten Belohnungen konvergiert die unendliche
  Summe ($|G_t|\le R_{\max}/(1-\gamma)$).
- **Modellierung:** unmittelbare Belohnung ist mehr wert als ferne (wie ein Zinssatz);
  $\gamma$ nahe 0 = „myopisch/gierig", $\gamma$ nahe 1 = „weitsichtig".
- Bei **episodischen** Aufgaben (mit Terminalzustand, z. B. ein Spiel endet) kann
  $\gamma=1$ sinnvoll sein; bei **kontinuierlichen** Aufgaben braucht man $\gamma<1$.

Praktische Rekursion (Basis für alles Folgende):
$$G_t = R_{t+1} + \gamma\,G_{t+1}.$$

### 1.4 Wertfunktionen

Die **Zustandswertfunktion** $V^\pi(s)$ ist der **erwartete Return**, wenn man in $s$ startet
und danach $\pi$ folgt:
$$V^\pi(s) = \mathbb E_\pi\!\left[G_t \mid S_t=s\right].$$

Die **Aktionswertfunktion** $Q^\pi(s,a)$ ist der erwartete Return, wenn man in $s$ **zuerst
$a$** nimmt und danach $\pi$ folgt:
$$Q^\pi(s,a) = \mathbb E_\pi\!\left[G_t \mid S_t=s, A_t=a\right].$$

Zusammenhang: $V^\pi(s)=\sum_a \pi(a\mid s)\,Q^\pi(s,a)$. $Q$ ist praktischer für **Kontrolle**,
weil man ohne Modell direkt die beste Aktion ablesen kann: $\arg\max_a Q(s,a)$.

### 1.5 Die Bellman-Gleichungen

Setzt man die Return-Rekursion $G_t = R_{t+1}+\gamma G_{t+1}$ in die Definition ein, erhält man
die **Bellman-Erwartungsgleichung** — ein *lineares* Gleichungssystem, das $V^\pi$ eindeutig
festlegt:
$$\boxed{\;V^\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\big[r + \gamma V^\pi(s')\big]\;}$$
und analog
$$Q^\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\Big[r + \gamma \sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a')\Big].$$

Intuition: „Der Wert eines Zustands = sofortige Belohnung + diskontierter Wert des
Folgezustands, gemittelt über Policy und Dynamik." Es ist eine **Konsistenzbedingung**: der
Wert *jetzt* muss zum Wert *danach* passen.

**Optimalität.** Es existiert (bei endlichem MDP) eine **optimale Policy** $\pi_*$, die $V^\pi(s)$
für *alle* $s$ gleichzeitig maximiert. Ihre Wertfunktionen $V_*=V^{\pi_*}$, $Q_*=Q^{\pi_*}$
erfüllen die **Bellman-Optimalitätsgleichungen** (jetzt *nichtlinear* durch das $\max$):
$$\boxed{\;V_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma V_*(s')\big]\;}$$
$$\boxed{\;Q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma \max_{a'} Q_*(s',a')\big]\;}$$

Aus $Q_*$ liest man die **optimale Policy** *greedy* ab: $\pi_*(s)=\arg\max_a Q_*(s,a)$. Das ist
der Kern: **kennt man $Q_*$, hat man optimales Verhalten** — ganz ohne Modell.

---

## 2 · Aufbau — Von Planung zu Lernen

Wir ordnen alle Verfahren nach **zwei Achsen**:

| | **Modell bekannt** ($p$ gegeben) | **Modell unbekannt** (nur Erfahrung) |
|---|---|---|
| **Prädiktion** (evaluiere gegebenes $\pi$) | Policy Evaluation (DP) | **Monte Carlo**, **TD(0)** |
| **Kontrolle** (finde bestes $\pi$) | **Value/Policy Iteration** (DP) | **MC-Control**, **SARSA**, **Q-Learning** |

Die linke Spalte („**Planung**") ist Modul 07. Die rechte Spalte („**Lernen**") ist der Kern
dieses Moduls.

### 2.1 Rückblick: Dynamische Programmierung (Modell bekannt)

Wenn $p$ und $R$ bekannt sind, macht man aus den Bellman-Gleichungen **Update-Regeln**:

- **Policy Evaluation:** iteriere $V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$
  bis zur Konvergenz (Fixpunkt = $V^\pi$).
- **Policy Iteration:** abwechselnd (1) evaluieren, (2) *greedy verbessern*
  $\pi'(s)=\arg\max_a Q^\pi(s,a)$ — endet in endlich vielen Schritten.
- **Value Iteration:** wende direkt das Bellman-**Optimalitäts**-Update an,
  $V_{k+1}(s)\leftarrow\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$.

> **Modul-07-Brücke.** Genau das hast du in **Modul 07, Projekt 03** auf der 4×3-Gridworld
> gebaut: Value Iteration konvergierte in 34, Policy Iteration in 5 Iterationen zur *gleichen*
> optimalen Policy, mit den AIMA-Utilities. Der **Bellman-Operator** ist eine **Kontraktion**
> mit Faktor $\gamma$ (Banach-Fixpunktsatz) → garantierte Konvergenz. **Der Haken:** man
> braucht $p$ und $R$ **explizit**. In der echten Welt kennt man sie selten. RL löst das.

### 2.2 Monte-Carlo-Methoden (lernen aus vollständigen Episoden)

**Idee:** Wenn man $p$ nicht kennt, *mittelt* man einfach **beobachtete Returns**. $V^\pi(s)$ ist
ein Erwartungswert — also schätze ihn durch den Stichprobenmittelwert der Returns, die nach
Besuchen von $s$ tatsächlich eintraten.

Für jede Episode berechne $G_t$ rückwärts; für jeden besuchten Zustand $s$ (**first-visit**:
nur beim ersten Auftreten pro Episode; **every-visit**: bei jedem):
$$N(s)\leftarrow N(s)+1,\qquad V(s)\leftarrow V(s) + \tfrac1{N(s)}\big(G_t - V(s)\big).$$
Das ist der **inkrementelle Mittelwert**. Die Form $\text{neu}\leftarrow\text{alt}+\alpha(\text{Ziel}-\text{alt})$
mit **Fehler** $(\text{Ziel}-\text{alt})$ ist das **Grundmuster jedes RL-Updates**; mit fester
Lernrate $\alpha$ statt $1/N$ „vergisst" man alte Erfahrung exponentiell (gut für
nichtstationäre Umgebungen).

**Eigenschaften:** MC ist **unverzerrt** (der Return ist eine echte Stichprobe von $G_t$), aber
**hochvariant** (der ganze Zufall einer Episode steckt drin). MC braucht **vollständige,
terminierende** Episoden (kein Bootstrapping) und lernt **erst am Episodenende**.

### 2.3 Temporal-Difference-Lernen (TD) — der Durchbruch

**TD** kombiniert das Beste aus DP und MC: es lernt **aus Erfahrung** (wie MC), aber
**bootstrappt** (wie DP) — es aktualisiert eine Schätzung mit einer *anderen* Schätzung, ohne
das Episodenende abzuwarten. Das **TD(0)**-Update für Prädiktion:
$$\boxed{\;V(S_t) \leftarrow V(S_t) + \alpha\underbrace{\big[\,\overbrace{R_{t+1}+\gamma V(S_{t+1})}^{\text{TD-Ziel}} - V(S_t)\,\big]}_{\delta_t\ =\ \text{TD-Fehler}}\;}$$

Der **TD-Fehler** $\delta_t = R_{t+1}+\gamma V(S_{t+1}) - V(S_t)$ misst die Überraschung
(„war der Folgezustand besser oder schlechter als erwartet?"). TD kann **online**, nach *jedem
Schritt* lernen, auch in **nicht-terminierenden** Aufgaben.

**Bias/Varianz-Vergleich:**

| | MC | TD(0) |
|---|---|---|
| Ziel | echter Return $G_t$ | $R_{t+1}+\gamma V(S_{t+1})$ (geschätzt) |
| Bias | **unverzerrt** | **verzerrt** (bootstrapping) |
| Varianz | **hoch** | **niedrig** |
| braucht Episodenende? | ja | nein |
| Markov ausgenutzt? | nein | ja (nutzt Struktur) |

TD ist in der Praxis meist **schneller** und **dateneffizienter** — der Preis ist der Bias, der
mit besserer Schätzung verschwindet.

**Dazwischen: n-step & TD(λ).** Es gibt ein ganzes Spektrum. Der **n-step-Return**
$$G_{t:t+n}=R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n})$$
interpoliert zwischen TD ($n=1$) und MC ($n=\infty$). **TD(λ)** mittelt geometrisch **alle**
n-step-Returns mit Gewichten $(1-\lambda)\lambda^{n-1}$ zum **λ-Return** $G_t^\lambda$. Effizient
implementiert man das *nicht* durch Vorausschauen, sondern durch **Eligibility Traces**
$e(s)$, die rückwärts „Verantwortlichkeit" verteilen (Abschnitt 3.2).

### 2.4 Modellfreie Kontrolle: SARSA vs. Q-Learning

Für **Kontrolle** (bestes $\pi$ finden) lernen wir $Q(s,a)$ statt $V(s)$, denn ohne Modell
braucht man $Q$, um greedy handeln zu können. Beide folgen dem TD-Muster, unterscheiden sich
aber im **Ziel**.

**SARSA (on-policy).** Der Name kommt vom Tupel $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$. Man wählt
$A_{t+1}$ **mit der aktuellen (explorierenden) Policy** und bootstrappt mit *diesem* Wert:
$$\boxed{\;Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\big]\;}$$
SARSA lernt den Wert der Policy, **die es tatsächlich ausführt** (inkl. Exploration).

**Q-Learning (off-policy).** Man bootstrappt mit der **greedy** (besten) Aktion, unabhängig
davon, was tatsächlich gewählt wurde:
$$\boxed{\;Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\,\max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)\big]\;}$$
Q-Learning approximiert direkt $Q_*$ — es lernt die **optimale** Policy, während es einer
anderen (explorierenden) **Verhaltens-Policy** folgt. Das ist der Kern von *off-policy*: die
Policy, die man *lernt*, ≠ die Policy, die man *ausführt*.

**Expected SARSA.** Ersetzt das $Q(S_{t+1},A_{t+1})$ durch den **Erwartungswert** über die
Policy — reduziert die Varianz aus der zufälligen Wahl von $A_{t+1}$:
$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a') - Q(S_t,A_t)\Big].$$
Bei greedy $\pi$ wird Expected SARSA **zu** Q-Learning — es ist die verallgemeinernde Klammer.

> **Das Cliff-Walking-Aha (Projekt 02).** Auf einer Gridworld mit einer „Klippe" (Absturz =
> −100) findet **Q-Learning** die *optimale, riskante* Route direkt an der Kante entlang —
> lernt sie aber, während es gelegentlich (durch ε-Exploration) abstürzt, also mit **schlechterem
> Online-Ertrag**. **SARSA** lernt eine *sicherere* Route mit Abstand zur Klippe, weil es die
> Explorationskosten in seinen Werten *mit einpreist*, und erzielt **online mehr Belohnung**.
> Fazit: „optimal im Grenzwert" (Q-Learning) ≠ „gut, während man noch exploriert" (SARSA).

### 2.5 Generalized Policy Iteration & ε-greedy-Kontrolle

Alle Kontrollverfahren sind Instanzen von **Generalized Policy Iteration (GPI)**: ein Tanz aus
(a) **Evaluation** (Werte an die Policy annähern) und (b) **Improvement** (Policy greedy bzgl.
Werten machen). Sie treiben sich gegenseitig zum gemeinsamen Fixpunkt $\pi_*,Q_*$.

Damit Kontrolle funktioniert, muss der Agent **explorieren** — würde er immer nur greedy
handeln, sähe er nie Alternativen. Die einfachste Lösung ist eine **ε-greedy**-Policy:
$$\pi(a\mid s)=\begin{cases}1-\varepsilon+\varepsilon/|\mathcal A| & a=\arg\max_{a'}Q(s,a')\\[2pt]\varepsilon/|\mathcal A| & \text{sonst.}\end{cases}$$
Mit Wahrscheinlichkeit $1-\varepsilon$ die beste bekannte Aktion, mit $\varepsilon$ eine
zufällige. Für **Konvergenz zur Optimal-Policy** verlangt man **GLIE** (*Greedy in the Limit
with Infinite Exploration*): jeder $(s,a)$ wird unendlich oft besucht **und** $\varepsilon\to0$
(z. B. $\varepsilon_k=1/k$). Zusammen mit den **Robbins-Monro**-Bedingungen an die Lernrate,
$$\sum_k \alpha_k = \infty,\qquad \sum_k \alpha_k^2 < \infty,$$
konvergiert tabellarisches Q-Learning **mit Wahrscheinlichkeit 1** gegen $Q_*$ (Watkins &
Dayan 1992). In der Praxis nimmt man oft ein konstantes, kleines $\alpha$ und ein langsam
fallendes $\varepsilon$ — Theorie und Praxis weichen hier bewusst ab.

---

## 3 · Advanced-Themen

### 3.1 Das Explore-Exploit-Dilemma & Multi-armed Bandits

Der **Bandit** ist ein MDP mit *einem* Zustand: $k$ „Arme" (Aktionen), jeder liefert Belohnung
aus einer unbekannten Verteilung mit Mittelwert $q_*(a)$. Er isoliert das **Explore-Exploit-
Dilemma** in Reinform: **exploitieren** (den bisher besten Arm ziehen) vs. **explorieren**
(einen unsicheren Arm testen, der *vielleicht* besser ist). Zu viel Exploit → man bleibt in
einem lokalen Optimum stecken; zu viel Explore → man verschenkt Belohnung. Man misst die Güte
über den **Regret** $\rho_T = T\,q_*(a^*) - \sum_{t=1}^{T}\mathbb E[q_*(A_t)]$ (entgangene
Belohnung gegenüber dem besten Arm).

Strategien (Projekt 01 vergleicht sie):
- **ε-greedy** — simpel, aber exploriert *uniform* und *ewig* (linearer Regret bei festem ε).
- **Optimistische Initialisierung** — setze $Q_0(a)$ absichtlich zu hoch; jede noch nicht
  probierte Aktion wirkt attraktiv → *automatische* Anfangsexploration, danach greedy. Trick,
  kein Allheilmittel (wirkt nur früh, nicht bei nichtstationär).
- **UCB (Upper Confidence Bound)** — wähle $A_t=\arg\max_a\big[Q(a)+c\sqrt{\ln t / N(a)}\big]$:
  „Optimismus angesichts von Unsicherheit". Der Bonusterm ist groß für selten gezogene Arme und
  schrumpft mit Wissen. UCB1 erreicht **logarithmischen** Regret $O(\ln T)$ — beweisbar
  fast-optimal.
- **Boltzmann/Softmax** — wähle $a$ mit Wahrscheinlichkeit $\propto e^{Q(a)/\tau}$; die
  **Temperatur** $\tau$ steuert von gierig ($\tau\to0$) zu uniform ($\tau\to\infty$).
- **Thompson Sampling** (bayesianisch) — halte eine Posterior-Verteilung über jedes $q_*(a)$,
  ziehe eine Stichprobe je Arm, spiele den Sieger. Oft empirisch das Beste; hier nur erwähnt.

### 3.2 Eligibility Traces — TD(λ) effizient

Ein **Eligibility Trace** $e_t(s)$ (bzw. $e_t(s,a)$) ist ein Kurzzeitgedächtnis, das markiert,
welche Zustände *kürzlich und häufig* besucht wurden und daher für einen aktuellen TD-Fehler
**„verantwortlich"** sind. **Accumulating trace:**
$$e_t(s)=\gamma\lambda\,e_{t-1}(s)+\mathbb 1[S_t=s],\qquad V(s)\leftarrow V(s)+\alpha\,\delta_t\,e_t(s)\ \ \forall s.$$
Der eine TD-Fehler $\delta_t$ wird auf *alle* kürzlich besuchten Zustände verteilt, abgestuft
nach $ (\gamma\lambda)^{k}$. Für $\lambda=0$ ergibt sich TD(0), für $\lambda=1$ (näherungsweise)
MC. Das löst das **credit-assignment**-Problem elegant und beschleunigt oft das Lernen. Die
Kontroll-Varianten heißen **SARSA(λ)** und **Watkins's Q(λ)**.

### 3.3 Funktionsapproximation — warum Tabellen nicht reichen

Bisher war $Q$ eine **Tabelle** $|\mathcal S|\times|\mathcal A|$. Das scheitert, sobald der
Zustandsraum groß/kontinuierlich ist (Schach $\sim10^{47}$, Go $\sim10^{170}$, Bilder als
Zustand). Lösung: **parametrisiere** $\hat V(s;\mathbf w)\approx V^\pi(s)$ mit wenigen
Gewichten $\mathbf w$ — linear $\hat V(s;\mathbf w)=\mathbf w^\top\mathbf x(s)$ mit
**Feature-Vektor** $\mathbf x(s)$, oder nichtlinear (neuronales Netz → Deep RL). Man
**verallgemeinert** dann über ähnliche Zustände.

Das Lernziel wird ein Regressionsproblem; das **semi-gradient TD(0)**-Update lautet
$$\mathbf w \leftarrow \mathbf w + \alpha\big[R_{t+1}+\gamma\hat V(S_{t+1};\mathbf w) - \hat V(S_t;\mathbf w)\big]\,\nabla_{\mathbf w}\hat V(S_t;\mathbf w).$$
„Semi"-Gradient, weil man den Gradienten des bootstrapped Ziels **ignoriert** (es hängt selbst
von $\mathbf w$ ab) — man behandelt das TD-Ziel als fix.

> **Die „deadly triad".** Kombiniert man **(1) Funktionsapproximation + (2) Bootstrapping +
> (3) off-policy-Training**, kann RL **divergieren** (Werte laufen ins Unendliche). Jede
> Komponente einzeln ist ok, aber alle drei zusammen sind gefährlich. Deep Q-Networks (Modul 14)
> haben genau diese Triade und brauchen deshalb Tricks (**Experience Replay**, **Target
> Network**), um sie zu zähmen. Das ist die konzeptuelle Brücke zu Modul 14.

### 3.4 Policy-Gradient-Methoden (Ausblick)

Statt Werte zu lernen und daraus greedy eine Policy abzuleiten, kann man die **Policy direkt**
parametrisieren, $\pi_\theta(a\mid s)$, und $\theta$ per Gradientenaufstieg auf die erwartete
Rendite $J(\theta)=\mathbb E_{\pi_\theta}[G_0]$ optimieren. Das **Policy-Gradient-Theorem** gibt
$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(A_t\mid S_t)\,Q^{\pi_\theta}(S_t,A_t)\big],$$
woraus der **REINFORCE**-Algorithmus folgt (Monte-Carlo-Schätzung von $Q$ durch $G_t$, oft mit
einer **Baseline** $b(s)$ zur Varianzreduktion). Vorteile: natürliche **stochastische** Policies,
**kontinuierliche** Aktionsräume. **Actor-Critic** verbindet beides (ein *Actor* $\pi_\theta$, ein
*Critic* $\hat V$). Das ist die Grundlage moderner Verfahren (A2C/A3C, PPO, DDPG, SAC) — Thema
von **Modul 14 (Deep RL for Optimal Control)**.

### 3.5 Wenn Markov verletzt ist: POMDPs

Sieht der Agent nicht den vollen Zustand, sondern nur eine **Beobachtung** $O_t$, hat man einen
**Partially Observable MDP (POMDP)**. Optimales Handeln erfordert dann einen **Belief State**
(Posterior über den wahren Zustand) — exakt oft unlösbar. Praktische RL-Antworten: den Zustand
aus einem **Fenster** vergangener Beobachtungen bauen oder ein **rekurrentes** Netz (RNN/LSTM,
Modul 09) die Historie zusammenfassen lassen.

---

## 4 · Zusammenfassung / Cheat-Sheet

**Begriffe.** MDP $(\mathcal S,\mathcal A,P,R,\gamma)$ · Policy $\pi(a|s)$ · Return
$G_t=\sum_k\gamma^k R_{t+k+1}$ · $V^\pi,Q^\pi$ · Optimal $V_*,Q_*$, $\pi_*(s)=\arg\max_a Q_*(s,a)$.

**Bellman.**
- Erwartung: $V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V^\pi(s')]$
- Optimalität: $Q_*(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_*(s',a')]$

**Update-Regeln (alle: $\text{alt}\leftarrow\text{alt}+\alpha(\text{Ziel}-\text{alt})$).**

| Methode | Ziel für Update |
|---|---|
| MC | $G_t$ (echter Return) |
| TD(0) Prädiktion | $R_{t+1}+\gamma V(S_{t+1})$ |
| **SARSA** (on-policy) | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ |
| **Expected SARSA** | $R_{t+1}+\gamma\sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a')$ |
| **Q-Learning** (off-policy) | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ |

**Exploration.** ε-greedy · optimistische Init · UCB $Q(a)+c\sqrt{\ln t/N(a)}$ · Softmax($\tau$).

**Konvergenz.** GLIE (∞ Exploration + $\varepsilon\to0$) + Robbins-Monro ($\sum\alpha=\infty,\sum\alpha^2<\infty$).

**Achsen.** modellbasiert (Planung, DP) ↔ modellfrei (Lernen) · on-policy ↔ off-policy ·
Bootstrapping (TD/DP) ↔ Sampling volle Returns (MC) · Tabelle ↔ Funktionsapproximation.

**Deadly Triad.** FA + Bootstrapping + off-policy → mögliche Divergenz (→ Deep RL braucht Replay/Target-Net).

---

## 5 · Selbsttest

<details>
<summary><b>1.</b> Worin unterscheidet sich RL grundlegend von überwachtem Lernen?</summary>

Kein Supervisor/Label mit „richtiger Aktion" — nur ein **evaluatives**, oft **verzögertes**
Belohnungssignal. Die Daten sind **nicht i.i.d.**: der Agent beeinflusst durch sein Handeln,
welche Zustände er als Nächstes sieht, und muss aktiv **explorieren**. Ziel ist die **langfristige**
kumulierte Belohnung (Return), nicht die momentane.
</details>

<details>
<summary><b>2.</b> Warum diskontiert man den Return, und was bewirkt γ?</summary>

Damit die (bei kontinuierlichen Aufgaben unendliche) Summe **konvergiert** ($|G|\le R_{\max}/(1-\gamma)$)
und um **nähere Belohnungen höher zu gewichten**. $\gamma\to0$ = myopisch/gierig, $\gamma\to1$ =
weitsichtig. Bei episodischen Aufgaben ist $\gamma=1$ oft ok.
</details>

<details>
<summary><b>3.</b> Was besagt die Bellman-Optimalitätsgleichung, und warum ist sie „schwerer" als die Erwartungsgleichung?</summary>

$V_*(s)=\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_*(s')]$ — der optimale Wert ist der **beste**
erreichbare erwartete Return. Durch das $\max$ ist sie **nichtlinear** (kein lineares
Gleichungssystem mehr), daher löst man sie iterativ (Value Iteration) statt direkt.
</details>

<details>
<summary><b>4.</b> Nenne den Kernunterschied zwischen Monte Carlo und TD(0) in Bias/Varianz.</summary>

MC nutzt den **echten Return** → **unverzerrt**, aber **hohe Varianz**, braucht Episodenende.
TD(0) nutzt ein **bootstrapped Ziel** $R+\gamma V(S')$ → **verzerrt** (nutzt eigene Schätzung),
aber **niedrige Varianz**, lernt **online** nach jedem Schritt.
</details>

<details>
<summary><b>5.</b> SARSA vs. Q-Learning: Was ist on- bzw. off-policy, und was heißt das praktisch?</summary>

**SARSA (on-policy)** bootstrappt mit der **tatsächlich gewählten** nächsten Aktion
$A_{t+1}\sim\pi$ → lernt den Wert der ausgeführten (explorierenden) Policy.
**Q-Learning (off-policy)** bootstrappt mit $\max_{a'}Q(S_{t+1},a')$ → lernt direkt $Q_*$
(die *optimale* Policy), obwohl es einer explorierenden Verhaltens-Policy folgt. Praktisch: bei
Cliff Walking wählt SARSA die **sichere**, Q-Learning die **optimale-aber-riskante** Route.
</details>

<details>
<summary><b>6.</b> Erkläre das Explore-Exploit-Dilemma und wie UCB es adressiert.</summary>

**Exploit** = beste bekannte Aktion nutzen; **Explore** = unsichere Aktionen testen, die
*vielleicht* besser sind. UCB wählt $\arg\max_a[Q(a)+c\sqrt{\ln t/N(a)}]$ — der Bonus ist groß
für **selten** gezogene Arme und schrumpft mit Erfahrung („Optimismus bei Unsicherheit"),
erreicht **logarithmischen** Regret.
</details>

<details>
<summary><b>7.</b> Was ist GLIE, und warum braucht man es?</summary>

*Greedy in the Limit with Infinite Exploration*: (a) jeder $(s,a)$ wird **unendlich oft**
besucht **und** (b) die Policy wird im Grenzwert **greedy** ($\varepsilon\to0$). Nötig, damit
ε-greedy-Kontrolle (z. B. SARSA) gegen die **optimale** Policy konvergiert — sonst bleibt
Rest-Exploration und man ist nur ε-optimal.
</details>

<details>
<summary><b>8.</b> Was ist die „deadly triad" und warum ist sie relevant für Deep RL?</summary>

Die Kombination **Funktionsapproximation + Bootstrapping + off-policy** kann zu **Divergenz**
führen. Deep Q-Networks (Modul 14) haben alle drei → man braucht **Experience Replay** und ein
**Target Network**, um zu stabilisieren.
</details>

<details>
<summary><b>9.</b> Wozu dienen Eligibility Traces?</summary>

Sie lösen das **credit-assignment**-Problem und realisieren **TD(λ)** effizient: ein Trace
$e(s)$ merkt sich kürzlich besuchte Zustände, sodass ein aktueller TD-Fehler $\delta_t$
rückwirkend auf **alle** verantwortlichen Zustände (abgestuft nach $(\gamma\lambda)^k$) verteilt
wird — interpoliert zwischen TD(0) und MC.
</details>

<details>
<summary><b>10.</b> Warum ist tabellarisches Q-Learning für Schach/Go ungeeignet, und was ist die Alternative?</summary>

Die Tabelle hätte $|\mathcal S|\times|\mathcal A|$ Einträge — bei $10^{47}$/$10^{170}$ Zuständen
unmöglich zu speichern *oder* zu besuchen. Alternative: **Funktionsapproximation** $\hat Q(s,a;\mathbf w)$
mit Features/neuronalem Netz, die über ähnliche Zustände **verallgemeinert** (→ Deep RL).
</details>

---

## 6 · Literatur & Quellen

**Das Standardwerk (kostenlos!):**
- 📗 **Sutton & Barto — *Reinforcement Learning: An Introduction* (2. Aufl., 2018)** —
  *die* Referenz, didaktisch hervorragend, **frei als PDF** (incompleteideas.net/book/the-book.html).
  Einsteigerfreundlich *und* vollständig. Kap. 2 (Bandits), 3 (MDP), 4 (DP), 5 (MC), 6 (TD),
  7 (n-step), 12 (Eligibility Traces), 13 (Policy Gradient) decken genau dieses Modul ab.
  **Beste Einzelquelle.**

**Vorlesungen (frei, online):**
- 🎥 **David Silver — *RL Course* (DeepMind/UCL, 10 Vorlesungen, YouTube)** — der Klassiker,
  folgt grob Sutton & Barto. *Einsteiger→vertiefend.*
- 🎥 **Stanford CS234 *Reinforcement Learning* (Emma Brunskill)** — Vorlesungsvideos + Folien frei.
  *Vertiefend.*
- 🎥 **UC Berkeley CS285 *Deep RL* (Sergey Levine)** — für den Sprung zu Modul 14. *Vertiefend.*

**Interaktiv/Blog (einsteigerfreundlich):**
- 🌐 **Gymnasium-Dokumentation** (gymnasium.farama.org) — Standard-API für RL-Umgebungen
  (Nachfolger von OpenAI Gym). Wir bauen im Modul zwar *ohne* Gym (didaktisch), aber für die
  Praxis solltest du die API kennen.
- 🌐 **Andrej Karpathy — *Deep Reinforcement Learning: Pong from Pixels*** (karpathy.github.io) —
  legendärer Blogpost zu Policy Gradient. *Einsteiger→vertiefend.*
- 🌐 **Lilian Weng — *A (Long) Peek into Reinforcement Learning*** (lilianweng.github.io) —
  kompakte, präzise Übersicht. *Vertiefend.*
- 🌐 **Spinning Up in Deep RL** (OpenAI, spinningup.openai.com) — für Modul 14, saubere
  Implementierungen + Theorie. *Vertiefend, kostenlos.*

**Klassische Paper (vertiefend):**
- Watkins & Dayan (1992), *Q-learning* — Konvergenzbeweis.
- Sutton (1988), *Learning to Predict by the Methods of Temporal Differences* — TD-Ursprung.
- Auer, Cesa-Bianchi & Fischer (2002), *Finite-time Analysis of the Multiarmed Bandit* — UCB1.

---

## Nächstes Modul

**Modul 14 — Deep Reinforcement Learning for Optimal Control** ersetzt die Q-**Tabelle** durch
ein **neuronales Netz** (DQN, PyTorch aus Modul 05/09), zähmt die *deadly triad* mit Replay &
Target-Network und geht zu **Policy-Gradient/Actor-Critic** (REINFORCE → PPO) über — für
**kontinuierliche** Steuerung (Optimal Control). Dieses Modul liefert dafür das komplette
konzeptuelle Fundament.
