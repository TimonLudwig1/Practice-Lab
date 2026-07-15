# Projekt 03 (final) — RL trifft Optimalsteuerung: Policy Gradient vs. LQR

**Format: Python-Projekt, _ohne Code-Vorgabe_.** Dieses Abschlussprojekt bekommt **kein
Gerüst** — du baust alles selbst: das System, die exakte Referenzlösung, den modellfreien
Lerner, die Auswertung. Das ist die Master-Prüfungsleistung des Moduls und der Abschluss des
gesamten RL-Blocks (13 + 14).

**Warum ein `.py`-Projekt?** Vier Komponenten (System, Riccati-Referenz, Policy-Gradient-Agent,
Auswertung) greifen sauber getrennt und testbar ineinander — die Architektur echter
Regelungs-/RL-Software. Ein Notebook würde das verwischen.

---

## Die Idee — und warum sie den Modultitel einlöst

Das Modul heißt „Deep RL **for Optimal Control**". Dieses Projekt macht die Brücke konkret:

> Es gibt eine **Problemklasse**, in der die optimale Steuerung **exakt und in geschlossener
> Form** bekannt ist: **lineare Dynamik + quadratische Kosten** → der **LQR**, gelöst über die
> **Riccati-Gleichung**. Damit hast du eine **perfekte Messlatte**. Ein modellfreier RL-Agent,
> der $A,B,Q,R$ **nicht kennt**, muss versuchen, sie allein aus Erfahrung zu erreichen.
> **Wie nah kommt er — und zu welchem Preis?**

Das ist exakt die Struktur des Modul-13-Finales (dort: Value Iteration als exakte Referenz für
einen diskreten MDP), nur eine Ebene höher: **kontinuierlicher** Zustand *und* **kontinuierliche**
Stellgröße.

## Das System (doppelter Integrator)

Eine Masse auf reibungsfreier Schiene — der „Hello World" der Regelungstechnik, und ein
realistisches Modell für Positionierungsaufgaben (Roboterachse, Kamerazoom, Fahrzeuglängsregelung).

- **Zustand** $x=[\text{Position},\ \text{Geschwindigkeit}]\in\mathbb R^2$
- **Stellgröße** $u\in\mathbb R$ (Kraft) — **kontinuierlich**! (DQN aus Projekt 01 scheidet damit aus.)
- **Dynamik** $x_{t+1}=Ax_t+Bu_t$ mit $A=\begin{pmatrix}1&dt\\0&1\end{pmatrix}$, $B=\begin{pmatrix}dt^2/2\\dt\end{pmatrix}$
- **Kosten** $c(x,u)=x^\top Q x+u^\top R u$, Belohnung $r=-c$

Ziel: aus einem zufälligen Startzustand schnell **und** energiesparend in den Ursprung.
$Q$ gewichtet den Zustandsfehler, $R$ die Stellenergie — der klassische Zielkonflikt.

## Aufgabenstellung (Schritt für Schritt)

Baue (Dateinamen als Vorschlag):

1. **`lqr_env.py` — das System.** Gym-artige API `reset()`/`step(u) → (x', r, done)` mit
   Episodenlänge `horizon`. $A,B,Q,R$ als Attribute — aber **nur die Referenz darf sie lesen**;
   der Lerner sieht ausschließlich $(x,u,r,x')$. Plus einen Helfer, der eine lineare Rückführung
   $u=w^\top x$ deterministisch ausrollt.
2. **`lqr_reference.py` — die exakte Lösung.** Löse die **diskrete algebraische
   Riccati-Gleichung** per Fixpunktiteration
   $$P \leftarrow Q + A^\top P A - A^\top P B\,(R+B^\top P B)^{-1}B^\top P A$$
   und gewinne daraus $K=(R+B^\top PB)^{-1}B^\top PA$ mit $u^*=-Kx$ sowie $J^*(x_0)=x_0^\top P x_0$.
   *(Das ist inhaltlich **Value Iteration auf der quadratischen Wertfunktion** $V(x)=x^\top Px$
   statt auf einer Tabelle — die direkte Fortsetzung von Modul 13.)*
   Ergänze eine Stabilitätsprüfung über die Eigenwerte von $A-BK$.
3. **`policy_gradient_continuous.py` — der modellfreie Lerner.** Eine **gaußsche** Politik
   $\pi_\theta(u\mid x)=\mathcal N(w^\top x,\ \sigma^2)$. **Wähle die Mittelwertfunktion linear** —
   dann sind die gelernten Gewichte $w$ **direkt** mit $-K$ vergleichbar (Parameter gegen
   Parameter!). Trainiere per REINFORCE.
4. **`run.py` — das Experiment.** Referenz rechnen, modellfrei lernen, vergleichen:
   $w$ vs. $-K$, mittlere Kosten vs. LQR-Optimum (**Optimalitätslücke in %**), Lernkurve und
   Trajektorienvergleich plotten.
5. **`test_*.py` — Tests.** U. a.: Riccati-Fixpunkt erfüllt die DARE, $P$ symmetrisch positiv
   definit, Regelkreis stabil ($|\lambda(A-BK)|<1$), $J^*(x_0)=x_0^\top Px_0$ stimmt mit der
   Simulation überein, **kein anderer linearer Regler schlägt $-K$** (Optimalitätstest), der
   gelernte Regler ist besser als Nichtstun und hat negative Rückführung.

**Parameter-Vorschlag:** $dt=0.1$, $Q=\mathrm{diag}(1,\ 0.1)$, $R=(0.1)$, `horizon`$=50$,
$x_0\sim\mathcal N(0,I)$, $\gamma=1$.

### Drei Fallen, an denen dieses Projekt hängt

Plane sie ein — ohne sie konvergiert REINFORCE hier **nicht**:

1. **Batch von Episoden pro Update.** Ein Einzel-Episoden-Gradient ist viel zu verrauscht.
2. **Per-Zeitschritt-Baseline.** Returns bei $t=0$ sind betragsmäßig *viel* größer als bei
   $t=T-1$. Normierst du **global** über den Batch, dominiert dieser **Zeittrend** das
   Lernsignal. Zentriere/skaliere **an jedem $t$ separat** über den Batch. (Das war bei uns der
   Unterschied zwischen „lernt kaum" und „3 % vom Optimum".)
3. **$\log\sigma$ nach unten klemmen.** Sonst kollabiert die Exploration ($\sigma\to0$), die
   log-Wahrscheinlichkeiten explodieren und das Training **divergiert** (bei uns reproduzierbar:
   Optimalitätslücke sprang auf 350–770 %).

## Was am Ende herauskommen soll

- **Riccati** konvergiert in ~110 Iterationen (Millisekunden) zu $-K \approx [-2.76,\ -2.51]$,
  Regelkreis stabil ($|\lambda|\approx0.87<1$).
- **Validierung der Referenz:** simulierte Kosten von $u=-Kx$ ab $x_0=[1,0]$ **= 9.0776**
  = $x_0^\top Px_0$ **= 9.0776** ✓ (auf 4 Nachkommastellen — so *weißt* du, dass die Referenz stimmt).
- **Modellfrei gelernt:** $w\approx[-2.04,\ -2.23]$, mittlere Kosten **12.43** vs. LQR-Optimum
  **12.02** → **Optimalitätslücke ≈ 3.4 %**, in ~70 s / ~19 200 Episoden.

### Die Interpretation, die du liefern sollst

- **Der Preis der Modellfreiheit.** LQR löst das Problem **exakt** in ~110 Riccati-Iterationen
  (Millisekunden) — *weil es $A,B,Q,R$ kennt*. Der RL-Agent braucht **~19 200 Episoden**, um auf
  ~3 % heranzukommen — *weil er nichts weiß*. **Das ist die zentrale Botschaft des Moduls:**
  Kennst du dein Modell, nimm Optimalsteuerung. RL ist die Antwort für die Fälle, in denen du
  es **nicht** kennst (oder es nichtlinear/unbekannt ist) — dann ist es unschlagbar, aber teuer.
- **Warum bleibt eine Lücke?** Die gelernte Rückführung ist systematisch etwas *weniger
  aggressiv* als $-K$. Zwei echte Gründe: (a) die Politik behält **Explorationsrauschen**
  $\sigma\approx0.22$ — die optimale *stochastische* Politik ist milder als der optimale
  *deterministische* Regler; (b) endlicher Horizont ($T=50$) vs. $K$ aus dem
  **unendlichen** Horizont. Die Kostenlandschaft ist um das Optimum herum **flach** — deshalb
  ist die Lücke in den *Kosten* (3 %) viel kleiner als der Abstand in den *Parametern*
  ($\|w-(-K)\|\approx0.77$). **Dieselbe Beobachtung wie im Modul-13-Finale.**

## Referenzlösung

Vollständig und getestet in [`loesung/`](loesung/) (System, Riccati-Referenz, gaußscher
Policy-Gradient, `run.py`, **12 Tests**). **Erst selbst versuchen!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python loesung/test_lqr.py   # 12 Tests, ~20 s
/.../xtAI_Learn-Repository/.venv/bin/python loesung/run.py        # Experiment + Plots, ~70 s
```
`numpy` + `torch` (+ `matplotlib`). CPU, lineares System, winzige Politik (2 Gewichte + $\log\sigma$).

## Erweiterungen (für die besonders Motivierten)

- **Prozessrauschen** $x_{t+1}=Ax_t+Bu_t+w_t$, $w\sim\mathcal N(0,W)$: LQR bleibt optimal
  (*certainty equivalence*!) — bestätige das empirisch. Wie ändert sich $J^*$?
- **Actor-Critic**: ersetze die Batch-Baseline durch einen gelernten **quadratischen** Critic
  $\hat V(x)=x^\top \hat P x$ — konvergiert er gegen das Riccati-$P$?
- **Nichtlinear**: mache das System nichtlinear (z. B. Pendel-Aufschwung). Jetzt gibt es **keine**
  exakte Lösung mehr — genau hier gewinnt RL. Vergleiche gegen einen um die Ruhelage
  **linearisierten** LQR: wo scheitert er?
- **MPC** (Skript 4.3): plane bei bekanntem Modell über einen endlichen Horizont, führe nur den
  ersten Schritt aus. Vergleiche gegen LQR und RL.
- **Unbekanntes Modell schätzen** (modellbasiertes RL, Skript 4.4): schätze $A,B$ per linearer
  Regression aus Daten, rechne LQR darauf. Wie viele Daten brauchst du, um die 3 % zu schlagen?
