# Modul 21 — Robotics 1

> **Worum geht es?** Ein Roboter muss drei Fragen beantworten: **„Wo ist meine Hand?"** (Kinematik), **„Wie komme ich dorthin, ohne anzustoßen?"** (Bewegungsplanung) und **„Wo bin ich überhaupt?"** (Zustandsschätzung/Lokalisierung). Dieses Modul behandelt die Mathematik hinter allen dreien: die **kinematische Kette** und ihre Ableitung (die **Jacobi-Matrix**), die **inverse Kinematik**, den **Konfigurationsraum** mit **RRT/A\***-Planung, und den **Bayes-Filter** in seinen zwei berühmten Ausprägungen (**Kalman-** und **Partikelfilter**) — plus die einfachste und meistgenutzte Regelung der Welt, den **PID-Regler**.
>
> **Vorkenntnisse**: lineare Algebra (Matrizen, Ableitungen/Jacobi, Pseudoinverse), Wahrscheinlichkeitsrechnung. Aus diesem Repo bauen direkt auf: **Modul 19** (homogene $4\times4$-Transformationen — das Fundament der Kinematik), **Modul 06** (Suchalgorithmen, A\*), **Modul 07** (Bayes-Netze/Inferenz — der Bayes-Filter ist deren rekursive Form), **Modul 17** (Komplementärfilter — der Kalman-Filter ist seine optimale Version) und **Modul 18** (inverse-Varianz-Fusion — mathematisch identisch zum Kalman-Gain).

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–20 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, orientiert an den Standard-Curricula der Robotik-Einführung (Siciliano/Craig für Kinematik, LaValle für Planung, Thrun für probabilistische Robotik). **Bewusst ohne echten Roboter und ohne Physik-Engine** (`mujoco`/`pygame` fehlen hier): Der lehrbare, übertragbare Kern sind die **Algorithmen und ihre Mathematik** — DH-Ketten, die Jacobi-Matrix und ihre Singularitäten, gedämpfte Least-Squares-IK, Sampling-basierte Planung, der Bayes-Filter. Wer einen Roboterarm per SDK bewegt, versteht Kinematik nicht; wer die Jacobi-Matrix herleitet und sieht, *warum* die Pseudoinverse in einer Singularität explodiert und die Dämpfung sie rettet, schon. Alle Projekte simulieren from scratch mit `numpy`/`matplotlib` — CPU-Sekunden.

---

## Inhalt

1. [Lernziele](#lernziele)
2. [Grundlagen (Basics)](#grundlagen-basics)
3. [Aufbau (Intermediate)](#aufbau-intermediate)
4. [Advanced-Themen](#advanced-themen)
5. [Zusammenfassung / Cheat-Sheet](#zusammenfassung--cheat-sheet)
6. [Selbsttest](#selbsttest)
7. [Literatur & Quellen](#literatur--quellen)

---

## Lernziele

Nach diesem Modul solltest du …

- den **Sense-Plan-Act-Zyklus** und die Begriffe **Freiheitsgrade, Arbeitsraum, Konfigurationsraum** sauber unterscheiden können.
- die **Vorwärtskinematik** einer kinematischen Kette aufstellen — als Produkt homogener Transformationen, inklusive der **Denavit-Hartenberg-Konvention** (die vier Parameter und ihre Matrix).
- die **Jacobi-Matrix** herleiten (analytisch und numerisch), ihre Bedeutung ($\dot{\mathbf x} = \mathbf J(\mathbf q)\,\dot{\mathbf q}$) erklären und **Singularitäten** ($\det\mathbf J = 0$) erkennen und interpretieren.
- **inverse Kinematik** lösen: **analytisch** für den 2-Gelenk-Arm (mit *beiden* Lösungen — Ellbogen oben/unten) und **numerisch** über Jacobi-Iteration — inklusive der Frage, warum man die **Pseudoinverse** durch **Damped Least Squares** ersetzen muss.
- den **Konfigurationsraum** verstehen (der Roboter wird zum *Punkt*) und **Sampling-basierte Planung** (**RRT**, PRM) gegen Gittersuche (**A\***, Modul 06) abwägen — inklusive *Vollständigkeit* vs. *probabilistischer Vollständigkeit*.
- den **rekursiven Bayes-Filter** (Prädiktion + Korrektur) aufschreiben und seine zwei Ausprägungen beherrschen: den **Kalman-Filter** (linear-gaußsch, mit vollständigen Gleichungen und der Einsicht, dass der **Kalman-Gain die inverse-Varianz-Gewichtung aus Modul 18 ist**) und den **Partikelfilter** (Monte-Carlo, für multimodale Verteilungen).
- einen **PID-Regler** aufstellen und seine drei Terme interpretieren.

---

## Grundlagen (Basics)

### 1. Was ein Roboter tut: Sense — Plan — Act

Die klassische Architektur der Robotik ist ein Kreislauf:

```
        +-----------+      +-----------+      +-----------+
   ---> |   SENSE   | ---> |   PLAN    | ---> |    ACT    | ---+
   |    | (messen,  |      | (Pfad,    |      | (regeln,  |    |
   |    | schaetzen)|      | Trajektorie)|    | Motoren)  |    |
   |    +-----------+      +-----------+      +-----------+    |
   +-----------------------------------------------------------+
                        (Rueckkopplung)
```

- **Sense**: Sensoren liefern verrauschte, unvollständige Messungen → **Zustandsschätzung** (Abschnitt 9–11).
- **Plan**: Aus Start und Ziel einen **kollisionsfreien Weg** finden (Abschnitt 7–8).
- **Act**: Den Plan in Motorbefehle umsetzen und Abweichungen **ausregeln** (Abschnitt 12).

Dieses Modul geht alle drei Stufen durch — und die drei Projekte spiegeln sie.

### 2. Freiheitsgrade, Arbeitsraum, Konfigurationsraum

Drei Begriffe, die man ständig verwechselt und deshalb sauber trennen muss:

- **Freiheitsgrade (DoF)**: die Anzahl unabhängiger Bewegungsmöglichkeiten. Ein Roboterarm mit $n$ Drehgelenken hat $n$ DoF. Ein starrer Körper im Raum hat 6 (3 Translation + 3 Rotation, vgl. Modul 19).
- **Arbeitsraum (workspace)**: die Menge der **Positionen/Posen im kartesischen Raum**, die der Endeffektor erreichen kann. Ein 2-Gelenk-Arm mit Längen $l_1, l_2$ erreicht einen Kreisring mit Außenradius $l_1+l_2$ und Innenradius $|l_1-l_2|$.
- **Konfigurationsraum (C-Space)**: die Menge **aller Gelenkstellungen** $\mathbf q = (q_1,\dots,q_n)$. Für $n$ Drehgelenke ist das der $n$-Torus $[0,2\pi)^n$. **Der entscheidende Trick der Bewegungsplanung**: Im C-Space ist der (ausgedehnte) Roboter ein **einziger Punkt** — Planung wird zur Punktsuche in einem Hindernisfeld (Abschnitt 7).

> **Merke die Richtung:** Vorwärtskinematik geht **C-Space → Arbeitsraum** ($\mathbf q \mapsto \mathbf x$, immer eindeutig). Inverse Kinematik geht **Arbeitsraum → C-Space** ($\mathbf x \mapsto \mathbf q$, oft mehrdeutig oder unlösbar). Diese Asymmetrie ist der Grund, warum IK das schwierige Problem ist.

### 3. Koordinatenframes und die kinematische Kette

Ein Roboter ist eine **Kette starrer Glieder (links)**, verbunden durch **Gelenke (joints)**. An jedem Glied klebt ein Koordinatensystem. Die Transformation vom Frame $i$ zum Frame $i-1$ ist eine homogene $4\times4$-Matrix (Modul 19):

$$^{i-1}\mathbf T_i \;=\; \begin{pmatrix}\mathbf R_i & \mathbf p_i\\ \mathbf 0^\top & 1\end{pmatrix}.$$

Die **Pose des Endeffektors** relativ zur Basis ist das **Produkt der Kette**:

$$\boxed{\;^{0}\mathbf T_n(\mathbf q) \;=\; ^{0}\mathbf T_1(q_1)\;^{1}\mathbf T_2(q_2)\cdots\,^{n-1}\mathbf T_n(q_n)\;}$$

Das ist die **Vorwärtskinematik (FK)**: Gelenkwinkel rein, Endeffektor-Pose raus. Mehr steckt konzeptuell nicht dahinter — die gesamte Schwierigkeit liegt darin, die einzelnen $^{i-1}\mathbf T_i$ systematisch und fehlerfrei aufzustellen. Genau dafür gibt es die DH-Konvention.

---

## Aufbau (Intermediate)

### 4. Denavit-Hartenberg: vier Parameter statt sechs

Eine allgemeine Transformation zwischen zwei Frames bräuchte 6 Parameter (3 Translation + 3 Rotation). **Denavit-Hartenberg (DH)** legt die Frames so geschickt an die Gelenkachsen, dass **vier** genügen. Die Konvention (Standard-DH):

| Parameter | Bedeutung |
|---|---|
| $\theta_i$ | Drehung um die alte $z$-Achse (**Gelenkvariable bei Drehgelenken**) |
| $d_i$ | Verschiebung entlang der alten $z$-Achse (**Gelenkvariable bei Schubgelenken**) |
| $a_i$ | Verschiebung entlang der neuen $x$-Achse (**Gliedlänge**) |
| $\alpha_i$ | Drehung um die neue $x$-Achse (**Gliedverwindung**) |

Die zugehörige Transformationsmatrix ist das Produkt $\mathrm{Rot}_z(\theta_i)\,\mathrm{Trans}_z(d_i)\,\mathrm{Trans}_x(a_i)\,\mathrm{Rot}_x(\alpha_i)$, ausmultipliziert:

$$^{i-1}\mathbf T_i=\begin{pmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i\\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i\\
0 & \sin\alpha_i & \cos\alpha_i & d_i\\
0&0&0&1\end{pmatrix}$$

**Beispiel: der planare 2-Gelenk-Arm** (alles in der Ebene, $\alpha_i=0$, $d_i=0$, $a_i=l_i$). Die Kette kollabiert zu einer einfachen Formel für die Endeffektor-Position:

$$\boxed{\;x = l_1\cos q_1 + l_2\cos(q_1+q_2),\qquad y = l_1\sin q_1 + l_2\sin(q_1+q_2)\;}$$

Man beachte das $\cos(q_1+q_2)$: Die Winkel **addieren sich entlang der Kette** — jedes Glied dreht mit allen vorherigen mit. Das **Basic-Projekt** baut genau diese Kette (und die allgemeine DH-Matrix).

### 5. Die Jacobi-Matrix: die Ableitung der Kinematik

Die Vorwärtskinematik $\mathbf x = f(\mathbf q)$ ist nichtlinear. Ihre **Ableitung** ist die **Jacobi-Matrix**:

$$\mathbf J(\mathbf q) = \frac{\partial f}{\partial \mathbf q} = \begin{pmatrix}
\frac{\partial x_1}{\partial q_1} & \cdots & \frac{\partial x_1}{\partial q_n}\\
\vdots & & \vdots\\
\frac{\partial x_m}{\partial q_1} & \cdots & \frac{\partial x_m}{\partial q_n}\end{pmatrix} \in\mathbb R^{m\times n}$$

und sie verknüpft **Geschwindigkeiten**:

$$\boxed{\;\dot{\mathbf x} = \mathbf J(\mathbf q)\,\dot{\mathbf q}\;}$$

„Wie schnell bewegt sich die Hand, wenn ich die Gelenke mit gegebener Rate drehe?" Zusätzlich gilt die **Kraft-Dualität** $\boldsymbol\tau = \mathbf J^\top\mathbf F$ (Gelenkmomente aus einer Kraft am Endeffektor).

**Für den planaren 2-Gelenk-Arm** ergibt Ableiten der Formeln aus Abschnitt 4:

$$\mathbf J = \begin{pmatrix}
-l_1\sin q_1 - l_2\sin(q_1{+}q_2) & -l_2\sin(q_1{+}q_2)\\
\;\;\,l_1\cos q_1 + l_2\cos(q_1{+}q_2) & \;\;\,l_2\cos(q_1{+}q_2)\end{pmatrix}$$

**Singularitäten.** Interessant wird es, wo $\mathbf J$ den Rang verliert. Für den 2-Gelenk-Arm ist

$$\det\mathbf J = l_1 l_2 \sin q_2 .$$

Das wird **null bei $q_2 = 0$ oder $q_2 = \pi$** — also bei **vollständig gestrecktem oder vollständig eingeklapptem** Arm. Anschaulich: Im gestreckten Zustand kann der Endeffektor sich **nicht mehr radial nach außen** bewegen, egal wie man die Gelenke dreht — eine Bewegungsrichtung geht verloren. In der Nähe der Singularität wird $\mathbf J$ schlecht konditioniert: Um eine kleine Handbewegung zu erzielen, wären **riesige Gelenkgeschwindigkeiten** nötig. Das ist kein numerisches Artefakt, sondern echte Physik — und der Grund für Damped Least Squares (Abschnitt 6). Ein Maß dafür ist die **Manipulierbarkeit** $w = \sqrt{\det(\mathbf J\mathbf J^\top)}$ (Yoshikawa), die in Singularitäten auf 0 fällt.

### 6. Inverse Kinematik

**Gesucht**: Gelenkwinkel $\mathbf q$ für eine gewünschte Endeffektor-Pose $\mathbf x^\ast$. Drei Komplikationen: es kann **keine** Lösung geben (Ziel außerhalb des Arbeitsraums), **mehrere** (Ellbogen oben/unten) oder **unendlich viele** (redundante Arme, $n > m$).

**(a) Analytisch — der 2-Gelenk-Arm.** Mit dem Kosinussatz auf das Dreieck aus $l_1$, $l_2$ und dem Abstand $r=\sqrt{x^2+y^2}$:

$$\cos q_2 = \frac{r^2 - l_1^2 - l_2^2}{2 l_1 l_2}, \qquad q_2 = \pm\arccos(\cdot)$$

Das $\pm$ **ist** die Doppellösung: **Ellbogen oben** und **Ellbogen unten**. Danach folgt $q_1$ aus

$$q_1 = \operatorname{atan2}(y,x) - \operatorname{atan2}\big(l_2\sin q_2,\; l_1 + l_2\cos q_2\big).$$

Existenzbedingung: $|l_1-l_2| \le r \le l_1+l_2$ (sonst liegt das Ziel außerhalb des Kreisrings). **`atan2` statt `atan`** ist Pflicht — nur so bleibt der Quadrant korrekt.

**(b) Numerisch — Jacobi-Iteration.** Für allgemeine Arme gibt es keine geschlossene Formel. Man linearisiert und iteriert (Newton-artig): Mit dem Fehler $\mathbf e = \mathbf x^\ast - f(\mathbf q)$ sucht man ein $\Delta\mathbf q$ mit $\mathbf J\,\Delta\mathbf q \approx \mathbf e$:

- **Jacobi-Transponierte**: $\Delta\mathbf q = \alpha\,\mathbf J^\top\mathbf e$ — billig, robust, aber langsam (das ist Gradientenabstieg auf $\tfrac12\|\mathbf e\|^2$).
- **Pseudoinverse**: $\Delta\mathbf q = \mathbf J^{+}\mathbf e$ mit $\mathbf J^{+}=\mathbf J^\top(\mathbf J\mathbf J^\top)^{-1}$ — schnell (quadratische Konvergenz), liefert bei Redundanz die **kleinste** Gelenkänderung. **Aber**: In der Nähe einer Singularität wird $\mathbf J\mathbf J^\top$ fast singulär, und $\Delta\mathbf q$ **explodiert**.
- **Damped Least Squares (DLS / Levenberg-Marquardt)** — die Praxislösung:

$$\boxed{\;\Delta\mathbf q = \mathbf J^\top\big(\mathbf J\mathbf J^\top + \lambda^2\mathbf I\big)^{-1}\mathbf e\;}$$

Der Dämpfungsterm $\lambda^2\mathbf I$ macht die Inverse **immer** wohldefiniert. Er minimiert nicht mehr nur den Fehler, sondern $\|\mathbf J\Delta\mathbf q - \mathbf e\|^2 + \lambda^2\|\Delta\mathbf q\|^2$ — man **tauscht ein wenig Genauigkeit gegen beschränkte Gelenkgeschwindigkeiten**. Genau dieser Trade-off (kleines $\lambda$: schnell, aber instabil nahe Singularitäten; großes $\lambda$: stabil, aber langsam) wird im **Medium-Projekt** vermessen.

**(c) Redundanz und Nullraum.** Bei $n>m$ (mehr Gelenke als Aufgabendimensionen) hat $\mathbf J$ einen **Nullraum**: Gelenkbewegungen, die den Endeffektor *nicht* bewegen. Man nutzt sie für Nebenziele (Hindernisse meiden, Gelenkgrenzen einhalten):
$$\Delta\mathbf q = \mathbf J^{+}\mathbf e + (\mathbf I - \mathbf J^{+}\mathbf J)\,\mathbf z,$$
wobei $(\mathbf I - \mathbf J^{+}\mathbf J)$ auf den Nullraum projiziert und $\mathbf z$ das Nebenziel kodiert.

### 7. Bewegungsplanung I: der Konfigurationsraum

Die zentrale Idee (Lozano-Pérez): Statt den **ausgedehnten** Roboter durch den Arbeitsraum zu schieben, arbeitet man im **Konfigurationsraum**, wo der Roboter ein **Punkt** ist. Hindernisse im Arbeitsraum bilden sich auf **C-Space-Hindernisse** $\mathcal C_{\text{obs}}$ ab (die im Allgemeinen komplizierter geformt sind), und der freie Raum ist $\mathcal C_{\text{free}}$.

> **Planung = einen Pfad für einen Punkt durch $\mathcal C_{\text{free}}$ finden.** Das ist die Vereinfachung, die das ganze Feld trägt. Der Preis: Der C-Space ist hochdimensional ($n$ DoF → $n$ Dimensionen), und ihn *explizit* zu berechnen ist meist unmöglich. Deshalb prüft man Kollisionen nur **punktweise** (ein Kollisionstest pro Stichprobe) — das führt direkt zu den Sampling-Verfahren.

### 8. Bewegungsplanung II: Gitter vs. Sampling

**Gittersuche (A\*, Dijkstra — Modul 06).** Man diskretisiert den C-Space zu einem Gitter und sucht mit **A\***. **Vorteil**: *vollständig* und *optimal* (bei zulässiger Heuristik) — bezogen auf die Auflösung. **Nachteil**: Die Zellzahl wächst **exponentiell** mit der Dimension ($k^n$) — der **Fluch der Dimensionalität**. Ab etwa 4–6 DoF praktisch unbrauchbar.

**Sampling-basierte Planung.** Man baut den freien Raum nie explizit auf, sondern **erkundet ihn stichprobenartig**:

- **PRM (Probabilistic Roadmap)**: Ziehe $N$ zufällige Konfigurationen, behalte die kollisionsfreien, verbinde nahe Paare mit kollisionsfreien Kanten → ein Graph („Roadmap"), in dem man dann mit A\* sucht. Gut für **viele Anfragen** in derselben statischen Umgebung (multi-query).
- **RRT (Rapidly-exploring Random Tree)**: Wachse einen **Baum** von der Startkonfiguration:

```
wiederhole:
  1. q_rand  <- zufaellige Konfiguration (mit Wahrscheinlichkeit p direkt das ZIEL: goal bias)
  2. q_near  <- der Baumknoten, der q_rand am naechsten ist
  3. q_new   <- gehe von q_near einen Schritt der Laenge eps in Richtung q_rand
  4. wenn die Kante q_near -> q_new kollisionsfrei ist: fuege q_new dem Baum hinzu
  5. wenn q_new nahe genug am Ziel: Pfad durch Zurueckverfolgen der Eltern ausgeben
```

Der Kern ist Schritt 1–2: Weil $\mathbf q_{\text{rand}}$ **uniform** gezogen wird, werden große unerforschte Gebiete mit höherer Wahrscheinlichkeit getroffen — der Baum wächst **bevorzugt in unerforschtes Terrain** (daher „rapidly-exploring"). Der **Goal-Bias** (mit z. B. 5–10 % Wahrscheinlichkeit direkt aufs Ziel zielen) beschleunigt massiv.

**Vollständigkeit.** RRT/PRM sind nicht *vollständig*, sondern **probabilistisch vollständig**: Existiert eine Lösung, geht die Wahrscheinlichkeit, sie zu finden, mit der Stichprobenzahl gegen 1. Sie garantieren aber **keine Optimalität** — die Pfade sind typisch zackig und zu lang. Abhilfen: **Pfadglättung** (Shortcutting: versuche, Zwischenpunkte zu überspringen) und **RRT\***, das durch Umverdrahten (rewiring) *asymptotisch optimal* wird.

---

## Advanced-Themen

### 9. Zustandsschätzung: der rekursive Bayes-Filter

Ein Roboter kennt seinen Zustand $\mathbf x_t$ (Position, Orientierung) nie exakt: Odometrie **driftet**, Sensoren **rauschen**. Die Lösung ist, statt eines Punktschätzers eine **Wahrscheinlichkeitsverteilung** zu führen — den **Belief** $bel(\mathbf x_t) = p(\mathbf x_t \mid \mathbf z_{1:t}, \mathbf u_{1:t})$. Er wird **rekursiv** in zwei Schritten fortgeschrieben (Rückverweis Modul 07):

**1. Prädiktion (Bewegungsmodell $\mathbf u_t$ anwenden)** — der Belief wird *unschärfer*:
$$\overline{bel}(\mathbf x_t) = \int p(\mathbf x_t \mid \mathbf u_t, \mathbf x_{t-1})\; bel(\mathbf x_{t-1})\;\mathrm d\mathbf x_{t-1}$$

**2. Korrektur (Messung $\mathbf z_t$ einarbeiten)** — der Belief wird *schärfer*:
$$bel(\mathbf x_t) = \eta\; p(\mathbf z_t\mid \mathbf x_t)\;\overline{bel}(\mathbf x_t)$$

Das ist **die** Grundgleichung der probabilistischen Robotik. Kalman- und Partikelfilter sind nur zwei verschiedene **Darstellungen** dieses Beliefs.

### 10. Der Kalman-Filter (linear-gaußsch)

Nimmt man an, dass alles **linear** und **gaußsch** ist, bleibt der Belief eine Gaußverteilung $\mathcal N(\boldsymbol\mu, \boldsymbol\Sigma)$, und die Integrale werden zu Matrixoperationen. Mit Bewegungsmodell $\mathbf x_t = \mathbf A\mathbf x_{t-1}+\mathbf B\mathbf u_t + \boldsymbol\varepsilon$ ($\boldsymbol\varepsilon\sim\mathcal N(0,\mathbf R)$) und Messmodell $\mathbf z_t = \mathbf C\mathbf x_t + \boldsymbol\delta$ ($\boldsymbol\delta\sim\mathcal N(0,\mathbf Q)$):

**Prädiktion**
$$\bar{\boldsymbol\mu}_t = \mathbf A\boldsymbol\mu_{t-1}+\mathbf B\mathbf u_t, \qquad \bar{\boldsymbol\Sigma}_t = \mathbf A\boldsymbol\Sigma_{t-1}\mathbf A^\top + \mathbf R$$

**Korrektur**
$$\mathbf K_t = \bar{\boldsymbol\Sigma}_t\mathbf C^\top\big(\mathbf C\bar{\boldsymbol\Sigma}_t\mathbf C^\top+\mathbf Q\big)^{-1}$$
$$\boldsymbol\mu_t = \bar{\boldsymbol\mu}_t + \mathbf K_t(\mathbf z_t - \mathbf C\bar{\boldsymbol\mu}_t), \qquad \boldsymbol\Sigma_t = (\mathbf I - \mathbf K_t\mathbf C)\bar{\boldsymbol\Sigma}_t$$

$\mathbf K_t$ ist der **Kalman-Gain**. Er gewichtet zwischen Vorhersage und Messung — und zwar **genau nach Reliabilität**:

> **Der Bogen zu Modul 17 und 18.** Im skalaren Fall ($\mathbf C=1$) wird der Gain $K = \frac{\bar\sigma^2}{\bar\sigma^2+\sigma_z^2}$, und die Korrektur $\mu = \bar\mu + K(z-\bar\mu)$ ist **exakt die inverse-Varianz-gewichtete Fusion** aus Modul 18 (Ernst & Banks) — mit $1/\sigma^2 = 1/\bar\sigma^2 + 1/\sigma_z^2$. Der **Komplementärfilter aus Modul 17** ist derselbe Ausdruck mit *heuristisch* statt *optimal* gewähltem Gewicht. Drei Module, eine Formel: Der Kalman-Filter ist die rekursive, optimale Form der Sensorfusion.

**Nichtlinearität.** Reale Roboterbewegung (Drehungen!) ist nicht linear. Der **Extended Kalman Filter (EKF)** linearisiert Bewegungs- und Messmodell pro Schritt über ihre **Jacobi-Matrizen** (dieselbe Idee wie in Abschnitt 5) und wendet dann die Kalman-Gleichungen an. Der **Unscented Kalman Filter (UKF)** propagiert stattdessen gezielt gewählte Sigma-Punkte — meist genauer, ohne Ableitungen.

### 11. Der Partikelfilter (Monte-Carlo-Lokalisierung)

Der Kalman-Filter kann nur **eine** Gaußglocke darstellen. Bei der **globalen Lokalisierung** („wo bin ich in dieser Karte?") ist der Belief aber oft **multimodal** — mehrere Räume sehen gleich aus. Der **Partikelfilter** stellt den Belief durch eine **Stichprobe** dar: $M$ gewichtete Partikel $\{(\mathbf x^{[m]}, w^{[m]})\}$. Jeder Zyklus:

```
1. PRAEDIKTION: bewege jedes Partikel gemaess Bewegungsmodell + Rauschen
2. GEWICHTUNG:  w[m] <- p(z_t | x[m])     (wie gut passt die Messung zu diesem Partikel?)
3. NORMIEREN:   w <- w / sum(w)
4. RESAMPLING:  ziehe M neue Partikel MIT Zuruecklegen proportional zu w
```

Der Kern ist **Schritt 4**: Partikel, die die Messungen gut erklären, werden **vervielfacht**, schlechte **sterben aus** — „survival of the fittest". So konzentriert sich die Stichprobe auf die wahrscheinlichen Zustände, ohne je eine Verteilungsform anzunehmen.

**Zwei praktische Fallen:**
- **Partikelverarmung (particle depletion)**: Zu häufiges Resampling reduziert die Vielfalt; im Extremfall sind alle Partikel identisch, und der Filter kann sich nicht mehr korrigieren. Gegenmittel: nur resampeln, wenn die **effektive Stichprobengröße** $N_{\text{eff}} = 1/\sum_m (w^{[m]})^2$ unter eine Schwelle (z. B. $M/2$) fällt.
- **Zu wenige Partikel** → der Filter „verliert" die wahre Position und divergiert. Der Zusammenhang zwischen Partikelzahl und Genauigkeit wird im **Final-Projekt** vermessen.

### 12. Regelung: der PID-Regler

Ist der Pfad geplant, muss der Roboter ihm **folgen** — trotz Störungen und Modellfehlern. Der meistgenutzte Regler der Technik ist der **PID**. Mit dem Regelfehler $e(t) = x_{\text{soll}}(t) - x_{\text{ist}}(t)$:

$$\boxed{\;u(t) = \underbrace{K_p\,e(t)}_{\text{Gegenwart}} + \underbrace{K_i\!\int_0^t\! e(\tau)\,\mathrm d\tau}_{\text{Vergangenheit}} + \underbrace{K_d\,\frac{\mathrm de(t)}{\mathrm dt}}_{\text{Zukunft}}\;}$$

- **P** reagiert proportional zum aktuellen Fehler. Allein bleibt oft eine **bleibende Regelabweichung**; zu großes $K_p$ → Überschwingen/Schwingen.
- **I** summiert den Fehler über die Zeit und **eliminiert die bleibende Abweichung** — neigt aber zu **Integral-Windup** (Aufsummieren bei gesättigtem Stellglied).
- **D** reagiert auf die **Änderungsrate** und wirkt **dämpfend** (bremst vor dem Ziel ab) — verstärkt aber Messrauschen.

Für die Pfadverfolgung mobiler Roboter nutzt man häufig geometrische Varianten wie **Pure Pursuit** (ziele auf einen Punkt im Abstand $L$ voraus auf dem Pfad).

### 13. Odometrie-Drift und der Ausblick auf SLAM

**Odometrie** (Radumdrehungen integrieren) ist ein **Dead-Reckoning**-Verfahren: Jeder kleine Fehler wird **aufintegriert**, der Fehler wächst **unbeschränkt** — besonders Orientierungsfehler, die sich als wachsende Positionsabweichung auswirken. Deshalb *muss* eine externe Messung (Landmarken, GPS, Scan-Matching per **ICP aus Modul 20**) den Drift korrigieren — genau das leistet der Korrekturschritt des Bayes-Filters. Kennt man die Karte **nicht**, muss man Karte und Pose **gleichzeitig** schätzen: das ist **SLAM** (Simultaneous Localization and Mapping) — Thema von Robotics 2.

---

## Zusammenfassung / Cheat-Sheet

**Begriffe**: DoF (unabhängige Bewegungen) · Arbeitsraum (erreichbare *Posen*) · **C-Space** (alle *Gelenkstellungen*; dort ist der Roboter ein **Punkt**). Sense–Plan–Act.

**Vorwärtskinematik**: $^{0}\mathbf T_n(\mathbf q)=\prod_i {}^{i-1}\mathbf T_i(q_i)$. DH: 4 Parameter $(\theta_i, d_i, a_i, \alpha_i)$.
2-Gelenk planar: $x=l_1\cos q_1 + l_2\cos(q_1{+}q_2)$, $y=l_1\sin q_1+l_2\sin(q_1{+}q_2)$.

**Jacobi**: $\dot{\mathbf x}=\mathbf J(\mathbf q)\dot{\mathbf q}$; Kraft-Dualität $\boldsymbol\tau=\mathbf J^\top\mathbf F$.
2-Gelenk: $\det\mathbf J = l_1l_2\sin q_2$ → **Singularität bei $q_2=0,\pi$** (gestreckt/eingeklappt). Manipulierbarkeit $w=\sqrt{\det(\mathbf J\mathbf J^\top)}$.

**Inverse Kinematik**
| Methode | Update | Eigenschaft |
|---|---|---|
| analytisch (2-Gelenk) | Kosinussatz, $q_2=\pm\arccos(\cdot)$ | exakt, **zwei** Lösungen (Ellbogen oben/unten) |
| Jacobi-Transponierte | $\Delta\mathbf q=\alpha\mathbf J^\top\mathbf e$ | robust, langsam (Gradientenabstieg) |
| Pseudoinverse | $\Delta\mathbf q=\mathbf J^{+}\mathbf e$ | schnell, **explodiert** an Singularitäten |
| **DLS** | $\Delta\mathbf q=\mathbf J^\top(\mathbf J\mathbf J^\top+\lambda^2\mathbf I)^{-1}\mathbf e$ | immer wohldefiniert; Genauigkeit ↔ Stabilität |
| Redundanz | $+\,(\mathbf I-\mathbf J^{+}\mathbf J)\mathbf z$ | Nullraum für Nebenziele |

**Planung**: A\* auf Gitter = vollständig+optimal, aber $k^n$ (Fluch der Dimension). **RRT**: sample → nächster Knoten → Schritt $\varepsilon$ → Kollisionstest; **Goal-Bias**; **probabilistisch vollständig**, nicht optimal → Glätten/RRT\*.

**Bayes-Filter**: Prädiktion (unschärfer) $\overline{bel}=\int p(\mathbf x_t|\mathbf u_t,\mathbf x_{t-1})bel\,\mathrm d\mathbf x$; Korrektur (schärfer) $bel=\eta\,p(\mathbf z_t|\mathbf x_t)\overline{bel}$.

**Kalman**: $\bar{\boldsymbol\Sigma}=\mathbf A\boldsymbol\Sigma\mathbf A^\top+\mathbf R$; $\mathbf K=\bar{\boldsymbol\Sigma}\mathbf C^\top(\mathbf C\bar{\boldsymbol\Sigma}\mathbf C^\top+\mathbf Q)^{-1}$; $\boldsymbol\mu=\bar{\boldsymbol\mu}+\mathbf K(\mathbf z-\mathbf C\bar{\boldsymbol\mu})$.
**Skalar: $K=\bar\sigma^2/(\bar\sigma^2+\sigma_z^2)$ = inverse-Varianz-Fusion (Modul 18) = optimaler Komplementärfilter (Modul 17).** Nichtlinear → EKF (Jacobi) / UKF (Sigma-Punkte).

**Partikelfilter**: bewegen → gewichten $w\propto p(\mathbf z|\mathbf x)$ → normieren → **resampeln**. Multimodal möglich. Fallen: **Partikelverarmung** (nur resampeln bei $N_{\text{eff}}=1/\sum w^2 < M/2$), zu wenige Partikel → Divergenz.

**PID**: $u=K_pe+K_i\int e+K_d\dot e$ (Gegenwart/Vergangenheit/Zukunft). I killt bleibende Abweichung (Windup!), D dämpft (rauschempfindlich).

**Odometrie driftet unbeschränkt** → externe Messung nötig; ohne Karte → SLAM.

---

## Selbsttest

<details>
<summary><b>1.</b> Unterscheide Arbeitsraum und Konfigurationsraum. Warum ist der C-Space für die Planung so nützlich?</summary>

Der **Arbeitsraum** ist die Menge der **Posen im kartesischen Raum**, die der Endeffektor erreichen kann (beim 2-Gelenk-Arm ein Kreisring mit Radien $|l_1-l_2|$ bis $l_1+l_2$). Der **Konfigurationsraum** ist die Menge **aller Gelenkstellungen** $\mathbf q$ (bei $n$ Drehgelenken ein $n$-Torus).

Nützlich, weil im C-Space der **ausgedehnte Roboter zu einem einzigen Punkt** wird: Planung reduziert sich auf „finde einen Weg für einen *Punkt* durch den hindernisfreien Raum $\mathcal C_{\text{free}}$". Der Preis ist die hohe Dimension — deshalb testet man Kollisionen nur punktweise (Sampling) statt $\mathcal C_{\text{obs}}$ explizit zu konstruieren.
</details>

<details>
<summary><b>2.</b> Warum ist die Vorwärtskinematik immer eindeutig, die inverse aber nicht?</summary>

Die **FK** ist eine Funktion $\mathbf q\mapsto\mathbf x$: Setzt man Gelenkwinkel ein, ergibt das Produkt der Transformationen **genau eine** Pose. Die **IK** ist die *Umkehrung* dieser Abbildung, und die ist im Allgemeinen nicht injektiv: Es kann **keine** Lösung geben (Ziel außerhalb des Arbeitsraums), **mehrere** (z. B. Ellbogen oben/unten beim 2-Gelenk-Arm) oder **unendlich viele** (redundante Arme mit $n>m$, deren Nullraum ganze Lösungsmannigfaltigkeiten erzeugt).
</details>

<details>
<summary><b>3.</b> Was ist die Jacobi-Matrix, und was bedeutet $\det\mathbf J = 0$ physikalisch?</summary>

$\mathbf J = \partial f/\partial\mathbf q$ ist die Ableitung der Vorwärtskinematik; sie verknüpft **Gelenkgeschwindigkeiten mit Endeffektor-Geschwindigkeiten**: $\dot{\mathbf x}=\mathbf J\dot{\mathbf q}$ (dual: $\boldsymbol\tau=\mathbf J^\top\mathbf F$).

$\det\mathbf J=0$ heißt **Singularität**: $\mathbf J$ verliert Rang, also gibt es eine **Bewegungsrichtung des Endeffektors, die mit keiner Gelenkbewegung erreichbar ist** — ein Freiheitsgrad geht (lokal) verloren. Beim 2-Gelenk-Arm ist $\det\mathbf J=l_1l_2\sin q_2$, also singulär bei $q_2=0$ (gestreckt) und $q_2=\pi$ (eingeklappt); im gestreckten Zustand kann die Hand sich nicht weiter radial nach außen bewegen. In der Nähe werden die nötigen Gelenkgeschwindigkeiten beliebig groß.
</details>

<details>
<summary><b>4.</b> Ein 2-Gelenk-Arm mit $l_1=l_2=1$ soll $(x,y)=(1,1)$ erreichen. Wie viele Lösungen gibt es, und wie berechnest du $q_2$?</summary>

$r=\sqrt{1^2+1^2}=\sqrt2\approx1.414$, und $|l_1-l_2|=0 \le \sqrt2 \le 2 = l_1+l_2$ → das Ziel ist **erreichbar**.

$\cos q_2 = \dfrac{r^2-l_1^2-l_2^2}{2l_1l_2} = \dfrac{2-1-1}{2}=0 \;\Rightarrow\; q_2=\pm\dfrac{\pi}{2}$.

Also **zwei Lösungen** (Ellbogen oben und unten). Danach $q_1=\operatorname{atan2}(y,x)-\operatorname{atan2}(l_2\sin q_2,\;l_1+l_2\cos q_2)$; für $q_2=+\pi/2$: $q_1=\frac{\pi}{4}-\operatorname{atan2}(1,1)=\frac{\pi}{4}-\frac{\pi}{4}=0$.
</details>

<details>
<summary><b>5.</b> Warum ersetzt man in der numerischen IK die Pseudoinverse durch Damped Least Squares?</summary>

Weil $\mathbf J^{+}=\mathbf J^\top(\mathbf J\mathbf J^\top)^{-1}$ nahe einer **Singularität** versagt: $\mathbf J\mathbf J^\top$ wird fast singulär, die Inverse riesig, und $\Delta\mathbf q$ **explodiert** (unrealistische Gelenksprünge, numerische Instabilität). **DLS** ersetzt sie durch $\mathbf J^\top(\mathbf J\mathbf J^\top+\lambda^2\mathbf I)^{-1}$ — die Dämpfung macht die Inverse **immer** wohldefiniert. Minimiert wird $\|\mathbf J\Delta\mathbf q-\mathbf e\|^2+\lambda^2\|\Delta\mathbf q\|^2$: Man opfert etwas Genauigkeit für **beschränkte** Schrittweiten. Großes $\lambda$ = stabil aber langsam, kleines $\lambda$ = schnell aber nahe Singularitäten instabil.
</details>

<details>
<summary><b>6.</b> Beschreibe RRT in vier Schritten. Was heißt „probabilistisch vollständig"?</summary>

(1) Ziehe eine zufällige Konfiguration $\mathbf q_{\text{rand}}$ (mit kleiner Wahrscheinlichkeit direkt das Ziel — **Goal-Bias**). (2) Finde den **nächsten Baumknoten** $\mathbf q_{\text{near}}$. (3) Gehe von dort einen **Schritt $\varepsilon$** in Richtung $\mathbf q_{\text{rand}}$ → $\mathbf q_{\text{new}}$. (4) Ist die Kante **kollisionsfrei**, füge $\mathbf q_{\text{new}}$ dem Baum hinzu; ist das Ziel nah genug, verfolge die Eltern zurück.

**Probabilistisch vollständig** heißt: Existiert eine Lösung, dann geht die Wahrscheinlichkeit, sie zu finden, mit wachsender Stichprobenzahl **gegen 1** — es gibt aber keine endliche Schranke und **keine Garantie**, wenn keine Lösung existiert (der Algorithmus terminiert dann nicht mit „unlösbar"). Optimalität ist ebenfalls nicht garantiert (→ Glätten, RRT\*).
</details>

<details>
<summary><b>7.</b> Schreibe die zwei Schritte des Bayes-Filters auf und sage, was jeder mit der Unsicherheit macht.</summary>

**Prädiktion**: $\overline{bel}(\mathbf x_t)=\int p(\mathbf x_t\mid\mathbf u_t,\mathbf x_{t-1})\,bel(\mathbf x_{t-1})\,\mathrm d\mathbf x_{t-1}$ — die Bewegung wird angewandt; da das Bewegungsmodell rauscht, wird die Unsicherheit **größer** (der Belief verschmiert).

**Korrektur**: $bel(\mathbf x_t)=\eta\,p(\mathbf z_t\mid\mathbf x_t)\,\overline{bel}(\mathbf x_t)$ — die Messung wird multiplikativ eingearbeitet; das macht den Belief **schärfer** (Unsicherheit sinkt). Kalman- und Partikelfilter sind nur zwei Darstellungen dieses selben Beliefs.
</details>

<details>
<summary><b>8.</b> Zeige, dass der skalare Kalman-Gain die inverse-Varianz-Fusion aus Modul 18 ist.</summary>

Skalar mit $C=1$: $K=\dfrac{\bar\sigma^2}{\bar\sigma^2+\sigma_z^2}$ und $\mu=\bar\mu+K(z-\bar\mu)=(1-K)\bar\mu+Kz$.

Einsetzen: $1-K=\dfrac{\sigma_z^2}{\bar\sigma^2+\sigma_z^2}$, also $\mu=\dfrac{\sigma_z^2\,\bar\mu+\bar\sigma^2 z}{\bar\sigma^2+\sigma_z^2} = \dfrac{\bar\mu/\bar\sigma^2 + z/\sigma_z^2}{1/\bar\sigma^2+1/\sigma_z^2}$ — genau die **reliabilitätsgewichtete** (inverse-Varianz-) Mittelung. Und die Posterior-Varianz erfüllt $1/\sigma^2=1/\bar\sigma^2+1/\sigma_z^2$ (Präzisionen addieren sich). Das ist identisch zu Ernst & Banks (Modul 18); der Komplementärfilter aus Modul 17 ist derselbe Ausdruck mit heuristisch gewähltem Gewicht.
</details>

<details>
<summary><b>9.</b> Wann brauchst du einen Partikelfilter statt eines Kalman-Filters? Was ist Partikelverarmung?</summary>

Wenn der Belief **nicht** durch eine einzelne Gaußverteilung darstellbar ist — vor allem bei **globaler Lokalisierung** (multimodal: mehrere Orte in der Karte sehen gleich aus) oder stark **nichtlinearen/nicht-gaußschen** Modellen. Der Partikelfilter repräsentiert beliebige Verteilungen durch gewichtete Stichproben.

**Partikelverarmung**: Durch (zu häufiges) Resampling verschwindet die Vielfalt — Partikel werden mehrfach kopiert, bis im Extremfall alle identisch sind; der Filter kann sich dann nicht mehr korrigieren und divergiert. Gegenmittel: nur resampeln, wenn die **effektive Stichprobengröße** $N_{\text{eff}}=1/\sum_m (w^{[m]})^2$ unter eine Schwelle (z. B. $M/2$) fällt.
</details>

<details>
<summary><b>10.</b> Wozu dient jeder der drei PID-Terme, und was ist die typische Schwäche jedes einzelnen?</summary>

- **P** ($K_p e$): reagiert auf den **aktuellen** Fehler. Schwäche: allein bleibt oft eine **bleibende Regelabweichung**; zu groß → Überschwingen/Schwingen.
- **I** ($K_i\int e$): summiert **vergangene** Fehler und eliminiert die bleibende Abweichung. Schwäche: **Integral-Windup** (Aufsummieren, wenn das Stellglied gesättigt ist) → träges Überschwingen.
- **D** ($K_d\dot e$): reagiert auf die **Änderungsrate** (antizipiert), wirkt **dämpfend**. Schwäche: **verstärkt Messrauschen**.
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **Siciliano, Sciavicco, Villani & Oriolo, *Robotics: Modelling, Planning and Control*** (Springer). Das umfassende Standardwerk zu Kinematik, Jacobi, Planung, Regelung. *Vertiefend, sehr gründlich.*
- **Craig, *Introduction to Robotics: Mechanics and Control***. Der Klassiker für **DH-Parameter** und Kinematik, angenehm konkret. *Einsteigerfreundlich.*
- **Thrun, Burgard & Fox, *Probabilistic Robotics*** (MIT Press). Die Referenz für Bayes-/Kalman-/Partikelfilter und Lokalisierung — Kapitel 2–4 und 8 decken Abschnitt 9–11 dieses Skripts ab. *Sehr gut geschrieben; Pflicht für die Schätz-Seite.*
- **LaValle, *Planning Algorithms*** — **komplett frei online** unter lavalle.pl/planning/. Die Referenz für C-Space, PRM, RRT (RRT stammt von LaValle selbst). *Kostenlos, vertiefend.*
- **Lynch & Park, *Modern Robotics*** — frei als PDF + **kostenlose Vorlesungsvideos** (Northwestern/Coursera). Moderne Darstellung (Screw-Theorie statt DH). *Kostenlos, einsteigerfreundlich.*

**Schlüssel-Papers**
- **Denavit & Hartenberg (1955)** — die Konvention. *Historisch.*
- **LaValle, „Rapidly-Exploring Random Trees: A New Tool for Path Planning"**, TR 1998. Das RRT-Originalpaper, kurz und lesbar. *Einsteigerfreundlich.*
- **Kavraki et al., „Probabilistic Roadmaps"**, *IEEE T-RA 1996*. *Vertiefend.*
- **Karaman & Frazzoli, „Sampling-based Algorithms for Optimal Motion Planning" (RRT\*)**, *IJRR 2011*. Asymptotische Optimalität. *Vertiefend.*
- **Buss, „Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares"** (Notizen, frei online). Genau Abschnitt 6 — sehr empfehlenswert. *Kostenlos, kompakt.*

**Frei verfügbare Kurse**
- **Modern Robotics** (Kevin Lynch, Northwestern) — Videoreihe + Buch, kostenlos.
- **Cyrill Stachniss (Uni Bonn)**, YouTube — hervorragende Vorlesungen zu Kalman-/Partikelfilter, SLAM und Planung. *Kostenlos, sehr zu empfehlen.*

**Zum Ausprobieren**
- Die **drei Projekte** bauen FK+Jacobi+Arbeitsraum (basic), analytische und numerische **IK inkl. DLS/Singularitäten** (medium) und eine vollständige **Sense-Plan-Act-Navigation** mit RRT + Partikelfilter + PID (final) — alles from scratch.

---

> **Nächstes Modul:** Modul 22 „Robotics 2" — Fortgeschrittene Robotik: Dynamik (Kräfte/Momente statt nur Geometrie), **SLAM**, fortgeschrittene Regelung und lernbasierte Ansätze. Die Kinematik, Planung und Filterung aus diesem Modul sind die direkte Grundlage.
