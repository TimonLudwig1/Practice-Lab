# Modul 20 — 3D Point Cloud Processing

> **Worum geht es?** Ein **LiDAR**, eine **Tiefenkamera** (Kinect/RealSense) oder **Photogrammetrie** liefern die Welt nicht als sauberes Gitter, sondern als **Punktwolke (point cloud)** — eine ungeordnete Menge von 3D-Punkten $\{\mathbf p_1, \dots, \mathbf p_n\}$, jeder mit optionalen Attributen (Farbe, Intensität, Normale). Dieses Modul behandelt, wie man aus dieser rohen Geometrie Bedeutung gewinnt: **Nachbarschaften** finden, **Normalen** schätzen, zwei Scans **registrieren** (das berühmte **ICP**), Ebenen und Objekte **segmentieren**, und wie **Deep Learning** (PointNet) mit der ungeordneten Struktur umgeht.
>
> **Vorkenntnisse**: lineare Algebra (Eigenwerte, SVD, Skalarprodukt/Kreuzprodukt), etwas Statistik. Aus diesem Repo bauen direkt auf: **Modul 05** (PCA — Normalen sind eine lokale PCA; Eigenzerlegung der Kovarianz), **Modul 10** (Procrustes-Alignment — ICP löst pro Iteration ein Procrustes-Problem), **Modul 19** (homogene/rigid Transformationen — das Ergebnis von ICP ist eine $4\times4$-Rigid-Matrix). **Modul 19 ist Pflicht-Vormodul.**

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–19 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, eng an der Standardpraxis (Open3D/PCL-Ökosystem, die Registrierungs- und Segmentierungspipelines der Robotik/autonomen Fahrzeuge) und konsistent mit dem 3D-Block dieses Repos. **Bewusst ohne echte Sensorhardware und ohne die Bibliothek Open3D** (die hier fehlt): Der lehrbare Kern sind die **Algorithmen und ihre Mathematik** — kd-Baum-Nachbarschaften, PCA-Normalen, die **SVD-Lösung von ICP**, die **RANSAC-Statistik**, das **Permutationsinvarianz-Prinzip** von PointNet. Wer `open3d.registration.icp()` aufruft, versteht ICP nicht; wer die Kabsch-Rotation aus der SVD herleitet und den Konvergenz-Basin vermisst, schon. Alle Projekte arbeiten **from scratch** mit reinem `numpy`/`scipy` auf synthetischen, reproduzierbaren Punktwolken — CPU-Sekunden.

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

- erklären können, was eine Punktwolke **fundamental schwierig** macht: **Ungeordnetheit (Permutationsinvarianz)**, irreguläre Dichte, fehlende Topologie/Konnektivität, Rauschen, Teilüberlappung.
- die **Kern-Nachbarschaftsoperationen** (kNN, Radius-Suche) und ihre effiziente Umsetzung per **kd-Baum** verstehen.
- **Normalen und Krümmung** über eine **lokale PCA** (Eigenzerlegung der Kovarianzmatrix) schätzen — inklusive der Vorzeichen-/Orientierungsmehrdeutigkeit.
- **Voxel-Downsampling** und andere Sampling-Strategien (FPS) motivieren und anwenden.
- das **ICP-Verfahren** vollständig herleiten und implementieren können: die **geschlossene SVD-Lösung** des Procrustes-Problems (Kabsch, mit Determinanten-Korrektur gegen Spiegelungen), die Iteration Korrespondenz→Lösung→Anwendung, **point-to-point vs. point-to-plane**, und **warum ICP nur lokal konvergiert**.
- **RANSAC** für robuste Ebenenschätzung herleiten — inklusive der **Iterationszahl-Formel** $N=\log(1-p)/\log(1-w^s)$ — und mit Clustering zu einer **Segmentierungspipeline** kombinieren.
- **Feature-Deskriptoren** (PFH/FPFH, Spin Images) und die **globale Registrierungs-Pipeline** (Feature-Matching → RANSAC → ICP-Verfeinerung) einordnen.
- das **PointNet-Prinzip** erklären: warum eine **symmetrische Funktion (Max-Pooling)** über per-Punkt-MLPs Permutationsinvarianz erzeugt, und wo seine Grenzen liegen (→ PointNet++).

---

## Grundlagen (Basics)

### 1. Was ist eine Punktwolke — und warum ist sie schwer?

Eine Punktwolke ist eine **Menge** $P = \{\mathbf p_1, \dots, \mathbf p_n\}$ mit $\mathbf p_i \in \mathbb R^3$ (oft plus Attribute). Das Wort **Menge** ist der Kern aller Schwierigkeiten:

1. **Ungeordnet / permutationsinvariant.** Es gibt keine „erste" oder „zehnte" Zeile wie bei einem Bild. Zwei Wolken mit denselben Punkten in anderer Reihenfolge sind **identisch**. Jeder Algorithmus (und jedes neuronale Netz!) muss **invariant gegenüber der Punktreihenfolge** sein — das ist die zentrale Designbedingung (Abschnitt 13).
2. **Irreguläre Abtastung / variable Dichte.** LiDAR-Punkte sind nah am Sensor dicht, fern dünn. Kein gleichmäßiges Raster → man kann keine Standard-Faltung (Modul 11) anwenden.
3. **Keine Topologie.** Ein Mesh hat Kanten/Flächen; eine Punktwolke nur Punkte. „Welche Punkte gehören zur selben Fläche?" muss man erst *ableiten* (Nachbarschaften, Normalen, Segmentierung).
4. **Rauschen, Ausreißer, Teilüberlappung, Verdeckung.** Echte Scans sind verrauscht, haben Fehlmessungen (Ausreißer), und ein einzelner Scan sieht immer nur *eine Seite* eines Objekts.

Diese vier Punkte motivieren praktisch das gesamte Modul.

### 2. Woher kommen Punktwolken?

- **LiDAR** (Light Detection and Ranging): misst Laufzeit/Phase von Laserpulsen → Distanz pro Strahlrichtung. Kern der autonomen Fahrzeuge und der Vermessung.
- **Tiefenkameras** (structured light: Kinect v1; time-of-flight: Kinect v2, RealSense): liefern ein Tiefenbild, das per Kameramodell (Modul 19: Rückprojektion mit der inversen Projektionsmatrix) in 3D-Punkte umgerechnet wird.
- **Stereo / Photogrammetrie / SfM**: aus mehreren 2D-Bildern (Modul 11/12) durch Triangulation.
- **Simulation** (wie in diesem Modul): aus bekannten Flächen sampeln — erlaubt *ground truth* für die Evaluation.

### 3. Nachbarschaften und der kd-Baum

Fast jede Punktwolken-Operation braucht **lokale Nachbarschaften** — „gib mir die $k$ nächsten Punkte zu $\mathbf p$" (**kNN**) oder „alle Punkte im Radius $r$" (**Radius-Suche**). Naiv kostet das $O(n)$ pro Anfrage, also $O(n^2)$ für alle Punkte — untragbar bei Millionen Punkten.

Der **kd-Baum (k-dimensional tree)** ist die Standard-Datenstruktur: ein binärer Baum, der den Raum rekursiv entlang alternierender Achsen an Medianen aufteilt. Aufbau in $O(n\log n)$, eine Nachbarschaftsanfrage im Mittel in $O(\log n)$. Bei einer Suche schneidet man Teilbäume ab, deren Region weiter als der bisher beste Kandidat entfernt ist (branch-and-bound). Wir nutzen `scipy.spatial.cKDTree` — die Datenstruktur selbst zu implementieren ist nicht das Lernziel, ihre *Verwendung* für kNN/Radius schon.

> **Merke:** Der kd-Baum ist der stille Motor unter fast allem, was folgt — Normalen-Schätzung (lokale Nachbarschaft), ICP (nächste Korrespondenzen), Clustering (Radius-Nachbarn). Effiziente Nachbarschaft = Voraussetzung für alles.

### 4. Downsampling: Voxel-Grid und Farthest Point Sampling

Rohe Scans sind riesig und ungleichmäßig dicht. **Downsampling** reduziert die Punktzahl *und* vergleichmäßigt die Dichte:

- **Voxel-Grid**: Lege ein 3D-Gitter der Kantenlänge $v$ über die Wolke; **ersetze alle Punkte in einem Voxel durch ihren Schwerpunkt** (centroid). Ergebnis: höchstens ein Punkt pro besetztem Voxel, näherungsweise gleichmäßige Dichte. Der De-facto-Standard-Vorverarbeitungsschritt.
- **Farthest Point Sampling (FPS)**: Wähle iterativ den Punkt, der am **weitesten** von der bisher gewählten Menge entfernt ist. Ergibt eine gut verteilte Teilmenge fester Größe — genutzt in PointNet++ (Abschnitt 14).
- **Uniform/Random**: einfach jeden $k$-ten oder zufällig — schnell, aber respektiert die Dichte nicht.

### 5. Normalen-Schätzung: eine lokale PCA

Eine **Oberflächennormale** $\mathbf n_i$ pro Punkt ist die wichtigste abgeleitete Größe (für Beleuchtung, point-to-plane-ICP, Features, Segmentierung). Ohne Mesh schätzt man sie aus der **lokalen Nachbarschaft** — und das ist **exakt eine PCA** (Rückverweis Modul 05):

1. Nimm die $k$ nächsten Nachbarn $\mathcal N(\mathbf p_i)$.
2. Bilde ihren Schwerpunkt $\bar{\mathbf p}$ und die **Kovarianzmatrix**
   $$\mathbf C = \frac{1}{|\mathcal N|}\sum_{\mathbf q \in \mathcal N} (\mathbf q - \bar{\mathbf p})(\mathbf q - \bar{\mathbf p})^\top \in \mathbb R^{3\times3}.$$
3. Eigenzerlegung $\mathbf C = \sum_j \lambda_j \mathbf v_j \mathbf v_j^\top$ mit $\lambda_0 \le \lambda_1 \le \lambda_2$. Die lokale Fläche spannt sich in Richtung der **zwei großen** Eigenvektoren auf; die Richtung der **kleinsten Varianz** steht **senkrecht** darauf — das ist die **Normale**:
   $$\boxed{\;\mathbf n_i = \mathbf v_0 \quad (\text{Eigenvektor zum kleinsten Eigenwert } \lambda_0)\;}$$

**Vorzeichen-Ambiguität.** $\mathbf v_0$ und $-\mathbf v_0$ sind beide gültige Eigenvektoren — die PCA legt die *Orientierung* nicht fest. Man orientiert konsistent, z. B. **zum Sensor/Betrachterpunkt hin**: ist $\mathbf n_i \cdot (\mathbf p_{\text{view}} - \mathbf p_i) < 0$, drehe $\mathbf n_i \mapsto -\mathbf n_i$.

**Krümmung als Bonus.** Die Eigenwerte liefern gratis ein Krümmungsmaß (surface variation):
$$\sigma = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}.$$
$\sigma \approx 0$: flach (ein Eigenwert winzig); $\sigma \to 1/3$: isotrop/verrauscht/Ecke. Das **Basic-Projekt** rechnet Normalen und Krümmung so von Hand.

---

## Aufbau (Intermediate)

### 6. Registrierung und ICP — das zentrale Problem

**Registrierung** heißt: Zwei Punktwolken $P$ (source) und $Q$ (target), die dasselbe Objekt aus **verschiedenen Blickwinkeln** zeigen (teilüberlappend), durch eine **starre Transformation** $(\mathbf R, \mathbf t)$ zur Deckung bringen. Anwendungen: 3D-Scans zu einem Modell fusionieren, Roboter-Lokalisierung (Scan gegen Karte), SLAM.

Das **Iterative Closest Point (ICP)** (Besl & McKay 1992) ist der Klassiker. Es zerfällt in zwei abwechselnde Schritte, die je für sich lösbar sind:

```
wiederhole bis Konvergenz:
  (1) KORRESPONDENZ:  fuer jeden Punkt p_i in P finde den naechsten Punkt q_{c(i)} in Q  (kd-Baum!)
  (2) TRANSFORMATION: loese (R,t) = argmin  sum_i || R p_i + t - q_{c(i)} ||^2   (geschlossen, s.u.)
  (3) wende (R,t) auf P an; miss den Fehler; wenn kaum noch Aenderung -> fertig
```

Der Trick: Schritt (2) hat bei *festen* Korrespondenzen eine **geschlossene Lösung** — das ist das Procrustes/Kabsch-Problem.

### 7. Die SVD-Lösung des Procrustes-Problems (Kabsch)

Gegeben Korrespondenzpaare $(\mathbf p_i, \mathbf q_i)$, gesucht $(\mathbf R, \mathbf t)$ mit $\mathbf R^\top\mathbf R = \mathbf I$, $\det\mathbf R = +1$ (echte Rotation), die
$$E(\mathbf R, \mathbf t) = \sum_i \|\mathbf R\mathbf p_i + \mathbf t - \mathbf q_i\|^2$$
minimiert. Die Herleitung (baut auf Modul 10):

**Schritt 1 — Translation ausrechnen.** Für optimales $\mathbf R$ ergibt $\partial E/\partial \mathbf t = 0$, dass die Schwerpunkte aufeinander abgebildet werden: $\mathbf t = \bar{\mathbf q} - \mathbf R\bar{\mathbf p}$. Also **zentriere** beide Wolken: $\tilde{\mathbf p}_i = \mathbf p_i - \bar{\mathbf p}$, $\tilde{\mathbf q}_i = \mathbf q_i - \bar{\mathbf q}$. Dann bleibt nur noch die Rotation.

**Schritt 2 — Rotation via SVD.** Minimiere $\sum_i\|\mathbf R\tilde{\mathbf p}_i - \tilde{\mathbf q}_i\|^2$. Ausmultiplizieren und Weglassen konstanter Terme zeigt: das ist **äquivalent zu $\max_{\mathbf R}\ \mathrm{tr}(\mathbf R^\top \mathbf H)$** mit der **Kreuz-Kovarianzmatrix**
$$\mathbf H = \sum_i \tilde{\mathbf p}_i\,\tilde{\mathbf q}_i^\top \in \mathbb R^{3\times3}.$$
Mit der Singulärwertzerlegung $\mathbf H = \mathbf U\,\mathbf \Sigma\,\mathbf V^\top$ ist die optimale Rotation
$$\boxed{\;\mathbf R = \mathbf V\,\mathrm{diag}(1,\,1,\,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top,\qquad \mathbf t = \bar{\mathbf q} - \mathbf R\bar{\mathbf p}\;}$$

**Die Determinanten-Korrektur** $\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))$ ist essenziell: Ohne sie könnte $\mathbf R = \mathbf V\mathbf U^\top$ eine **Spiegelung** ($\det = -1$) sein, wenn die Daten sehr verrauscht/entartet sind. Die Korrektur erzwingt $\det\mathbf R = +1$ (echte Rotation, keine Spiegelung). Das ist die berühmte **Kabsch-Formel** (auch: Umeyama, wenn zusätzlich Skalierung geschätzt wird) — dieselbe Struktur wie das orthogonale Procrustes aus Modul 10, hier für 3D-Rigid-Body. Das **Medium-Projekt** implementiert genau diese Lösung und baut ICP darauf.

### 8. ICP-Konvergenz, point-to-plane und Robustheit

**Konvergenz.** Jeder ICP-Schritt senkt den Fehler **monoton** (beide Teilschritte sind optimal für ihre Variable) → ICP konvergiert garantiert. **Aber nur zu einem lokalen Minimum**: Startet man mit einer stark falschen Ausgangspose, findet ICP eine falsche Deckung (die nächsten Nachbarn sind dann die falschen Punkte). ICP braucht daher eine **gute Initialisierung** — die liefert die globale Registrierung (Abschnitt 12).

**Point-to-plane.** Statt Punkt-zu-Punkt-Distanz minimiert man den Abstand **entlang der Ziel-Normalen**:
$$E_{\perp} = \sum_i \big(\mathbf n_{q_i}\cdot(\mathbf R\mathbf p_i + \mathbf t - \mathbf q_i)\big)^2.$$
Das erlaubt den Punkten, **entlang der Oberfläche zu gleiten**, statt Punkt-auf-Punkt zu rasten — was bei nicht deckungsgleicher Abtastung (Punkte liegen nie exakt aufeinander) **deutlich schneller konvergiert** (oft in wenigen Iterationen) und einen **größeren Konvergenz-Basin** hat. Preis: Es braucht Normalen (Abschnitt 5) und wird per Gauß-Newton linearisiert gelöst (kleine Winkel).

**Robustheit gegen Ausreißer/Teilüberlappung.** Vanilla-ICP nimmt *alle* Korrespondenzen gleich ernst — ein einziger Ausreißer oder der nicht-überlappende Teil zieht die Lösung weg. Gegenmittel:
- **Distanzschwelle / Trimming**: verwirf Korrespondenzpaare, deren Abstand zu groß ist (**Trimmed ICP**, nutze nur die besten $x\%$).
- **Robuste Gewichte** (Huber/Tukey statt quadratischer Verlust).
- **Reziproke Korrespondenzen** (nur behalten, wenn $\mathbf p$ nächster zu $\mathbf q$ *und* umgekehrt).

Das **Final-Projekt** baut Segmentierung; die Robustheitsmechanismen von ICP werden im Medium-Projekt behandelt.

### 9. Segmentierung I: RANSAC-Ebenenschätzung

Viele Szenen bestehen aus **Ebenen** (Boden, Wände, Tische) plus Objekten. Ebenen findet man robust mit **RANSAC** (RANdom SAmple Consensus, Fischler & Bolles 1981):

```
wiederhole N mal:
  1. ziehe zufaellig 3 Punkte -> definieren eine Ebene (Normale = Kreuzprodukt zweier Kantenvektoren)
  2. zaehle INLIER: Punkte mit |Abstand zur Ebene| < tau
  3. merke das Modell mit den meisten Inliern
danach: passe die Ebene per PCA an ALLE Inlier an (Verfeinerung)
```

Der Abstand eines Punktes $\mathbf x$ zur Ebene mit Normale $\mathbf n$ (Einheitsvektor) durch $\mathbf p_0$ ist $|\mathbf n\cdot(\mathbf x - \mathbf p_0)|$. RANSAC ist robust, weil ein Modell aus einem **ausreißerfreien Minimalsample** (3 Punkte) genügt und über viele Versuche irgendwann getroffen wird.

**Wie viele Iterationen $N$?** Bei einem Inlier-Anteil $w$ und Minimalsample-Größe $s$ (hier $s=3$) ist die Wahrscheinlichkeit, dass ein Sample **komplett aus Inliern** besteht, $w^s$. Um mit Konfidenz $p$ (z. B. 0.99) **mindestens einmal** ein sauberes Sample zu ziehen, braucht man
$$\boxed{\;N = \frac{\log(1-p)}{\log(1 - w^s)}\;}$$
Beispiel $w=0.5$, $s=3$, $p=0.99$: $N = \log(0.01)/\log(1-0.125) \approx 35$. Bei $w=0.3$ schon $\approx 169$. Diese Formel ist die Rechtfertigung, warum RANSAC mit *wenigen* Iterationen auskommt — und wird im **Final-Projekt** empirisch verifiziert.

### 10. Segmentierung II: Clustering der Objekte

Nach dem Entfernen der großen Ebene(n) bleiben die **Objekt-Punkte**. Diese trennt man in einzelne Objekte per **Clustering** — hier ist die räumliche Struktur der Schlüssel:

- **Euclidean Clustering / Region Growing**: Starte an einem Punkt, füge alle Punkte im Radius $r$ hinzu, wachse rekursiv → ein Cluster; wiederhole für unbesuchte Punkte. Das ist im Kern **DBSCAN** (Modul 05) mit `min_samples` klein — Punkte, deren $r$-Nachbarschaften sich berühren, bilden ein Objekt.
- **DBSCAN** (Modul 05) direkt: robust gegen Rauschen (Ausreißer werden als „noise" markiert), findet beliebig geformte Cluster, braucht keine Clusterzahl.

Die Pipeline **RANSAC-Ebene entfernen → Rest clustern** ist der klassische „tabletop segmentation"-Ansatz der Robotik-Perzeption. Das **Final-Projekt** baut sie vollständig und evaluiert sie gegen ground-truth-Labels.

### 11. Feature-Deskriptoren

Für die **globale** Registrierung (ohne gute Initialisierung) und für Objekterkennung braucht man **lokale Deskriptoren**, die die Geometrie um einen Punkt **rotations- und translationsinvariant** beschreiben:

- **PFH / FPFH** (Fast Point Feature Histograms, Rusu et al.): Histogramme der **Winkelbeziehungen zwischen den Normalen** benachbarter Punktpaare in der Nachbarschaft. FPFH ist die schnelle, in der Praxis dominante Variante. Zwei Punkte mit ähnlichem FPFH haben ähnliche lokale Geometrie → Kandidaten für eine Korrespondenz.
- **Spin Images** (Johnson & Hebert): projizieren die lokale Nachbarschaft in ein 2D-Histogramm bzgl. der Normalen-Achse.

---

## Advanced-Themen

### 12. Die globale Registrierungs-Pipeline

ICP braucht eine gute Initialisierung (Abschnitt 8). Diese liefert die **globale (grobe) Registrierung**, die *ohne* Anfangspose auskommt:

```
1. Downsampling (Voxel) beider Wolken
2. Normalen + FPFH-Deskriptoren berechnen
3. FEATURE-MATCHING: fuer jeden Punkt in P den FPFH-naechsten in Q -> Kandidaten-Korrespondenzen
4. RANSAC ueber Korrespondenzen: ziehe 3 Matches, loese Kabsch, zaehle Inlier -> grobe (R,t)
5. ICP-VERFEINERUNG (point-to-plane) ausgehend von der groben Pose -> praezise (R,t)
```

Das ist der Standard (z. B. Open3Ds `global_registration` + `icp`). Der Kern: **Features geben die grobe Ausrichtung (großer Einzugsbereich, aber ungenau), ICP verfeinert (präzise, aber kleiner Einzugsbereich)** — eine schöne Arbeitsteilung „grob-zu-fein". Die neuere Alternative **FGR** (Fast Global Registration) verzichtet auf RANSAC und optimiert einen robusten Verlust direkt.

### 13. Deep Learning auf Punktwolken: das Permutationsinvarianz-Problem

Ein Bild-CNN (Modul 11) setzt ein **geordnetes Gitter** voraus — für eine ungeordnete Punktmenge unbrauchbar. Die Kernfrage: **Wie baut man ein neuronales Netz $f$, dessen Ausgabe sich nicht ändert, wenn man die Punkte umordnet?** Formal: $f$ muss **permutationsinvariant** sein, $f(\{\mathbf x_1,\dots,\mathbf x_n\}) = f(\{\mathbf x_{\pi(1)},\dots,\mathbf x_{\pi(n)}\})$ für jede Permutation $\pi$.

**PointNet** (Qi et al. 2017) löst das mit einer **symmetrischen Funktion**. Die zentrale Konstruktion:
$$\boxed{\;f(\{\mathbf x_1,\dots,\mathbf x_n\}) = \gamma\!\Big(\underset{i=1,\dots,n}{\text{MAX}}\ h(\mathbf x_i)\Big)\;}$$
- $h$ ist ein **geteiltes MLP**, das **jeden Punkt einzeln** in einen hochdimensionalen Merkmalsraum abbildet (identisch für alle Punkte).
- **MAX** ist ein **elementweises Max-Pooling** über *alle* Punkte → ein einziger globaler Merkmalsvektor.
- $\gamma$ ist ein weiteres MLP für die finale Vorhersage (Klasse / Segmentierung).

**Warum funktioniert das?** Weil **Max (wie Summe/Mittel) symmetrisch** ist: Das Maximum einer Menge hängt nicht von der Reihenfolge ab. Also ist die ganze Pipeline permutationsinvariant — *by construction*. Qi et al. bewiesen sogar, dass diese Form jede stetige mengeninvariante Funktion approximieren kann (Universalität), wenn $h$ hochdimensional genug ist. Zusätzlich richtet ein **T-Net** (ein kleines Netz, das eine Transformationsmatrix vorhersagt) die Eingabe/Features aus (Invarianz gegen Rigid-Transformationen).

**Grenze und Nachfolger.** PointNet aggregiert **global** (ein Max über *alle* Punkte) und erfasst deshalb **keine lokale Struktur**. **PointNet++** behebt das **hierarchisch**: wiederholt **Farthest Point Sampling** (Abschnitt 4) → **Gruppierung** lokaler Nachbarschaften (kd-Baum/Radius) → PointNet auf jeder lokalen Gruppe — genau wie ein CNN lokale Faltungen stapelt, nur auf Punkten. Weitere Familien: **voxel-basiert** (3D-CNN auf Voxelgittern), **graph-basiert** (DGCNN — kNN-Graph + Graph-Convolution, Rückverweis Modul 16), und punktbasierte Transformer.

> **Praxis-Hinweis (Modul-Regel):** Ein PointNet auf ModelNet/ShapeNet zu trainieren braucht GPU-Stunden — auf einem Laptop weder nötig noch sinnvoll. Wir verstehen das **Prinzip** (symmetrische Funktion → Permutationsinvarianz) und können es an einem **winzigen** Beispiel *demonstrieren* (ein Mini-Max-Pooling-Netz, das eine mengeninvariante Eigenschaft lernt) — das ist die eigentliche Master-Einsicht, unabhängig von der Skalierung.

### 14. Oberflächen-Rekonstruktion und Registrierungstheorie (kurz)

- **Oberflächen-Rekonstruktion** (Punktwolke → Mesh): **Poisson-Rekonstruktion** (löst eine Poisson-Gleichung aus den orientierten Normalen — deshalb sind Normalen so wichtig), **Ball-Pivoting**, **Marching Cubes** auf einer impliziten Funktion.
- **Globale Optimalität von ICP**: Vanilla-ICP ist nur lokal; **Go-ICP** garantiert per Branch-and-Bound über den Rotationsraum $SO(3)$ das **globale** Optimum (teurer). Zeigt, dass die Initialisierungsabhängigkeit ein *lösbares*, aber grundsätzliches Problem ist.

---

## Zusammenfassung / Cheat-Sheet

**Was eine Punktwolke schwer macht**: ungeordnet (permutationsinvariant) · irreguläre Dichte · keine Topologie · Rauschen/Ausreißer/Teilüberlappung.

**Nachbarschaften**: kNN & Radius via **kd-Baum** ($O(\log n)$/Anfrage). Motor unter Normalen, ICP, Clustering.

**Normalen (lokale PCA)**: $\mathbf C=\frac1{|\mathcal N|}\sum(\mathbf q-\bar{\mathbf p})(\mathbf q-\bar{\mathbf p})^\top$; $\mathbf n=\mathbf v_0$ (kleinster Eigenwert); Vorzeichen zum Betrachter; Krümmung $\sigma=\lambda_0/\sum\lambda$.

**Downsampling**: Voxel-Grid (Schwerpunkt/Voxel), FPS (gleichverteilte Auswahl).

**ICP** (iteriere): (1) nächste Korrespondenz (kd-Baum), (2) Kabsch lösen, (3) anwenden. Konvergiert **monoton**, aber nur **lokal** → gute Init nötig.

**Kabsch/Procrustes (SVD)**: zentrieren; $\mathbf H=\sum\tilde{\mathbf p}_i\tilde{\mathbf q}_i^\top$; $\mathbf H=\mathbf U\Sigma\mathbf V^\top$; $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$; $\mathbf t=\bar{\mathbf q}-\mathbf R\bar{\mathbf p}$. **Det-Korrektur** verhindert Spiegelung.

**point-to-plane**: minimiere $(\mathbf n_q\cdot(\mathbf R\mathbf p+\mathbf t-\mathbf q))^2$ → schneller, größerer Basin, braucht Normalen.

**RANSAC-Ebene**: 3 Punkte → Ebene (Normale = Kreuzprodukt); Inlier $|\mathbf n\cdot(\mathbf x-\mathbf p_0)|<\tau$; Iterationen $N=\log(1-p)/\log(1-w^s)$.

**Segmentierungspipeline**: RANSAC-Ebene entfernen → Rest per DBSCAN/Euclidean clustern.

**Globale Registrierung**: Voxel → FPFH → Feature-Match → RANSAC (grob) → ICP (fein). Grob-zu-fein.

**PointNet**: $f=\gamma(\mathrm{MAX}_i\,h(\mathbf x_i))$ — geteiltes per-Punkt-MLP + **symmetrisches Max-Pooling** = **Permutationsinvarianz by construction**. Grenze: keine lokale Struktur → **PointNet++** (FPS + Gruppierung, hierarchisch).

---

## Selbsttest

<details>
<summary><b>1.</b> Warum kann man ein Standard-CNN (Modul 11) nicht direkt auf eine Punktwolke anwenden, und was ist die fundamentale Eigenschaft, die jedes Punktwolken-Netz erfüllen muss?</summary>

Ein CNN setzt ein **reguläres, geordnetes Gitter** voraus (Faltung über feste Nachbar-Pixel). Eine Punktwolke ist **ungeordnet** und **irregulär abgetastet** — es gibt keine feste Nachbarschaft und keine Reihenfolge. Die fundamentale Eigenschaft ist **Permutationsinvarianz**: Die Ausgabe darf sich nicht ändern, wenn man die Punkte umordnet, da dieselbe Menge dieselbe Wolke ist.
</details>

<details>
<summary><b>2.</b> Wie schätzt man eine Punktnormale, und welche Größe der lokalen PCA liefert sie? Was ist mit dem Vorzeichen?</summary>

Man nimmt die **lokale Nachbarschaft**, bildet ihre **Kovarianzmatrix** $\mathbf C$ und zerlegt sie in Eigenwerte/-vektoren. Die Fläche liegt in Richtung der zwei größten Eigenvektoren; die Normale ist der **Eigenvektor zum kleinsten Eigenwert** $\lambda_0$ (Richtung minimaler Varianz, senkrecht zur Fläche). Das **Vorzeichen** ist mehrdeutig ($\pm\mathbf v_0$ sind beide Eigenvektoren) — man orientiert konsistent, meist **zum Sensor/Betrachter hin** (Vorzeichen umdrehen, falls $\mathbf n\cdot(\mathbf p_{\text{view}}-\mathbf p)<0$).
</details>

<details>
<summary><b>3.</b> Beschreibe die zwei Schritte einer ICP-Iteration. Warum konvergiert ICP, und warum nur lokal?</summary>

(1) **Korrespondenz**: Für jeden Quellpunkt den nächsten Zielpunkt suchen (kd-Baum). (2) **Transformation**: Bei diesen festen Korrespondenzen die optimale $(\mathbf R,\mathbf t)$ per Kabsch/SVD lösen und anwenden. ICP **konvergiert**, weil beide Schritte den Fehler **monoton senken** (jeder ist optimal für seine Variable) und der Fehler nach unten beschränkt ist. Nur **lokal**, weil die Korrespondenzen von der aktuellen Pose abhängen: Startet man weit von der Lösung, sind die „nächsten" Punkte die *falschen* → ICP rastet in einem lokalen Minimum ein. Deshalb braucht ICP eine gute Initialisierung.
</details>

<details>
<summary><b>4.</b> Leite die Rolle der SVD in der Kabsch-Lösung nach: Welche Matrix wird zerlegt, und wofür ist die Determinanten-Korrektur?</summary>

Nach dem Zentrieren beider Wolken reduziert sich das Problem auf $\max_{\mathbf R}\mathrm{tr}(\mathbf R^\top\mathbf H)$ mit der **Kreuz-Kovarianz** $\mathbf H=\sum_i\tilde{\mathbf p}_i\tilde{\mathbf q}_i^\top$. Mit $\mathbf H=\mathbf U\Sigma\mathbf V^\top$ ist die optimale Rotation $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$. Die **Determinanten-Korrektur** stellt sicher, dass $\det\mathbf R=+1$ ist (echte Rotation). Ohne sie könnte $\mathbf V\mathbf U^\top$ bei verrauschten/entarteten Daten eine **Spiegelung** ($\det=-1$) sein, die physikalisch keine gültige Starrkörperbewegung ist.
</details>

<details>
<summary><b>5.</b> Was ist der Vorteil von point-to-plane gegenüber point-to-point ICP?</summary>

Point-to-plane minimiert den Abstand **entlang der Ziel-Normalen** statt Punkt-auf-Punkt. Das lässt die Punkte **entlang der Oberfläche gleiten**, statt exakt aufeinander rasten zu müssen — was realistisch ist (die Abtastpunkte zweier Scans liegen nie exakt übereinander). Folge: **schnellere Konvergenz** (oft wenige Iterationen) und ein **größerer Konvergenz-Basin**. Preis: Es braucht Oberflächennormalen und wird linearisiert (Gauß-Newton) gelöst.
</details>

<details>
<summary><b>6.</b> Ein RANSAC-Ebenendetektor läuft bei Inlier-Anteil $w=0.4$. Wie viele Iterationen für 99 % Konfidenz? Formel und Größenordnung.</summary>

$N=\dfrac{\log(1-p)}{\log(1-w^s)}$ mit $p=0.99$, $s=3$, $w=0.4$: $w^s=0.064$, $\log(0.01)/\log(0.936)\approx -4.605/-0.0661\approx \mathbf{70}$ Iterationen. (Die Formel wächst stark, wenn $w$ sinkt — bei $w=0.2$ schon ~570.) Kernpunkt: RANSAC kommt mit **überraschend wenigen** Iterationen aus, solange der Inlier-Anteil nicht winzig ist.
</details>

<details>
<summary><b>7.</b> Beschreibe die klassische „tabletop"-Segmentierungspipeline.</summary>

(1) **RANSAC-Ebene** finden und entfernen (die dominante Ebene = Tisch/Boden). (2) Die verbleibenden Punkte per **Euclidean Clustering / DBSCAN** in einzelne **Objekt-Cluster** trennen (Punkte, deren Radius-Nachbarschaften sich berühren, gehören zusammen). Optional vorher **Voxel-Downsampling** und **Normalen**. Ergebnis: Boden abgetrennt, jedes Objekt ein eigenes Cluster.
</details>

<details>
<summary><b>8.</b> Warum ist Max-Pooling der Schlüssel zu PointNets Permutationsinvarianz? Skizziere die Architektur.</summary>

$f(\{\mathbf x_i\}) = \gamma(\mathrm{MAX}_i\,h(\mathbf x_i))$: Ein **geteiltes MLP** $h$ bildet jeden Punkt **einzeln** in einen Merkmalsraum ab; ein **elementweises Max** über *alle* Punkte aggregiert zu einem globalen Vektor; ein MLP $\gamma$ macht die Vorhersage. Das **Max ist symmetrisch** — sein Ergebnis hängt nicht von der Reihenfolge der Eingaben ab —, also ist die ganze Funktion **permutationsinvariant by construction**. (Summe/Mittel täten es prinzipiell auch; Max funktioniert empirisch am besten.)
</details>

<details>
<summary><b>9.</b> Was ist PointNets Hauptschwäche, und wie behebt PointNet++ sie?</summary>

PointNet aggregiert mit **einem** globalen Max über alle Punkte und erfasst deshalb **keine lokale Struktur** (feine geometrische Details, lokale Nachbarschaften). **PointNet++** behebt das **hierarchisch**: **Farthest Point Sampling** wählt Zentren, um jedes wird eine **lokale Nachbarschaft gruppiert** (Radius/kNN), auf der ein kleines PointNet ein lokales Merkmal berechnet; das wird über mehrere Ebenen gestapelt — analog zu den lokalen Faltungen eines CNN, nur auf Punkten.
</details>

<details>
<summary><b>10.</b> Warum braucht ICP eine gute Initialisierung, und woher kommt sie in der Praxis?</summary>

Weil ICP nur **lokal** konvergiert: Bei stark falscher Startpose sind die nächsten-Nachbar-Korrespondenzen falsch, und ICP rastet im falschen Minimum ein. Die grobe Init liefert die **globale Registrierung**: Voxel-Downsampling → **FPFH-Deskriptoren** → Feature-Matching → **RANSAC** über die Matches → grobe $(\mathbf R,\mathbf t)$. Diese grobe Pose (großer Einzugsbereich, ungenau) wird dann per **ICP verfeinert** (präzise, kleiner Einzugsbereich) — die Grob-zu-fein-Arbeitsteilung.
</details>

---

## Literatur & Quellen

**Lehrbücher / Übersichten**
- **Rusu & Cousins, „3D is here: Point Cloud Library (PCL)"**, *ICRA 2011*, sowie die **PCL-Tutorials** (pointclouds.org) — die praktische Referenz für Normalen, Segmentierung, Registrierung, FPFH. *Einsteigerfreundlich.*
- **Die Open3D-Dokumentation** (open3d.org) — exzellente, gut erklärte Tutorials zu ICP, globaler Registrierung, RANSAC-Segmentierung. Ideal, um die hier from-scratch gebauten Verfahren gegen eine ausgereifte Bibliothek zu spiegeln. *Einsteiger- bis fortgeschrittenenfreundlich, kostenlos.*

**Schlüssel-Papers (frei auffindbar)**
- **Besl & McKay, „A Method for Registration of 3-D Shapes" (ICP)**, *IEEE TPAMI 1992*. Der ICP-Ursprung. *Vertiefend.*
- **Arun, Huang & Blostein, „Least-Squares Fitting of Two 3-D Point Sets"**, *TPAMI 1987* — die SVD-Lösung (Kabsch/Umeyama-Familie). *Kompakt, die Mathematik aus Abschnitt 7.*
- **Chen & Medioni, „Object modelling by registration of multiple range images"**, 1992 — point-to-plane ICP. *Vertiefend.*
- **Fischler & Bolles, „Random Sample Consensus (RANSAC)"**, *CACM 1981*. Der RANSAC-Ursprung. *Einsteigerfreundlich, klassisch.*
- **Rusu, Blodow & Beetz, „Fast Point Feature Histograms (FPFH)"**, *ICRA 2009*. *Vertiefend.*
- **Qi et al., „PointNet: Deep Learning on Point Sets…"**, *CVPR 2017* und **„PointNet++"**, *NeurIPS 2017*. Die Deep-Learning-Grundlagen (Abschnitt 13). *Vertiefend, aber sehr gut geschrieben — der PointNet-Beweis der Universalität lohnt sich.*

**Frei verfügbare Kurse / Materialien**
- **Open3D „Getting Started"- und „Pipelines"-Tutorials** — Schritt für Schritt Registrierung/Segmentierung. *Kostenlos.*
- **Nicolai Nielsens / diverse YouTube-Serien zu Point Cloud Registration & ICP** und **Scratchapixel** (Geometrie). *Kostenlos.*
- Vorlesungen zu **3D Computer Vision / Photogrammetrie** (z. B. Cyrill Stachniss, Uni Bonn — frei auf YouTube, exzellent zu ICP/Features/SLAM). *Kostenlos, vertiefend.*

**Zum Ausprobieren**
- Die **drei Projekte** bauen Nachbarschaften+Normalen (basic), ICP mit Kabsch-SVD (medium) und eine RANSAC+Clustering-Segmentierungspipeline (final) — alles from scratch, die beste Vertiefung.

---

> **Nächstes Modul:** Modul 21 „Robotics 1" — Grundlagen der Robotik (Kinematik, Bewegungsplanung, Sensorik). Die 3D-Geometrie/Transformationen (Modul 19) und Punktwolken-Perzeption (dieses Modul) sind zentrale Bausteine der Roboter-Wahrnehmung.
