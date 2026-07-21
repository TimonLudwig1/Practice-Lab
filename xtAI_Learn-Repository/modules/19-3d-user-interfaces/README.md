# Modul 19 — 3D User Interfaces

> **Worum geht es?** Sobald Inhalte im **dreidimensionalen Raum** liegen — in VR/AR, in einem CAD-Programm, in einem Spiel —, reicht die aus dem 2D-Desktop gewohnte Interaktion (Maus, Fenster, Menü) nicht mehr. Ein **3D User Interface (3DUI)** ist eine Schnittstelle, über die der Mensch Objekte im 3D-Raum **auswählt, manipuliert, sich darin bewegt und orientiert**. Dieses Modul behandelt die *Prinzipien* und die *Mathematik* dieser Interaktion: Wie hängen die Koordinatensysteme zusammen? Wie trifft ein virtueller Zeigestrahl ein Objekt? Wie erreicht man mit einem kurzen Arm ein weit entferntes Objekt? Und warum ist Zeigen in 3D *fundamental schwerer* als in 2D?
>
> **Vorkenntnisse**: lineare Algebra (Matrizen, Vektoren, Skalarprodukt), etwas 3D-Geometrie. Aus diesem Repo bauen direkt auf: **Modul 17** (Core XR — Rotationsmathematik/Quaternionen, Fitts' Law, Tracking, Cybersickness) und **Modul 18** (Multimodal — Referenzauflösung, Zeige-Ambiguität, Nutzerstudien-Methodik). Modul 17 ist **Pflicht-Vormodul**; vieles hier ist dessen direkte Fortsetzung.

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–18 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, eng an der maßgeblichen Referenz (Bowman, LaViola, Kruijff, Poupyrev: *3D User Interfaces — Theory and Practice*) und konsistent mit dem XR-Block dieses Repos. **Wieder ohne konkrete Hardware** (kein VR-Headset, kein Tracked Controller): Der lehrbare, übertragbare Kern sind die **Transformationsmathematik, die Interaktionstechniken als Algorithmen, die Zeige-Präzisionsmodelle (Fitts in 3D) und die Evaluationsmethodik**. Ein Controller zu halten lernt man in Minuten; zu verstehen, *warum* Ray-Casting mit der Distanz zusammenbricht und *wie* Go-Go den Arm nichtlinear verlängert, ist die Master-Kompetenz. Die Projekte simulieren Zeigen und Auswahl realistisch mit reiner Geometrie/Statistik auf der CPU.

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

- die **vier universellen 3D-Interaktionsaufgaben** nach Bowman benennen und abgrenzen können: **Selektion, Manipulation, Travel (Fortbewegung), Wayfinding (Orientierung)** — plus System Control.
- **Koordinatensysteme und homogene Transformationen** vollständig beherrschen: die $4\times4$-Matrix, warum man Translation nur homogen als Matrix schreiben kann, die **Transformationskette** Objekt→Welt→Kamera→Bild, und wie man sie invertiert.
- die **Ray-Objekt-Schnitt-Mathematik** herleiten und implementieren können: Ray-Kugel, Ray-AABB (Slab-Methode), Ray-Dreieck (Möller-Trumbore).
- die wichtigsten **Selektions- und Manipulationstechniken** als Algorithmen verstehen: Ray-Casting, Cone/Bubble, Virtual Hand, **Go-Go** (die nichtlineare Armverlängerung — mit Formel), HOMER, sowie **isomorphe vs. „magische" (non-isomorphic)** Techniken und **DOF-Separation**.
- erklären können, warum **Zeigen in 3D schwerer** ist als in 2D (**angulares Fitts' Law**, Heisenberg-Effekt, fehlende physische Constraints, Tiefenwahrnehmung) und wie Control-Display-Gain hilft.
- die **Travel-Metaphern** (Walking, Steering, Target-based/Teleport, Manipulation-based) und ihre Kopplung an **Cybersickness** (Rückverweis Modul 17) einordnen.
- ein 3D-Selektionsexperiment **sauber gestalten und auswerten** (ISO-9241-9-Throughput, Fitts in 3D, Docking, within-subject-Statistik wie in Modul 17).

---

## Grundlagen (Basics)

### 1. Warum ist 3D-Interaktion schwer?

Der Desktop hat den Menschen 40 Jahre lang mit **2D, WIMP** (Windows, Icons, Menus, Pointer) und harten physischen Constraints (die Maus liegt auf dem Tisch) verwöhnt. Im 3D-Raum fällt das alles weg. Vier fundamentale Schwierigkeiten:

1. **Sechs Freiheitsgrade (6 DoF).** Ein Objekt im Raum hat drei Positions- und drei Orientierungsfreiheitsgrade. Ein Menü hat zwei. Der Mensch muss sechs gleichzeitig kontrollieren — mehr, als die Wahrnehmung sauber trennen kann.
2. **Keine physischen Constraints / keine passive Haptik.** Auf dem Tisch stoppt die Maus an der Tischkante; in der Luft gibt es keinen Anschlag, keine Auflage, kein Feedback beim „Berühren" eines virtuellen Objekts. Die Hand ermüdet („gorilla arm").
3. **Präzision.** Ohne Auflage zittert die Hand (Tremor); beim Klicken verrutscht der Zeiger (der **Heisenberg-Effekt** der 3D-Interaktion, Abschnitt 13).
4. **Tiefenwahrnehmung.** Wie weit ist das Objekt? Menschen **unterschätzen Distanzen in VR** systematisch (oft 70–80 % der echten Distanz), was Greifen und Bewegen verzerrt.

Deshalb braucht 3D-Interaktion *eigene* Techniken — man kann die Maus nicht einfach „in die Luft heben".

### 2. Die vier universellen Interaktionsaufgaben (Bowman)

Fast jede 3D-Interaktion lässt sich in vier **kanonische Aufgaben** zerlegen. Diese Taxonomie strukturiert das ganze Modul:

| Aufgabe | Frage | Beispieltechniken |
|---|---|---|
| **Selektion** | *Welches* Objekt meine ich? | Ray-Casting, Virtual Hand, Go-Go, Bubble/Cone |
| **Manipulation** | Objekt *bewegen/drehen/skalieren* | Virtual Hand, HOMER, Scaled-World Grab, Widgets |
| **Travel** (motorische Fortbewegung) | Wie *bewege* ich meinen Blickpunkt? | Walking, Steering, Teleport, Grab-the-world |
| **Wayfinding** (kognitive Orientierung) | *Wo* bin ich, wie komme ich *hin*? | Karten, Landmarken, Kompass, Wegmarkierungen |

Dazu **System Control** (Befehle/Modi wählen — 3D-Menüs, Gesten, Sprache; hier greift Modul 18) und **symbolische Eingabe** (Text in 3D — notorisch schwer).

> **Merke die Travel/Wayfinding-Unterscheidung:** *Travel* ist die **motorische** Komponente (das Bewegen selbst), *Wayfinding* die **kognitive** (das Wissen, wohin). Beide zusammen = *Navigation*. Ein gutes 3DUI unterstützt beide getrennt.

### 3. Koordinatensysteme und homogene Transformationen

Das mathematische Fundament. Ein 3D-Objekt existiert in mehreren **Bezugssystemen (frames)** gleichzeitig, und Interaktion heißt ständig, zwischen ihnen umzurechnen:

```
 Objekt-       Welt-         Kamera-/Eye-    Bild-/Screen-
 koordinaten → koordinaten → koordinaten  → koordinaten
   (model)      (world)        (view)         (projection)
      M_model       M_view         M_proj
```

Ein Punkt wird durch die Kette transformiert: $\mathbf{p}_{\text{screen}} = M_{\text{proj}}\,M_{\text{view}}\,M_{\text{model}}\,\mathbf{p}_{\text{object}}$.

**Homogene Koordinaten.** Der Trick, der alles zusammenhält: Man erweitert einen 3D-Punkt $\mathbf{p}=(x,y,z)$ um eine vierte Koordinate $w=1$: $\tilde{\mathbf{p}} = (x,y,z,1)^\top$. Der Grund ist rein pragmatisch, aber tiefgreifend: **Translation ist keine lineare Abbildung** (sie fixiert den Ursprung nicht) und lässt sich *nicht* als $3\times3$-Matrix schreiben. In homogenen Koordinaten geht es doch — als $4\times4$-Matrix:

$$T(\mathbf{t}) = \begin{pmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad
R = \begin{pmatrix} & & & 0 \\ & \mathbf{R}_{3\times3} & & 0 \\ & & & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad
S(\mathbf{s}) = \begin{pmatrix} s_x & & & 0 \\ & s_y & & 0 \\ & & s_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Jetzt sind **Rotation, Translation und Skalierung alle $4\times4$-Matrizen** und lassen sich durch **Matrixmultiplikation verketten** — das ist der ganze Grund für homogene Koordinaten. Eine starre Transformation (rigid body, Rotation + Translation) ist

$$M = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}, \qquad
  M^{-1} = \begin{pmatrix} \mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}.$$

Die Inverse nutzt $\mathbf{R}^{-1}=\mathbf{R}^\top$ (Rotationsmatrizen sind orthogonal, Rückverweis Modul 17) — man muss nie numerisch invertieren. **Reihenfolge ist entscheidend** (Matrixmultiplikation ist nicht kommutativ, wie schon bei den Rotationen in Modul 17): $T\,R$ (erst rotieren, dann verschieben) $\neq R\,T$.

> **Achtung, Konvention:** Angewandt wird von rechts nach links auf den Spaltenvektor: $M\mathbf{p} = T(R(S\mathbf{p}))$ heißt „erst skalieren, dann rotieren, dann translatieren". Die $\mathbf{R}$ hier kann direkt aus einem Quaternion (Modul 17) gebaut werden — 3DUIs speichern Orientierungen fast immer als Quaternion und konvertieren nur zum Rendern in die Matrix.

### 4. Tiefenwahrnehmung (depth cues)

Damit der Mensch im 3D-Raum zeigen und greifen kann, muss er **Distanz schätzen**. Das Gehirn nutzt dafür viele **Tiefenhinweise (depth cues)**, die man kennen muss, weil VR-Systeme sie unterschiedlich gut liefern:

- **Okklusion** (Verdeckung): der stärkste Hinweis — was verdeckt, ist näher. Immer korrekt in VR.
- **Perspektive / relative Größe**: Fernes erscheint kleiner; parallele Linien konvergieren.
- **Stereopsis** (binokulare Disparität): die zwei Augenbilder differieren — Hauptvorteil eines VR-Headsets, wirkt aber nur im Nahbereich (< ~10 m).
- **Bewegungsparallaxe**: bei Kopfbewegung wandert Nahes schneller als Fernes — braucht gutes Head-Tracking (Modul 17).
- **Akkommodation/Konvergenz**: Fokus- und Augenstellung — Quelle des **Vergenz-Akkommodations-Konflikts** (VAC) aus Modul 17.

**Konsequenz:** VR liefert Stereopsis und Parallaxe gut, aber der VAC und fehlende weitere Cues führen zur systematischen **Distanzunterschätzung**. Für ein 3DUI heißt das: Objekte wirken näher, als sie sind → Greiftechniken müssen das kompensieren (siehe Go-Go, Abschnitt 6).

---

## Aufbau (Intermediate)

### 5. Selektion I: Ray-Casting und die Ray-Objekt-Schnitt-Mathematik

Die mit Abstand häufigste Selektionstechnik: Vom Controller (oder der Hand) geht ein **virtueller Strahl (ray)** aus; das erste getroffene Objekt wird selektiert. Ein Strahl ist parametrisch

$$\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}, \quad t \ge 0,$$

mit Ursprung $\mathbf{o}$ und **normierter** Richtung $\mathbf{d}$ ($\|\mathbf{d}\|=1$, dann ist $t$ die euklidische Distanz). Selektion = **nächster Schnittpunkt** über alle Objekte. Die Mathematik pro Objekttyp:

**Ray-Kugel** (Objekt als Bounding Sphere, Zentrum $\mathbf{c}$, Radius $R$). Setze $\mathbf{r}(t)$ in $\|\mathbf{p}-\mathbf{c}\|^2 = R^2$ ein. Mit $\mathbf{m}=\mathbf{o}-\mathbf{c}$:

$$\|\mathbf{m}+t\mathbf{d}\|^2 = R^2 \;\Longrightarrow\; t^2\underbrace{(\mathbf{d}\cdot\mathbf{d})}_{=1} + 2t(\mathbf{m}\cdot\mathbf{d}) + (\mathbf{m}\cdot\mathbf{m}-R^2)=0.$$

Eine quadratische Gleichung $t^2 + 2bt + c = 0$ mit $b=\mathbf{m}\cdot\mathbf{d}$, $c=\mathbf{m}\cdot\mathbf{m}-R^2$. Diskriminante $\Delta = b^2 - c$. Ist $\Delta<0$: kein Treffer. Sonst $t = -b - \sqrt{\Delta}$ (der nähere, vordere Schnittpunkt); ist dieser $<0$ und der andere $>0$, ist der Ursprung *innerhalb* der Kugel.

**Ray-AABB** (achsenparallele Bounding Box, Slab-Methode). Für jede Achse $i\in\{x,y,z\}$ ist die Box ein „Slab" $[\min_i, \max_i]$. Der Strahl tritt bei $t_{i,1}=(\min_i - o_i)/d_i$ ein und bei $t_{i,2}=(\max_i-o_i)/d_i$ aus (Vorzeichen von $d_i$ beachten → tauschen). Der Strahl trifft die Box gdw. das **Intervall-Maximum der Eintritte $\le$ das Intervall-Minimum der Austritte** ist:

$$t_{\text{enter}} = \max_i \min(t_{i,1}, t_{i,2}), \quad t_{\text{exit}} = \min_i \max(t_{i,1}, t_{i,2}), \quad \text{Treffer} \iff t_{\text{enter}} \le t_{\text{exit}} \wedge t_{\text{exit}}\ge 0.$$

**Ray-Dreieck (Möller-Trumbore)** — für echte Mesh-Geometrie. Ein Dreieck mit Ecken $\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2$; ein Punkt darin ist $\mathbf{v}_0 + u\,\mathbf{e}_1 + v\,\mathbf{e}_2$ mit $\mathbf{e}_1=\mathbf{v}_1-\mathbf{v}_0$, $\mathbf{e}_2=\mathbf{v}_2-\mathbf{v}_0$ und **baryzentrischen** Koordinaten $u,v\ge0$, $u+v\le1$. Gleichsetzen mit $\mathbf{r}(t)$ ergibt ein $3\times3$-System, das man ohne Matrixinversion über Spatprodukte löst:

$$\mathbf{p} = \mathbf{d}\times\mathbf{e}_2,\quad \det = \mathbf{e}_1\cdot\mathbf{p}, \quad \mathbf{s}=\mathbf{o}-\mathbf{v}_0,\quad
u = \frac{\mathbf{s}\cdot\mathbf{p}}{\det},\quad \mathbf{q}=\mathbf{s}\times\mathbf{e}_1,\quad v=\frac{\mathbf{d}\cdot\mathbf{q}}{\det},\quad t=\frac{\mathbf{e}_2\cdot\mathbf{q}}{\det}.$$

Treffer gdw. $u\ge0,\ v\ge0,\ u+v\le1,\ t>0$ (und $|\det|$ nicht ~0, sonst Strahl parallel zum Dreieck). Das ist der Standard-Algorithmus jedes Raytracers und Selektions-Systems.

**Warum Ray-Casting so beliebt ist** — und wo es scheitert: Es erlaubt, **weit entfernte** Objekte mühelos zu selektieren (der Strahl ist beliebig lang). Aber: Bei **Distanz** subtendiert ein Objekt einen winzigen Winkel, und schon minimales Handzittern verfehlt es (Abschnitt 12, angulares Fitts). Bei **Verdeckung/Dichte** ist unklar, welches von mehreren hintereinander/nah beieinanderliegenden Objekten gemeint ist — das ist ein **Disambiguierungsproblem** wie in Modul 18. Genau hier setzen Cone/Bubble-Techniken an.

### 6. Selektion II: Virtual Hand und Go-Go (nichtlineare Reichweitenverlängerung)

**Virtual Hand.** Die direkteste Technik: Eine virtuelle Hand folgt **isomorph** (1:1) der echten Hand; man selektiert durch **Berühren** (Kollision Hand ↔ Objekt). Intuitiv und präzise — aber die Reichweite ist **auf die Armlänge begrenzt**. Man kann nur greifen, was in Armnähe ist.

**Go-Go** (Poupyrev et al. 1996) — die eleganteste Lösung des Reichweitenproblems. Idee: Die virtuelle Hand folgt der echten **nichtlinear**. Innerhalb einer Schwelle $D$ (typisch ~2/3 der Armlänge) ist die Abbildung 1:1 (präzise Nahinteraktion); *jenseits* von $D$ wächst die virtuelle Reichweite **quadratisch**, sodass kleine reale Armstreckungen große virtuelle Reichweiten erzeugen. Sei $r_r$ die reale Handdistanz vom Körper und $r_v$ die virtuelle:

$$r_v = \begin{cases} r_r & r_r < D \\[4pt] r_r + k\,(r_r - D)^2 & r_r \ge D \end{cases}$$

mit einem Verstärkungskoeffizienten $k$ (steuert, wie schnell die Reichweite wächst). Die Funktion ist **stetig und stetig differenzierbar** an $r_r=D$ (Wert $D$, Ableitung $1$) — kein Sprung, kein Knick in der Geschwindigkeit, was sie „natürlich" anfühlen lässt. Go-Go behält so die **Präzision** der Virtual Hand im Nahbereich und gewinnt die **Reichweite** des Ray-Castings in der Ferne. Das **Basic-** und **Medium-Projekt** implementieren diese Funktion.

> Go-Go ist der Prototyp einer **„magischen" (non-isomorphic)** Technik: Sie bricht bewusst mit der 1:1-Realität, um eine bessere Usability zu erzielen. Das ist ein Kernprinzip der 3D-Interaktion — der *Isomorphismus* (Realtreue) ist *nicht* immer optimal.

### 7. Selektion III: Volumen-Techniken (Cone/Bubble) gegen Ambiguität

Wenn Ray-Casting an kleinen/dichten/fernen Zielen scheitert, hilft ein **Selektionsvolumen** statt einer Linie:

- **Cone/Flashlight**: Statt eines Strahls ein **Kegel**; alle Objekte im Kegel sind Kandidaten. Ein Objekt wird gewählt, das den **kleinsten Winkelabstand zur Kegelachse** hat. Erleichtert das Treffen kleiner/ferner Ziele — aber bei **dichten** Szenen sind viele Objekte im Kegel → neue Ambiguität.
- **Bubble Cursor** (Grossman & Balakrishnan, ursprünglich 2D, 3D-Varianten existieren): Ein Cursor mit **dynamischem Radius**, der sich so anpasst, dass **genau ein** Ziel umschlossen wird — er umfasst immer das *nächstgelegene* Objekt vollständig. Macht effektiv jedes Ziel „so groß wie sein Voronoi-Gebiet" → drastisch bessere Fitts-Performance in dünn besetzten Szenen.

Die Wahl zwischen Punkt- und Volumentechnik ist ein **Präzisions-vs-Ambiguitäts-Trade-off**, den das **Final-Projekt** empirisch vermisst: Ray-Casting ist präzise, aber unbrauchbar bei kleinen fernen Zielen; Cone/Bubble greifen die leicht, über-selektieren aber im Gedränge. Die Disambiguierung „welches der Kandidaten-Objekte?" ist konzeptuell dasselbe Problem wie die Referenzauflösung in Modul 18 (dort mit Zeit + Semantik, hier mit Winkel + Distanz).

### 8. Manipulation: DOF, isomorph vs. magisch, HOMER, DOF-Separation

Nach der Selektion folgt oft die **Manipulation** (bewegen, drehen, skalieren). Kernkonzepte:

- **6-DoF-Manipulation isomorph**: Die virtuelle Hand überträgt Translation + Rotation 1:1 aufs Objekt. Natürlich, aber wieder reichweiten- und präzisionsbegrenzt.
- **HOMER** (*Hand-centered Object Manipulation Extending Ray-casting*): selektiert per Ray-Casting (große Reichweite), dann **springt die virtuelle Hand zum Objekt** und manipuliert es hand-zentriert. Kombiniert Ray-Casting-Reichweite mit Virtual-Hand-Manipulation.
- **Scaled-World Grab**: skaliert die Welt beim Greifen so, dass das ferne Objekt in Reichweite kommt — mathematisch elegant, kann aber desorientieren.
- **DOF-Separation**: Oft will man *nicht* alle 6 DoF gleichzeitig kontrollieren (z. B. ein Bild nur *entlang der Wand* verschieben). Techniken **beschränken (constrain)** DoF, **rasten (snap)** an Gitter/Kanten oder trennen Translation von Rotation über Widgets/Handles. Das kompensiert die menschliche Schwäche, viele DoF simultan präzise zu führen.

> **Perzeptuelle Struktur (Jacob et al.):** Freiheitsgrade sollten so gruppiert werden, wie der Mensch sie *wahrnimmt*. Position (x,y,z) ist **integral** (man bewegt die Hand als Ganzes), Position-vs-Farbe wäre **separabel**. Eine Manipulationstechnik, die integrale DoF trennt oder separable koppelt, fühlt sich falsch an. Design-Regel: **DOF-Struktur der Aufgabe = DOF-Struktur der Technik.**

### 9. Travel & Wayfinding

**Travel-Metaphern** (die motorische Fortbewegung):

- **Physical Walking / Real Walking**: am immersivsten, am wenigsten sickness-anfällig — aber durch den realen Raum begrenzt (Redirected Walking als Trick).
- **Steering**: kontinuierliche Richtungsangabe — *gaze-directed* (Blickrichtung) oder *pointing-directed* (Handrichtung). Einfach, aber kontinuierliche visuelle Bewegung ohne körperliche → **Vection → Cybersickness** (Rückverweis Modul 17, Sensory-Conflict-Theorie).
- **Target-based / Teleport**: man wählt ein Ziel, wird **sofort dorthin versetzt**. Der De-facto-Standard in VR, **weil er Cybersickness minimiert** (keine kontinuierliche Vection) — auf Kosten des räumlichen Verständnisses.
- **Manipulation-based** („grab the world"): man greift die Welt und zieht sich hindurch.

**Wayfinding-Unterstützung** (die kognitive Orientierung): **Landmarken**, **Karten** (You-are-here, mit der heiklen Frage der Ausrichtung — track-up vs. north-up), **Kompass/Trails/Breadcrumbs**, gute **Sichtlinien**. Ziel ist der Aufbau einer mentalen **cognitive map**.

---

## Advanced-Themen

### 10. Fitts' Law in 3D: warum Zeigen mit der Distanz zusammenbricht

In Modul 17 haben wir Fitts' Law für 2D-Zeigen eingeführt: die Bewegungszeit

$$MT = a + b\,\underbrace{\log_2\!\Big(\tfrac{D}{W}+1\Big)}_{\text{Index of Difficulty }ID}$$

mit Zieldistanz $D$ und Zielbreite $W$. In 3D — speziell beim **Ray-Casting** — ist die relevante Größe **nicht die lineare, sondern die *anguläre* Ausdehnung**. Ein Ziel der (Quer-)Breite $W$ in Distanz $L$ vom Auge/Controller subtendiert den Winkel

$$\theta_W \approx 2\arctan\!\Big(\frac{W}{2L}\Big) \approx \frac{W}{L} \quad (\text{für } W \ll L),$$

und der Zeiger muss über einen **Winkel** $\theta_D$ schwenken. Das **angulare Fitts' Law** ersetzt Distanz/Breite durch Winkel:

$$MT = a + b\,\log_2\!\Big(\frac{\theta_D}{\theta_W}+1\Big).$$

> **Die entscheidende Konsequenz:** $\theta_W \approx W/L$ **schrumpft mit der Distanz $L$**. Ein Objekt doppelt so weit weg ist *anguär* halb so groß → $ID$ steigt → Selektion dauert länger und wird fehleranfälliger. **Ray-Casting-Präzision verschlechtert sich linear mit der Zieldistanz** — das ist der harte, quantitative Grund, warum ferne kleine Ziele in VR so schwer zu treffen sind, und warum Go-Go (bringt das Ziel „näher") oder Bubble (vergrößert das effektive $\theta_W$) helfen. Das **Medium-Projekt** misst genau diese angulare Fitts-Beziehung.

### 11. Control-Display-Gain und adaptive Verstärkung (PRISM)

**Control-Display (C/D) Gain** ist das Verhältnis von Anzeige-Bewegung zu Kontroll-Bewegung. Gain $>1$ (kleine Handbewegung → große Zeigerbewegung) gibt **Reichweite/Geschwindigkeit**, kostet **Präzision**; Gain $<1$ umgekehrt. Ein fester Gain ist ein Kompromiss. **Adaptive Verfahren** wie **PRISM** (Frees et al.) senken den Gain bei *langsamer* Handbewegung (präzises Zielen) und heben ihn bei schneller (großräumiges Bewegen) — sie nutzen die Handgeschwindigkeit als Absichtsindikator. Das ist die kontinuierliche Verallgemeinerung des Go-Go-Gedankens.

### 12. Der Heisenberg-Effekt der 3D-Interaktion

Ein subtiler, praxisrelevanter Effekt: Beim **Betätigen des Auswahl-Buttons** (Trigger drücken) verwackelt die Hand — der Zeiger springt im Moment des Klicks weg vom Ziel. Je kleiner/ferner das Ziel (kleines $\theta_W$), desto fataler. Gegenmaßnahmen: den Zeiger-Zustand **kurz vor** dem Klick einfrieren, Klick-Ereignisse zeitlich filtern, oder Bestätigung über eine **andere Modalität** (Sprache statt Trigger — wieder Modul 18). Das ist das 3D-Analogon zum Wackeln beim Maus-Klick, nur ohne die stabilisierende Tischauflage viel stärker.

### 13. Ray-Casting-Präzision als stochastisches Modell

Für die Evaluation modelliert man die Zeigepräzision als **angulares Rauschen**: Die reale Zeigerichtung streut gaußförmig um die intendierte, mit Standardabweichung $\sigma_\theta$ (Hand-Tremor + Tracking-Rauschen + Heisenberg). Ein Ziel mit angularem Radius $\theta_W/2$ in Distanz $L$ wird getroffen, wenn die Winkelabweichung kleiner ist:

$$P(\text{Treffer}) = P\!\big(|\epsilon_\theta| < \tfrac{\theta_W}{2}\big) = P\!\big(|\epsilon_\theta| < \tfrac{W}{2L}\big), \quad \epsilon_\theta \sim \mathcal{N}(0,\sigma_\theta^2).$$

Das koppelt direkt an das inverse-Varianz-Denken aus Modul 18: Präzision ist $1/\sigma_\theta^2$, und alles, was $\sigma_\theta$ senkt (Auflage, Stabilisierung, C/D-Gain, Prediction wie in Modul 17), erhöht die Trefferwahrscheinlichkeit. Die **Projekte** bauen genau dieses Modell.

### 14. Evaluation: ISO 9241-9 und Throughput in 3D

Der Standard zur Bewertung von Zeigegeräten/-techniken ist **ISO 9241-9**, dessen Kern der **Throughput** (Durchsatz, in bits/s) ist:

$$TP = \frac{ID_e}{MT}, \qquad ID_e = \log_2\!\Big(\frac{D_e}{W_e}+1\Big),$$

mit **effektiver** Breite $W_e = 4.133\,\sigma_x$ (aus der Streuung der tatsächlichen Klickpositionen — die „effektive" Breite korrigiert für die vom Nutzer real genutzte Genauigkeit, sodass ~96 % der Klicks im Ziel liegen) und effektiver Distanz $D_e$. Throughput fasst **Geschwindigkeit und Genauigkeit in einer Zahl** zusammen und macht Techniken vergleichbar. Typische Aufgaben: **reciprocal tapping** (zwischen zwei Zielen hin und her), **Docking** (ein Objekt in Zielpose bringen — testet Manipulation inkl. Rotation). Die statistische Auswertung (within-subject, Counterbalancing, Effektstärken, Wilcoxon/ANOVA) folgt exakt der Methodik aus **Modul 17** — das **Final-Projekt** wendet sie auf einen Selektionstechnik-Vergleich an.

---

## Zusammenfassung / Cheat-Sheet

**Die vier Aufgaben (Bowman)**: Selektion · Manipulation · Travel (motorisch) · Wayfinding (kognitiv) · [+ System Control].

**Homogene Transformationen**
- Punkt homogen: $\tilde{\mathbf p}=(x,y,z,1)$. Grund: Translation wird als $4\times4$-Matrix schreibbar → alles verkettbar.
- Kette: $\mathbf p_{\text{screen}} = M_{\text{proj}} M_{\text{view}} M_{\text{model}}\,\mathbf p_{\text{obj}}$. Reihenfolge zählt (nicht kommutativ).
- Rigid-Body-Inverse: $M^{-1}=\begin{psmallmatrix}\mathbf R^\top & -\mathbf R^\top\mathbf t\\ 0 & 1\end{psmallmatrix}$ (nutzt $\mathbf R^{-1}=\mathbf R^\top$).

**Ray $\mathbf r(t)=\mathbf o+t\mathbf d$ ($\|\mathbf d\|=1$), Schnitt**
| Objekt | Kern |
|---|---|
| Kugel | $t^2+2bt+c=0$, $b=\mathbf m\cdot\mathbf d$, $c=\|\mathbf m\|^2-R^2$, $\mathbf m=\mathbf o-\mathbf c$; $t=-b-\sqrt{b^2-c}$ |
| AABB | Slabs: Treffer $\iff \max_i\min(t_{i1},t_{i2}) \le \min_i\max(t_{i1},t_{i2})$ |
| Dreieck | Möller-Trumbore: $u,v$ baryzentrisch $\ge0$, $u+v\le1$, $t>0$ |

**Selektionstechniken**: Ray-Casting (Reichweite, aber angular-präzisionsschwach) · Virtual Hand (präzise, kurze Reichweite) · **Go-Go** $r_v=r_r+k(r_r-D)^2$ für $r_r\ge D$ (beides) · Cone/Bubble (gegen kleine/dünne Ziele, aber Ambiguität im Gedränge).

**Fitts in 3D**: angular, $ID=\log_2(\theta_D/\theta_W+1)$ mit $\theta_W\approx W/L$ → **Präzision fällt mit Distanz $L$**.

**Throughput (ISO 9241-9)**: $TP=ID_e/MT$, $W_e=4.133\,\sigma_x$ (effektive Breite aus Klickstreuung).

**Travel & Sickness**: Teleport minimiert Vection/Sickness (Modul 17); Steering ist einfach aber sickness-anfällig; Real Walking am besten, aber raumbegrenzt.

**Design-Regeln**: „magische" (non-isomorphic) Technik schlägt oft die realtreue; DOF-Struktur der Technik = DOF-Struktur der Aufgabe; Heisenberg-Effekt beim Klick behandeln.

---

## Selbsttest

<details>
<summary><b>1.</b> Warum braucht man homogene ($4\times4$) Koordinaten — reichen nicht $3\times3$-Rotationsmatrizen?</summary>

Weil **Translation keine lineare Abbildung** ist (sie bildet den Ursprung nicht auf sich ab) und sich daher *nicht* als $3\times3$-Matrix schreiben lässt. In homogenen Koordinaten (vierte Komponente $w=1$) wird Translation zur $4\times4$-Matrix mit dem Verschiebungsvektor in der letzten Spalte. Damit sind **Rotation, Translation und Skalierung alle Matrizen** und lassen sich durch Multiplikation zu **einer** Transformationskette verketten — das ist der ganze Zweck.
</details>

<details>
<summary><b>2.</b> Nenne die vier universellen 3D-Interaktionsaufgaben und den Unterschied zwischen Travel und Wayfinding.</summary>

**Selektion, Manipulation, Travel, Wayfinding** (+ System Control). **Travel** ist die *motorische* Fortbewegung (das Bewegen des Blickpunkts selbst), **Wayfinding** die *kognitive* Orientierung (wissen, wo man ist und wie man zum Ziel kommt). Zusammen bilden sie *Navigation*.
</details>

<details>
<summary><b>3.</b> Leite die Ray-Kugel-Schnittgleichung her. Wann gibt es keinen Treffer?</summary>

Strahl $\mathbf r(t)=\mathbf o+t\mathbf d$ ($\|\mathbf d\|=1$) in die Kugelgleichung $\|\mathbf p-\mathbf c\|^2=R^2$ einsetzen. Mit $\mathbf m=\mathbf o-\mathbf c$: $\|\mathbf m+t\mathbf d\|^2=R^2 \Rightarrow t^2 + 2t(\mathbf m\cdot\mathbf d) + (\|\mathbf m\|^2-R^2)=0$. Das ist $t^2+2bt+c=0$ mit $b=\mathbf m\cdot\mathbf d$, $c=\|\mathbf m\|^2-R^2$. Diskriminante $\Delta=b^2-c$. **Kein Treffer, wenn $\Delta<0$** (Strahl verfehlt die Kugel). Sonst ist der nähere Schnittpunkt $t=-b-\sqrt\Delta$.
</details>

<details>
<summary><b>4.</b> Schreibe die Go-Go-Funktion auf und erkläre, welches Problem sie löst und warum ihre Form (quadratisch, mit Schwelle) sinnvoll ist.</summary>

$$r_v = \begin{cases} r_r & r_r<D\\ r_r + k(r_r-D)^2 & r_r\ge D\end{cases}$$

Sie löst das **Reichweitenproblem** der Virtual Hand: Im Nahbereich ($r_r<D$) bleibt die Abbildung **1:1** → volle Präzision; jenseits der Schwelle $D$ wächst die virtuelle Reichweite **quadratisch** → kleine reale Armstreckungen erreichen ferne Objekte. Die quadratische Form ist an $r_r=D$ **stetig und mit Ableitung 1 differenzierbar** (kein Sprung, kein Geschwindigkeitsknick) → fühlt sich natürlich an. Go-Go vereint so Nahpräzision und Fernreichweite und ist ein Beispiel für eine „magische" (non-isomorphic) Technik.
</details>

<details>
<summary><b>5.</b> Warum wird Ray-Casting mit zunehmender Zieldistanz schlechter? Nutze das angulare Fitts' Law.</summary>

Weil die für das Zeigen relevante Größe der **anguläre** Radius ist: Ein Ziel der Breite $W$ in Distanz $L$ subtendiert nur $\theta_W\approx W/L$. Das **schrumpft mit $L$** — doppelte Distanz = halbe anguläre Größe. Im angularen Fitts' Law $ID=\log_2(\theta_D/\theta_W+1)$ steigt damit der Schwierigkeitsindex, die Bewegungszeit wächst und (bei festem Winkelrauschen $\sigma_\theta$) die Trefferwahrscheinlichkeit $P(|\epsilon_\theta|<\theta_W/2)$ sinkt. Fazit: Ray-Casting-Präzision fällt mit der Distanz.
</details>

<details>
<summary><b>6.</b> Was ist der Heisenberg-Effekt der 3D-Interaktion, und wie begegnet man ihm?</summary>

Beim **Drücken des Auswahl-Buttons** verwackelt die Hand, sodass der Zeiger im Klick-Moment vom Ziel wegspringt — besonders schlimm bei kleinen/fernen (angular winzigen) Zielen. Gegenmittel: den Zeiger-Zustand **kurz vor dem Klick einfrieren**, Klicks zeitlich filtern, oder die Bestätigung in eine **andere Modalität** verlagern (z. B. Sprachkommando statt Trigger — Modul 18).
</details>

<details>
<summary><b>7.</b> Warum ist Teleport-Travel in VR so verbreitet, und was ist der Preis?</summary>

Weil Teleport (target-based) **keine kontinuierliche visuelle Eigenbewegung** erzeugt → keine **Vection** → minimale **Cybersickness** (Sensory-Conflict-Theorie, Modul 17). Der Preis ist **schlechteres räumliches Verständnis / Wayfinding**: Durch das „Springen" baut der Nutzer eine weniger zusammenhängende cognitive map auf und kann die Distanz/Richtung des Wegs schlechter einschätzen.
</details>

<details>
<summary><b>8.</b> Was ist die „effektive Breite" $W_e$ im ISO-9241-9-Throughput, und warum benutzt man sie statt der Zielbreite $W$?</summary>

$W_e = 4.133\,\sigma_x$ wird aus der **Streuung der tatsächlichen Klickpositionen** berechnet. Sie erfasst die **real vom Nutzer genutzte Genauigkeit**: Zielt jemand schlampig (breite Streuung), steigt $W_e$; zielt er präziser als nötig, sinkt sie. Der Faktor 4.133 normiert so, dass ~96 % der Klicks ins effektive Ziel fallen. Dadurch wird der **Speed-Accuracy-Tradeoff herausgerechnet** — Throughput $TP=ID_e/MT$ bewertet Techniken fair, egal ob ein Nutzer eher schnell-ungenau oder langsam-genau agiert.
</details>

<details>
<summary><b>9.</b> Wann ist ein Bubble Cursor / eine Cone-Technik dem Ray-Casting überlegen, wann nicht?</summary>

**Überlegen** bei **kleinen, fernen oder isolierten** Zielen: Der Bubble Cursor vergrößert das effektive Ziel auf sein Voronoi-Gebiet, die Cone-Technik senkt die nötige Winkelpräzision → beide erleichtern das Treffen dramatisch (bessere Fitts-Performance). **Nicht überlegen** in **dichten** Szenen: Dann liegen viele Objekte im Kegel/Radius, und es entsteht ein **Disambiguierungsproblem** — welches Objekt ist gemeint? Ray-Casting ist dort präziser. Es ist ein Präzisions-vs-Ambiguitäts-Trade-off.
</details>

<details>
<summary><b>10.</b> Was besagt das Prinzip „DOF-Struktur der Technik = DOF-Struktur der Aufgabe"?</summary>

Freiheitsgrade sollten in der Technik **so gruppiert** werden, wie der Mensch sie in der Aufgabe **wahrnimmt und kontrolliert** (integral vs. separabel, nach Jacob et al.). Position (x,y,z) ist *integral* — man bewegt die Hand als Ganzes; solche DoF zu trennen (z. B. jede Achse einzeln stellen zu müssen) fühlt sich falsch an. Umgekehrt sollte man *separable* Aufgabenanteile nicht künstlich koppeln. Praktisch heißt das: Constraints/Widgets/Snapping so wählen, dass sie der natürlichen DoF-Wahrnehmung der Aufgabe entsprechen.
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **LaViola, Kruijff, McMahan, Bowman & Poupyrev, *3D User Interfaces: Theory and Practice* (2. Aufl., 2017).** *Das* Standardwerk — Taxonomie der vier Aufgaben, alle Techniken, Evaluation. Pflicht. *Einsteiger- bis fortgeschrittenenfreundlich.*
- **Foley/van Dam o. ä. Computergrafik-Lehrbuch** für homogene Transformationen und Ray-Objekt-Schnitt; alternativ **Marschner & Shirley, *Fundamentals of Computer Graphics*** (Kap. Transformationen, Ray Tracing). *Einsteigerfreundlich.*
- **Ericson, *Real-Time Collision Detection*** — die maßgebliche Referenz für Ray-Kugel/AABB/Dreieck-Tests. *Vertiefend.*

**Schlüssel-Papers (frei auffindbar)**
- **Poupyrev, Billinghurst, Weghorst & Ichikawa, „The Go-Go Interaction Technique"**, *UIST 1996*. Die Reichweiten-Verlängerung. *Kurz, lesenswert.*
- **Bowman & Hodges, „An Evaluation of Techniques for Grabbing and Manipulating Remote Objects" (HOMER)**, *I3D 1997*. *Vertiefend.*
- **Grossman & Balakrishnan, „The Bubble Cursor"**, *CHI 2005*. Dynamischer Cursor-Radius, Fitts-optimal. *Einsteigerfreundlich.*
- **Möller & Trumbore, „Fast, Minimum Storage Ray-Triangle Intersection"**, *Journal of Graphics Tools 1997*. Der Standard-Algorithmus. *Vertiefend, aber kompakt.*
- **Frees, Kessler & Kay, „PRISM Interaction for Enhancing Control in Immersive Virtual Environments"**, *ACM TOCHI 2007*. Adaptiver C/D-Gain. *Vertiefend.*
- **MacKenzie, „Fitts' Law as a Research and Design Tool in HCI"**, *HCI 1992* — die Referenz für Fitts/Throughput (auch für ISO 9241-9). *Einsteigerfreundlich.*

**Frei verfügbare Kurse / Materialien**
- **Scratchapixel** (scratchapixel.com) — hervorragende, kostenlose Tutorials zu Transformationen und Ray-Object-Intersection mit vollständiger Herleitung. *Kostenlos, einsteigerfreundlich.*
- Diverse **VR/3DUI-Vorlesungen** (z. B. von Doug Bowman / Virginia Tech) mit frei verfügbaren Folien. *Kostenlos, vertiefend.*

**Zum Ausprobieren**
- Die **drei Projekte dieses Moduls** bauen Transformationsketten + Ray-Casting (basic), das angulare Fitts-Modell + Go-Go (medium) und einen kompletten Selektionstechnik-Vergleich unter Clutter mit ISO-Throughput (final) — die beste Vertiefung ist, sie zu implementieren.

---

> **Nächstes Modul:** Modul 20 „3D Point Cloud Processing" — das Verarbeiten von 3D-Punktwolken (Registrierung/ICP, Segmentierung, Feature-Deskriptoren, PointNet). Die 3D-Geometrie und Transformationsmathematik aus diesem Modul (Abschnitt 3) ist die direkte Grundlage.
