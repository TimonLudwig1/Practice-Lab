# Modul 17 — Core XR: Principles of Interactive Systems

> **Worum geht es?** **XR** (Extended Reality — VR, AR, MR) ist der Versuch, einem Menschen
> vorzugaukeln, er sei woanders. Das Erstaunliche daran: Ob das gelingt, entscheidet sich kaum
> an der Grafik. Es entscheidet sich an **Millisekunden** und an **Mathematik** — an der Frage,
> ob das Bild schnell und korrekt genug auf eine Kopfbewegung reagiert. Ist es 20 ms zu spät,
> wird dem Nutzer übel. Dieses Modul behandelt die **Prinzipien** dahinter: Wahrnehmung,
> **Tracking** und Rotationsmathematik, **Motion-to-Photon-Latenz**, Interaktionstechniken,
> **Cybersickness** — und wie man interaktive Systeme **empirisch evaluiert**, denn die einzige
> Instanz, die über XR urteilt, ist ein Mensch.

**Hilfreiche Vorkenntnisse:** Lineare Algebra (Matrizen, Vektoren, Basiswechsel), Trigonometrie,
etwas Statistik.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 02/03 (Data Science)** — für den Evaluationsteil: EDA, Hypothesentests, Bootstrap.
  Abschnitt 5.2 knüpft direkt an die A/B-Test-Logik aus Modul 03 an.
- Sonst **keine**. Dieses Modul startet ein neues Feld (Block F) und baut nicht auf ML/RL auf.

> **Hinweis zur Ausgestaltung.** Wie bei Modul 15/16 lag keine offizielle Modulbeschreibung vor.
> Ich schneide „Core XR: Principles of Interactive Systems" auf die **Prinzipien** zu, die
> hardware-unabhängig gelten und die man ohne VR-Brille wirklich *durchdringen* kann:
> **Wahrnehmung, Tracking-Mathematik, Latenz, Interaktion, Cybersickness, Evaluation**.
>
> **Werkzeug-Entscheidung:** In dieser Umgebung gibt es **keine VR-Hardware** und **keine
> 3D-Engine** (`open3d`, `trimesh`, `pygame` fehlen). Das ist weniger schlimm, als es klingt:
> Der intellektuelle Kern von XR ist **Mathematik und Zeitverhalten**, und beides lässt sich mit
> `numpy`/`scipy` exakt nachrechnen und **testen** — besser sogar als in einer Engine, wo alles
> hinter fertigen Funktionen verschwindet. `scipy.spatial.transform.Rotation` (inkl. `Slerp`)
> ist vorhanden; wo eine Engine nötig wäre (Rendering, Shader), erkläre ich **theoretisch**.
> Für die Praxis: **Unity + OpenXR** ist der Industriestandard.

---

## Lernziele

Nach diesem Modul kannst du …

- XR im **Reality-Virtuality-Kontinuum** (Milgram) verorten und **Immersion** (technisch) von
  **Präsenz** (psychologisch) unterscheiden — inkl. Slaters *place illusion* / *plausibility illusion*;
- die relevanten Eigenschaften des **menschlichen Wahrnehmungssystems** benennen (FoV,
  Auflösung, **Vergenz-Akkommodations-Konflikt**, Vestibularsystem, Propriozeption) und daraus
  Design-Anforderungen ableiten;
- **Tracking** einordnen: 3 vs. **6 DoF**, outside-in vs. inside-out, SLAM, IMU-Sensorik — und
  erklären, warum ein Gyroskop **driftet** und wie **Sensorfusion** das repariert;
- **Rotationen** sicher handhaben: Euler-Winkel und ihr **Gimbal Lock**, Rotationsmatrizen,
  **Quaternionen** (mit Herleitung), **SLERP** — und begründen, warum XR Quaternionen nutzt;
- die **Motion-to-Photon-Latenz** in ihre Glieder zerlegen, ihr Budget berechnen und die
  Gegenmittel erklären (**Prediction**, **Timewarp/Reprojection**);
- **Interaktionstechniken** (Ray-Casting, Go-Go, Locomotion) vergleichen und **Fitts' Law**
  anwenden;
- **Cybersickness** über die **Sensory-Conflict-Theorie** erklären und Gegenmaßnahmen begründen;
- eine **Nutzerstudie** methodisch sauber planen und auswerten: Within-/Between-Subject,
  etablierte Fragebögen (**IPQ, SSQ, SUS, NASA-TLX**), passende Tests, **Effektstärke**,
  Mehrfachvergleiche.

---

## 1 · Grundlagen — Was XR ausmacht

### 1.1 Das Reality-Virtuality-Kontinuum

**Milgram & Kishino (1994)** ordnen alles auf einer Achse:

```
Reale          Augmented        Augmented         Virtuelle
Umgebung  ───  Reality (AR) ─── Virtuality  ───   Umgebung (VR)
   |                |                |                 |
 nichts        Virtuelles       Reales in         alles
 virtuell      in real          virtuell          virtuell
        └──────── Mixed Reality (MR) ────────┘
```

**XR** ist der Sammelbegriff für alles davon. Der Unterschied ist nicht bloß graduell — er
ändert die **Anforderungen**: In **VR** muss man die ganze Welt liefern, aber niemand sieht, ob
sie *falsch* zur Realität steht. In **AR** ist die Realität die Referenz — Virtuelles muss
**registriert** sein (an der richtigen Stelle bleiben), und schon 1 mm Versatz oder 5 ms
Verzögerung fallen sofort auf, weil das echte Objekt daneben liegt. **AR ist deshalb technisch
härter als VR.**

### 1.2 Immersion ≠ Präsenz

Die wichtigste begriffliche Unterscheidung des Modulnamens (**Slater**):

- **Immersion** ist eine **objektive Eigenschaft der Technik**: FoV, Auflösung, Latenz,
  Tracking-Genauigkeit, Anzahl angesprochener Sinne. Messbar, in Datenblättern nachlesbar.
- **Präsenz** ist die **subjektive Reaktion des Menschen**: *das Gefühl, dort zu sein*. Nicht
  messbar außer durch den Nutzer selbst (→ Abschnitt 5).

Slater zerlegt Präsenz weiter in zwei Illusionen, die **unabhängig** voneinander kippen können:
- **Place Illusion (PI)** — „Ich bin an diesem Ort." Entsteht aus **sensomotorischer
  Kontingenz**: Ich bewege den Kopf und die Welt reagiert **so, wie sie es in echt täte**. PI
  hängt fast vollständig an **Tracking und Latenz** — nicht an der Grafik.
- **Plausibility Illusion (Psi)** — „Was hier passiert, passiert wirklich." Entsteht, wenn die
  Welt **auf mich reagiert** und sich glaubwürdig verhält.

> **Die zentrale Einsicht des Moduls:** Ein grafisch primitiver, aber perfekt getrackter,
> latenzarmer Comic-Raum erzeugt **mehr Präsenz** als eine fotorealistische Szene, die 50 ms
> hinterherhinkt. **Deshalb geht es in diesem Modul um Mathematik und Millisekunden, nicht um
> Shader.** PI bricht sofort zusammen, wenn die sensomotorische Kontingenz verletzt wird — und
> der Körper bemerkt das gnadenlos zuverlässig.

### 1.3 Der Mensch als Systemspezifikation

XR baut nicht für Displays, sondern für ein **Wahrnehmungssystem**. Dessen Eckdaten *sind* die
Anforderungen:

| Größe | Mensch | typisches Headset | Konsequenz |
|---|---|---|---|
| **Sichtfeld (FoV)** | ~200–220° horizontal (binokular ~114° überlappend) | ~90–110° | „Taucherbrillen-Effekt", schwächt Präsenz |
| **Auflösung** | ~60 Pixel/Grad (Fovea) | ~15–35 Pixel/Grad | Screen-Door-Effekt, unscharfe Ferne |
| **Zeitliche Auflösung** | Flimmern bis ~60–90 Hz sichtbar, Latenz ab ~20 ms spürbar | 90–120 Hz | **das harte Kriterium** (Abschnitt 3) |
| **Stereo-Tiefe** | Vergenz + Akkommodation **gekoppelt** | Akkommodation fix auf ~2 m | **Vergenz-Akkommodations-Konflikt** |

**Der Vergenz-Akkommodations-Konflikt (VAC)** verdient eine eigene Erklärung, weil er ein
**physikalisch unlösbares** Problem herkömmlicher Headsets ist:
- **Vergenz** = beide Augen drehen nach innen, um ein nahes Objekt zu fixieren.
- **Akkommodation** = die Augenlinse stellt scharf.

In der Realität sind beide **fest gekoppelt** (was ich fixiere, stelle ich scharf). Im Headset
sitzt das Display **immer** in derselben optischen Entfernung (~1,5–2 m), während die Vergenz
dem virtuellen Objekt folgt — auch wenn es 30 cm vor der Nase schwebt. **Die Augen konvergieren
auf 30 cm, fokussieren aber auf 2 m.** Das Gehirn bekommt widersprüchliche Tiefensignale →
Ermüdung, Kopfschmerz, Unschärfe. Deshalb die Design-Regel: **keine wichtigen Objekte näher als
~50 cm**. Echte Lösungen (Lichtfeld-Displays, varifokale Optik) sind Forschung.

**Vestibularsystem & Propriozeption:** Das Innenohr misst **Beschleunigung** und **Drehung**,
die Propriozeption meldet Gelenkstellungen. Beide kann man **nicht** täuschen — ein Display
erreicht sie nicht. Genau daraus entsteht Abschnitt 4.1.

---

## 2 · Tracking und die Mathematik der Orientierung

### 2.1 Freiheitsgrade: 3 DoF vs. 6 DoF

- **3 DoF**: nur **Orientierung** (yaw, pitch, roll). Reicht für 360°-Video. Beugt man sich vor,
  passiert **nichts** — die Welt klebt am Kopf. Das ist ein direkter Bruch der sensomotorischen
  Kontingenz (1.2) und ein zuverlässiger Übelkeits-Generator.
- **6 DoF**: Orientierung **+ Position** ($x,y,z$). Erst damit kann man sich um ein Objekt herum
  beugen. **6 DoF ist die Untergrenze für echte Präsenz.**

**Wie wird getrackt?**
- **Outside-in**: externe Basisstationen/Kameras beobachten das Headset (z. B. Lighthouse). Sehr
  präzise, aber Aufbau nötig, begrenztes Volumen, Verdeckung möglich.
- **Inside-out**: Kameras **im** Headset beobachten die Umgebung und lösen **SLAM**
  (*Simultaneous Localization and Mapping*) — Karte bauen und sich gleichzeitig darin
  lokalisieren, ein Henne-Ei-Problem. Kein Aufbau, beliebiges Volumen; dafür abhängig von Licht
  und Textur (weiße Wand = keine Merkmale = Tracking-Verlust). **Heute Standard.**

**Sensorik:** Eine **IMU** liefert **Gyroskop** (Winkelgeschwindigkeit, ~1000 Hz) und
**Accelerometer** (Beschleunigung inkl. **Schwerkraft**). Kameras liefern ~30–60 Hz. Das ergibt
eine natürliche Arbeitsteilung — und genau daraus folgt Abschnitt 2.4.

### 2.2 Rotationen darstellen — und warum Euler-Winkel scheitern

Eine Orientierung im Raum hat **3 Freiheitsgrade**. Es gibt mehrere Darstellungen:

**Euler-Winkel** $(\text{yaw},\text{pitch},\text{roll})$ — drei Drehungen nacheinander. Intuitiv
lesbar, kompakt (3 Zahlen). Drei Probleme:

1. **Reihenfolge ist Konvention, nicht Natur.** „ZYX" ≠ „XYZ". Rotationen **kommutieren nicht**:
   $R_A R_B \neq R_B R_A$. (Projekt 01 zeigt: dieselben zwei 90°-Drehungen auf $\hat z$ ergeben
   je nach Reihenfolge $[0,-1,0]$ oder $[1,0,0]$ — verschiedene Punkte.) Fast jeder
   XR-Schnittstellen-Bug hat hier seine Wurzel.
2. **Gimbal Lock** — der Killer (siehe unten).
3. **Interpolation ist kaputt.** Zwischen zwei Euler-Tripeln linear zu interpolieren erzeugt
   Taumeln, keine kürzeste Drehung.

> ### ⚠️ Gimbal Lock — präzise formuliert
> Steht der **Pitch auf ±90°** (Blick senkrecht nach oben/unten), fallen die yaw- und die
> roll-Achse **zusammen**. Man verliert einen Freiheitsgrad: **3 DoF → 2 DoF**.
>
> Konkret (Konvention ZYX, Pitch = 90°) hängt die Rotation **nur noch von der Differenz
> $(\text{yaw}-\text{roll})$** ab. Alle diese Kombinationen ergeben **exakt dieselbe** Rotation:
>
> | yaw | roll | yaw − roll | Quaternion |
> |---|---|---|---|
> | 0° | 0° | 0 | $[0,\ 0{,}7071,\ 0,\ 0{,}7071]$ |
> | 40° | 40° | 0 | $[0,\ 0{,}7071,\ 0,\ 0{,}7071]$ |
> | 90° | 90° | 0 | $[0,\ 0{,}7071,\ 0,\ 0{,}7071]$ |
>
> Der Winkelabstand zwischen ihnen ist **0,000000°** — sie sind nicht *ähnlich*, sondern
> **identisch**. Ein Nutzer, der den Kopf ganz nach oben legt, kann yaw und roll nicht mehr
> unabhängig steuern; die Rückrechnung `as_euler` muss raten und faltet alles in einen Winkel.
> *(Ein verbreiteter Irrtum: „(yaw=0, roll=40) und (yaw=40, roll=0) sind dann gleich." **Falsch** —
> ihre Differenz unterscheidet sich um 80°, sie liegen 80° auseinander. Es ist die **Differenz**,
> die überlebt, nicht der Einzelwert.)*
>
> **Rotationsmatrizen** ($3\times3$, orthogonal, $\det=1$) haben kein Gimbal Lock, brauchen aber
> 9 Zahlen für 3 DoF und driften numerisch aus der Orthogonalität.

### 2.3 Quaternionen — die Lösung

Ein **Quaternion** $q = w + x\,i + y\,j + z\,k$ mit $i^2=j^2=k^2=ijk=-1$. Für Rotationen nutzt
man **Einheitsquaternionen** ($\|q\|=1$). Der Zusammenhang zur Anschauung ist die
**Achse-Winkel-Darstellung**: Drehung um die Einheitsachse $\hat{\mathbf n}$ um den Winkel $\theta$:
$$\boxed{\;q = \Big(\cos\tfrac{\theta}{2},\ \hat{\mathbf n}\sin\tfrac{\theta}{2}\Big)\;}$$

Ein Punkt $\mathbf v$ wird rotiert durch $\mathbf v' = q\,\mathbf v\,q^{-1}$ (mit $\mathbf v$ als
reinem Quaternion). Verkettung ist schlicht **Multiplikation**: $q_{AB} = q_B q_A$.

**Warum XR sie nutzt:**
- **Kein Gimbal Lock** — die Parametrisierung ist überall regulär.
- **Kompakt** (4 Zahlen) und **numerisch stabil** — Drift korrigiert man durch schlichtes
  **Normieren**, nicht durch Re-Orthogonalisierung.
- **Interpolierbar** — siehe SLERP.
- **Verkettung billig** (16 Multiplikationen statt 27 bei Matrizen).

**Die Kuriosität, die man kennen muss:** $q$ und $-q$ beschreiben **dieselbe** Rotation (die
*doppelte Überdeckung* $SU(2)\to SO(3)$; das $\theta/2$ oben ist der Grund). Praktische Folge:
Beim Interpolieren muss man das **Vorzeichen prüfen** — sonst nimmt der Agent den **langen Weg**
(359° statt 1°). Ein klassischer Bug.

**SLERP** (*Spherical Linear Interpolation*) interpoliert auf der Einheitssphäre entlang des
**Großkreises** — die kürzeste Drehung mit **konstanter Winkelgeschwindigkeit**:
$$\text{Slerp}(q_0,q_1;t)=\frac{\sin\big((1-t)\Omega\big)}{\sin\Omega}\,q_0+\frac{\sin(t\Omega)}{\sin\Omega}\,q_1,
\qquad \cos\Omega = q_0\!\cdot\! q_1$$
Naives **LERP** (komponentenweise mitteln + normieren) läuft die **Sehne** statt des Bogens →
die Winkelgeschwindigkeit **schwankt**, in der Mitte zu schnell. Projekt 01 misst das: bei einer
Drehung 0°→170° hat SLERP exakt konstante Schritte (Streuung **0,000**), LERP nicht (**5,659**).
Bei kleinen Winkeln ist der Unterschied vernachlässigbar — deshalb ist LERP für
Netzwerk-Interpolation zwischen dichten Frames durchaus üblich.

### 2.4 Sensorfusion: warum Gyro allein driftet

Die zwei Sensoren einer IMU haben **komplementäre** Fehler:

| | Gyroskop | Accelerometer |
|---|---|---|
| misst | Winkel**geschwindigkeit** | Beschleunigung + **Schwerkraft** |
| Rate | schnell (~1000 Hz) | schnell |
| kurzfristig | **präzise, glatt** | **verrauscht** (jede Bewegung stört) |
| langfristig | **driftet weg** | **driftfrei** (Schwerkraft zeigt immer nach unten) |

**Warum driftet das Gyro?** Man braucht den *Winkel*, hat aber die *Geschwindigkeit* — also
**integriert** man: $\theta_k=\theta_{k-1}+\omega_k\Delta t$. Jeder noch so kleine **Bias**
summiert sich dabei **unbegrenzt** auf. Ein Bias von nur 0,5 °/s ergibt nach einer Minute **30°**
Fehler. (Projekt 01 misst: RMSE **16,98°**, Endfehler **29,95°** — die virtuelle Welt kippt weg.)

Das Accelerometer misst über die **Schwerkraft** einen absoluten Bezug („wo ist unten?"), ist
aber verrauscht (RMSE **3,01°**) und bei Bewegung unbrauchbar.

**Der Komplementärfilter** kombiniert beide in einer Zeile — Hochpass aufs Gyro, Tiefpass aufs
Accel:
$$\boxed{\;\theta_k = \alpha\big(\theta_{k-1}+\omega_k\Delta t\big) + (1-\alpha)\,\theta_{\text{accel},k}\;}$$
mit $\alpha$ nahe 1 (z. B. 0,98). Kurzfristig folgt er dem glatten Gyro, langfristig zieht ihn
das Accel zurück auf die Wahrheit. Ergebnis: RMSE **0,42°** — **besser als beide Einzelsensoren**.
Ein **Kalman-Filter** macht dasselbe optimal (mit geschätzter Unsicherheit und Bias-Schätzung);
der Komplementärfilter ist sein armer, erstaunlich guter Verwandter. Die **Magnetometer**-Ergänzung
liefert zusätzlich absoluten **yaw** (Kompass) — den das Accel *nicht* liefern kann, weil eine
Drehung um die Schwerkraftachse die Schwerkraft nicht ändert.

---

## 3 · Motion-to-Photon: das Millisekunden-Budget

### 3.1 Die Kette

**Motion-to-Photon-Latenz** = die Zeit von „Kopf bewegt sich" bis „passendes Photon trifft die
Netzhaut". Sie ist **die** kritische Größe von XR. Die Kette:

```
Kopf bewegt sich
   → IMU/Kamera misst        ~1-2 ms
   → Sensorfusion/Pose       ~1 ms
   → Anwendung/Physik        ~2-5 ms
   → Rendering (GPU)         ~5-11 ms   (bei 90 Hz = 11.1 ms pro Frame)
   → Übertragung/Scanout     ~3-11 ms
   → Display (Pixel-Response) ~1-5 ms
   = Motion-to-Photon        ~15-40 ms
```

**Das Ziel: < 20 ms.** Darüber wird der Konflikt zwischen Vestibularsystem und Augen spürbar
(Abschnitt 4.1). Ab ~50 ms ist es für viele unerträglich. Zum Vergleich: Ein normales Spiel am
Monitor mit 60 ms Latenz stört niemanden — **weil dort kein Vestibularsystem widerspricht.**

Man beachte: Bei **90 Hz** ist allein ein Frame **11,1 ms**. Das Budget ist also nach *einem*
Frame plus Scanout praktisch aufgebraucht — hier ist kein Platz für „das optimieren wir später".

### 3.2 Die zwei Gegenmittel

Weil man die Kette nicht beliebig verkürzen kann, **betrügt** man — auf zwei Arten:

**1. Prediction.** Man rendert nicht für *jetzt*, sondern für den Zeitpunkt, zu dem das Photon
erscheinen wird: Aus der aktuellen Winkelgeschwindigkeit extrapoliert man die Pose um die
Latenz $\Delta t$ nach vorn. Bei gleichmäßiger Bewegung funktioniert das verblüffend gut. Der
Preis: Bei **Richtungswechseln** liegt die Vorhersage daneben → **Overshoot**. Deshalb sagt man
nur ~20–40 ms voraus, nicht mehr.

**2. Timewarp / Reprojection (ASW).** Der eigentliche Trick, und der Grund, warum moderne
Headsets erträglich sind: **Nachdem** das Bild gerendert ist, aber **bevor** es angezeigt wird,
holt man die **allerneueste** Pose und **verschiebt/verzerrt das fertige Bild** entsprechend.
- **Orientational Timewarp** ist quasi gratis und sehr wirksam: Eine reine *Drehung* lässt sich
  auf einem fertigen Bild fast perfekt nachkorrigieren (man schiebt den Bildausschnitt).
- **Positionale** Korrektur ist schwerer: Bewegt sich der Kopf **seitlich**, ändert sich die
  **Verdeckung** — hinter dem Vordergrund müsste Information auftauchen, die nie gerendert wurde
  (Disokklusion). Man bekommt Artefakte oder muss raten.
- Reicht die GPU nicht für 90 Hz, rendert man mit 45 Hz und **erfindet** jedes zweite Bild per
  Reprojection dazu (*Asynchronous Spacewarp*) — sichtbar an Artefakten an bewegten Kanten,
  aber besser als Ruckeln.

> **Merke:** Timewarp macht die **Latenz nicht kleiner** — es macht sie **unsichtbar für die
> Orientierung**, den Kanal, auf den das Vestibularsystem am empfindlichsten reagiert. Es ist ein
> Wahrnehmungs-Trick, kein Performance-Fix.

---

## 4 · Interaktion und Cybersickness

### 4.1 Cybersickness: die Sensory-Conflict-Theorie

**Das Symptom:** Übelkeit, Schwindel, Schwitzen, Augenermüdung, Desorientierung — bei manchen
Menschen nach Minuten, noch Stunden nachwirkend.

**Die führende Erklärung (Sensory Conflict / Reason & Brady):** Übelkeit entsteht, wenn die
Sinne **widersprüchliche** Bewegungsinformation liefern:
- **Augen** sagen: „Wir bewegen uns" (die Welt zieht vorbei → **Vection**, die illusorische
  Eigenbewegung).
- **Vestibularsystem** sagt: „Wir sitzen still." Es misst echte Beschleunigung — und die ist null.

Das Gehirn kann den Widerspruch nicht auflösen. Die (evolutionär plausible) Hypothese: Ein
solcher Konflikt entsteht natürlicherweise durch **Neurotoxine** → das Gehirn schließt auf
Vergiftung → **Erbrechen**. **Man wird in VR schlecht, weil das Gehirn glaubt, man sei
vergiftet.** — Bemerkenswert: **Seekrankheit ist derselbe Konflikt mit vertauschten Rollen**
(Vestibularsystem meldet Bewegung, die Augen sehen in der Kabine Stillstand).

**Die Ursachen, nach Wirksamkeit sortiert:**
1. **Latenz** — die Welt hinkt der Kopfbewegung hinterher. Der stärkste, aber auch der am
   besten lösbare Hebel (Abschnitt 3).
2. **Künstliche Fortbewegung** — Bewegung per Stick, während der Körper stillsteht. Der
   inhärente Konflikt.
3. **Beschleunigung** — konstante Geschwindigkeit ist relativ harmlos; **Beschleunigung**,
   Drehung und Treppen/Rampen sind es nicht (das Vestibularsystem misst genau *Beschleunigung*).
4. **3 DoF statt 6 DoF**, falscher Augenabstand (IPD), niedrige Bildrate.

**Gegenmaßnahmen — und warum sie wirken:**
- **Teleportation** statt kontinuierlicher Bewegung: **kein** optischer Fluss → **kein** Konflikt.
  Der Goldstandard für Komfort, mit dem Preis, dass räumliches Verständnis leidet.
- **Snap-Turn** (ruckartige 30°-Sprünge) statt weichem Drehen — Drehung ist der schlimmste Fall.
- **Vignetting/Tunnelblick** während der Bewegung: reduziert den optischen Fluss in der
  Peripherie (dort sitzt die Vection-Empfindlichkeit).
- **Statischer Bezugsrahmen** (virtuelles Cockpit, Nase, Gitter): etwas, das mit dem Kopf
  mitgeht und „Stillstand" bestätigt.
- **Hohe Bildrate, niedrige Latenz** — die Basis, ohne die alles andere egal ist.

> **Ehrliche Einordnung:** Die Sensory-Conflict-Theorie erklärt viel, aber **nicht alles** (etwa
> nicht gut, warum die Anfälligkeit zwischen Menschen so **massiv** streut — Frauen berichten in
> Studien häufiger Symptome, was teils auf IPD-Passform zurückgeführt wird). Konkurrierend ist
> die **Postural-Instability-Theorie** (Riccio & Stoffregen): Übelkeit folgt aus länger
> anhaltender Unfähigkeit, die Körperhaltung zu stabilisieren. Beide sind vermutlich Teilwahrheiten.

### 4.2 Interaktionstechniken

**Selektion & Manipulation:**
- **Virtual Hand** — direkt zugreifen. Natürlich, aber nur in Armreichweite.
- **Ray-Casting** — ein Strahl aus der Hand, „Laserpointer". Reichweite unbegrenzt; aber die
  **Winkelpräzision** limitiert: In der Ferne bedeutet 1° Handzittern viele Zentimeter Versatz.
- **Go-Go** — nichtlineare Armverlängerung: bis zu einer Schwelle 1:1, darüber wächst der
  virtuelle Arm überproportional. Verbindet Natürlichkeit mit Reichweite.
- **Fitts' Law** gilt auch hier und quantifiziert die Zielzeit:
  $$MT = a + b\log_2\!\Big(\frac{D}{W}+1\Big)$$
  ($D$ = Distanz, $W$ = Zielbreite). Der Term $\log_2(D/W+1)$ ist der **Index of Difficulty**.
  Praktische Folge: Zielgröße hilft **logarithmisch** — kleine Ziele sind teuer, und in 3D nutzt
  man die **Winkelgröße**, nicht die metrische.

**Locomotion** (siehe 4.1 für den Komfort-Aspekt): Teleport · kontinuierlich (Stick) ·
**Room-Scale** (echtes Gehen — bester Komfort, begrenzt durch das Zimmer) · **Redirected
Walking** (die Welt wird unmerklich gedreht, sodass der Nutzer im Kreis läuft, aber geradeaus zu
gehen glaubt — verblüffend wirksam, braucht aber viel Platz).

---

## 5 · Evaluation: XR ist eine empirische Disziplin

### 5.1 Warum überhaupt Nutzerstudien?

Für „ist die Latenz < 20 ms?" reicht eine Messung. Aber die eigentlichen Fragen — *Fühlt sich
das präsent an? Wird jemandem übel? Ist es benutzbar?* — sind **nur am Menschen** beantwortbar.
Es gibt keine Offline-Metrik für Präsenz. **Das ist der Grund, warum XR-Forschung im Kern
experimentelle Psychologie mit Technik ist.**

**Etablierte Instrumente** (nimm die validierten, erfinde keine eigenen Fragebögen):
- **IPQ** (Igroup Presence Questionnaire) — Präsenz.
- **SSQ** (Simulator Sickness Questionnaire) — Cybersickness, mit den Subskalen *Nausea*,
  *Oculomotor*, *Disorientation*. **Wichtig: vorher *und* nachher** erheben (Differenz zählt).
- **SUS** (System Usability Scale) — 10 Items, Score 0–100. *(Verwirrend: „SUS" heißt in der
  Präsenzliteratur auch *Slater-Usoh-Steed*-Fragebogen. Kontext beachten.)*
- **NASA-TLX** — subjektive Beanspruchung (mental, körperlich, zeitlich, Leistung, Anstrengung,
  Frustration).
- **Objektiv** daneben: Task-Completion-Time, Fehlerrate, Trajektorien — und physiologisch
  (Herzrate, Hautleitwert) als Sickness-Korrelat.

### 5.2 Studiendesign

- **Within-Subject** (jede Person testet **alle** Bedingungen): weniger Teilnehmer nötig,
  kontrolliert für individuelle Unterschiede — die in XR **riesig** sind (Anfälligkeit,
  VR-Erfahrung). **In XR meist die richtige Wahl.** Preis: **Reihenfolgeeffekte** (Lernen,
  Ermüdung, kumulierte Übelkeit) → **Counterbalancing** (z. B. Latin Square) ist Pflicht.
- **Between-Subject** (jede Person **eine** Bedingung): keine Reihenfolgeeffekte, aber viel mehr
  Teilnehmer nötig. Nötig, wenn eine Bedingung die andere „verdirbt" (wer einmal 6 DoF hatte,
  bewertet 3 DoF anders).

**Auswertung** — das knüpft direkt an Modul 03 an:
- **Skalenniveau beachten:** Fragebogen-Items sind **ordinal** (Likert). Für Einzel-Items sind
  nichtparametrische Tests angebracht: **Wilcoxon signed-rank** (within), **Mann-Whitney U**
  (between). Für gemittelte Subskalen mit vielen Items argumentiert man oft intervallskaliert
  → **paired t-test** / ANOVA. *Beides ist vertretbar — man muss es nur begründen.*
- **Effektstärke berichten, nicht nur p.** Ein signifikanter, aber winziger Effekt ist
  irrelevant. **Cohen's d** bzw. $r=Z/\sqrt{N}$. Faustregel: $d\approx0{,}2$ klein, $0{,}5$
  mittel, $0{,}8$ groß.
- **Mehrfachvergleiche korrigieren.** Wer IPQ, SSQ, SUS, TLX und Zeit einzeln testet, macht 5+
  Tests — bei $\alpha=0{,}05$ ist ein Fehlalarm dann fast garantiert (**Bonferroni**: $\alpha/m$;
  oder Holm/FDR). *Das ist derselbe Gedanke wie die Basisraten-Diskussion in Modul 15: viele
  Tests × kleine Fehlerrate = viele Fehlalarme.*
- **Stichprobengröße vorher** planen (Power-Analyse). N=8 findet nur Elefanten. Typische
  XR-Studien: N=20–40.
- **Ethik:** Cybersickness ist eine reale Belastung. Abbruchmöglichkeit ohne Begründung,
  Aufklärung, Pausen, keine Fahrt nach Hause direkt nach einer Sickness-Studie.

---

## 6 · Zusammenfassung / Cheat-Sheet

**Begriffe.** Reality-Virtuality-Kontinuum (Milgram) · **Immersion** = Technik (objektiv) ·
**Präsenz** = Erleben (subjektiv) · **Place Illusion** (Tracking+Latenz!) + **Plausibility
Illusion** (Slater).

**Mensch.** FoV ~200° vs. Headset ~110° · **VAC**: Vergenz folgt dem Objekt, Akkommodation klebt
auf ~2 m → nichts näher als ~50 cm · Vestibularsystem **kann man nicht täuschen**.

**Tracking.** 3 DoF (nur Orientierung) vs. **6 DoF** (+ Position) · outside-in vs. **inside-out
(SLAM)** · IMU = Gyro (schnell, **driftet**) + Accel (verrauscht, **driftfrei**).

**Rotationen.** Euler: intuitiv, aber **nicht kommutativ** + **Gimbal Lock** (Pitch ±90° ⟹ nur
(yaw−roll) überlebt, 3→2 DoF) · Matrizen: 9 Zahlen, driften · **Quaternionen**:
$q=(\cos\frac\theta2,\ \hat{\mathbf n}\sin\frac\theta2)$, $\mathbf v'=q\mathbf vq^{-1}$,
$q\equiv-q$ (Vorzeichen prüfen!) · **SLERP** = konstante Winkelgeschwindigkeit, LERP nicht.

**Fusion.** $\theta_k=\alpha(\theta_{k-1}+\omega_k\Delta t)+(1-\alpha)\theta_{\text{accel},k}$
⟹ besser als beide Sensoren einzeln.

**Latenz.** **Motion-to-Photon < 20 ms** · 90 Hz ⟹ 11,1 ms je Frame · **Prediction**
(extrapolieren, Overshoot bei Richtungswechsel) · **Timewarp** (fertiges Bild nachkorrigieren;
Rotation ~gratis, Position ⟹ **Disokklusion**).

**Cybersickness.** **Sensory Conflict**: Augen sehen Bewegung (**Vection**), Vestibularsystem
nicht ⟹ Gehirn vermutet Gift. Hebel: Latenz > künstliche Fortbewegung > **Beschleunigung**.
Gegenmittel: Teleport, Snap-Turn, Vignetting, statischer Bezugsrahmen.

**Interaktion.** Virtual Hand · **Ray-Casting** (Winkelpräzision!) · Go-Go · **Fitts**:
$MT=a+b\log_2(D/W+1)$.

**Evaluation.** **IPQ** (Präsenz) · **SSQ** (Sickness, vorher/nachher!) · SUS · NASA-TLX ·
**within-subject + Counterbalancing** · ordinal ⟹ **Wilcoxon/Mann-Whitney** · **Effektstärke**
(Cohen's d) · **Bonferroni** · Power vorher planen.

---

## 7 · Selbsttest

<details>
<summary><b>1.</b> Immersion vs. Präsenz — und warum ist ein hässliches, schnelles System besser als ein schönes, langsames?</summary>

**Immersion** = objektive Technik-Eigenschaft (FoV, Latenz, Tracking). **Präsenz** = subjektives
Gefühl, dort zu sein. Die **Place Illusion** entsteht aus **sensomotorischer Kontingenz** — die
Welt muss auf Kopfbewegung reagieren *wie in echt*. Das hängt an **Tracking und Latenz**, nicht
an der Grafik. Ein latenzarmer Comic-Raum erzeugt daher mehr Präsenz als eine fotorealistische
Szene mit 50 ms Verzug (die zusätzlich krank macht).
</details>

<details>
<summary><b>2.</b> Was ist der Vergenz-Akkommodations-Konflikt und was folgt daraus fürs Design?</summary>

**Vergenz** (Augen drehen nach innen) folgt dem virtuellen Objekt, **Akkommodation** (Linse
stellt scharf) klebt auf der festen Displaydistanz (~2 m). In der Realität sind beide gekoppelt —
im Headset widersprechen sie sich ⟹ Ermüdung, Kopfschmerz. **Design-Regel: nichts Wichtiges
näher als ~50 cm.** Echte Lösungen (varifokale/Lichtfeld-Optik) sind Forschung.
</details>

<details>
<summary><b>3.</b> Erkläre Gimbal Lock präzise. Welche Kombinationen werden ununterscheidbar?</summary>

Bei **Pitch = ±90°** fallen yaw- und roll-Achse zusammen ⟹ **3 DoF → 2 DoF**. Es überlebt nur
die **Differenz** $(\text{yaw}-\text{roll})$: (0°,0°), (40°,40°), (90°,90°) ergeben **exakt
dieselbe** Rotation (0,000000° Abstand). **Nicht** gleich sind dagegen (0°,40°) und (40°,0°) —
ihre Differenzen unterscheiden sich um 80°. Quaternionen haben das Problem nicht.
</details>

<details>
<summary><b>4.</b> Warum $\theta/2$ im Quaternion — und warum ist $q\equiv-q$ praktisch relevant?</summary>

Weil die Rotation als $\mathbf v'=q\mathbf vq^{-1}$ **zweimal** wirkt (einmal $q$, einmal
$q^{-1}$) — jede Hälfte trägt $\theta/2$ bei. Folge ist die **doppelte Überdeckung**: $q$ und
$-q$ sind dieselbe Rotation. **Praktisch:** Beim Interpolieren muss man das Vorzeichen prüfen
(ggf. $q_1 \to -q_1$), sonst nimmt SLERP den **langen Weg** (359° statt 1°).
</details>

<details>
<summary><b>5.</b> SLERP vs. LERP — was ist der Unterschied, und wann ist LERP trotzdem ok?</summary>

**SLERP** läuft den **Großkreis** auf der Einheitssphäre ⟹ kürzeste Drehung mit **konstanter
Winkelgeschwindigkeit**. **LERP** (mitteln + normieren) läuft die **Sehne** ⟹ die
Winkelgeschwindigkeit schwankt (gemessen: Streuung der Schritte 0,000 vs. 5,659 bei 0°→170°).
Bei **kleinen** Winkeln ist der Fehler vernachlässigbar — deshalb ist LERP zwischen dichten
Frames üblich (billiger).
</details>

<details>
<summary><b>6.</b> Warum driftet ein Gyroskop, und wie repariert der Komplementärfilter das?</summary>

Das Gyro misst Winkel**geschwindigkeit**; für den Winkel muss man **integrieren** — dabei
summiert sich jeder **Bias** unbegrenzt auf (0,5 °/s ⟹ 30° nach einer Minute; gemessen RMSE
16,98°). Das **Accelerometer** liefert über die **Schwerkraft** einen absoluten, driftfreien,
aber verrauschten Bezug (RMSE 3,01°). Der Filter
$\theta_k=\alpha(\theta_{k-1}+\omega_k\Delta t)+(1-\alpha)\theta_{\text{accel}}$ nimmt kurzfristig
das Gyro, langfristig das Accel ⟹ RMSE **0,42°**, besser als beide.
</details>

<details>
<summary><b>7.</b> Was ist Motion-to-Photon-Latenz, wo liegt die Grenze, und warum stört 60 ms am Monitor nicht?</summary>

Zeit von der Kopfbewegung bis zum passenden Photon auf der Netzhaut; Kette aus Sensor →
Fusion → App → Rendering → Scanout → Display. **Ziel < 20 ms.** Am Monitor stört 60 ms nicht,
weil dort **kein Vestibularsystem widerspricht** — im Headset erzeugt die Verzögerung genau den
sensorischen Konflikt, der übel macht.
</details>

<details>
<summary><b>8.</b> Was macht Timewarp — und warum hilft es bei Drehung besser als bei Translation?</summary>

Es korrigiert das **fertig gerenderte** Bild kurz vor der Anzeige anhand der **neuesten** Pose.
Eine reine **Drehung** lässt sich fast perfekt nachschieben (nur ein anderer Bildausschnitt). Bei
**Translation** ändert sich die **Verdeckung**: hinter Vordergrundobjekten müsste Information
auftauchen, die nie gerendert wurde (**Disokklusion**) ⟹ Artefakte/Raten. Timewarp senkt die
Latenz **nicht**, es macht sie für die Orientierung **unsichtbar**.
</details>

<details>
<summary><b>9.</b> Erkläre Cybersickness über die Sensory-Conflict-Theorie. Warum hilft Teleportation?</summary>

Augen melden Eigenbewegung (**Vection**), das Vestibularsystem meldet Stillstand. Der Widerspruch
ähnelt dem Muster einer **Vergiftung** ⟹ Übelkeit. **Teleportation** erzeugt **keinen optischen
Fluss** ⟹ kein Konflikt. Weitere Hebel: Latenz senken, **Beschleunigung** vermeiden, Snap-Turn,
Vignetting, statischer Bezugsrahmen. (Seekrankheit ist derselbe Konflikt mit vertauschten Rollen.)
</details>

<details>
<summary><b>10.</b> Du vergleichst 3 DoF vs. 6 DoF und misst IPQ, SSQ, SUS, TLX und Zeit. Nenne drei methodische Pflichten.</summary>

Beliebige drei: **Within-Subject mit Counterbalancing** (Reihenfolgeeffekte: Lernen, Ermüdung,
kumulierte Übelkeit) · **SSQ vorher und nachher** erheben (Differenz zählt) · **ordinale**
Likert-Items ⟹ **Wilcoxon** statt t-Test (oder begründen) · **Effektstärke** berichten, nicht nur
p · **Mehrfachvergleiche korrigieren** (5 Tests ⟹ Bonferroni/Holm) · **Power vorher** planen
(N=20–40) · **Ethik** (Abbruch jederzeit).
</details>

---

## 8 · Literatur & Quellen

**Standardwerke:**
- 📗 **LaValle — *Virtual Reality*** (Cambridge; **frei online**: lavalle.pl/vr/). *Die*
  Referenz für genau dieses Modul: Wahrnehmung, Tracking, Rotationsmathematik, Latenz — mit
  Mathematik, aber lesbar. **Beste Einzelquelle.** Kap. 3 (Transformationen), 9 (Tracking),
  12 (Sickness).
- 📗 **Jerald — *The VR Book: Human-Centered Design for VR***. Stark auf Wahrnehmung,
  Sickness und Interaktionsdesign. *Einsteigerfreundlich.*
- 📗 **Bowman et al. — *3D User Interfaces: Theory and Practice***. Das Standardwerk zu
  Interaktionstechniken (→ vertieft in Modul 19). *Vertiefend.*

**Schlüsselpaper:**
- 📄 **Milgram & Kishino (1994), *A Taxonomy of Mixed Reality Visual Displays*** — das Kontinuum.
- 📄 **Slater (2009), *Place Illusion and Plausibility Illusion*** (Phil. Trans. R. Soc.) —
  die begriffliche Grundlage von 1.2. *Frei, kurz, lesenswert.*
- 📄 **Reason & Brady (1975), *Motion Sickness*** — Sensory-Conflict-Theorie.
- 📄 **Riccio & Stoffregen (1991), *An Ecological Theory of Motion Sickness*** — die
  konkurrierende Postural-Instability-Theorie.
- 📄 **Shoemake (1985), *Animating Rotation with Quaternion Curves*** (SIGGRAPH) — **SLERP**.
- 📄 **Van Waveren (2016), *The Asynchronous Time Warp for VR on Mobile Hardware***.
- 📄 **Poupyrev et al. (1996), *The Go-Go Interaction Technique***.
- 📄 **Razzaque et al. (2001), *Redirected Walking***.

**Mathematik/Praxis:**
- 🌐 **3Blue1Brown — *Visualizing quaternions*** (eater.net/quaternions) — interaktiv,
  hervorragend, wenn Quaternionen abstrakt bleiben. *Einsteigerfreundlich, frei.*
- 🌐 **scipy `spatial.transform.Rotation`-Doku** — die API der Projekte.
- 🌐 **OpenXR-Spezifikation** (khronos.org/openxr) — der offene Industriestandard; die
  Begriffe (Pose, Space, Predicted Display Time) tauchen dort genau so auf.
- 🌐 **Unity XR Interaction Toolkit** / **Oculus Developer Blog** (Carmack/Abrash zu Latenz) —
  für die Praxis.

**Evaluation:**
- 📄 **Kennedy et al. (1993), *Simulator Sickness Questionnaire (SSQ)***.
- 📄 **Schubert et al. (2001), *The Experience of Presence: Factor Analytic Insights*** (IPQ;
  igroup.org/pq/ipq — frei verfügbar, inkl. Items).
- 📄 **Hart & Staveland (1988), *NASA-TLX***; **Brooke (1996), *SUS***.
- 📗 **Hyndman-artig fürs Handwerk:** *Field — Discovering Statistics* für die Testwahl, oder
  Modul 03 dieses Repos.

---

## Nächstes Modul

**Modul 18 — Multimodal Interfaces** erweitert die Interaktion über Hände hinaus (Sprache,
Blick, Gesten, Haptik) und fragt, wie man Modalitäten **fusioniert**. **Modul 19 — 3D User
Interfaces** vertieft dann systematisch die Interaktionstechniken aus 4.2. Was du hier gelernt
hast — **Präsenz hängt an Latenz und Tracking**, Quaternionen, und dass am Ende ein **Mensch**
im Experiment entscheidet — trägt durch den gesamten XR-Block.
