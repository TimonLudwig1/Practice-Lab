# Modul 18 — Multimodal Interfaces

> **Worum geht es?** Menschen kommunizieren nie über nur *einen* Kanal: Wir sprechen und zeigen gleichzeitig, wir sehen und fühlen ein Objekt zur selben Zeit, wir hören eine Stimme und lesen die Lippen dazu. **Multimodale Interfaces** sind Mensch-Computer-Schnittstellen, die mehrere solcher **Modalitäten** (Sprache, Blick, Geste, Haptik, …) gleichzeitig aufnehmen oder ausgeben — und, das ist der Kern, sie **fusionieren**. Dieses Modul behandelt die *Prinzipien* und vor allem die *Mathematik* dieser Fusion: Wann macht es zwei Signale gemeinsam besser als einzeln? Wie gewichtet man widersprüchliche Sinne optimal? Wie richtet man zwei asynchrone Ereignisströme zeitlich aufeinander aus?
>
> **Vorkenntnisse**: Wahrscheinlichkeitsrechnung (bedingte Wahrscheinlichkeit, Bayes, Gauß-Verteilung, Varianz), etwas lineare Algebra. Aus diesem Repo helfen: **Modul 04/05** (Naive Bayes, Klassifikation, Konfidenzen), **Modul 07** (Bayes-Netze, bedingte Unabhängigkeit), **Modul 09** (Attention/Transformer — für Cross-Modal-Attention), und ganz besonders **Modul 17** (Core XR: der *Komplementärfilter* zur Sensorfusion ist der direkte Vorläufer der hier zentralen **inverse-Varianz-Gewichtung**).
>
> **Ideale Vormodule**: 17 (Core XR) unmittelbar davor; 07 (Theorie der KI 2 — Bayes-Netze) als probabilistisches Fundament.

> **Hinweis zum Zuschnitt dieses Moduls.** Für „Multimodal Interfaces" liegt keine offizielle Modulbeschreibung vor, deshalb habe ich den Inhalt selbst zugeschnitten — konsistent mit dem, was dieser Titel an einschlägigen HCI/XAI-Lehrstühlen bedeutet (Fortsetzung von Core XR, Richtung Interaktion über die Hand hinaus). **Wie in Modul 17 verzichte ich bewusst auf konkrete Hardware** (kein Mikrofon, kein Eye-Tracker, keine Datenhandschuhe). Der Grund ist didaktisch: Die eigentlich *lehrbaren, prüfbaren und übertragbaren* Inhalte sind die **Fusionsmathematik, die Wahrnehmungsmodelle und die Evaluationsmethodik**. Ein Eye-Tracker zu kalibrieren lernt man in einer Stunde am Gerät; zu verstehen, *warum* die optimale Verschmelzung zweier Sinne eine inverse-Varianz-Gewichtung ist und wann sie versagt, ist die Master-Kompetenz. Die Projekte simulieren die Signale realistisch und arbeiten mit reiner Mathematik/ML auf der CPU.

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

- **erklären können, was eine Modalität ist** und wie sie sich von Kanal, Medium und Gerät unterscheidet — und warum „multimodal" mehr heißt als „mehrere Eingabegeräte".
- die **klassischen Klassifikationsschemata** multimodaler Systeme beherrschen: **CASE** (Nebenläufigkeit von Modalitäten auf Systemseite) und **CARE** (Complementarity, Assignment, Redundancy, Equivalence — die Beziehungen von Modalitäten zueinander).
- die **menschliche Seite** verstehen: *Multiple Resource Theory*, der *McGurk-Effekt*, *Redundancy Gain*, das *Midas-Touch-Problem* beim Blick.
- den **Kern des Moduls** — die **probabilistische Fusion** — vollständig herleiten können: von der Bayes-Produktregel unter bedingter Unabhängigkeit über die **inverse-Varianz-Gewichtung** (Maximum-Likelihood-Integration) bis zur Einsicht, dass **das menschliche Gehirn nachweislich so fusioniert** (Ernst & Banks 2002).
- die **Fusionsebenen** (early / late / hybrid) gegeneinander abwägen können — inklusive der Frage, welche bei fehlenden Modalitäten robust ist.
- **Mutual Disambiguation** verstehen: warum zwei je *unsichere* Erkenner zusammen *sicherer* sind als jeder allein, und dass Fusion die Fehlerrate senken *kann*, aber unter Korrelation auch *nicht muss*.
- **zeitliche Fusion** beherrschen: das *temporal binding window*, Zeitfenster-Alignment asynchroner Ereignisströme, deiktische Referenzauflösung à la „Put-that-there".
- moderne **Deep-Multimodal-Learning**-Konzepte einordnen: joint vs coordinated representations, Cross-Modal Attention, contrastive Alignment (CLIP-Idee), Modality Dropout.
- multimodale Systeme **methodisch sauber evaluieren** und die Fallen kennen (Korrelation der Modalitäten, Konfidenz-Kalibrierung, Redundanz ≠ Gewinn).

---

## Grundlagen (Basics)

### 1. Was ist eine Modalität? (und was nicht)

Der Begriff **Modalität** wird schlampig verwendet. Sauber unterschieden werden drei Ebenen:

- **Modalität (modality)**: ein menschlicher **Wahrnehmungs- oder Ausdruckskanal**, gebunden an einen Sinn bzw. eine motorische Fähigkeit — *Sehen, Hören, Tasten (Haptik), Sprechen, Zeigen (Deixis), Blicken (Gaze)*. Modalität ist eine Aussage über *Bedeutungsrepräsentation*, nicht über Technik.
- **Medium / Kanal**: das *physikalische Trägermedium* (Schallwelle, Lichtwelle) bzw. der Übertragungsweg.
- **Gerät (device)**: die *technische Umsetzung* (Mikrofon, Kamera, Touchscreen, Force-Feedback-Motor).

> **Warum die Unterscheidung wichtig ist.** Ein Touchscreen und eine Maus sind **dieselbe Modalität** (2D-Zeigen), obwohl es verschiedene Geräte sind. Umgekehrt kann *ein* Gerät (eine Kamera) *mehrere* Modalitäten liefern (Gesten *und* Blickrichtung *und* Mimik). „Multimodal" meint **mehrere Modalitäten**, nicht mehrere Geräte. Ein System mit zwei Mäusen ist nicht multimodal; ein System mit Sprache **und** Zeigegeste ist es.

Eine gängige, brauchbare Definition (nach Oviatt): Ein **multimodales Interface** verarbeitet zwei oder mehr kombinierte Nutzer-Eingabemodalitäten **in koordinierter Weise mit multimedialer Systemausgabe**. Das Schlüsselwort ist **koordiniert** — das reine Nebeneinander (erst tippen, dann klicken) ist trivial; die interessante (und schwierige) Sache ist die **Fusion** koordinierter, oft *gleichzeitiger* Eingaben.

### 2. Das Gründungsbeispiel: „Put-that-there" (Bolt, 1980)

Richard Bolts System am MIT ist das kanonische Beispiel und zieht sich als roter Faden durch dieses Modul (das **Final-Projekt** baut es nach). Der Nutzer sitzt vor einer Großprojektion und sagt:

> *„Put **that** … **there**."*

… während er beim Wort *„that"* auf ein Objekt zeigt und beim Wort *„there"* auf eine Zielposition. Weder die Sprache allein (*„that"* / *„there"* sind **deiktische**, d. h. auf den Kontext zeigende, für sich bedeutungslose Wörter) noch die Geste allein (ein Zeigen ohne Verb ist mehrdeutig) genügt. Erst die **zeitlich koordinierte Fusion** ergibt einen ausführbaren Befehl.

Daraus lernen wir drei Kernprobleme, die dieses Modul strukturieren:

1. **Referenzauflösung**: Welches Objekt meint *„that"*? → verknüpfe das deiktische Wort mit dem Zeigeziel.
2. **Zeitliche Fusion**: Das Zeigen und das Wort *„that"* fallen nicht exakt zusammen — es gibt ein **Zeitfenster**. Wie breit? (Empirisch: Geste **leicht vor** oder um das Wort herum.)
3. **Ambiguität & Fehler**: Spracherkennung und Zeigeerkennung sind je *unsicher*. Wie kombiniert man zwei unsichere Hypothesen zu einer *sichereren*?

### 3. Warum überhaupt multimodal? Die menschliche Seite

Multimodalität ist kein technischer Selbstzweck, sondern spiegelt, **wie Menschen wahrnehmen und kommunizieren**. Vier Befunde begründen das Gebiet:

**(a) Multiple Resource Theory (Wickens).** Der Mensch hat nicht *einen* Aufmerksamkeitspool, sondern **mehrere getrennte Ressourcen** entlang von Dimensionen (Modalität visuell/auditiv, Kode räumlich/verbal, Verarbeitungsstufe). Zwei Aufgaben stören sich **weniger**, wenn sie *verschiedene* Ressourcen nutzen. Praxis: Eine Navigationsanweisung *auditiv* auszugeben, während der Fahrer *visuell* die Straße überwacht, überlastet weniger als sie ins Sichtfeld zu blenden. Multimodale Ausgabe kann Last **verteilen**.

**(b) Der McGurk-Effekt.** Zeigt man das Video einer Person, die *„ga"* artikuliert, spielt aber die Tonspur *„ba"*, hören viele Menschen ein drittes: *„da"*. Das ist der Beweis, dass das Gehirn Audio und Video **nicht getrennt, sondern verschmolzen** verarbeitet — die Wahrnehmung ist bereits fusioniert, bevor sie bewusst wird, und lässt sich nicht „wegwollen". Multimodale Integration ist im Menschen **obligatorisch und automatisch**, nicht optional.

**(c) Redundancy Gain / Redundant Target Effect.** Präsentiert man ein Signal **gleichzeitig** in zwei Modalitäten (Ton *und* Blitz), reagieren Menschen **schneller** als auf jedes einzeln — und zwar schneller, als reine Statistik ("nimm das schnellere der beiden") erlaubt (**race model inequality**, Miller 1982). Die Modalitäten werden also **koaktiv aufsummiert**, nicht bloß parallel abgewartet. Redundanz *nützt*.

**(d) Fehlerreduktion (Oviatt).** In echten Systemen zeigt sich: Menschen wechseln **von selbst** die Modalität, wenn eine versagt (Spracherkennung scheitert am Eigennamen → man buchstabiert oder tippt). Und die Fusion zweier fehlerbehafteter Erkenner ergibt oft **weniger** Fehler als der beste einzelne — **Mutual Disambiguation** (Abschnitt Intermediate). Das ist das ökonomische Kernargument für Multimodalität.

### 4. Ein Steckbrief der wichtigsten Modalitäten

| Modalität | Stärke | typisches Problem |
|---|---|---|
| **Sprache (ASR)** | ausdrucksstark, hände-/augenfrei, gut für abstrakte/benannte Objekte | Erkennungsfehler, Rauschen, Umgebungslärm, keine räumliche Präzision, Datenschutz |
| **Zeigen/Deixis (Geste)** | räumlich präzise, direkt, sprachunabhängig | mehrdeutig ohne Kontext, Ermüdung („gorilla arm"), keine abstrakten Begriffe |
| **Blick (Gaze)** | schnellster Zeiger, „natürlich" bei Aufmerksamkeit | **Midas-Touch-Problem**: der Blick zeigt *immer* irgendwohin — woher weiß das System, wann ein Blick *Absicht* ist und wann nur Umschauen? Braucht ein zweites Signal (Blinzeln, Verweildauer, Sprache) zur *Selektion*. |
| **Haptik (Berührung/Kraft)** | unmittelbares Feedback, hohe zeitliche Auflösung, funktioniert ohne Sicht | begrenzte Informationsdichte, geräteabhängig |
| **Mimik / Prosodie** | emotionaler/pragmatischer Kontext | schwer eindeutig zu interpretieren |

Das **Midas-Touch-Problem** des Blicks ist selbst schon ein Multimodalitäts-Argument: Blick allein taugt kaum zur Selektion, aber *Blick zur groben Vorauswahl* + *Sprache/Klick zur Bestätigung* ist eine hochwirksame Kombination (das ist reliability-based fusion in Aktion — die grobe, schnelle Modalität wird durch die präzise, langsame bestätigt).

### 5. Zwei klassische Klassifikationsschemata: CASE und CARE

Um über multimodale Systeme *präzise* zu reden, braucht man Vokabular. Coutaz & Nigay (1990er) liefern zwei orthogonale Schemata:

**CASE** — beschreibt die **Systemfähigkeit**, Modalitäten *zeitlich* und *fusioniert* zu verarbeiten. Zwei Achsen: *Fusion* (kombiniert / unabhängig) und *Nebenläufigkeit* (sequentiell / parallel):

|                       | **fusioniert (combined)** | **unabhängig (independent)** |
|-----------------------|---------------------------|------------------------------|
| **sequentiell (use one at a time)** | **Alternate** | **Exclusive** |
| **parallel (concurrent)** | **Synergistic** | **Concurrent** |

- **Exclusive**: eine Modalität nach der anderen, keine Fusion (klassisches Menü: erst Objekt anklicken, dann Aktion wählen).
- **Alternate**: nacheinander, aber die Bedeutungen werden verschmolzen („markiere das" *[Pause]* „lösche es").
- **Concurrent**: gleichzeitig, aber getrennte Bedeutungen (mit einer Hand scrollen, mit der Stimme diktieren — unabhängig).
- **Synergistic**: gleichzeitig **und** verschmolzen — *„Put that there"* mit simultanem Zeigen. **Das ist der anspruchsvollste und interessanteste Fall** und Gegenstand der Fusionsmathematik.

**CARE** — beschreibt die **Beziehung mehrerer Modalitäten zu *einem Ziel*** (können sie dieselbe Aufgabe erfüllen?):

- **Complementarity (Komplementarität)**: Die Modalitäten liefern **verschiedene, sich ergänzende** Teile der Bedeutung — keine allein reicht. *„Put **that**"* (Sprache: Aktion) + *Zeigen* (Geste: Objekt). ⇒ Fusion ist **notwendig**.
- **Assignment (Zuweisung)**: Eine bestimmte Modalität ist einer Aufgabe **fest zugeordnet** (Lautstärke *nur* per Regler).
- **Redundancy (Redundanz)**: Mehrere Modalitäten drücken **dasselbe** aus („speichern" sagen *und* auf den Speichern-Button zeigen). ⇒ Fusion erhöht **Sicherheit/Robustheit**, ist aber nicht *nötig*.
- **Equivalence (Äquivalenz)**: Man **kann wählen**, welche Modalität — beide führen zum selben Ziel (Text tippen *oder* diktieren). ⇒ Flexibilität, Barrierefreiheit.

> **Merksatz:** *Complementarity* und *Redundancy* sind die zwei Gründe, überhaupt zu fusionieren — und sie führen zu **verschiedener Mathematik**. Komplementäre Signale werden **verkettet/gemeinsam interpretiert** (jedes trägt einen Teil bei); redundante Signale werden **gemittelt/gewichtet** (jedes ist eine verrauschte Schätzung desselben). Diese Zweiteilung durchzieht das ganze Modul.

---

## Aufbau (Intermediate)

Jetzt zum Herzstück: **Wie** fusioniert man mathematisch? Wir bauen von der allgemeinen Bayes-Regel zur konkreten inverse-Varianz-Gewichtung und zeigen, dass das kein Ingenieurstrick ist, sondern beschreibt, wie das Gehirn arbeitet.

### 6. Die drei Fusionsebenen: early, late, hybrid

Wenn zwei Modalitäten $A$ (z. B. Audio) und $B$ (z. B. Bild) eine gemeinsame Entscheidung stützen sollen, kann die Verschmelzung an verschiedenen Stellen der Verarbeitungskette passieren:

```
        ┌── Feature-Extraktion A ──┐
Signal A┤                          ├─► [FUSION]  ◄─ EARLY (feature-level)
        └──────────────────────────┘      │
                                           ▼
                                     gemeinsames Modell ──► Entscheidung

Signal A ─► Modell A ─► Entscheidung/Konfidenz A ┐
                                                 ├─► [FUSION] ◄─ LATE (decision-level)
Signal B ─► Modell B ─► Entscheidung/Konfidenz B ┘        │
                                                          ▼
                                                     Endentscheidung
```

- **Early Fusion (feature-level)**: Die Feature-Vektoren beider Modalitäten werden **früh zusammengeführt** (z. B. konkateniert: $\mathbf{x} = [\mathbf{x}_A; \mathbf{x}_B]$) und **ein** Modell lernt auf dem gemeinsamen Raum.
  - **Vorteil**: kann **Korrelationen zwischen** den Modalitäten ausnutzen (z. B. Lippenbewegung ↔ Laut). Ein einziges Modell.
  - **Nachteil**: erfordert **zeitliche Synchronisation** und gleiche Abtastrate; **fehlt eine Modalität**, ist der Feature-Vektor unvollständig → das Modell bricht ein. Der gemeinsame Raum ist hochdimensional (Fluch der Dimensionalität, Rückverweis Modul 04).

- **Late Fusion (decision-level)**: Jede Modalität hat ihr **eigenes** Modell; erst die **Entscheidungen/Konfidenzen** werden kombiniert (gewichtete Summe, Produkt, Voting, gelernter Meta-Klassifikator).
  - **Vorteil**: modular; jede Modalität kann getrennt trainiert werden (auch auf verschiedenen Datensätzen); **robust gegen fehlende Modalitäten** (fällt eine aus, gewichtet man die andere hoch); Modelle beliebig heterogen.
  - **Nachteil**: **verwirft die Feinkorrelation** zwischen den Modalitäten — die McGurk-artige gemeinsame Struktur geht verloren.

- **Hybrid / Intermediate Fusion**: Kompromiss — teils gemeinsame, teils getrennte Verarbeitung; im Deep Learning die häufigste Form (getrennte Encoder → **Cross-Modal-Attention** in mittleren Schichten → gemeinsamer Kopf).

**Faustregel**: Sind die Modalitäten **eng gekoppelt und synchron** (Audio+Lippen) → *early* gewinnt. Sind sie **heterogen, asynchron oder oft fehlend** (Sprachbefehl + gelegentliche Geste) → *late* ist robuster. Das **Medium-Projekt** vergleicht beide empirisch und zeigt genau diesen Trade-off inklusive Missing-Modality-Test.

### 7. Probabilistische Late Fusion: die Bayes-Produktregel

Der sauberste Rahmen für late fusion ist Bayes. Sei $y$ die gesuchte Klasse/Bedeutung, und die Modalitäten liefern Beobachtungen $z_A, z_B$. Gesucht ist die Posterior:

$$P(y \mid z_A, z_B) = \frac{P(z_A, z_B \mid y)\, P(y)}{P(z_A, z_B)}.$$

Der entscheidende Schritt ist die **Annahme bedingter Unabhängigkeit** der Modalitäten *gegeben die Klasse* (dieselbe Annahme wie bei Naive Bayes, Modul 04/08):

$$P(z_A, z_B \mid y) = P(z_A \mid y)\, P(z_B \mid y).$$

Damit wird die Posterior zu einem **Produkt der modalitätsspezifischen Likelihoods**:

$$\boxed{\,P(y \mid z_A, z_B) \;\propto\; P(z_A \mid y)\,P(z_B\mid y)\,P(y)\,}$$

Und mit den Einzel-Posteriors $P(y\mid z_A) \propto P(z_A\mid y)P(y)$ lässt sich das als **normalisiertes Produkt der Posteriors** schreiben (bei uniformem oder herausgerechnetem Prior):

$$P(y \mid z_A, z_B) \;\propto\; \frac{P(y\mid z_A)\, P(y \mid z_B)}{P(y)}.$$

Nimmt man den **Logarithmus**, wird das Produkt zur **Summe** — die *log-Konfidenzen addieren sich*:

$$\log P(y\mid z_A,z_B) = \log P(y\mid z_A) + \log P(y\mid z_B) - \log P(y) + \text{const}.$$

> **Das ist die formale Basis von "zwei Meinungen sind besser als eine".** Jede Modalität liefert eine Verteilung über Bedeutungen; Fusion **multipliziert** sie. Wo *beide* dieselbe Bedeutung stützen, verstärkt sich die Wahrscheinlichkeit; wo sie sich widersprechen, dämpft das Produkt. **Achtung — die kritische Annahme ist die bedingte Unabhängigkeit.** Sind die Modalitäten in Wahrheit *korreliert* (z. B. beide Erkenner scheitern am selben Hintergrundlärm), ist das Produkt **überkonfident** und die Fusion kann *schlechter* als die beste Einzelmodalität werden. Diese Falle ist ein zentraler Lernpunkt (Medium- und Final-Projekt).

### 8. Redundante Fusion kontinuierlicher Größen: inverse-Varianz-Gewichtung

Der wichtigste Spezialfall — und die direkte Fortsetzung von **Modul 17** (Komplementärfilter). Zwei Modalitäten schätzen **denselben kontinuierlichen Wert** $s$ (z. B. die Position eines Objekts): eine visuelle Schätzung $\hat{s}_V$ und eine haptische $\hat{s}_H$. Beide sind verrauscht, modelliert als Gauß:

$$\hat{s}_V \sim \mathcal{N}(s, \sigma_V^2), \qquad \hat{s}_H \sim \mathcal{N}(s, \sigma_H^2),$$

mit **Varianzen** (= Unzuverlässigkeit) $\sigma_V^2, \sigma_H^2$. Die inverse Varianz $r = 1/\sigma^2$ heißt **Reliabilität (Präzision)**.

**Frage:** Was ist die *beste* (Maximum-Likelihood-) Schätzung von $s$ aus beiden? Bei unabhängigem Gauß-Rauschen ist die gemeinsame Log-Likelihood

$$\log L(s) = -\frac{(\hat{s}_V - s)^2}{2\sigma_V^2} - \frac{(\hat{s}_H - s)^2}{2\sigma_H^2} + \text{const}.$$

Ableiten nach $s$, gleich null setzen:

$$\frac{\hat{s}_V - s}{\sigma_V^2} + \frac{\hat{s}_H - s}{\sigma_H^2} = 0 \;\Longrightarrow\; s\left(\frac{1}{\sigma_V^2}+\frac{1}{\sigma_H^2}\right) = \frac{\hat{s}_V}{\sigma_V^2}+\frac{\hat{s}_H}{\sigma_H^2}.$$

Damit ergibt sich die **inverse-Varianz-gewichtete Fusion**:

$$\boxed{\;\hat{s} = w_V\,\hat{s}_V + w_H\,\hat{s}_H, \qquad w_V = \frac{1/\sigma_V^2}{1/\sigma_V^2 + 1/\sigma_H^2}, \quad w_H = \frac{1/\sigma_H^2}{1/\sigma_V^2 + 1/\sigma_H^2}\;}$$

Jede Modalität wird **proportional zu ihrer Reliabilität** (= inverser Varianz) gewichtet — die *zuverlässigere* Modalität dominiert. Und die **Varianz der Fusion** ist:

$$\boxed{\;\frac{1}{\sigma_{\text{fus}}^2} = \frac{1}{\sigma_V^2} + \frac{1}{\sigma_H^2}\;} \quad\Longrightarrow\quad \sigma_{\text{fus}}^2 = \frac{\sigma_V^2\,\sigma_H^2}{\sigma_V^2+\sigma_H^2} \le \min(\sigma_V^2, \sigma_H^2).$$

> **Zwei fundamentale Aussagen:**
> 1. **Präzisionen (nicht Varianzen) addieren sich.** Die fusionierte Schätzung ist **immer präziser als jede einzelne** — selbst wenn eine Modalität *deutlich* schlechter ist, verschlechtert sie das Ergebnis nie (sie bekommt nur wenig Gewicht). Das ist der mathematische Gehalt von *Redundancy Gain*.
> 2. Das ist **exakt** der stationäre Kalman-Filter / die Sensorfusion aus Modul 17 — nur dort war es „Gyroskop vs. Beschleunigungsmesser", hier ist es „Sehen vs. Tasten". **Dieselbe Formel.** (Der Komplementärfilter $\theta = \alpha\theta_{\text{Gyro}} + (1-\alpha)\theta_{\text{Accel}}$ ist der Spezialfall mit heuristisch statt aus Varianzen gewähltem $\alpha$.)

### 9. Der Brückenschlag zur Kognition: Ernst & Banks (2002)

Der Grund, warum diese Formel im *Interface*-Kontext so zentral ist: **Das menschliche Gehirn fusioniert Sinne nachweislich genau so.** Ernst & Banks (Nature, 2002) ließen Probanden die Höhe einer Kante **sehen** und **ertasten**. Der Clou des Experiments:

1. **Vorhersage der Gewichte.** Man misst zuerst die *einzelnen* Diskriminationsschwellen (JND) → daraus die Varianzen $\sigma_V^2, \sigma_H^2$. Die Theorie sagt voraus: im kombinierten Fall müsste die wahrgenommene Höhe das **reliability-gewichtete Mittel** sein.
2. **Konflikt-Trials.** Man baut heimlich einen **Konflikt** ein (das Gesehene sagt 55 mm, das Getastete 57 mm). Wohin verschiebt sich die Wahrnehmung? → **exakt zur inverse-Varianz-Vorhersage.** Macht man das Sehen künstlich unzuverlässig (Rauschen im Bild), *wandert das Gewicht messbar zur Haptik* — der wahrgenommene Wert folgt der zuverlässigeren Modalität.
3. **Präzisionsgewinn.** Die kombinierte JND ist **kleiner** als jede einzelne — und zwar um genau den von $\frac{1}{\sigma_{\text{fus}}^2}=\frac{1}{\sigma_V^2}+\frac{1}{\sigma_H^2}$ vorhergesagten Betrag.

**Konsequenz für Interface-Design.** Multimodale Wahrnehmung ist **statistisch optimal** (MLE) und **reliabilitätsadaptiv**. Ein gut gestaltetes multimodales Interface sollte diese Erwartung *bedienen*: konsistente, kalibrierte, sich ergänzende Signale liefern. Widersprüchliche Modalitäten (das Auge sagt A, die Hand sagt B) erzeugen dieselbe Art Konflikt wie im Labor — und das kann, wie in Modul 17 die sensorische Konflikt-Theorie der Cybersickness, **Unbehagen** verursachen. Das **Basic-Projekt** rechnet dieses Experiment vollständig nach.

### 10. Mutual Disambiguation: warum unsicher + unsicher = sicherer

Der ökonomische Kernbefund (Oviatt, 1999) für **klassifikatorische** (statt kontinuierliche) Fusion. Angenommen, die Spracherkennung hört *„send"* oder *„spend"* (unsicher), und die Geste zeigt auf ein E-Mail-Icon (unsicher, könnte auch das daneben sein). Jede Modalität für sich hat eine **N-best-Liste** mit Konfidenzen. Die Bayes-Produktfusion (Abschnitt 7) **multipliziert** diese Listen: nur die Hypothese, die in *beiden* plausibel ist („send" × „E-Mail"), überlebt. Der Fehler der einen Modalität wird durch die andere **herausgefiltert** — das Signal, das beide teilen, verstärkt sich, die idiosynkratischen Fehler löschen sich weg.

Formal: Ist die Fehlerrate von Modalität $A$ gleich $\varepsilon_A$ und von $B$ gleich $\varepsilon_B$, und sind die Fehler **unabhängig**, dann tritt ein *gemeinsamer, unkorrigierbarer* Fehler nur mit $\approx \varepsilon_A \cdot \varepsilon_B$ auf — das **Produkt**, also viel seltener. Oviatt maß empirisch, dass Fusion die Fehlerrate um **19–41 %** gegenüber der besseren Einzelmodalität senkte, besonders für Nutzer mit Akzent oder in lauter Umgebung.

> **Die Falle (unbedingt merken):** Der Gewinn $\varepsilon_A\cdot\varepsilon_B$ gilt **nur bei unabhängigen Fehlern**. Sind die Fehler **korreliert** (beide Modalitäten scheitern an denselben schwierigen Fällen — z. B. ein neuer Nutzer, den *beide* Modelle schlecht kennen), dann ist der gemeinsame Fehler viel größer als das Produkt, und Fusion bringt wenig bis nichts. **Mutual Disambiguation lebt von *Diversität* der Fehler.** Das ist dieselbe Einsicht wie bei Ensemble-Methoden (Modul 04/05): unkorrelierte schwache Lerner ergänzen sich, korrelierte nicht. Das **Medium-Projekt** demonstriert beide Seiten dieser Medaille.

### 11. Zeitliche Fusion: das temporal binding window

Bisher taten wir so, als kämen $z_A, z_B$ gleichzeitig. In der Realität sind es **asynchrone Ereignisströme** mit Zeitstempeln. Wann „gehören" ein Sprach-Event und ein Gesten-Event zusammen?

**Wahrnehmungsseitig** gibt es das **temporal binding window**: ein Zeitfenster (beim Menschen grob **±100–300 ms**, je nach Modalitätspaar), innerhalb dessen zwei Ereignisse als *ein* Ereignis wahrgenommen werden. Außerhalb: getrennt. Dies erklärt auch, warum leichte Audio-Video-Desynchronisation im Film unbemerkt bleibt, starke aber stört.

**Systemseitig** braucht man einen **Alignment-Mechanismus**. Die einfachste Form ist ein **Zeitfenster-Match**: Ereignis $a$ (Zeitstempel $t_a$) und $b$ (Zeitstempel $t_b$) werden fusioniert, wenn $|t_a - t_b| \le \Delta$. Besser als eine harte Schwelle ist eine **weiche, probabilistische** Formulierung — ein zeitlicher Kompatibilitäts-Score, z. B. gaußförmig:

$$P_{\text{temp}}(a \leftrightarrow b) = \exp\!\left(-\frac{(t_a - t_b - \mu)^2}{2\tau^2}\right),$$

wobei $\mu$ den typischen **Versatz** modelliert (empirisch bei „Put-that-there" zeigt die **Geste leicht *vor* oder um** das deiktische Wort — $\mu \neq 0$!) und $\tau$ die Fensterbreite. Dieser zeitliche Score wird dann **mit** dem semantischen Konfidenz-Score **multipliziert** (wieder Bayes-Produkt: räumlich-semantische × zeitliche Plausibilität). Das **Final-Projekt** implementiert genau diese zeitgewichtete Referenzauflösung.

**Multimodal Turn-Taking / Endpointing.** Verwandtes Problem: Wann ist eine Äußerung *fertig*? Ein rein sprachbasiertes System nutzt Stille (Endpointing), doch Menschen signalisieren das Ende auch über **Blick** (Blickabwendung → „ich bin fertig, du bist dran") und Gestik. Multimodale Turn-Taking-Modelle fusionieren diese Hinweise, um flüssigere Dialoge zu erreichen — ein aktuelles Forschungsthema bei Sprachassistenten.

### 12. Referenzauflösung: „that" mit einem Zeigeziel verbinden

Die konkrete Aufgabe aus „Put-that-there". Gegeben: ein deiktisches Wort mit Zeitstempel $t_{\text{word}}$, und eine Menge von Objekten mit ihren Bildschirmpositionen; dazu ein kontinuierlicher **Zeigestrahl** (Gestenrichtung über die Zeit). Gesucht: das gemeinte Objekt. Man kombiniert **drei** Wahrscheinlichkeiten multiplikativ:

$$P(\text{obj} = o \mid \ldots) \;\propto\; \underbrace{P_{\text{sem}}(o)}_{\text{passt der Typ zum Verb?}} \cdot \underbrace{P_{\text{point}}(o)}_{\text{zeigt die Geste dorthin?}} \cdot \underbrace{P_{\text{temp}}(o)}_{\text{fällt Zeigen \& Wort zusammen?}}$$

- $P_{\text{point}}(o)$ modelliert man z. B. als Gauß um den Zeigestrahl (Winkelabstand Objekt↔Strahl), Breite = Zeigepräzision.
- $P_{\text{temp}}(o)$ ist der Zeitfenster-Score aus Abschnitt 11, ausgewertet zum Zeitpunkt, zu dem die Geste am nächsten an $o$ war.
- $P_{\text{sem}}(o)$ filtert nach Verträglichkeit (auf ein „lösche das" folgend sind nur löschbare Objekte plausibel).

Das gemeinte Objekt ist $\arg\max_o P(\text{obj}=o\mid\ldots)$. Das Schöne: Ist der Zeigestrahl **mehrdeutig** (zwei Objekte fast gleich nah), entscheidet der **zeitliche** und **semantische** Faktor — Mutual Disambiguation in Reinform. Genau das baut das **Final-Projekt** als vollständigen Multimodal-Interpreter.

---

## Advanced-Themen

### 13. Fusion als probabilistisches grafisches Modell

Die Late-Fusion-Produktregel ist ein Spezialfall eines **Bayes-Netzes** (Rückverweis Modul 07). Klasse $y$ als verborgene Wurzel, Modalitäten $z_A, z_B$ als bedingt unabhängige Kinder:

```
        (y)
       /   \
    (z_A) (z_B)      →  P(y,z_A,z_B) = P(y)P(z_A|y)P(z_B|y)   (Naive-Bayes-Struktur)
```

Der Vorteil dieser Sicht: Sie macht die **Modellannahmen explizit** und erlaubt Erweiterungen:

- **Korrelierte Modalitäten** modelliert man durch eine zusätzliche Kante $z_A \to z_B$ oder eine gemeinsame verborgene „Umgebungs"-Variable (z. B. Lärmpegel), die beide beeinflusst. Das **repariert** die Überkonfidenz aus Abschnitt 7.
- **Zeitliche Dynamik**: Erweitert man das Netz über die Zeit, erhält man ein **Dynamisches Bayes-Netz (DBN)**. Zwei modalitätsspezifische **HMMs** (Rückverweis Modul 07/08), deren verborgene Zustände sich gegenseitig beeinflussen, heißen **Coupled HMM** — das klassische Modell für audiovisuelle Spracherkennung (Lippen + Ton), bevor Deep Learning kam.
- **Kontinuierlich**: Das lineare Gauß-DBN *ist* der **Kalman-Filter**; die Mehr-Sensor-Variante ist genau die inverse-Varianz-Fusion aus Abschnitt 8, rekursiv über die Zeit.

### 14. Deep Multimodal Learning

Die moderne (lernbasierte) Sicht. Baltrušaitis et al. (2019) ordnen das Feld in **fünf Kernprobleme**, die man kennen sollte:

1. **Representation** — wie repräsentiert man mehrere Modalitäten gemeinsam?
   - **Joint representation**: alle Modalitäten in *einen* gemeinsamen Raum projizieren und dort verschmelzen (early-fusion-artig). Gut, wenn zur Inferenzzeit *alle* Modalitäten da sind.
   - **Coordinated representation**: jede Modalität behält ihren *eigenen* Raum, aber die Räume werden **koordiniert** (z. B. durch eine Ähnlichkeits-/Kontrastiv-Loss so ausgerichtet, dass zusammengehörige Bild- und Text-Paare nah beieinander liegen). **CLIP** ist das prominente Beispiel: zwei Encoder (Bild, Text), trainiert mit **contrastive loss**, sodass Bild und passende Bildunterschrift ähnliche Vektoren bekommen. Erlaubt *cross-modal retrieval* und *zero-shot*-Klassifikation.
2. **Translation** — eine Modalität in eine andere übersetzen (Bild → Bildunterschrift, Text → Sprache). Baut auf seq2seq/Transformer aus Modul 09.
3. **Alignment** — Teilelemente zweier Modalitäten einander zuordnen (welches Wort gehört zu welchem Bildbereich?). Genau das leistet **Cross-Modal Attention**: Die Attention-Matrix aus Modul 09 wird *zwischen* zwei Modalitäten aufgespannt (Query aus Text, Key/Value aus Bild) — das ist das architektonische Herz moderner Vision-Language-Modelle.
4. **Fusion** — die Entscheidung selbst (early/late/hybrid, wie oben, jetzt im neuronalen Kontext).
5. **Co-Learning** — Wissen von einer (datenreichen) Modalität nutzen, um eine andere (datenarme) besser zu lernen.

**Modality Dropout & fehlende Modalitäten.** Ein praktisches Deep-Learning-Rezept: Beim Training zufällig **ganze Modalitäten weglassen** (wie Dropout aus Modul 05, nur auf Modalitätsebene). Das zwingt das Netz, nicht von einer einzelnen Modalität abhängig zu werden, und macht es **robust gegen Ausfall** zur Testzeit — die neuronale Entsprechung des Late-Fusion-Robustheitsarguments aus Abschnitt 6.

> **Praxis-Hinweis (im Sinne der Modul-Regel):** Ein CLIP-Modell oder ein Cross-Modal-Transformer wird in der Praxis auf GPU-Clustern mit Millionen Paaren trainiert — das ist auf einem Laptop weder machbar noch nötig. Wir verstehen die **Architektur und die Loss-Funktion** theoretisch (contrastive loss, cross-attention aus Modul 09) und demonstrieren die *Prinzipien* — Fusion, Mutual Disambiguation, Robustheit — an kleinen, exakt kontrollierten Modellen, wo man den Effekt **sauber isoliert messen** kann. Genau das tun die drei Projekte.

### 15. Multimodale Ausgabe (Fission) und Barrierefreiheit

Bisher ging es um **Fusion** (Eingabe verschmelzen). Das Gegenstück ist **Fission**: eine Systemnachricht *auf* mehrere Ausgabemodalitäten *verteilen* (Text + Sprache + Vibration). Design-Fragen: Welche Information in welche Modalität (nach *Multiple Resource Theory*, Abschnitt 3)? Redundant (dieselbe Info mehrfach → Robustheit) oder komplementär (Karte visuell + Richtung auditiv)?

**Barrierefreiheit (Universal Design)** ist ein Hauptargument für **Äquivalenz** (CARE): Bietet ein System mehrere *äquivalente* Modalitäten, kann jeder die für sich passende wählen — Untertitel für Gehörlose, Screenreader/Audio für Blinde, Sprachsteuerung für motorisch Eingeschränkte. Multimodalität ist damit nicht nur „natürlicher", sondern **inklusiver**. Ein gutes multimodales System **erzwingt** keine Modalität, sondern **erlaubt** sie.

### 16. Evaluation multimodaler Systeme — und ihre Fallen

Wie in Modul 17 gilt: Ein System zu bauen ist die halbe Miete, es *sauber zu bewerten* die andere. Besonderheiten und Fallen bei multimodalen Systemen:

- **Der richtige Vergleichsmaßstab.** Multimodale Fusion muss sich an der **besten Einzelmodalität** messen, nicht am Durchschnitt. „Fusion schlägt Modalität A" ist wertlos, wenn Modalität B allein schon besser war.
- **Korrelation zerstört den erwarteten Gewinn.** (Abschnitte 7, 10.) Man muss die **Fehlerkorrelation** der Modalitäten *messen*, nicht Unabhängigkeit *annehmen*. Ein scheinbar enttäuschendes Fusionsergebnis ist oft nur korrelierte Redundanz.
- **Konfidenz-Kalibrierung.** Late-Fusion per Produkt setzt voraus, dass die Konfidenzen *kalibriert* sind (eine 0.9 heißt wirklich 90 %). Ein überkonfidenter Erkenner **überstimmt** in der Produktfusion fälschlich den anderen. Kalibrierung (Modul 04/15: Reliability-Diagramme) ist Voraussetzung.
- **Fehlende Modalitäten testen.** Robustheit gegen Ausfall ist ein Hauptversprechen von late fusion — also *muss* man den Ausfall-Fall messen (eine Modalität abschalten und die Genauigkeit prüfen).
- **Nutzungsverhalten.** Auf Nutzerseite (mit echten Probanden, Methodik wie Modul 17): Wie *oft* nutzen Menschen überhaupt beide Modalitäten gleichzeitig (synergistisch)? Oviatts Befund: seltener als erwartet — Menschen sind oft sequentiell und **integrieren nur bei Bedarf** (schwierige Objekte, Fehlerbehebung). Ein System darf **nicht** erzwingen, dass immer beide Modalitäten kommen.

Diese Fallen sind kein Beiwerk, sondern der Kern der Master-Kompetenz: **Fusion ist kein Gratis-Gewinn.** Sie gewinnt unter *unabhängigen, kalibrierten, komplementären* Signalen und kann sonst verlieren. Alle drei Projekte machen mindestens einen dieser Effekte explizit messbar.

---

## Zusammenfassung / Cheat-Sheet

**Begriffe**
- **Modalität**: Wahrnehmungs-/Ausdruckskanal (Sehen, Hören, Sprechen, Zeigen, Tasten). ≠ Gerät. „Multimodal" = mehrere *Modalitäten*, koordiniert.
- **CASE** (Systemfähigkeit): Exclusive · Alternate · Concurrent · **Synergistic** (parallel + fusioniert = der harte Fall).
- **CARE** (Modalitätsbeziehung): **Complementarity** (ergänzend → verketten) · Assignment · **Redundancy** (gleich → gewichten/mitteln) · **Equivalence** (wählbar → Barrierefreiheit).
- **Midas-Touch-Problem**: Blick zeigt immer irgendwohin → braucht zweites Signal zur Selektion.
- **McGurk / Redundancy Gain / Multiple Resource Theory**: menschliche Integration ist obligatorisch, beschleunigend, ressourcenverteilend.

**Die zentralen Formeln**

| Was | Formel |
|---|---|
| Late Fusion (Bayes, bed. Unabh.) | $P(y\mid z_A,z_B) \propto P(z_A\mid y)P(z_B\mid y)P(y)$ |
| … in Log-Form | $\log$-Konfidenzen **addieren** sich |
| Inverse-Varianz-Gewicht | $w_i = \dfrac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$ |
| Fusions-Präzision | $\dfrac{1}{\sigma_{\text{fus}}^2}=\sum_i \dfrac{1}{\sigma_i^2}$ → **immer** $\le \min_i \sigma_i^2$ |
| Zeitliche Kompatibilität | $P_{\text{temp}} = \exp\!\big(-(t_a-t_b-\mu)^2/2\tau^2\big)$ |
| Referenzauflösung | $P(o)\propto P_{\text{sem}}(o)\,P_{\text{point}}(o)\,P_{\text{temp}}(o)$ |
| Mutual Disambiguation | gemeinsamer Fehler $\approx \varepsilon_A\varepsilon_B$ — **nur bei unabhängigen Fehlern!** |

**Fusionsebenen**: early (feature, nutzt Korrelation, braucht Sync, bricht bei fehlender Modalität) · late (decision, modular, robust, verwirft Feinkorrelation) · hybrid (DL-Standard: Encoder + Cross-Attention).

**Die drei goldenen Warnungen**
1. Fusion gewinnt nur bei **unabhängigen** Fehlern/Signalen — Korrelation macht sie überkonfident oder nutzlos.
2. Late-Fusion braucht **kalibrierte** Konfidenzen.
3. Miss immer gegen die **beste Einzelmodalität** und teste **fehlende Modalitäten**.

**Roter Faden zu Modul 17**: Der Komplementärfilter (Gyro+Accel) ist der heuristische Vorläufer der inverse-Varianz-Fusion; Ernst & Banks zeigen, dass das Gehirn *exakt* diese optimale Fusion durchführt.

---

## Selbsttest

<details>
<summary><b>1.</b> Ein System hat einen Touchscreen und eine Maus. Ist es multimodal? Begründe mit der Modalitäts-Definition.</summary>

**Nein.** Touchscreen und Maus realisieren **dieselbe Modalität** — 2D-Zeigen/Selektion —, nur mit verschiedenen *Geräten*. Multimodal wäre es erst mit **verschiedenen Modalitäten**, z. B. Zeigen **plus Sprache** oder **plus Blick**. Der Fehler ist, „multimodal" mit „mehrere Eingabegeräte" zu verwechseln.
</details>

<details>
<summary><b>2.</b> Ordne „Put-that-there" (simultanes Sprechen + Zeigen) in CASE und CARE ein.</summary>

**CASE: Synergistic** — die Modalitäten kommen *parallel* (gleichzeitig) **und** werden *fusioniert*. Das ist der anspruchsvollste CASE-Quadrant.

**CARE: Complementarity** — die Sprache liefert die *Aktion* („put"), die Geste das *Objekt/Ziel* (das Zeigen); **keine allein** genügt, sie ergänzen sich. (Nicht Redundancy — sie sagen nicht dasselbe.)
</details>

<details>
<summary><b>3.</b> Zwei Sensoren schätzen eine Position: $\sigma_A = 2$ mm, $\sigma_B = 4$ mm. Berechne die Fusionsgewichte und die Fusionsvarianz. Ist die Fusion präziser als der bessere Sensor allein?</summary>

Reliabilitäten: $1/\sigma_A^2 = 1/4 = 0{,}25$, $1/\sigma_B^2 = 1/16 = 0{,}0625$. Summe $= 0{,}3125$.

Gewichte: $w_A = 0{,}25/0{,}3125 = 0{,}8$, $w_B = 0{,}0625/0{,}3125 = 0{,}2$. Der bessere Sensor bekommt 80 % Gewicht.

Fusionsvarianz: $\sigma_{\text{fus}}^2 = 1/0{,}3125 = 3{,}2\,\text{mm}^2$, also $\sigma_{\text{fus}} \approx 1{,}79$ mm.

**Ja** — $1{,}79 < 2 = \sigma_A$. Die Fusion ist präziser als *jeder* einzelne Sensor. Genau das ist Redundancy Gain / die Aussage $\sigma_{\text{fus}}^2 \le \min(\sigma_A^2,\sigma_B^2)$.
</details>

<details>
<summary><b>4.</b> Was sagt das Experiment von Ernst & Banks (2002) für das Interface-Design aus?</summary>

Dass menschliche multimodale Wahrnehmung **statistisch optimal** ist: Das Gehirn kombiniert Sehen und Tasten per **inverse-Varianz-Gewichtung** (MLE) und verschiebt das Gewicht **adaptiv** zur zuverlässigeren Modalität (macht man das Sehen verrauscht, dominiert das Tasten). Für Design heißt das: Nutzer *erwarten* konsistente, kalibrierte, sich ergänzende Signale; **widersprüchliche** Modalitäten erzeugen einen sensorischen Konflikt (wie bei der Cybersickness in Modul 17) und stören.
</details>

<details>
<summary><b>5.</b> Warum kann die Bayes-Produktfusion zweier Erkenner *schlechter* sein als der beste einzelne? Nenne die Bedingung und die Ursache.</summary>

Weil die Produktregel **bedingte Unabhängigkeit** der Modalitäten voraussetzt. Sind die Fehler in Wahrheit **korreliert** (beide scheitern an denselben schwierigen Fällen — gleicher Lärm, gleicher unbekannter Nutzer), ist das Produkt **überkonfident**: Es behandelt zwei abhängige Stimmen, als wären es zwei unabhängige, und verstärkt so den gemeinsamen Fehler. Mutual Disambiguation (Fehlergewinn $\approx \varepsilon_A\varepsilon_B$) gilt nur bei **unabhängigen** Fehlern; Fusion lebt von **Diversität** der Fehler — dieselbe Logik wie bei Ensembles (Modul 04/05).
</details>

<details>
<summary><b>6.</b> Wann wählst du early, wann late fusion?</summary>

**Early** (Feature-Konkatenation, ein Modell), wenn die Modalitäten **eng gekoppelt, synchron und fein korreliert** sind (Audio + Lippenbild) — dann nutzt man die gemeinsame Struktur. **Late** (getrennte Modelle, Entscheidungen kombinieren), wenn die Modalitäten **heterogen, asynchron oder oft fehlend** sind — late fusion ist **modular** und **robust gegen Modalitätsausfall** (fällt eine weg, gewichtet man die andere hoch), verwirft aber die Feinkorrelation. Kompromiss: hybrid/Cross-Attention.
</details>

<details>
<summary><b>7.</b> Was ist das Midas-Touch-Problem, und wie löst Multimodalität es?</summary>

Der **Blick zeigt permanent irgendwohin** — auch beim bloßen Umschauen. Ein blickgesteuertes Interface kann daher nicht wissen, *wann* ein Blick eine **Selektionsabsicht** ist und wann nur Wahrnehmung. Lösung: ein **zweites Signal** zur Bestätigung — Sprache („das da"), ein Knopf, langes Verweilen (dwell) oder Blinzeln. Der Blick liefert schnell die grobe Vorauswahl, die zweite Modalität den präzisen Auslöser: klassische **komplementäre** Fusion.
</details>

<details>
<summary><b>8.</b> Bei der zeitlichen Fusion in „Put-that-there": Warum setzt man den Versatz $\mu$ im Zeitfenster-Score nicht auf null?</summary>

Weil Zeigen und Sprechen **empirisch nicht exakt zusammenfallen**: Die Geste (das Zeigen) beginnt typischerweise **leicht vor** dem deiktischen Wort „that" (die Hand ist schon am Ziel, wenn das Wort fällt). Ein Modell mit $\mu = 0$ würde diesen systematischen Versatz als „Fehlpaarung" bestrafen. Mit $\mu \neq 0$ (dem gemessenen typischen Versatz) und Fensterbreite $\tau$ trifft der zeitliche Kompatibilitäts-Score $\exp(-(t_a-t_b-\mu)^2/2\tau^2)$ die reale Koordination.
</details>

<details>
<summary><b>9.</b> Warum ist Multimodalität ein Barrierefreiheits-Argument? Welche CARE-Eigenschaft ist relevant?</summary>

**Equivalence**: Bietet ein System mehrere *äquivalente* Modalitäten für dieselbe Aufgabe (tippen ODER diktieren, sehen ODER hören), kann jeder Nutzer die für sich zugängliche wählen — Untertitel für Gehörlose, Audio/Screenreader für Blinde, Sprache für motorisch Eingeschränkte. Ein gutes multimodales System **erlaubt** Modalitäten, statt eine zu **erzwingen** (Universal Design).
</details>

<details>
<summary><b>10.</b> Du fusionierst zwei Klassifikatoren per Konfidenz-Produkt, und einer ist chronisch überkonfident (gibt fast immer 0.99). Was passiert und was ist die Abhilfe?</summary>

Der überkonfidente Erkenner **dominiert** die Produktfusion und **überstimmt** den anderen fälschlich, selbst wenn er falsch liegt — die Produktregel vertraut seiner (unverdienten) 0.99. Abhilfe: **Konfidenz-Kalibrierung** (Reliability-Diagramme, Temperature Scaling, Platt-Scaling aus Modul 04/15), damit eine ausgegebene 0.9 tatsächlich 90 % Trefferwahrscheinlichkeit bedeutet. Late-Fusion per Produkt ist nur so gut wie die **Kalibrierung** seiner Eingänge.
</details>

---

## Literatur & Quellen

**Lehrbücher / Übersichten**
- **Sharon Oviatt, „Multimodal Interfaces"**, in: *The Human-Computer Interaction Handbook* (Jacko & Sears, Hrsg.). Der kanonische Übersichtstext — CASE/CARE, Mutual Disambiguation, Nutzungsmuster. *Einsteigerfreundlich, Pflichtlektüre.*
- **Oviatt & Cohen, *The Paradigm Shift to Multimodality in Contemporary Computer Interfaces*** (Morgan & Claypool, 2015). Buchlange, gut lesbare Vertiefung. *Vertiefend.*
- **Baltrušaitis, Ahuja & Morency, „Multimodal Machine Learning: A Survey and Taxonomy"**, *IEEE TPAMI* 2019. Die Standard-Taxonomie (representation/translation/alignment/fusion/co-learning) für die ML-Seite. *Frei als arXiv:1705.09406. Vertiefend.*

**Schlüssel-Papers (frei auffindbar)**
- **Bolt, „Put-that-there: Voice and gesture at the graphics interface"**, *SIGGRAPH 1980*. Das Gründungspaper. Kurz, lesenswert. *Einsteigerfreundlich.*
- **Ernst & Banks, „Humans integrate visual and haptic information in a statistically optimal fashion"**, *Nature* 2002. Der Beweis der inverse-Varianz-Fusion im Menschen. *Kompakt, sehr wirkungsvoll — unbedingt lesen.*
- **Oviatt, „Ten myths of multimodal interaction"**, *Communications of the ACM* 1999. Räumt mit verbreiteten Irrtümern auf (u. a. „Nutzer kombinieren immer alles gleichzeitig"). *Einsteigerfreundlich, unterhaltsam.*
- **Miller, „Divided attention: Evidence for coactivation with redundant signals"**, *Cognitive Psychology* 1982. Das Race-Model / Redundancy-Gain-Paper. *Vertiefend.*
- **McGurk & MacDonald, „Hearing lips and seeing voices"**, *Nature* 1976. Der McGurk-Effekt. *Kurz, klassisch.*
- **Radford et al., „Learning Transferable Visual Models From Natural Language Supervision" (CLIP)**, 2021 (arXiv:2103.00020). Contrastive coordinated representation. *Vertiefend, für die Deep-Learning-Seite.*

**Frei verfügbare Kurse / Materialien**
- **CMU 11-777 „Multimodal Machine Learning"** (Louis-Philippe Morency) — Vorlesungsvideos + Folien frei online. Die maßgebliche Vorlesung zur ML-Seite dieses Moduls. *Kostenlos, vertiefend.*
- **Kevin Murphy, *Probabilistic Machine Learning*** (frei als PDF, probml.github.io) — Kapitel zu Bayes, Gauß-Fusion, grafischen Modellen unterfüttern Abschnitte 7–13. *Kostenlos, vertiefend.*

**Interaktiv / zum Ausprobieren**
- Zahlreiche **McGurk-Effekt-Demovideos** (BBC/YouTube) — man *fühlt* die obligatorische Fusion am eigenen Kopf. *Kostenlos.*
- Die **drei Projekte dieses Moduls** rechnen Ernst-Banks-Fusion, Mutual Disambiguation und die „Put-that-there"-Referenzauflösung selbst nach — die beste Vertiefung ist, sie zu bauen.

---

> **Nächstes Modul im XR-Block:** Modul 19 „3D User Interfaces" — Interaktion im dreidimensionalen Raum (Selektion/Manipulation/Navigation in 3D, Zeigetechniken wie Ray-Casting/Go-Go aus Modul 17 vertieft, Bezugssysteme). Die Referenzauflösung und Zeigemathematik aus diesem Modul (Abschnitt 12) sind direkte Vorarbeit.
