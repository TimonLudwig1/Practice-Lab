# Modul 15 — Machine Learning for Networks 1

> **Worum geht es?** Ein Kommunikationsnetz ist eine der größten Datenquellen der Informatik:
> jede Sekunde entstehen Millionen Pakete, Flows und Messwerte. Dieses Modul wendet **Machine
> Learning auf Kommunikationsnetze** an — Verkehr **klassifizieren**, **Angriffe erkennen**,
> **Dienstgüte (QoS/QoE) vorhersagen**, **Last prognostizieren**. Der Reiz liegt nicht in
> exotischen Modellen (es sind meist dieselben wie in Modul 04/05), sondern in den **brutalen
> Eigenheiten der Domäne**: extreme Klassenungleichgewichte, nicht-stationärer Verkehr,
> Verschlüsselung, Line-Rate-Anforderungen — und einer statistischen Falle, die in der Praxis
> ganze Security-Produkte unbrauchbar macht: der **Base-Rate-Fallacy**.

**Hilfreiche Vorkenntnisse:** Klassifikation/Regression, Pipelines, Kreuzvalidierung, Metriken
(Modul 04/05); Grundbegriffe Rechnernetze (IP, TCP/UDP, Ports) sind nützlich, werden aber hier
eingeführt.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 04 (ML 1)** — Klassifikation, Pipelines, CV, GridSearch, **Schwellenwahl & Kosten**
  (das Adult-Projekt mit Kosten-Schwelle ist hier die direkte Vorlage);
- **Modul 05 (ML 2)** — Ensembles/Netze, Clustering & **unüberwachte** Verfahren (für
  Anomalieerkennung im Finalprojekt);
- **Modul 02/03 (Data Science)** — EDA, Datenbereinigung, Umgang mit realen, schmutzigen Daten.
  *(RL aus Modul 13/14 wird hier **nicht** gebraucht.)*

> **Hinweis zur inhaltlichen Ausgestaltung.** Für dieses Modul lag keine offizielle
> Modulbeschreibung vor. Ich habe es entlang der Würzburger Verortung (Lehrstuhl für
> **Kommunikationsnetze**) und der internationalen Standardliteratur zugeschnitten: „Networks"
> meint hier **Kommunikationsnetze**, *nicht* neuronale Netze und *nicht* primär Graph-Learning.
> Modul 16 (ML for Networks 2) vertieft später Richtung Graph-basierte Verfahren (GNN),
> Netzwerk-Zeitreihen und selbstlernende Netze.

---

## Lernziele

Nach diesem Modul kannst du …

- **Netzwerkdaten** einordnen: Paket- vs. **Flow**-Ebene, NetFlow/IPFIX, aktive vs. passive
  Messung, **Sampling**, und die daraus ableitbaren **Features**;
- die Entwicklung der **Traffic-Klassifikation** erklären: Port-basiert → **DPI** →
  statistisches ML → **verschlüsselter** Verkehr;
- **Intrusion/Anomaly Detection** als ML-Problem formulieren — überwacht, unüberwacht,
  **semi-überwacht** (nur Normalverkehr) — und den Unterschied zwischen *Missbrauchs-* und
  *Anomalieerkennung* benennen;
- mit **extremem Klassenungleichgewicht** korrekt umgehen und begründen, warum **Accuracy**
  und oft auch die **ROC-Kurve** hier täuschen (→ **PR-Kurve**);
- den **Base-Rate-Fallacy** (Axelsson) **quantitativ** herleiten und ausrechnen, warum ein
  „99,9 % genauer" Detektor im Betrieb an Fehlalarmen erstickt;
- **Concept Drift**/Nichtstationarität, **Deployment**-Randbedingungen (Line-Rate, Latenz),
  **adversariale** Evasion und **Datenschutz** als Praxisprobleme erklären;
- Netzwerk-Datensätze **kritisch** bewerten (warum KDD Cup 99 berühmt-berüchtigt ist).

---

## 1 · Grundlagen — Netzwerkdaten

### 1.1 Warum überhaupt ML im Netz?

Klassische Netzbetriebs-Werkzeuge sind **regelbasiert**: feste Portnummern, Signaturen,
Schwellwerte. Das bricht aus drei Gründen:

1. **Verschlüsselung.** Über 90 % des Web-Verkehrs ist heute TLS. In den Nutzdaten steht nichts
   mehr Lesbares → signaturbasierte **Deep Packet Inspection (DPI)** läuft ins Leere. Übrig
   bleiben **Metadaten** (Paketgrößen, Timing, Richtungen) — und die auszuwerten ist ein
   **statistisches** Problem, also ML.
2. **Skalierung & Dynamik.** Anwendungen wechseln Ports, tunneln über 443, verändern sich
   wöchentlich. Handgepflegte Regeln veralten schneller, als man sie schreibt.
3. **Unbekannte Angriffe.** Signaturen erkennen nur, was man schon kennt. **Zero-Days**
   erfordern, *Normalität* zu modellieren und Abweichungen zu melden.

### 1.2 Pakete, Flows, und was man messen kann

Netzwerkdaten gibt es auf mehreren Granularitätsstufen:

- **Paket-Ebene** (`pcap` via tcpdump/Wireshark): jedes einzelne Paket mit Header + ggf. Payload.
  Maximaler Informationsgehalt, aber **riesig** (10 Gbit/s ⇒ ~GB/s) und datenschutzkritisch.
- **Flow-Ebene** (**NetFlow**/**IPFIX**, sFlow): Pakete werden zu **Flows** aggregiert. Ein Flow
  ist klassisch das **5-Tupel**
  $$(\text{Quell-IP},\ \text{Ziel-IP},\ \text{Quell-Port},\ \text{Ziel-Port},\ \text{Protokoll})$$
  plus Zeitfenster. Pro Flow speichert man Aggregate: Dauer, Paketzahl, Bytes, Flags, …
  **Das ist der Sweet Spot für ML**: kompakt genug für Line-Rate, informativ genug zum Lernen,
  und ohne Payload deutlich datenschutzfreundlicher. *Genau auf dieser Ebene arbeiten alle drei
  Projekte.*
- **Aggregierte Zeitreihen** (SNMP-Zähler, Link-Auslastung pro 5 min): für Forecasting/Kapazitäts-
  planung.

**Messung** ist entweder **passiv** (mithören, z. B. an einem Router-Mirror-Port) oder **aktiv**
(selbst Pakete senden: `ping`, `traceroute`, Speedtests — verändert das Netz, das man misst).

**Sampling.** Bei hohen Raten kann man nicht jedes Paket erfassen; man nimmt z. B. jedes 1000-ste
(1:1000). Konsequenz für ML, die oft übersehen wird: **kleine Flows verschwinden** fast
vollständig (ein 3-Paket-Portscan wird bei 1:1000 mit Wahrscheinlichkeit ≈ 99,7 % nie gesehen),
während Elefanten-Flows überleben. Sampling **verzerrt** die Verteilung systematisch — und zwar
genau gegen die seltenen, sicherheitsrelevanten Ereignisse.

### 1.3 Typische Flow-Features

Aus einem Flow-Record lassen sich ableiten:

| Gruppe | Beispiele |
|---|---|
| **Volumen** | Bytes gesamt, Pakete gesamt, mittlere Paketgröße, Bytes je Richtung |
| **Zeit** | Dauer, Inter-Arrival-Times (Mittel/Std/Min/Max), Bytes pro Sekunde |
| **Richtung/Symmetrie** | Verhältnis Up-/Downstream, Anzahl Richtungswechsel |
| **Protokoll/Header** | Protokoll, TCP-Flags (SYN/FIN/RST), TTL, Fenstergröße |
| **Kontext/Host** | Anzahl Verbindungen derselben Quelle im Zeitfenster, Zahl verschiedener Ziel-Ports |

Die **Kontext-Features** sind oft die wertvollsten: ein einzelner SYN ist harmlos, aber „200 SYNs
an 200 verschiedene Ports desselben Hosts in 2 Sekunden" ist ein Portscan. **Angriffe sind
häufig erst im Aggregat sichtbar, nicht im Einzelereignis.**

### 1.4 Die Verteilungen sind *nicht* gutartig

Netzwerkverkehr verletzt fast alle bequemen Annahmen:

- **Schwere Ränder (heavy tails).** Flow-Größen sind extrem schief: wenige **Elefanten-Flows**
  tragen den Großteil der Bytes, Millionen **Mäuse-Flows** den Großteil der Flows. Mittelwerte
  sind hier fast bedeutungslos → **Log-Transformation** von Byte-/Paketzählern ist quasi Pflicht.
- **Nicht-stationär.** Tag/Nacht, Werktag/Wochenende, neue Apps → die Verteilung driftet
  permanent (**Concept Drift**, Abschnitt 3.3).
- **Nicht i.i.d.** Pakete eines Flows und Flows eines Hosts sind stark korreliert. Naiv
  zufälliges Splitten in Train/Test **leakt** deshalb massiv (Abschnitt 3.5).

---

## 2 · Aufbau — Die vier Kernaufgaben

### 2.1 Traffic-Klassifikation

**Frage:** Welche Anwendung/Dienstklasse erzeugt diesen Flow (Video, VoIP, Web, Gaming,
Filesharting)? **Wozu:** QoS-Priorisierung, Kapazitätsplanung, Abrechnung, Policy-Durchsetzung.

Die historische Entwicklung ist selbst die Lektion:

1. **Port-basiert** (bis ~2000): Port 80 = HTTP. **Tot** — alles läuft heute über 443, P2P nutzt
   dynamische Ports.
2. **DPI/Signaturen** (~2000–2010): in die Nutzdaten schauen. **Tot durch Verschlüsselung**
   (und rechtlich/datenschutzrechtlich heikel).
3. **Statistisches ML auf Flow-Features** (heute): klassifiziere anhand von *Größen und Timing*,
   nicht Inhalt. Funktioniert **auch bei TLS**, weil das Verschlüsseln zwar den Inhalt verbirgt,
   aber das **Muster** (Paketgrößen-Sequenz, Burstiness) kaum verändert.
4. **Deep Learning auf rohen Byte-/Paketsequenzen** (aktuell): CNNs/Transformer lernen die
   Features selbst.

> **Der zentrale Aha-Punkt:** Verschlüsselung schützt den **Inhalt**, nicht die **Metadaten**.
> Ein Video-Stream sieht auch verschlüsselt wie ein Video-Stream aus (periodische, große Bursts
> beim Puffer-Nachladen). Genau davon lebt sowohl die nützliche QoS-Klassifikation als auch
> das bedenkliche **Website Fingerprinting** — dieselbe Technik, zwei Vorzeichen.

### 2.2 Intrusion / Anomaly Detection

**Frage:** Ist dieser Verkehr bösartig? Zwei grundverschiedene Philosophien:

| | **Misuse/Signature Detection** | **Anomaly Detection** |
|---|---|---|
| Modelliert | das **Böse** (bekannte Angriffe) | das **Normale** |
| Erkennt Zero-Days? | **nein** | **ja** (im Prinzip) |
| Fehlalarme | wenige | **viele** |
| ML-Typ | überwachte Klassifikation | unüberwacht / **semi-überwacht** |

**Semi-überwacht** ist der praxisrelevante Fall und Thema des **Finalprojekts**: Man trainiert
**ausschließlich auf Normalverkehr** (den hat man reichlich und ungelabelt) und meldet
Abweichungen. Typische Verfahren: **Isolation Forest**, **One-Class SVM**, **Local Outlier
Factor**, **Autoencoder** (hoher Rekonstruktionsfehler = anomal), Gaußsche Modelle.

Die harte Realität: **Anomalie ≠ Angriff.** Ein Backup um 3 Uhr nachts ist hochgradig anomal und
völlig harmlos. Das ist der Grund für Abschnitt 3.1.

### 2.3 QoS- und QoE-Vorhersage

- **QoS (Quality of Service)** = die *technischen* Größen: Durchsatz, **Latenz**, **Jitter**,
  Paketverlust. Objektiv messbar.
- **QoE (Quality of Experience)** = die *subjektiv wahrgenommene* Qualität, klassisch als
  **MOS** (Mean Opinion Score, 1–5) durch Nutzerbefragung erhoben.

Die ML-Aufgabe: **QoS → QoE** abbilden (Regression), denn den Nutzer kann man im Betrieb nicht
ständig fragen. Der Zusammenhang ist **nichtlinear**: Das **IQX-Hypothese**-Modell beschreibt ihn
als exponentiell,
$$\text{QoE} = \alpha\,e^{-\beta\cdot \text{QoS-Störung}}+\gamma,$$
und das **Weber-Fechner-Gesetz** erklärt, warum: die Wahrnehmung reagiert auf *relative*, nicht
absolute Änderungen. Praktisch heißt das: Von 100 ms auf 200 ms Latenz ist ein Drama, von 2 s auf
2,1 s merkt niemand etwas. Beim Video-Streaming dominieren **Stalling** (Rebuffering) und
Qualitätswechsel den MOS weit stärker als die reine Auflösung. *(Beides — IQX und
Weber-Fechner — ist Würzburger Kernforschung.)*

### 2.4 Traffic-Forecasting

**Frage:** Wie viel Last liegt in 15 min / morgen auf diesem Link? **Wozu:** Kapazitätsplanung,
Energiesparen, Autoscaling, Anomalie-Baselines. Es ist ein **Zeitreihen**-Problem mit starker
**Saisonalität** (Tages-/Wochenrhythmus): klassisch **ARIMA/SARIMA**/Holtes-Winters, modern
Gradient Boosting auf Lag-Features oder **LSTM** (Modul 09). Für ein Baseline-Modell gilt fast
immer: **Saisonal-naiv** („so viel wie letzte Woche zur selben Zeit") ist erstaunlich stark —
schlägt man das nicht, lernt das Modell nichts Nützliches.

---

## 3 · Advanced — Wo Netzwerk-ML wirklich scheitert

### 3.1 Klassenungleichgewicht & der Base-Rate-Fallacy ⚠️

**Das wichtigste Kapitel dieses Moduls.** Angriffe sind **selten**. Genau daran scheitern die
meisten Papers und Produkte.

**Stufe 1 — die Accuracy-Falle.** Sind 99,99 % des Verkehrs normal, erreicht der Klassifikator
`return "normal"` eine **Accuracy von 99,99 %** — und ist vollkommen wertlos. **Accuracy ist bei
starkem Ungleichgewicht keine sinnvolle Metrik.** (Projekt 01 führt das vor.)

**Stufe 2 — der Base-Rate-Fallacy** (Axelsson, 2000). Subtiler und fataler. Gegeben:
- **Sensitivität/TPR** $P(A\mid I)$: Alarm, wenn Angriff — z. B. **0,99**
- **Falsch-Positiv-Rate/FPR** $P(A\mid \neg I)$: Alarm, obwohl harmlos — z. B. **0,001** (0,1 %!)
- **Basisrate/Prior** $\pi = P(I)$: Anteil Angriffe am Gesamtverkehr — realistisch **0,0001**

Gefragt ist, was den Analysten interessiert: **Wenn Alarm — wie wahrscheinlich ist es echt?**
Das ist der **positive Vorhersagewert (PPV/Precision)**, und Bayes liefert
$$\boxed{\;P(I\mid A)=\frac{P(A\mid I)\,\pi}{P(A\mid I)\,\pi + P(A\mid\neg I)\,(1-\pi)}\;}$$
Einsetzen:
$$P(I\mid A)=\frac{0{,}99\cdot 10^{-4}}{0{,}99\cdot 10^{-4} + 10^{-3}\cdot 0{,}9999} \approx \frac{0{,}000099}{0{,}001099}\approx \mathbf{9\,\%}.$$

**Über 90 % aller Alarme sind Fehlalarme** — bei einem Detektor mit 99 % Erkennung und nur 0,1 %
Fehlalarmrate. Nicht das Modell ist schuld, sondern die **winzige Basisrate**: die 0,1 % FPR
werden auf die *riesige* Menge harmlosen Verkehrs angewandt und erschlagen die wenigen echten
Treffer schlicht durch Masse.

**Konsequenzen für die Praxis:**
- Der begrenzende Faktor ist fast immer die **FPR**, nicht die Erkennungsrate. Will man PPV ≈ 50 %
  bei $\pi=10^{-4}$, braucht man FPR ≈ $10^{-4}$ — **hundertmal besser** als 0,1 %.
- **Alert Fatigue** ist die reale Folge: Analysten ignorieren Alarme, die zu 90 % falsch sind —
  und übersehen darin den echten.
- Rechne **immer in absoluten Zahlen**: 0,1 % FPR bei 10 Mio. Flows/Tag = **10 000 Fehlalarme
  pro Tag**. Kein Team der Welt bearbeitet das.

*Projekt 02 rechnet genau das durch — inklusive Alarme/Tag und Schwellenwahl.*

### 3.2 Metriken: warum ROC hier lügt

Die **ROC-Kurve** (TPR über FPR) ist bei starkem Ungleichgewicht **trügerisch optimistisch**,
weil die FPR im Nenner die **riesige** Negativmenge hat: 10 000 Fehlalarme unter 10 Mio.
Negativen sind FPR = 0,001 → die ROC-Kurve sieht weiterhin exzellent aus (AUC ≈ 0,99), obwohl
das Ergebnis operativ unbrauchbar ist.

Die **Precision-Recall-Kurve** verwendet stattdessen die **Precision**, in deren Nenner die
False Positives direkt gegen die *wenigen* True Positives antreten → sie **kollabiert sichtbar**
und zeigt die Wahrheit. **Faustregel: bei starkem Ungleichgewicht immer PR-Kurve + PR-AUC**
(Baseline der PR-Kurve ist die Basisrate $\pi$, nicht 0,5!). Ergänzend: **Precision@k**
(„von den k dringendsten Alarmen — wie viele echt?"), was der Arbeitsweise eines SOC entspricht.

### 3.3 Concept Drift

Das Modell wird auf März-Daten trainiert und im September eingesetzt — der Verkehr hat sich
längst geändert (neue Apps, neue Angriffe, neues Nutzerverhalten). Man unterscheidet:
- **virtueller Drift**: $P(x)$ ändert sich (anderer Verkehrsmix),
- **echter Drift**: $P(y\mid x)$ ändert sich (dasselbe Muster ist jetzt anders zu bewerten).

Gegenmittel: **zeitbasierte Evaluation** (nicht zufällig splitten!), laufendes **Monitoring** der
Score-Verteilung, periodisches **Retraining**, Drift-Detektoren (ADWIN, DDM). **Die
Modell-Halbwertszeit im Netz ist kurz** — ein IDS ist ein *Prozess*, kein Artefakt.

### 3.4 Deployment: Line-Rate, Latenz, Ort

Ein Modell im Netzpfad hat harte Randbedingungen, die es in Modul 04/05 nie gab:
- **Line-Rate**: bei 100 Gbit/s bleiben pro Paket **Nanosekunden**. Ein Random Forest mit 500
  Bäumen ist dort undenkbar → schlanke Modelle, Flow-statt-Paket-Ebene, Vorfilterung,
  Hardware (P4/SmartNIC/FPGA).
- **Wo?** Auf dem Router (schnell, dumm), an einem Kollektor (mittel), im Rechenzentrum
  (mächtig, aber sekundenlang verzögert).
- **Frühzeitigkeit:** für QoS-Priorisierung muss die Entscheidung nach den **ersten paar Paketen**
  fallen — nach dem Flow-Ende nützt sie nichts mehr. „Early classification" ist deshalb ein
  eigenes Forschungsthema.

### 3.5 Datenlecks, adversariale Angriffe, Privatsphäre

- **Data Leakage** ist in dieser Domäne endemisch. Splittet man Flows **zufällig**, landen Flows
  **desselben Angriffs** in Train *und* Test → das Modell „erkennt" den Angriff, den es schon
  gesehen hat, und die Metriken sind fantastisch **und wertlos**. Richtig: **zeitlich** splitten
  oder nach Host/Angriffstyp gruppieren (`GroupKFold`). *(Das Finalprojekt hält deshalb ganze
  Angriffstypen zurück.)*
- **Adversarial/Evasion:** Der Gegner ist **aktiv und intelligent** — anders als bei Katzenbildern.
  Er kann Padding einfügen, Timing verändern, den Angriff langsam fahren („low and slow"), um
  unter der Schwelle zu bleiben. Sogar **Poisoning** ist möglich: das Modell langsam an den
  Angriff gewöhnen.
- **Privatsphäre:** Verkehrsdaten sind personenbezogen (Metadaten verraten viel — siehe
  Website Fingerprinting). Flow- statt Payload-Ebene, Anonymisierung/Aggregation, ggf.
  Federated Learning.

### 3.6 Datensatz-Kritik — warum KDD Cup 99 berüchtigt ist

Die Projekte nutzen **KDD Cup 99** (via `sklearn.datasets.fetch_kddcup99`). Das muss man
**offen einordnen**:

**Probleme** (McHugh 2000; Tavallaee et al. 2009):
- **Uralt** (Simulation von 1998/99) — die Angriffe und der Verkehr haben mit heute wenig zu tun.
- **Synthetisch** erzeugt, nicht aus einem echten Produktivnetz.
- **Massiv redundant**: viele Duplikate → verzerrte Klassenanteile, Modelle merken sich Häufiges.
- **Zu leicht**: schon ein Random Forest erreicht ~99,99 % — die Klassen sind durch Artefakte
  fast perfekt trennbar (u. a. `src_bytes`). Ergebnisse hier sind **nicht** auf reale Netze
  übertragbar.
- **NSL-KDD** ist die bereinigte Fassung (Duplikate entfernt); **UNSW-NB15** (2015) und
  **CIC-IDS2017/2018** sind die modernen Nachfolger.

**Warum wir es trotzdem nehmen:** Es ist über sklearn **ohne jede Download-Hürde** verfügbar,
hat **echte Flow-Features** und ein **realistisches Ungleichgewicht** — es eignet sich also
hervorragend, um *Methodik* und *Fallstricke* zu lernen. **Was es nicht kann**, ist eine Aussage
über die reale Güte eines IDS zu treffen. Genau diese Unterscheidung — *gute Methodik* vs.
*belastbares Ergebnis* — ist selbst eine Lernleistung dieses Moduls. Für echte Arbeit: UNSW-NB15
oder CIC-IDS2017.

---

## 4 · Zusammenfassung / Cheat-Sheet

**Datenebenen.** Paket (`pcap`, riesig, Payload) → **Flow** (5-Tupel + Aggregate, NetFlow/IPFIX,
*der ML-Sweet-Spot*) → Zeitreihe (SNMP, Forecasting). Messung aktiv/passiv; **Sampling** löscht
kleine Flows.

**5-Tupel.** (Src-IP, Dst-IP, Src-Port, Dst-Port, Protokoll).

**Aufgaben.** Traffic-Klassifikation (Port→DPI→**statistisches ML**→verschlüsselt) · Intrusion
Detection (Misuse ↔ **Anomaly**) · **QoS→QoE** (IQX: $\alpha e^{-\beta x}+\gamma$;
Weber-Fechner) · Forecasting (Saisonalität; Baseline **saisonal-naiv**).

**Base-Rate-Fallacy** (das Herzstück):
$$P(I\mid A)=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}$$
TPR 0,99 · FPR 0,001 · $\pi=10^{-4}$ ⇒ **PPV ≈ 9 %**. Der Engpass ist die **FPR**, nicht die
Erkennungsrate. Immer **Alarme/Tag** ausrechnen.

**Metriken.** Accuracy ❌ · ROC/AUC ❌ (zu optimistisch, riesige Negativmenge) · **PR-Kurve +
PR-AUC ✅** (Baseline = $\pi$) · Precision@k ✅.

**Fallstricke.** zufälliger Split → **Leakage** (→ zeitlich/`GroupKFold`) · **Concept Drift**
(→ Retraining) · **Line-Rate** (→ schlanke Modelle) · **adversarialer** Gegner · heavy tails
(→ **log-transformieren**) · **KDD99 ist zu leicht & veraltet**.

---

## 5 · Selbsttest

<details>
<summary><b>1.</b> Warum reicht Port-basierte Klassifikation nicht mehr, und warum funktioniert ML trotz Verschlüsselung?</summary>

Ports sind unzuverlässig geworden (dynamische Ports, alles über 443 getunnelt). Verschlüsselung
verbirgt den **Inhalt**, aber nicht die **Metadaten**: Paketgrößen, Timing, Richtungen und
Burstiness bleiben sichtbar — und daraus lässt sich die Anwendung statistisch erkennen (ein
Video-Stream sieht auch verschlüsselt wie einer aus). DPI hingegen braucht lesbaren Payload und
ist damit tot.
</details>

<details>
<summary><b>2.</b> Was ist ein Flow, und warum ist die Flow-Ebene der Sweet Spot für ML?</summary>

Ein Flow ist die Aggregation aller Pakete mit demselben **5-Tupel** (Src-IP, Dst-IP, Src-Port,
Dst-Port, Protokoll) in einem Zeitfenster, gespeichert als Aggregat (Dauer, Bytes, Pakete,
Flags). Sweet Spot, weil: kompakt genug für hohe Datenraten, informativ genug zum Lernen,
und **ohne Payload** deutlich datenschutzfreundlicher als `pcap`.
</details>

<details>
<summary><b>3.</b> Ein IDS hat 99,99 % Accuracy. Warum sagt das nichts?</summary>

Weil bei einer Basisrate von z. B. 0,01 % Angriffen der triviale Klassifikator „alles normal"
bereits **99,99 %** Accuracy erreicht — ohne je einen Angriff zu finden. Bei starkem
Ungleichgewicht ist Accuracy von der Mehrheitsklasse dominiert und damit **nutzlos**; man braucht
Precision/Recall bzw. die PR-Kurve.
</details>

<details>
<summary><b>4.</b> Rechne: TPR = 0,99, FPR = 0,001, Basisrate π = 10⁻⁴. Wie viele Alarme sind echt?</summary>

$$P(I|A)=\frac{0{,}99\cdot10^{-4}}{0{,}99\cdot10^{-4}+0{,}001\cdot0{,}9999}=\frac{0{,}000099}{0{,}001099}\approx 9\,\%.$$
Nur **~9 %** der Alarme sind echt, **~91 % Fehlalarme** — trotz „99 % Erkennung, nur 0,1 %
Fehlalarmrate". Das ist der **Base-Rate-Fallacy**: die kleine FPR trifft auf eine gewaltige
Menge harmlosen Verkehrs.
</details>

<details>
<summary><b>5.</b> Warum PR- statt ROC-Kurve bei Angriffserkennung?</summary>

Die ROC nutzt die **FPR**, deren Nenner die riesige Negativmenge ist → auch 10 000 Fehlalarme
wirken als „FPR 0,001", die Kurve bleibt optisch exzellent. Die **Precision** stellt die False
Positives direkt den *wenigen* True Positives gegenüber → die PR-Kurve **kollabiert sichtbar**
und bildet die operative Realität ab. Die PR-Baseline ist die **Basisrate π**, nicht 0,5.
</details>

<details>
<summary><b>6.</b> Misuse- vs. Anomaly Detection — Unterschied und jeweiliger Preis?</summary>

**Misuse/Signature** modelliert bekannte **Angriffe** → wenige Fehlalarme, aber **blind für
Zero-Days**. **Anomaly Detection** modelliert das **Normale** und meldet Abweichungen → kann
Unbekanntes finden, produziert aber **viele Fehlalarme**, weil *anomal ≠ bösartig* (das nächtliche
Backup). Semi-überwacht (nur auf Normalverkehr trainieren) ist der praxisnahe Mittelweg.
</details>

<details>
<summary><b>7.</b> Warum ist ein zufälliger Train/Test-Split bei Netzwerkdaten gefährlich?</summary>

**Data Leakage**: Flows sind nicht i.i.d. Flows desselben Angriffs/Hosts landen in Train *und*
Test → das Modell erkennt Gesehenes wieder, die Metriken sind exzellent und **wertlos**. Richtig
ist ein **zeitlicher** Split (Zukunft vorhersagen) oder Gruppierung nach Host/Angriffstyp
(`GroupKFold`) — und für Zero-Day-Tests: ganze Angriffstypen zurückhalten.
</details>

<details>
<summary><b>8.</b> Was ist Concept Drift, und was folgt daraus organisatorisch?</summary>

Die Verteilung ändert sich über die Zeit — **virtuell** ($P(x)$, neuer Verkehrsmix) oder **echt**
($P(y|x)$, andere Bewertung desselben Musters). Folge: Modelle **veralten schnell**. Man braucht
zeitbasierte Evaluation, Monitoring der Score-Verteilung und periodisches **Retraining** — ein
IDS ist ein **Prozess**, kein einmalig geliefertes Artefakt.
</details>

<details>
<summary><b>9.</b> Was sagen IQX-Hypothese und Weber-Fechner über QoE?</summary>

**IQX:** QoE hängt **exponentiell** von der QoS-Störung ab, $\text{QoE}=\alpha e^{-\beta x}+\gamma$.
**Weber-Fechner:** Wahrnehmung reagiert auf **relative**, nicht absolute Änderungen. Praktisch:
100→200 ms Latenz ist gravierend, 2,0→2,1 s merkt niemand. Beim Video dominiert **Stalling** den
MOS stärker als die Auflösung.
</details>

<details>
<summary><b>10.</b> Nenne drei Gründe, warum KDD Cup 99 keine Aussage über reale IDS-Güte erlaubt.</summary>

Beliebige drei: **veraltet** (1998/99), **synthetisch** (kein Produktivnetz), **massiv redundant**
(Duplikate verzerren die Klassenanteile), **zu leicht** (RF ≈ 99,99 % durch Artefakte wie
`src_bytes`), unrealistische Basisrate. Moderne Alternativen: **NSL-KDD**, **UNSW-NB15**,
**CIC-IDS2017**.
</details>

---

## 6 · Literatur & Quellen

**Der Klassiker zum Kernthema (frei, unbedingt lesen):**
- 📄 **S. Axelsson (2000), *The Base-Rate Fallacy and the Difficulty of Intrusion Detection***
  (ACM TISSEC). Das Paper hinter Abschnitt 3.1 — kurz, rechnerisch, desillusionierend.
  *Einsteigerfreundlich, frei auffindbar.* **Beste Einzelquelle des Moduls.**
- 📄 **R. Sommer & V. Paxson (2010), *Outside the Closed World: On Using Machine Learning for
  Network Intrusion Detection*** (IEEE S&P). Warum ML im IDS-Kontext so oft scheitert —
  Pflichtlektüre, hervorragend geschrieben. *Frei.*

**Datensatz-Kritik:**
- 📄 **J. McHugh (2000), *Testing Intrusion Detection Systems*** — die Original-Kritik an DARPA/KDD.
- 📄 **Tavallaee et al. (2009), *A Detailed Analysis of the KDD CUP 99 Data Set*** — führt
  **NSL-KDD** ein. *Frei.*
- 🌐 **UNSW-NB15** (unsw.adfa.edu.au) und **CIC-IDS2017** (unb.ca/cic/datasets) — die modernen
  Datensätze für ernsthafte Arbeit. *Frei, Download nötig.*

**Traffic-Klassifikation & Messung:**
- 📄 **Nguyen & Armitage (2008), *A Survey of Techniques for Internet Traffic Classification
  using Machine Learning*** (IEEE Comm. Surveys) — der Standard-Überblick. *Vertiefend.*
- 📘 **M. Crotti et al. / Taylor et al., *AppScanner*** — Fingerprinting verschlüsselten
  Verkehrs. *Vertiefend.*
- 🌐 **Wireshark** (wireshark.org) — zum Anfassen: schau dir echten Verkehr selbst an. *Einsteiger.*

**QoE (Würzburger Kernthema):**
- 📄 **Fiedler, Hoßfeld & Tran-Gia (2010), *A Generic Quantitative Relationship between QoS and
  QoE*** (IEEE Network) — die **IQX-Hypothese**. *Einsteiger→vertiefend.*
- 📄 **Hoßfeld et al., *Quantification of YouTube QoE via Crowdsourcing*** — Stalling und MOS.

**Bücher/Kurse:**
- 📗 **Bishop / Hastie et al.** für die ML-Grundlagen (schon aus Modul 04/05 bekannt).
- 📘 **Kurose & Ross, *Computer Networking: A Top-Down Approach*** — falls dir Netzwerk-
  Grundlagen (TCP/IP, Ports, Router) fehlen. *Einsteigerfreundlich.*
- 🌐 **scikit-learn User Guide: *Imbalanced classification* / *Precision-Recall*** — praktisch
  und frei.

---

## Nächstes Modul

**Modul 16 — Machine Learning for Networks 2** vertieft: **Graph-basiertes** Lernen auf
Netztopologien (GNNs), Netzwerk-**Zeitreihen**/Forecasting im Detail, verschlüsselte
Verkehrsanalyse und selbstlernende/selbst-optimierende Netze. Das hier gelernte Fundament —
Flow-Features, Ungleichgewicht, Base-Rate, Drift, saubere Evaluation — gilt dort unverändert
weiter.
