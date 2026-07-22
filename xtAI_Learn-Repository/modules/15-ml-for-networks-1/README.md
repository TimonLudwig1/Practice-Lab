# Module 15 — Machine Learning for Networks 1

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** A communication network is one of the largest data sources in computer
> science: every second produces millions of packets, flows and measurements. This module applies
> **machine learning to communication networks** — **classifying** traffic, **detecting** attacks,
> **predicting quality of service/experience (QoS/QoE)**, **forecasting** load. The appeal does not
> lie in exotic models (they are mostly the same ones as in modules 04/05), but in the **brutal
> peculiarities of the domain**: extreme class imbalance, non-stationary traffic, encryption,
> line-rate requirements — and one statistical trap that renders entire security products useless
> in practice: the **base-rate fallacy**.

**Helpful prior knowledge:** classification/regression, pipelines, cross-validation, metrics
(modules 04/05); basic networking concepts (IP, TCP/UDP, ports) are useful but are introduced
here.

**Modules you should have done first:**
- **Module 04 (ML 1)** — classification, pipelines, CV, GridSearch, **threshold selection & costs**
  (the Adult project with its cost-based threshold is the direct blueprint here);
- **Module 05 (ML 2)** — ensembles/networks, clustering & **unsupervised** methods (for the
  anomaly detection in the final project);
- **Modules 02/03 (data science)** — EDA, data cleaning, dealing with real, dirty data.
  *(RL from modules 13/14 is **not** needed here.)*

> **Note on how the content was scoped.** No official module description was available for this
> module. I scoped it along the Würzburg affiliation (Chair of **Communication Networks**) and the
> international standard literature: "networks" here means **communication networks**, *not*
> neural networks and *not* primarily graph learning. Module 16 (ML for Networks 2) later goes
> deeper into graph-based methods (GNN), network time series and self-learning networks.

---

## Learning objectives

After this module you can …

- classify **network data**: packet vs. **flow** level, NetFlow/IPFIX, active vs. passive
  measurement, **sampling**, and the **features** derivable from them;
- explain the evolution of **traffic classification**: port-based → **DPI** → statistical ML →
  **encrypted** traffic;
- formulate **intrusion/anomaly detection** as an ML problem — supervised, unsupervised,
  **semi-supervised** (normal traffic only) — and state the difference between *misuse* and
  *anomaly* detection;
- handle **extreme class imbalance** correctly and justify why **accuracy** and often even the
  **ROC curve** are misleading here (→ **PR curve**);
- derive the **base-rate fallacy** (Axelsson) **quantitatively** and compute why a detector that
  is "99.9 % accurate" drowns in false alarms in production;
- explain **concept drift**/non-stationarity, **deployment** constraints (line rate, latency),
  **adversarial** evasion and **privacy** as practical problems;
- assess network datasets **critically** (why KDD Cup 99 is notorious).

---

## 1 · Basics — network data

### 1.1 Why use ML in the network at all?

Classical network operations tools are **rule-based**: fixed port numbers, signatures,
thresholds. That breaks for three reasons:

1. **Encryption.** More than 90 % of web traffic is TLS today. There is nothing readable left in
   the payload → signature-based **deep packet inspection (DPI)** runs into the void. What remains
   are the **metadata** (packet sizes, timing, directions) — and evaluating those is a
   **statistical** problem, i.e. ML.
2. **Scale & dynamics.** Applications change ports, tunnel over 443, change weekly. Hand-maintained
   rules become stale faster than you can write them.
3. **Unknown attacks.** Signatures only recognize what is already known. **Zero-days** require
   modelling *normality* and reporting deviations.

### 1.2 Packets, flows, and what can be measured

Network data exists at several levels of granularity:

- **Packet level** (`pcap` via tcpdump/Wireshark): every single packet with header plus, possibly,
  payload. Maximum information content, but **huge** (10 Gbit/s ⇒ ~GB/s) and privacy-critical.
- **Flow level** (**NetFlow**/**IPFIX**, sFlow): packets are aggregated into **flows**. Classically
  a flow is the **5-tuple**
  $$(\text{source IP},\ \text{destination IP},\ \text{source port},\ \text{destination port},\ \text{protocol})$$
  plus a time window. Per flow you store aggregates: duration, packet count, bytes, flags, …
  **This is the sweet spot for ML**: compact enough for line rate, informative enough to learn
  from, and much more privacy-friendly without the payload. *All three projects work at exactly
  this level.*
- **Aggregated time series** (SNMP counters, link utilization per 5 min): for forecasting/capacity
  planning.

**Measurement** is either **passive** (listening in, e.g. at a router mirror port) or **active**
(sending packets yourself: `ping`, `traceroute`, speed tests — which changes the network you are
measuring).

**Sampling.** At high rates you cannot capture every packet; you take, say, every 1000th one
(1:1000). A consequence for ML that is often overlooked: **small flows disappear** almost entirely
(a 3-packet port scan is never seen with probability ≈ 99.7 % at 1:1000), while elephant flows
survive. Sampling **biases** the distribution systematically — and precisely against the rare,
security-relevant events.

### 1.3 Typical flow features

From a flow record you can derive:

| Group | Examples |
|---|---|
| **Volume** | total bytes, total packets, mean packet size, bytes per direction |
| **Time** | duration, inter-arrival times (mean/std/min/max), bytes per second |
| **Direction/symmetry** | upstream/downstream ratio, number of direction changes |
| **Protocol/header** | protocol, TCP flags (SYN/FIN/RST), TTL, window size |
| **Context/host** | number of connections from the same source in the time window, number of distinct destination ports |

The **context features** are often the most valuable ones: a single SYN is harmless, but "200 SYNs
to 200 different ports of the same host within 2 seconds" is a port scan. **Attacks are frequently
visible only in the aggregate, not in the individual event.**

### 1.4 The distributions are *not* benign

Network traffic violates almost every convenient assumption:

- **Heavy tails.** Flow sizes are extremely skewed: a few **elephant flows** carry the bulk of the
  bytes, millions of **mice flows** the bulk of the flows. Means are almost meaningless here →
  a **log transform** of byte/packet counters is practically mandatory.
- **Non-stationary.** Day/night, weekday/weekend, new apps → the distribution drifts permanently
  (**concept drift**, section 3.3).
- **Not i.i.d.** Packets of one flow and flows of one host are strongly correlated. Naively
  splitting at random into train/test therefore **leaks** massively (section 3.5).

---

## 2 · Building up — the four core tasks

### 2.1 Traffic classification

**Question:** which application/service class produces this flow (video, VoIP, web, gaming, file
sharing)? **What for:** QoS prioritization, capacity planning, billing, policy enforcement.

The historical development is itself the lesson:

1. **Port-based** (until ~2000): port 80 = HTTP. **Dead** — everything runs over 443 today, P2P
   uses dynamic ports.
2. **DPI/signatures** (~2000–2010): look into the payload. **Killed by encryption** (and legally
   and privacy-wise delicate).
3. **Statistical ML on flow features** (today): classify based on *sizes and timing*, not content.
   Works **even with TLS**, because encryption hides the content but barely changes the **pattern**
   (packet size sequence, burstiness).
4. **Deep learning on raw byte/packet sequences** (current): CNNs/transformers learn the features
   themselves.

> **The central aha moment:** encryption protects the **content**, not the **metadata**. A video
> stream still looks like a video stream when encrypted (periodic, large bursts while the buffer
> refills). This is exactly what both the useful QoS classification and the worrying **website
> fingerprinting** live on — the same technique with two different signs.

### 2.2 Intrusion / anomaly detection

**Question:** is this traffic malicious? Two fundamentally different philosophies:

| | **Misuse/signature detection** | **Anomaly detection** |
|---|---|---|
| Models | the **bad** (known attacks) | the **normal** |
| Detects zero-days? | **no** | **yes** (in principle) |
| False alarms | few | **many** |
| ML type | supervised classification | unsupervised / **semi-supervised** |

**Semi-supervised** is the practically relevant case and the topic of the **final project**: you
train **exclusively on normal traffic** (of which you have plenty, unlabelled) and report
deviations. Typical methods: **isolation forest**, **one-class SVM**, **local outlier factor**,
**autoencoders** (high reconstruction error = anomalous), Gaussian models.

The hard reality: **anomaly ≠ attack.** A backup at 3 a.m. is highly anomalous and completely
harmless. That is the reason for section 3.1.

### 2.3 QoS and QoE prediction

- **QoS (quality of service)** = the *technical* quantities: throughput, **latency**, **jitter**,
  packet loss. Objectively measurable.
- **QoE (quality of experience)** = the *subjectively perceived* quality, classically collected as
  a **MOS** (mean opinion score, 1–5) via user surveys.

The ML task: map **QoS → QoE** (regression), because you cannot keep asking the user in
production. The relationship is **non-linear**: the **IQX hypothesis** model describes it as
exponential,
$$\text{QoE} = \alpha\,e^{-\beta\cdot \text{QoS impairment}}+\gamma,$$
and the **Weber-Fechner law** explains why: perception reacts to *relative*, not absolute changes.
In practice this means: going from 100 ms to 200 ms of latency is a drama, going from 2 s to 2.1 s
is unnoticeable. In video streaming, **stalling** (rebuffering) and quality switches dominate the
MOS far more strongly than the pure resolution. *(Both — IQX and Weber-Fechner — are core Würzburg
research.)*

### 2.4 Traffic forecasting

**Question:** how much load will be on this link in 15 min / tomorrow? **What for:** capacity
planning, energy saving, autoscaling, anomaly baselines. It is a **time series** problem with
strong **seasonality** (daily/weekly rhythm): classically **ARIMA/SARIMA**/Holt-Winters, in modern
form gradient boosting on lag features or **LSTMs** (module 09). For a baseline model, one thing
almost always holds: **seasonal-naive** ("as much as last week at the same time") is surprisingly
strong — if you do not beat it, your model is not learning anything useful.

---

## 3 · Advanced — where network ML really fails

### 3.1 Class imbalance & the base-rate fallacy ⚠️

**The most important chapter of this module.** Attacks are **rare**. This is exactly where most
papers and products fail.

**Stage 1 — the accuracy trap.** If 99.99 % of traffic is normal, the classifier
`return "normal"` achieves an **accuracy of 99.99 %** — and is entirely worthless. **Accuracy is
not a sensible metric under strong imbalance.** (Project 01 demonstrates this.)

**Stage 2 — the base-rate fallacy** (Axelsson, 2000). Subtler and more fatal. Given:
- **sensitivity/TPR** $P(A\mid I)$: alarm when there is an attack — e.g. **0.99**
- **false positive rate/FPR** $P(A\mid \neg I)$: alarm although harmless — e.g. **0.001** (0.1 %!)
- **base rate/prior** $\pi = P(I)$: share of attacks in total traffic — realistically **0.0001**

What is asked is what the analyst cares about: **given an alarm — how likely is it real?** That is
the **positive predictive value (PPV/precision)**, and Bayes gives
$$\boxed{\;P(I\mid A)=\frac{P(A\mid I)\,\pi}{P(A\mid I)\,\pi + P(A\mid\neg I)\,(1-\pi)}\;}$$
Substituting:
$$P(I\mid A)=\frac{0.99\cdot 10^{-4}}{0.99\cdot 10^{-4} + 10^{-3}\cdot 0.9999} \approx \frac{0.000099}{0.001099}\approx \mathbf{9\,\%}.$$

**More than 90 % of all alarms are false alarms** — with a detector that has 99 % detection and
only a 0.1 % false alarm rate. The model is not to blame, the **tiny base rate** is: the 0.1 % FPR
is applied to the *enormous* amount of harmless traffic and simply swamps the few genuine hits by
sheer mass.

**Consequences for practice:**
- The limiting factor is almost always the **FPR**, not the detection rate. If you want PPV ≈ 50 %
  at $\pi=10^{-4}$, you need FPR ≈ $10^{-4}$ — **a hundred times better** than 0.1 %.
- **Alert fatigue** is the real consequence: analysts ignore alarms that are 90 % wrong — and miss
  the real one among them.
- Always compute in **absolute numbers**: 0.1 % FPR at 10 million flows/day = **10,000 false
  alarms per day**. No team in the world works through that.

*Project 02 computes exactly this — including alarms per day and threshold selection.*

### 3.2 Metrics: why ROC lies here

The **ROC curve** (TPR over FPR) is **deceptively optimistic** under strong imbalance, because the
FPR has the **enormous** negative set in its denominator: 10,000 false alarms among 10 million
negatives means FPR = 0.001 → the ROC curve continues to look excellent (AUC ≈ 0.99), even though
the result is operationally useless.

The **precision-recall curve** uses **precision** instead, whose denominator pits the false
positives directly against the *few* true positives → it **collapses visibly** and shows the
truth. **Rule of thumb: under strong imbalance always use the PR curve + PR-AUC** (the baseline of
the PR curve is the base rate $\pi$, not 0.5!). In addition: **precision@k** ("of the k most urgent
alarms — how many are real?"), which matches the way a SOC works.

### 3.3 Concept drift

The model is trained on March data and deployed in September — the traffic has long since changed
(new apps, new attacks, new user behaviour). One distinguishes:
- **virtual drift**: $P(x)$ changes (different traffic mix),
- **real drift**: $P(y\mid x)$ changes (the same pattern must now be judged differently).

Countermeasures: **time-based evaluation** (do not split randomly!), continuous **monitoring** of
the score distribution, periodic **retraining**, drift detectors (ADWIN, DDM). **The half-life of
a model in the network is short** — an IDS is a *process*, not an artifact.

### 3.4 Deployment: line rate, latency, location

A model in the network path has hard constraints that never existed in modules 04/05:
- **Line rate**: at 100 Gbit/s only **nanoseconds** remain per packet. A random forest with 500
  trees is unthinkable there → lean models, flow instead of packet level, pre-filtering, hardware
  (P4/SmartNIC/FPGA).
- **Where?** On the router (fast, dumb), at a collector (medium), in the data centre (powerful,
  but delayed by seconds).
- **Earliness:** for QoS prioritization the decision must be made after the **first few packets** —
  after the flow has ended it is useless. "Early classification" is therefore a research topic of
  its own.

### 3.5 Data leakage, adversarial attacks, privacy

- **Data leakage** is endemic in this domain. If you split flows **at random**, flows of **the same
  attack** end up in train *and* test → the model "detects" the attack it has already seen, and the
  metrics are fantastic **and worthless**. Correct: split **temporally** or group by host/attack
  type (`GroupKFold`). *(This is why the final project holds back entire attack types.)*
- **Adversarial/evasion:** the adversary is **active and intelligent** — unlike with cat pictures.
  They can insert padding, change timing, run the attack slowly ("low and slow") to stay below the
  threshold. Even **poisoning** is possible: slowly accustoming the model to the attack.
- **Privacy:** traffic data is personal data (metadata reveal a great deal — see website
  fingerprinting). Flow instead of payload level, anonymization/aggregation, possibly federated
  learning.

### 3.6 Dataset criticism — why KDD Cup 99 is notorious

The projects use **KDD Cup 99** (via `sklearn.datasets.fetch_kddcup99`). This has to be classified
**openly**:

**Problems** (McHugh 2000; Tavallaee et al. 2009):
- **Ancient** (a simulation from 1998/99) — the attacks and the traffic have little to do with
  today.
- **Synthetically** generated, not from a real production network.
- **Massively redundant**: many duplicates → distorted class shares, models memorize what is
  frequent.
- **Too easy**: even a random forest reaches ~99.99 % — the classes are almost perfectly separable
  through artifacts (among others `src_bytes`). Results here are **not** transferable to real
  networks.
- **NSL-KDD** is the cleaned version (duplicates removed); **UNSW-NB15** (2015) and
  **CIC-IDS2017/2018** are the modern successors.

**Why we still take it:** it is available through sklearn **without any download hurdle**, has
**real flow features** and a **realistic imbalance** — so it is excellently suited to learning
*methodology* and *pitfalls*. **What it cannot do** is make a statement about the real quality of
an IDS. Precisely this distinction — *good methodology* vs. *trustworthy result* — is itself a
learning outcome of this module. For real work: UNSW-NB15 or CIC-IDS2017.

---

## 4 · Summary / cheat sheet

**Data levels.** Packet (`pcap`, huge, payload) → **flow** (5-tuple + aggregates, NetFlow/IPFIX,
*the ML sweet spot*) → time series (SNMP, forecasting). Measurement active/passive; **sampling**
erases small flows.

**5-tuple.** (src IP, dst IP, src port, dst port, protocol).

**Tasks.** Traffic classification (port→DPI→**statistical ML**→encrypted) · intrusion detection
(misuse ↔ **anomaly**) · **QoS→QoE** (IQX: $\alpha e^{-\beta x}+\gamma$; Weber-Fechner) ·
forecasting (seasonality; baseline **seasonal-naive**).

**Base-rate fallacy** (the heart of it all):
$$P(I\mid A)=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}$$
TPR 0.99 · FPR 0.001 · $\pi=10^{-4}$ ⇒ **PPV ≈ 9 %**. The bottleneck is the **FPR**, not the
detection rate. Always compute **alarms per day**.

**Metrics.** Accuracy ❌ · ROC/AUC ❌ (too optimistic, enormous negative set) · **PR curve +
PR-AUC ✅** (baseline = $\pi$) · precision@k ✅.

**Pitfalls.** random split → **leakage** (→ temporal/`GroupKFold`) · **concept drift** (→
retraining) · **line rate** (→ lean models) · **adversarial** opponent · heavy tails (→ **log
transform**) · **KDD99 is too easy & outdated**.

---

## 5 · Self-test

<details>
<summary><b>1.</b> Why is port-based classification no longer sufficient, and why does ML work despite encryption?</summary>

Ports have become unreliable (dynamic ports, everything tunnelled over 443). Encryption hides the
**content**, but not the **metadata**: packet sizes, timing, directions and burstiness remain
visible — and from those the application can be recognized statistically (a video stream still
looks like one when encrypted). DPI, by contrast, needs readable payload and is therefore dead.
</details>

<details>
<summary><b>2.</b> What is a flow, and why is the flow level the sweet spot for ML?</summary>

A flow is the aggregation of all packets with the same **5-tuple** (src IP, dst IP, src port, dst
port, protocol) within a time window, stored as an aggregate (duration, bytes, packets, flags).
Sweet spot because: compact enough for high data rates, informative enough to learn from, and
**without payload** much more privacy-friendly than `pcap`.
</details>

<details>
<summary><b>3.</b> An IDS has 99.99 % accuracy. Why does that say nothing?</summary>

Because with a base rate of, say, 0.01 % attacks, the trivial classifier "everything is normal"
already reaches **99.99 %** accuracy — without ever finding an attack. Under strong imbalance,
accuracy is dominated by the majority class and therefore **useless**; you need precision/recall
or the PR curve.
</details>

<details>
<summary><b>4.</b> Compute: TPR = 0.99, FPR = 0.001, base rate π = 10⁻⁴. How many alarms are real?</summary>

$$P(I|A)=\frac{0.99\cdot10^{-4}}{0.99\cdot10^{-4}+0.001\cdot0.9999}=\frac{0.000099}{0.001099}\approx 9\,\%.$$
Only **~9 %** of the alarms are real, **~91 % false alarms** — despite "99 % detection, only 0.1 %
false alarm rate". This is the **base-rate fallacy**: the small FPR meets an enormous amount of
harmless traffic.
</details>

<details>
<summary><b>5.</b> Why the PR curve instead of the ROC curve for attack detection?</summary>

The ROC uses the **FPR**, whose denominator is the enormous negative set → even 10,000 false alarms
appear as "FPR 0.001" and the curve stays visually excellent. **Precision** confronts the false
positives directly with the *few* true positives → the PR curve **collapses visibly** and reflects
the operational reality. The PR baseline is the **base rate π**, not 0.5.
</details>

<details>
<summary><b>6.</b> Misuse vs. anomaly detection — difference and respective price?</summary>

**Misuse/signature** models known **attacks** → few false alarms, but **blind to zero-days**.
**Anomaly detection** models the **normal** and reports deviations → can find the unknown, but
produces **many false alarms**, because *anomalous ≠ malicious* (the nightly backup).
Semi-supervised (training on normal traffic only) is the practical middle ground.
</details>

<details>
<summary><b>7.</b> Why is a random train/test split dangerous for network data?</summary>

**Data leakage**: flows are not i.i.d. Flows of the same attack/host end up in train *and* test →
the model recognizes what it has seen, and the metrics are excellent and **worthless**. The correct
approach is a **temporal** split (predict the future) or grouping by host/attack type
(`GroupKFold`) — and for zero-day tests: hold back entire attack types.
</details>

<details>
<summary><b>8.</b> What is concept drift, and what follows from it organizationally?</summary>

The distribution changes over time — **virtual** ($P(x)$, new traffic mix) or **real**
($P(y|x)$, different assessment of the same pattern). Consequence: models **become stale quickly**.
You need time-based evaluation, monitoring of the score distribution and periodic **retraining** —
an IDS is a **process**, not a one-off delivered artifact.
</details>

<details>
<summary><b>9.</b> What do the IQX hypothesis and Weber-Fechner say about QoE?</summary>

**IQX:** QoE depends **exponentially** on the QoS impairment, $\text{QoE}=\alpha e^{-\beta x}+\gamma$.
**Weber-Fechner:** perception reacts to **relative**, not absolute changes. In practice: 100→200 ms
of latency is severe, 2.0→2.1 s is unnoticeable. For video, **stalling** dominates the MOS more
strongly than the resolution.
</details>

<details>
<summary><b>10.</b> Name three reasons why KDD Cup 99 permits no statement about real IDS quality.</summary>

Any three: **outdated** (1998/99), **synthetic** (no production network), **massively redundant**
(duplicates distort the class shares), **too easy** (RF ≈ 99.99 % through artifacts such as
`src_bytes`), unrealistic base rate. Modern alternatives: **NSL-KDD**, **UNSW-NB15**,
**CIC-IDS2017**.
</details>

---

## 6 · Literature & sources

**The classic on the core topic (free, absolutely worth reading):**
- 📄 **S. Axelsson (2000), *The Base-Rate Fallacy and the Difficulty of Intrusion Detection***
  (ACM TISSEC). The paper behind section 3.1 — short, computational, disillusioning.
  *Beginner-friendly, freely findable.* **Best single source of the module.**
- 📄 **R. Sommer & V. Paxson (2010), *Outside the Closed World: On Using Machine Learning for
  Network Intrusion Detection*** (IEEE S&P). Why ML so often fails in the IDS context — mandatory
  reading, excellently written. *Free.*

**Dataset criticism:**
- 📄 **J. McHugh (2000), *Testing Intrusion Detection Systems*** — the original criticism of
  DARPA/KDD.
- 📄 **Tavallaee et al. (2009), *A Detailed Analysis of the KDD CUP 99 Data Set*** — introduces
  **NSL-KDD**. *Free.*
- 🌐 **UNSW-NB15** (unsw.adfa.edu.au) and **CIC-IDS2017** (unb.ca/cic/datasets) — the modern
  datasets for serious work. *Free, download required.*

**Traffic classification & measurement:**
- 📄 **Nguyen & Armitage (2008), *A Survey of Techniques for Internet Traffic Classification
  using Machine Learning*** (IEEE Comm. Surveys) — the standard overview. *In depth.*
- 📘 **M. Crotti et al. / Taylor et al., *AppScanner*** — fingerprinting of encrypted traffic.
  *In depth.*
- 🌐 **Wireshark** (wireshark.org) — hands-on: look at real traffic yourself. *Beginner.*

**QoE (a core Würzburg topic):**
- 📄 **Fiedler, Hoßfeld & Tran-Gia (2010), *A Generic Quantitative Relationship between QoS and
  QoE*** (IEEE Network) — the **IQX hypothesis**. *Beginner→in depth.*
- 📄 **Hoßfeld et al., *Quantification of YouTube QoE via Crowdsourcing*** — stalling and MOS.

**Books/courses:**
- 📗 **Bishop / Hastie et al.** for the ML foundations (already known from modules 04/05).
- 📘 **Kurose & Ross, *Computer Networking: A Top-Down Approach*** — in case you lack the
  networking basics (TCP/IP, ports, routers). *Beginner-friendly.*
- 🌐 **scikit-learn user guide: *Imbalanced classification* / *Precision-Recall*** — practical and
  free.

---

## Next module

**Module 16 — Machine Learning for Networks 2** goes deeper: **graph-based** learning on network
topologies (GNNs), network **time series**/forecasting in detail, encrypted traffic analysis and
self-learning/self-optimizing networks. The foundation learned here — flow features, imbalance,
base rate, drift, sound evaluation — continues to hold there unchanged.

---

# Modul 15 — Machine Learning for Networks 1 (deutsche Fassung)

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
