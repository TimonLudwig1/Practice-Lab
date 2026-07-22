# Module 16 — Machine Learning for Networks 2

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** Module 15 treated every network event on its own: one **flow**, one
> feature vector, one prediction. But a network is not a pile of independent rows — it is a
> **graph**, and traffic is a **time series**. This module takes seriously exactly the two
> structures module 15 ignored: **structure in space** (topology → *graph neural networks*) and
> **structure in time** (seasonality → *forecasting*) — and at the end both together
> (**spatio-temporal learning**). On top of that comes the look ahead: encrypted traffic analysis
> and **self-driving networks**.

**Helpful prior knowledge:** linear algebra (matrix multiplication, eigenvalues), probability
theory, PyTorch basics.

**Modules you should have done first:**
- **Module 15 (ML for Networks 1)** — *mandatory*. Flows, measurement, **class imbalance**,
  **base-rate fallacy**, PR instead of ROC, **concept drift**, leakage. All of that continues to
  hold here unchanged and is **not** repeated.
- **Module 05 (ML 2)** and **module 09 (NLP 2)** — neural networks, PyTorch, and the
  **attention** idea (GAT is attention on graphs; a transformer *is* a GNN on the complete
  graph).
- **Modules 13/14 (RL)** — only for section 4.3 (routing/traffic engineering as a decision
  problem).

> **Note on how the content was scoped.** As with module 15 no official module description was
> available. Here I deliver what the module 15 script announced as its continuation: **graph-based
> learning, network time series, encrypted traffic analysis, self-learning networks**.
>
> **Tooling decision:** `torch_geometric` (the standard GNN library) is **not installed** here,
> and neither is `statsmodels` (ARIMA). Didactically that is a stroke of luck: we build the
> **GCN from scratch** in plain PyTorch (it is ~15 lines — and afterwards the library holds no
> more mystery) and the forecasting likewise. ARIMA/SARIMA I explain completely **formally**,
> without a package demo.

---

## Learning objectives

After this module you can …

- model a network as a **graph** and characterize its structure: **degree distribution**,
  **power law/scale-free**, **small world**, **centralities**, clustering;
- explain **why the internet is robust against random failures and fragile against targeted hub
  attacks** — and show it on real data;
- **estimate power-law exponents correctly** (MLE instead of a naive log-log fit) and justify why
  the naive route is systematically wrong;
- place **graph representation learning**: handcrafted graph features → **node2vec** → **GNNs**;
- formulate the **message-passing framework** and derive **GCN** in full
  (including the question of why $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ looks *that*
  way); distinguish **GraphSAGE** (inductive) and **GAT** (attention); explain **over-smoothing**;
- formulate **link prediction** as a task and pit heuristics (common neighbors, Jaccard,
  **Adamic-Adar**, preferential attachment) against learned methods;
- forecast **network time series**: seasonality, the **seasonal-naive baseline**, lag features,
  ARIMA/SARIMA (formally), and **spatio-temporal GNNs**;
- place **encrypted traffic analysis** (website fingerprinting) and **self-driving networks**
  (intent-based networking, digital twin, RL routing).

---

## 1 · The network as a graph

### 1.1 Which graph exactly?

"The network" is ambiguous — depending on the level of abstraction you get a different graph:

| Level | Nodes | Edges | Order of magnitude |
|---|---|---|---|
| **AS level** | autonomous systems (providers, companies) | BGP peerings | ~75,000 today |
| **Router level** | individual routers | physical links | millions |
| **PoP/backbone** | sites of one operator | fibre routes | ~10–100 |
| **Overlay/traffic** | hosts | observed flows | huge, dynamic |

The projects use the **AS-level topology** (real data, see below). Important: the **traffic
graph** (who talks to whom) is *not* the **topology graph** (what is physically connected) — a
confusion many papers commit without noticing.

Formally: $G=(V,E)$, **adjacency matrix** $A\in\{0,1\}^{n\times n}$ with $A_{ij}=1$ ⟺ $(i,j)\in E$,
**degree matrix** $D=\operatorname{diag}(d_1,\dots,d_n)$, $d_i=\sum_j A_{ij}$. Undirected ⟹
$A=A^\top$. At $n=10\,670$ nodes a *dense* $A$ would already have 114 million entries (~455 MB) —
but with 22,000 edges **99.96 %** of them are zero (only 0.04 % filled). **Network graphs are
extremely sparse; you compute sparse on principle.**

### 1.2 The degree distribution: scale-free

The most famous property of the internet topology (Faloutsos³, 1999): the degree distribution
approximately follows a **power law**
$$P(d) \propto d^{-\alpha},\qquad \alpha \approx 2.1 \text{ (internet AS level)}.$$

That means: **most AS have 1–2 neighbours, a very few "hubs" have thousands.** No typical degree,
no meaningful "mean" scale — hence *scale-free*. On the real data of the projects: median degree
**2**, mean **4.12**, maximum **2312**. As with the flow sizes from module 15: **the mean is
meaningless** when the distribution has heavy tails.

Why does this arise? **Preferential attachment** (Barabási–Albert): the network *grows*, and new
nodes preferentially attach to already well connected nodes (a new ISP buys transit from a large
provider). "The rich get richer" ⟹ a power law with $\alpha=3$ in the basic BA model.

> ### ⚠️ How **not** to estimate the exponent
> The obvious route — histogram of the degrees, plot log-log, fit a line — is **systematically
> wrong**. Reason: in the tail there are only 0–3 observations per bin; their logarithmized noise
> is massively biased (and all empty bins drop out entirely). The least-squares line gets bent by
> this.
> **Clauset, Shalizi & Newman (2009)** show: you need the **maximum likelihood estimator**
> $$\hat\alpha = 1 + n\Big[\sum_{i=1}^{n}\ln\frac{d_i}{d_{\min}-\tfrac12}\Big]^{-1},$$
> applied only to the tail $d\ge d_{\min}$. On our real data:
>
> | Method | Result |
> |---|---|
> | naive log-log fit (PMF) | **−1.14** ❌ |
> | MLE, $d_{\min}=5$ | 2.12 |
> | **MLE, $d_{\min}=10$** | **2.08** ✅ (literature: ≈ 2.1) |
>
> A factor of **2** difference — and the naive method appears in a frightening number of papers.
> (Project 01 computes both.)

### 1.3 Small world, clustering, centralities

- **Small world**: despite 10,000+ nodes the mean shortest path is tiny (~3–4 hops in the AS
  graph). "Six degrees of separation" — in the internet rather three. Cause: the hubs.
- **Clustering coefficient** $C_i = \frac{2|\{(j,k)\in E: j,k\in N(i)\}|}{d_i(d_i-1)}$ — "how many
  of my neighbours are connected to each other?" Actually measured: **0.297** — far above the
  random graph, typical for real networks.
- **Centralities** — different answers to "who is important?":
  - **Degree**: many direct neighbours (local).
  - **Betweenness** $g(v)=\sum_{s\neq v\neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}$: on how many
    shortest paths does $v$ lie? ⟹ **the bottleneck/wiretapping point par excellence.**
    Expensive: $O(|V||E|)$ (Brandes).
  - **Closeness**: mean distance to all others ⟹ good for cache/server placement.
  - **Eigenvector/PageRank**: what matters is who has *important* neighbours (recursively).

### 1.4 Robustness: the most famous network result

**Albert, Jeong & Barabási (2000, Nature):** scale-free networks are **extremely robust against
random failures** and **extremely fragile against targeted attacks on hubs**. The reason is the
degree distribution itself: a node hit *at random* is with high probability a leaf of degree 1–2 —
its failure disturbs nobody. A hub hit *deliberately* tears thousands of connections down with it.

On the **real** AS topology (project 01 reproduces this):

| nodes removed | random failure | targeted (by degree) |
|---|---|---|
| 1 % | 0.989 | **0.413** |
| 5 % | 0.939 | **0.003** |
| 10 % | 0.800 | 0.001 |

*(share of nodes in the largest connected component.)*

**Remove 5 % of the hubs — and the internet crumbles to dust** (0.3 %), while 10 % of random
failures barely bother it. That is at once a security, a resilience and a regulatory statement.

---

## 2 · Graph representation learning

**The problem:** ML wants vectors, but a graph is combinatorial (and nodes have no canonical
ordering). How do you get usable features out of structure? Three generations:

### 2.1 Generation 1: handcrafted graph features

For every node: degree, clustering, centralities, neighbourhood statistics → an ordinary feature
vector → an ordinary sklearn model (module 04). **Simple, fast, interpretable** — and a serious
baseline that you first have to beat (project 01).

For **edge** (prediction) there are the classic similarity heuristics. With $N(u)$ = the
neighbourhood of $u$:

| Heuristic | Formula | Idea |
|---|---|---|
| **Common neighbors** | $\|N(u)\cap N(v)\|$ | mutual acquaintances |
| **Jaccard** | $\frac{\|N(u)\cap N(v)\|}{\|N(u)\cup N(v)\|}$ | normalized by neighbourhood size |
| **Adamic-Adar** | $\sum_{w\in N(u)\cap N(v)}\frac{1}{\log d_w}$ | rare common neighbours count more |
| **Preferential attachment** | $d_u\cdot d_v$ | hubs connect with hubs |

**Adamic-Adar** is considered the strongest of these heuristics — the weighting $1/\log d_w$ says:
a common neighbour of degree 3 is strong evidence, a common neighbour of degree 2000 (a tier-1 hub
that *everybody* is connected to) says almost nothing. Exactly the right intuition for scale-free
networks.

**Preferential attachment** is the odd one out: it looks **only** at the degrees and ignores the
neighbourhood completely. Remember that — in project 01 it "wins" anyway, and clearing up this
contradiction is one of the most important lessons of the module (section 2.5).

### 2.2 Generation 2: node2vec / DeepWalk

**The idea:** "word2vec for graphs" (module 08!). Run **random walks** on the graph, treat every
walk as a *sentence* and nodes as *words*, and train skip-gram. Nodes that appear in similar
contexts get similar embeddings.

**node2vec** steers the walks with two parameters between two extremes:
- **BFS-like** (local) ⟹ embeddings encode the **structural role** (hub? bridge? leaf?)
- **DFS-like** (far away) ⟹ embeddings encode **community membership**

**Limitation:** purely **transductive** and **feature-blind**. Every node gets one fixed learned
vector; if a new node arrives, you have to **retrain**. Node features (e.g. traffic statistics)
cannot be used by node2vec at all. That is exactly what GNNs solve.

### 2.3 Generation 3: graph neural networks

**The message-passing framework** — the common denominator of *all* GNNs. In layer $k$, for every
node $v$:
$$\mathbf h_v^{(k)} = \underbrace{\text{UPDATE}}_{\text{what do I do with it?}}\Big(\mathbf h_v^{(k-1)},\ \underbrace{\text{AGGREGATE}}_{\text{what do the neighbours say?}}\big(\{\mathbf h_u^{(k-1)}: u\in N(v)\}\big)\Big)$$

Every layer = **one hop** further. After $k$ layers a node "sees" its $k$-hop neighbourhood. Since
AGGREGATE runs over a **set**, it has to be **permutation invariant** (sum, mean, max) — nodes have
no ordering.

#### GCN (Kipf & Welling 2017) — derived in full

The naive approach "average the neighbours and multiply by a weight matrix" is
$H' = \sigma(A H W)$. Three problems, three repairs:

1. **The node forgets itself.** $A$ has a zero diagonal ⟹ $\mathbf h_v$ does not enter the sum.
   Repair: **self-loops**, $\tilde A = A + I$.
2. **The scales explode.** $\tilde A H$ *sums* the neighbours — a hub of degree 2312 gets
   activations ~1000× larger than a leaf of degree 2. After a few layers this diverges. Repair:
   **normalize**. Row-wise $\tilde D^{-1}\tilde A$ = the mean of the neighbours.
3. **Why then $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$?** The *symmetric* normalization splits an
   edge $(u,v)$ between **both** endpoints:
   $$\hat A_{uv}=\frac{1}{\sqrt{d_u}\sqrt{d_v}}.$$
   This has three advantages: (a) $\hat A$ stays **symmetric** (with $\tilde D^{-1}\tilde A$ it
   does not!), (b) the message of a hub to a leaf is damped by $\sqrt{d_{\text{hub}}}$ — a hub
   connected to *everybody* simply carries little information (the same intuition as
   **Adamic-Adar** and as **IDF** from module 08!), (c) the eigenvalues of $\hat A$ lie in
   $[-1,1]$ ⟹ **stable depth**. Formally this is a spectral graph theory approximation
   (first-order Chebyshev on the normalized Laplacian $L=I-\hat A$) — hence the name *spectral*
   GNN.

With that the **GCN layer** reads:
$$\boxed{\;H^{(k)}=\sigma\big(\hat A\,H^{(k-1)}\,W^{(k)}\big),\qquad \hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\;}$$

That is **all of it**. Two lines of mathematics, ~15 lines of PyTorch — and $\hat A$ is computed
**once** in advance (sparse). Project 02 builds exactly this.

#### GraphSAGE (Hamilton et al. 2017) — inductive

GCN is **transductive**: $\hat A$ contains the whole graph, a new node forces recomputation.
**GraphSAGE** instead *samples* a fixed number of neighbours and learns an **aggregator
function**:
$$\mathbf h_v^{(k)}=\sigma\Big(W\cdot\text{CONCAT}\big(\mathbf h_v^{(k-1)},\ \text{AGG}(\{\mathbf h_u^{(k-1)}\})\big)\Big)$$
Because the *function* is learned (not one vector per node), it can be applied to **never seen**
nodes/graphs — **inductively**. The neighbour sampling additionally makes it scalable on huge
graphs (hubs would otherwise blow up every batch). For networks with constantly new nodes this is
the practically relevant approach.

#### GAT (Veličković et al. 2018) — attention

GCN weights neighbours **fixedly** by degree. **GAT** *learns* the weights:
$$\alpha_{vu}=\frac{\exp\big(\text{LeakyReLU}(\mathbf a^\top[W\mathbf h_v\,\|\,W\mathbf h_u])\big)}{\sum_{w\in N(v)}\exp\big(\text{LeakyReLU}(\mathbf a^\top[W\mathbf h_v\,\|\,W\mathbf h_w])\big)},\qquad
\mathbf h_v'=\sigma\Big(\sum_{u\in N(v)}\alpha_{vu}W\mathbf h_u\Big)$$
That is **exactly** the attention from module 09 — only over the neighbourhood instead of over all
tokens. Conversely: **a transformer is a GAT on the complete graph** (every token is connected to
every other, positional encoding replaces the edges). That is not an analogy, it is the same
computation.

### 2.4 Over-smoothing: why GNNs stay shallow

Deep networks are normal in vision/NLP (50+ layers). GNNs mostly have **2–3**. Reason: every layer
averages over neighbours. After $k$ layers $H^{(k)}\approx\hat A^k H W\dots$ — and $\hat A^k$
converges for large $k$ to the stationary distribution of a random walk. **All node vectors become
similar to each other**, the information collapses (formally: the eigenvalues $<1$ die out, only
the one belonging to eigenvalue 1 survives). This is called **over-smoothing**. It is aggravated
in a **small-world** network: at a mean path of ~4, a 4-layer GNN already sees *the whole graph* —
every node then aggregates practically the same thing. Countermeasures: residual/skip connections,
**jumping knowledge**, PairNorm — or simply: stay shallow.

### 2.5 Link prediction

**Task:** given the graph without some edges — which ones are missing? **Applications in the
network:** topology inference (the measured AS topology is notoriously **incomplete** — many
peerings are seen by no route collector), predicting future peerings, anomaly detection (an edge
that "should not be there").

**The big advantage:** link prediction needs **no labels** — the graph labels itself. You remove
part of the edges (**positive** examples), draw an equal number of non-existent node pairs
(**negative**), and check whether the model separates them.

> ### ⚠️ Two traps that regularly ruin link prediction results
> 1. **Leakage through the graph itself.** The test edges have to be **removed from the graph**
>    before you compute features/embeddings. Otherwise you compute "common neighbors" on a graph
>    that already contains the edge you are looking for — and measure memorization. This is the
>    graph variant of the leakage problem from module 15 (3.5).
> 2. **The base rate — once again.** A graph with $n=10\,670$ has ~57 million possible pairs, but
>    only 22,000 edges ⟹ the true base rate is $\pi\approx 3.9\cdot10^{-4}$. If you evaluate (as
>    is customary) with **1:1** positive:negative, you are measuring at $\pi=0.5$ — and the
>    **base-rate fallacy from module 15 (3.1)** hits in full force: an AUC of 0.95 on balanced data
>    means, in real deployment (trying all pairs), almost nothing but false alarms. Balanced
>    sampling is convenient — but you have to say that it does **not** reflect reality.
> 3. **The degree confound.** If you draw the negative pairs **uniformly at random** (the
>    standard!), in a scale-free graph they are almost always **leaf × leaf** — real edges, by
>    contrast, involve hubs disproportionately often. Measured on the real AS topology: median
>    degree product **500** (real edges) vs. **4** (random pairs). Any degree-based quantity
>    separates that almost trivially, **without knowing anything about structure**. Consequence
>    (project 01 measures it):
>
>    | Heuristic | uniform negatives | **degree-matched** negatives |
>    |---|---|---|
>    | Adamic-Adar | 0.724 | **0.566** ← now the best |
>    | Common neighbors | 0.719 | 0.559 |
>    | **Preferential attachment** | **0.763** ← "the winner" | **0.406** ← chance level |
>
>    **PA crashes from 0.763 to 0.406** — its entire lead was an **artifact of the negative
>    sampling**, not a signal. And all methods drop to ~0.56: the task is *far* harder than the
>    naive evaluation suggests. **It is not only the model that can deceive — the construction of
>    the test set already decides what you measure.**

---

## 3 · Network time series & forecasting

### 3.1 The task and the structure

Predict the traffic $y_t$ on a link/node — for capacity planning, autoscaling, energy saving
(switching off links at night) and as a **baseline for anomaly detection** ("more traffic than
predicted = suspicious"). Network traffic has a very friendly property: **strong, stable
seasonality**.
- **Daily rhythm** (period 24 h): little at night, a peak in the evening ("prime time").
- **Weekly rhythm** (period 7 d): weekday ≠ weekend.
- **Trend**: long-term growth.
- **Noise/bursts**: network traffic is known to be **self-similar/long-range dependent**
  (Leland et al. 1994) — bursts on *all* time scales, not Poisson-smooth.

### 3.2 Baselines — and why they are half the battle

> **The most important rule of forecasting:** beat the trivial baseline first, then talk about
> models.

- **Naive**: $\hat y_{t+1}=y_t$ ("tomorrow like today").
- **Seasonal-naive**: $\hat y_{t+h}=y_{t+h-m}$ with season length $m$ ("next Tuesday 8 p.m. like
  last Tuesday 8 p.m."). **For network traffic this is frighteningly strong** — every expensive
  model has to measure itself against it first. Whoever does not report this baseline reports
  nothing.
- **Mean/drift**: a reference towards the bottom.

Metrics: **MAE**, **RMSE** (punishes outliers more strongly — relevant for bursty traffic),
**MAPE** (in percent, but explodes at $y_t\approx0$ ⟹ unusable at night), and
**MASE** = MAE relative to the naive baseline (**<1 = better than naive**; scale-free and
therefore the cleanest comparison across different links).

### 3.3 Classical: ARIMA / SARIMA (formally)

With the **backshift operator** $B y_t = y_{t-1}$:

- **AR(p)** — the value depends on its $p$ predecessors:
  $\phi(B)y_t=\varepsilon_t$ with $\phi(B)=1-\phi_1B-\dots-\phi_pB^p$.
- **MA(q)** — the value depends on the last $q$ **shocks**:
  $y_t=\theta(B)\varepsilon_t$ with $\theta(B)=1+\theta_1B+\dots+\theta_qB^q$.
- **I(d)** — **differencing** $\nabla^d=(1-B)^d$ makes the series **stationary** (removes the
  trend).

**ARIMA(p,d,q)**: $\phi(B)\,(1-B)^d y_t=\theta(B)\,\varepsilon_t$.

**SARIMA(p,d,q)(P,D,Q)$_m$** adds the season with period $m$:
$$\Phi(B^m)\,\phi(B)\,(1-B^m)^D(1-B)^d\,y_t=\Theta(B^m)\,\theta(B)\,\varepsilon_t.$$
The term $(1-B^m)^D$ is the **seasonal difference** — at its core the seasonal-naive baseline,
built into the model. The order is chosen classically via ACF/PACF or AIC/BIC (Box-Jenkins).

*(No code demo here: `statsmodels` is not installed in this environment. The projects use lag
features + ML instead — which in practice is often stronger and more flexible anyway.)*

### 3.4 ML on lag features

The pragmatic, usually best route: time series → **table**. Features per time point: lags
($y_{t-1},y_{t-2},\dots,y_{t-m},y_{t-2m}$), rolling means/stds, and **calendar features**
(hour, weekday) — the latter best **encoded cyclically**:
$$\sin\!\Big(\frac{2\pi\,\text{hour}}{24}\Big),\ \cos\!\Big(\frac{2\pi\,\text{hour}}{24}\Big)$$
so that 11 p.m. and midnight are adjacent (as raw numbers they would be maximally far apart!).
On top of that: ridge or gradient boosting (modules 04/05). For sequences: an LSTM (module 09) —
but with strong seasonality and little data often **worse** than ridge with good lag features.

> **Evaluation: time-based only.** For time series a random split is **fatal** — you train on the
> future and test on the past. Correct: a **temporal split** or **rolling-origin/walk-forward
> validation** (`TimeSeriesSplit`). This is the same leakage rule as in module 15 (3.5) — just
> even more obvious here.

### 3.5 The synthesis: spatio-temporal GNNs

Now both halves come together. Traffic on a network is **simultaneously** correlated in time
(seasonality) **and in space** (neighbouring links carry the same traffic — what flows over link A
flows over B right afterwards). One model per link ignores the topology; a GNN without time
ignores the seasonality.

**Spatio-temporal GNNs** combine both. The data structure is a tensor
$X\in\mathbb R^{T\times n\times f}$ (time × nodes × features), and the architecture alternates
between:
- **spatial** aggregation (a GNN layer over the topology) and
- **temporal** aggregation (1D convolution, GRU/LSTM or attention over $T$).

Well-known representatives: **STGCN** (graph conv + temporal conv), **DCRNN** (diffusion
convolution + GRU; models traffic explicitly as a **diffusion process** on the graph),
**Graph WaveNet** (learns the adjacency along the way — useful when the true topology is unknown).

> **An honest classification:** spatio-temporal GNNs are the flagship of this field — and at the
> same time it holds in many published comparisons that a **well-made seasonal-naive or lag-ridge
> baseline** comes astonishingly close. The topology only helps if the **spatial correlation** is
> really strong and the temporal structure does not already explain everything anyway.
> **Project 03 measures exactly that** — instead of claiming it.

---

## 4 · Advanced

### 4.1 Encrypted traffic analysis

Module 15 (2.1) established: encryption protects the **content**, not the **metadata**. Thought
through to the end this becomes **website fingerprinting**: an observer (a Wi-Fi operator, an ISP,
a Tor exit observer) sees only **packet sizes, directions and timing** — and recognizes from that
*which page* you are visiting. Because every page has a characteristic loading profile (number and
size of the resources). The features are size/direction sequences, burst patterns, inter-arrival
times; the models are classical (kNN/RF) or CNNs/transformers on the raw sequence.

**Countermeasures** all cost bandwidth or latency: **padding** (to fixed sizes), **traffic
shaping** (constant rate), **dummy traffic**, multiplexing. And: the lab results ("95 %
accuracy!") are mostly **closed world** (only 100 candidate pages). In the **open world**
(millions of pages, mostly uninteresting) the **base-rate fallacy** from module 15 strikes — of
course. The same technique for good: QoE monitoring by the operator ("is that video playing
smoothly right now?") without decrypting the payload.

### 4.2 Self-driving networks

The vision: networks that observe, understand and optimize **themselves** — the consequence of
modules 15+16.
- **Intent-based networking**: the operator says *what* they want ("video conferences < 50 ms"),
  not *how* (router configurations). The system translates, applies, verifies, corrects.
- **Digital twin**: a running simulation model of the real network — for "what if" without risk.
  This is exactly where **model-based learning** pays off (module 14, 4.4).
- **Closed loop**: measure → understand → decide → act → measure. The network variant of the
  agent-environment loop from module 13.
- **Reality check**: network operations are **safety-critical**. A model with 99 % accuracy that
  misconfigures the backbone 1 % of the time is unacceptable. That is why **explainability**,
  **verification**, conservative fallbacks and limited autonomy are not optional extras here but
  prerequisites. This explains why the vision has been discussed since ~2017 and is deployed only
  hesitantly.

### 4.3 Routing & traffic engineering as a decision problem

Here this module meets the RL block (13/14). **Traffic engineering**: how do I lay out
paths/weights in order to minimize utilization and delay? Classically: optimization
(multi-commodity flow, LP) on an **estimated** traffic matrix. With RL: state = utilization +
topology (→ **a GNN as the encoder**!), action = routing weights, reward = −utilization/delay.

**Why GNN + RL belong together here:** an MLP on a fixed adjacency matrix would have to relearn for
**every** topology. A GNN **generalizes across topologies** — exactly what networks that change
constantly need. (Cf. **RouteNet**, which predicts QoS metrics from topology + routing + traffic
matrix.)

**But:** the comparison has to run honestly against **classical optimization** — which, given a
known model, is often **exactly optimal**. That is precisely the lesson from the module 14 finale:
*if you know your model, take optimization. RL is for the case where you do not know it.*

### 4.4 Scaling and privacy

- **Scaling:** millions of nodes do not fit into one batch. Solutions: neighbour sampling
  (GraphSAGE), Cluster-GCN (partition the graph), historical embeddings.
- **Federated learning:** operators do not want to (or may not) share their traffic data (trade
  secrets, GDPR). Solution: share models instead of data. Marginally relevant: even the
  **topology** is a trade secret (peering relations).
- **Measurement bias:** **nobody** knows the "real" AS topology. Route collectors only see what
  BGP shows them; **peer-to-peer links between small AS are systematically missing**. So you train
  on an **incomplete, biasedly measured** graph — and the missing edges are not distributed at
  random. This has to be kept in mind for *every* statement about internet topology (including our
  robustness table).

---

## 5 · Summary / cheat sheet

**Graph.** $G=(V,E)$, $A$ (sparse!), $D=\operatorname{diag}(d_i)$. AS level ≠ router level ≠
traffic graph.

**Scale-free.** $P(d)\propto d^{-\alpha}$, internet AS $\alpha\approx2.1$. Arises through
**preferential attachment**. **Estimate the exponent by MLE**
($\hat\alpha=1+n[\sum\ln\frac{d_i}{d_{\min}-0.5}]^{-1}$), **never** by a log-log fit (a factor of 2
off!).

**Robustness.** Random failure: harmless (10 % → 0.80). **Hub attack: fatal (5 % → 0.003).**

**Message passing.** $\mathbf h_v^{(k)}=\text{UPDATE}(\mathbf h_v^{(k-1)},\text{AGGREGATE}(\{\mathbf h_u\}))$,
AGGREGATE **permutation invariant**.

**GCN.** $H^{(k)}=\sigma(\hat A H^{(k-1)}W^{(k)})$, $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$.
Self-loops (do not forget yourself) + symmetric normalization (tame the scales, damp the hubs,
eigenvalues in $[-1,1]$).

**Families.** GCN (transductive, fixed weights) · **GraphSAGE** (inductive, sampling) ·
**GAT** (attention = module 09; a transformer = GAT on the complete graph).

**Over-smoothing.** $\hat A^k$ → stationary ⟹ all nodes alike ⟹ GNNs stay at **2–3 layers**.

**Link prediction.** Common neighbors · Jaccard · **Adamic-Adar** ($\sum 1/\log d_w$) ·
pref. attachment. Three traps: **remove the test edges BEFORE computing features** · the true base
rate is $\approx3.9\cdot10^{-4}$ instead of 0.5 · the **degree confound** (uniformly drawn
negatives are leaf×leaf ⟹ PA 0.763 → **0.406** with degree-matched negatives).

**Forecasting.** Beat **seasonal-naive** first! · lags + cyclic calendar features
($\sin/\cos$) · **MASE < 1** = better than naive · SARIMA
$\Phi(B^m)\phi(B)(1-B^m)^D(1-B)^d y_t=\Theta(B^m)\theta(B)\varepsilon_t$ · **split temporally!**

**Spatio-temporal.** $X\in\mathbb R^{T\times n\times f}$; STGCN/DCRNN/Graph WaveNet =
spatial + temporal aggregation in alternation.

---

## 6 · Self-test

<details>
<summary><b>1.</b> What does "scale-free" mean, and why is the mean degree not very informative for it?</summary>

$P(d)\propto d^{-\alpha}$ — there is **no typical scale**: most nodes have degree 1–2, a few hubs
thousands. The mean lies (as with all heavy tails, cf. the flow sizes in module 15) somewhere in
between and describes **nobody**: in reality median 2, mean 4.12, maximum 2312. The cause:
**preferential attachment** during growth.
</details>

<details>
<summary><b>2.</b> Why is the naive log-log fit wrong for the power-law exponent?</summary>

In the tail there are only 0–3 observations per bin; their logarithmized noise is strongly biased,
empty bins disappear entirely — the least-squares line gets bent. The correct approach is the
**MLE** (Clauset et al. 2009) on the tail $d\ge d_{\min}$. Actually measured: naive **−1.14** vs.
MLE **2.08** — a factor of 2.
</details>

<details>
<summary><b>3.</b> Explain why the internet is robust against random failures and fragile against hub attacks.</summary>

Because of the degree distribution. A node hit **at random** is almost certainly a leaf (degree
1–2) — its failure disturbs nothing. A hub hit **deliberately** tears thousands of edges down with
it. In reality: 10 % at random → 80 % stay connected; **5 % of the hubs → 0.3 %**
(Albert/Jeong/Barabási 2000).
</details>

<details>
<summary><b>4.</b> Formulate message passing. Why does AGGREGATE have to be permutation invariant?</summary>

$\mathbf h_v^{(k)}=\text{UPDATE}(\mathbf h_v^{(k-1)},\ \text{AGGREGATE}(\{\mathbf h_u^{(k-1)}:u\in N(v)\}))$.
AGGREGATE runs over a **set** — neighbours have **no canonical ordering**. If the function depended
on the order, the output would depend on the arbitrary node numbering. Hence sum/mean/max.
</details>

<details>
<summary><b>5.</b> Why $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ — what does each part achieve?</summary>

**$+I$ (self-loops):** otherwise $\mathbf h_v$ drops out of its own update — the node forgets
itself. **$\tilde D^{-1/2}\cdot\tilde D^{-1/2}$:** normalizes, otherwise the scales explode (a hub
of degree 2312 vs. a leaf of degree 2). **Symmetric** instead of row-wise, because (a) $\hat A$
stays symmetric, (b) the message of a hub gets damped (hubs are uninformative — like
**Adamic-Adar**/IDF), (c) the eigenvalues stay in $[-1,1]$ ⟹ stable depth.
</details>

<details>
<summary><b>6.</b> GCN vs. GraphSAGE vs. GAT — one sentence each.</summary>

**GCN**: fixed, degree-based weights, needs the **whole** graph ⟹ **transductive**.
**GraphSAGE**: **samples** neighbours and learns an **aggregator function** ⟹ **inductive**
(works on new nodes/graphs) and scalable. **GAT**: **learns** the neighbour weights via
**attention** (= module 09, only over the neighbourhood).
</details>

<details>
<summary><b>7.</b> What is over-smoothing, and why are GNNs shallow because of it?</summary>

Every layer averages over neighbours; $\hat A^k$ converges to the stationary distribution of a
random walk ⟹ **all node vectors become equal**, the information collapses. Hence 2–3 layers. In a
**small-world** network (mean path ~4) this is aggravated: a 4-layer GNN already sees the whole
graph.
</details>

<details>
<summary><b>8.</b> Name the two traps in link prediction.</summary>

(1) **Leakage through the graph**: the test edges have to be removed **before** you compute
features/embeddings — otherwise common neighbors already "sees" the edge you are looking for.
(2) **The base rate**: in reality $\pi\approx4\cdot10^{-4}$ (22,000 edges among 57 million pairs),
but evaluation is mostly **1:1 balanced** ⟹ the base-rate fallacy (module 15): a great AUC, and in
real deployment almost nothing but false alarms.
</details>

<details>
<summary><b>9.</b> What is the seasonal-naive baseline, and what does MASE < 1 say?</summary>

$\hat y_{t+h}=y_{t+h-m}$ ("like last week at the same time"), $m$ = the season length. For network
traffic **very strong**. **MASE** = MAE relative to the naive baseline: **< 1 = better than
naive**, ≥ 1 = the model is not worth its money. Scale-free ⟹ comparable across different links.
</details>

<details>
<summary><b>10.</b> Why encode hour/weekday cyclically as sin/cos?</summary>

As a raw number, **11 p.m. and midnight would be maximally far apart** (distance 23), even though
they are adjacent. $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ map the hour onto a **circle** — 23 and 0
then lie right next to each other. Two components are needed because one alone is not unique.
</details>

<details>
<summary><b>11.</b> What is a spatio-temporal GNN, and when is it worth it?</summary>

A model on $X\in\mathbb R^{T\times n\times f}$ that alternates **spatial** aggregation (a GNN over
the topology) and **temporal** aggregation (conv/GRU/attention) (STGCN, DCRNN, Graph WaveNet). It
is only worth it if the **spatial correlation** really carries additional information — otherwise
the seasonality alone explains everything and the seasonal-naive baseline keeps up.
</details>

<details>
<summary><b>12.</b> Why does nobody know the real AS topology — and what does that mean for our results?</summary>

Route collectors only see what BGP shows them; **peer-to-peer links between small AS are
systematically missing** (they are announced nowhere). So the measured graph is **incomplete and
biased**, and the missing edges are **not** distributed at random. Every statement — including our
robustness table — stands under this caveat.
</details>

---

## 7 · Literature & sources

**Graph/network foundations:**
- 📗 **Barabási — *Network Science*** (networksciencebook.com) — **completely free online**,
  excellently illustrated; chapters 4 (scale-free) and 8 (robustness) cover section 1.
  *Beginner-friendly.* **Best single source for part 1.**
- 📄 **Faloutsos, Faloutsos & Faloutsos (1999), *On Power-Law Relationships of the Internet
  Topology*** — the paper that started it all.
- 📄 **Albert, Jeong & Barabási (2000), *Error and Attack Tolerance of Complex Networks***
  (Nature) — the robustness result from 1.4.
- 📄 **Clauset, Shalizi & Newman (2009), *Power-Law Distributions in Empirical Data*** — why the
  log-log fit is wrong and how to do it right. *In depth, very instructive.*

**Graph learning:**
- 📗 **Hamilton — *Graph Representation Learning Book*** (free, cs.mcgill.ca/~wlh/grl_book/) —
  the standard work. *Beginner→in depth.* **Best single source for part 2.**
- 📄 **Kipf & Welling (2017), *Semi-Supervised Classification with GCNs*** — the GCN paper.
- 📄 **Hamilton et al. (2017), *Inductive Representation Learning on Large Graphs*** (GraphSAGE).
- 📄 **Veličković et al. (2018), *Graph Attention Networks*** (GAT).
- 📄 **Grover & Leskovec (2016), *node2vec***.
- 🎥 **Stanford CS224W — *Machine Learning with Graphs*** (Leskovec; videos + slides free) — the
  best course on the topic. *In depth.*
- 🌐 **PyTorch Geometric docs** (pyg.org) — the standard library. In this module we deliberately
  build from scratch; for practical work you should know PyG.

**Time series:**
- 📗 **Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*** (otexts.com/fpp3) —
  **free online**, the standard work; chapters 5 (baselines, MASE) and 9 (ARIMA/SARIMA).
  *Beginner-friendly.* **Best single source for part 3.**
- 📄 **Leland et al. (1994), *On the Self-Similar Nature of Ethernet Traffic*** — the classic:
  network traffic is **not** Poisson.
- 📄 **Yu et al. (2018), *STGCN***; **Li et al. (2018), *DCRNN***; **Wu et al. (2019),
  *Graph WaveNet*** — spatio-temporal GNNs. *In depth.*

**Application/vision:**
- 📄 **Rusek et al. (2020), *RouteNet*** — a GNN predicts QoS from topology + routing + traffic.
- 📄 **Feamster & Rexford (2018), *Why (and How) Networks Should Run Themselves*** — self-driving
  networks. *Beginner-friendly.*
- 📄 **Panchenko et al. (2016), *Website Fingerprinting at Internet Scale*** — including the
  sobering open-world reality.

**Data:**
- 🌐 **SNAP** (snap.stanford.edu/data) — real network graphs, among them the **Oregon AS
  topology** of the projects. *Free, small downloads.*
- 🌐 **CAIDA** (caida.org) — the reference for internet measurement data (partly requires
  registration).
- 🌐 **SNDlib** (sndlib.zib.de) — real backbone topologies **including traffic matrices**.

---

## Next module

With this the network block (15+16) is complete. **Module 17 — Core XR: Principles of Interactive
Systems** follows, and with it a completely new field. What you have learned here — graphs as a
data structure, GNNs, time series, and above all the **discipline of measuring against honest
baselines** — carries far beyond networks: 3D point clouds (module 20) are graphs, robotics
(21/22) is time series + control, and self-aware computing (24) is the idea from 4.2 in general.

---

# Modul 16 — Machine Learning for Networks 2 (deutsche Fassung)

> **Worum geht es?** Modul 15 behandelte jedes Netzwerkereignis für sich: ein **Flow**, ein
> Feature-Vektor, eine Vorhersage. Aber ein Netz ist kein Haufen unabhängiger Zeilen — es ist
> ein **Graph**, und Verkehr ist eine **Zeitreihe**. Dieses Modul nimmt genau die zwei
> Strukturen ernst, die Modul 15 ignoriert hat: **Struktur im Raum** (Topologie → *Graph Neural
> Networks*) und **Struktur in der Zeit** (Saisonalität → *Forecasting*) — und am Ende beides
> zusammen (**Spatio-Temporal Learning**). Dazu kommt der Blick nach vorn: verschlüsselte
> Verkehrsanalyse und **selbstfahrende Netze**.

**Hilfreiche Vorkenntnisse:** Lineare Algebra (Matrixmultiplikation, Eigenwerte),
Wahrscheinlichkeitsrechnung, PyTorch-Grundlagen.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 15 (ML for Networks 1)** — *zwingend*. Flows, Messung, **Klassenungleichgewicht**,
  **Base-Rate-Fallacy**, PR-statt-ROC, **Concept Drift**, Leakage. Das alles gilt hier
  unverändert weiter und wird **nicht** wiederholt.
- **Modul 05 (ML 2)** und **Modul 09 (NLP 2)** — neuronale Netze, PyTorch, und die
  **Attention**-Idee (GAT ist Attention auf Graphen; ein Transformer *ist* ein GNN auf dem
  vollständigen Graphen).
- **Modul 13/14 (RL)** — nur für Abschnitt 4.3 (Routing/Traffic Engineering als
  Entscheidungsproblem).

> **Hinweis zur Ausgestaltung.** Wie bei Modul 15 lag keine offizielle Modulbeschreibung vor.
> Ich löse hier ein, was das Modul-15-Skript als Fortsetzung angekündigt hat: **Graph-basiertes
> Lernen, Netzwerk-Zeitreihen, verschlüsselte Verkehrsanalyse, selbstlernende Netze**.
>
> **Werkzeug-Entscheidung:** `torch_geometric` (die Standard-GNN-Bibliothek) ist hier **nicht
> installiert**, `statsmodels` (ARIMA) ebenfalls nicht. Das ist didaktisch ein Glücksfall: wir
> bauen das **GCN from scratch** in reinem PyTorch (es sind ~15 Zeilen — und danach ist die
> Bibliothek entzaubert) und das Forecasting ebenso. ARIMA/SARIMA erkläre ich vollständig
> **formal**, ohne Paket-Demo.

---

## Lernziele

Nach diesem Modul kannst du …

- ein Netz als **Graph** modellieren und seine Struktur charakterisieren: **Gradverteilung**,
  **Power-Law/Scale-Free**, **Small-World**, **Zentralitäten**, Clustering;
- erklären, **warum das Internet robust gegen Zufallsausfälle und fragil gegen gezielte
  Hub-Angriffe** ist — und es an echten Daten zeigen;
- **Power-Law-Exponenten korrekt schätzen** (MLE statt naivem log-log-Fit) und begründen,
  warum der naive Weg systematisch falsch liegt;
- **Graph Representation Learning** einordnen: handgebaute Graph-Features → **node2vec** →
  **GNNs**;
- das **Message-Passing-Framework** formulieren und **GCN** vollständig herleiten
  (inkl. der Frage, warum $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ *so* aussieht);
  **GraphSAGE** (induktiv) und **GAT** (Attention) abgrenzen; **Over-Smoothing** erklären;
- **Link Prediction** als Aufgabe formulieren und Heuristiken (Common Neighbors, Jaccard,
  **Adamic-Adar**, Preferential Attachment) gegen gelernte Verfahren stellen;
- **Netzwerk-Zeitreihen** vorhersagen: Saisonalität, die **saisonal-naive Baseline**,
  Lag-Features, ARIMA/SARIMA (formal), und **Spatio-Temporal GNNs**;
- **verschlüsselte Verkehrsanalyse** (Website Fingerprinting) und **Self-Driving Networks**
  (Intent-Based Networking, Digital Twin, RL-Routing) einordnen.

---

## 1 · Das Netz als Graph

### 1.1 Welcher Graph eigentlich?

„Das Netzwerk" ist mehrdeutig — je nach Abstraktionsebene bekommt man einen anderen Graphen:

| Ebene | Knoten | Kanten | Größenordnung |
|---|---|---|---|
| **AS-Level** | Autonome Systeme (Provider, Firmen) | BGP-Peerings | ~75 000 heute |
| **Router-Level** | einzelne Router | physische Links | Millionen |
| **PoP/Backbone** | Standorte eines Betreibers | Glasfaserstrecken | ~10–100 |
| **Overlay/Verkehr** | Hosts | beobachtete Flows | riesig, dynamisch |

Die Projekte nutzen die **AS-Level-Topologie** (echte Daten, siehe unten). Wichtig: Der
**Verkehrsgraph** (wer redet mit wem) ist *nicht* der **Topologiegraph** (was ist physisch
verbunden) — eine Verwechslung, die viele Papers unbemerkt begehen.

Formal: $G=(V,E)$, **Adjazenzmatrix** $A\in\{0,1\}^{n\times n}$ mit $A_{ij}=1$ ⟺ $(i,j)\in E$,
**Gradmatrix** $D=\operatorname{diag}(d_1,\dots,d_n)$, $d_i=\sum_j A_{ij}$. Ungerichtet ⟹
$A=A^\top$. Bei $n=10\,670$ Knoten hätte ein *dichtes* $A$ schon 114 Mio. Einträge (~455 MB) —
bei 22 000 Kanten sind aber **99,96 %** davon Null (nur 0,04 % besetzt). **Netzgraphen sind
extrem dünn besetzt; man rechnet grundsätzlich sparse.**

### 1.2 Die Gradverteilung: Scale-Free

Die berühmteste Eigenschaft der Internet-Topologie (Faloutsos³, 1999): die Gradverteilung folgt
näherungsweise einem **Potenzgesetz**
$$P(d) \propto d^{-\alpha},\qquad \alpha \approx 2{,}1 \text{ (Internet-AS-Level)}.$$

Das bedeutet: **die meisten AS haben 1–2 Nachbarn, einige wenige „Hubs" haben Tausende.** Kein
typischer Grad, keine sinnvolle „mittlere" Skala — daher *scale-free*. Auf den echten Daten der
Projekte: Median-Grad **2**, Mittelwert **4,12**, Maximum **2312**. Wie bei den Flow-Größen aus
Modul 15 gilt: **der Mittelwert ist bedeutungslos**, wenn die Verteilung schwere Ränder hat.

Warum entsteht das? **Preferential Attachment** (Barabási–Albert): Das Netz *wächst*, und neue
Knoten hängen sich bevorzugt an bereits gut verbundene Knoten (ein neuer ISP kauft Transit bei
einem großen Provider). „Reich wird reicher" ⟹ Potenzgesetz mit $\alpha=3$ im
BA-Grundmodell.

> ### ⚠️ Wie man den Exponenten **nicht** schätzt
> Der naheliegende Weg — Histogramm der Grade, log-log auftragen, Gerade fitten — ist
> **systematisch falsch**. Grund: im Schwanz sitzen pro Bin nur 0–3 Beobachtungen; deren
> logarithmiertes Rauschen ist massiv verzerrt (und alle leeren Bins fallen ganz raus). Die
> Kleinste-Quadrate-Gerade wird dadurch verbogen.
> **Clauset, Shalizi & Newman (2009)** zeigen: man braucht den **Maximum-Likelihood-Schätzer**
> $$\hat\alpha = 1 + n\Big[\sum_{i=1}^{n}\ln\frac{d_i}{d_{\min}-\tfrac12}\Big]^{-1},$$
> angewandt nur auf den Schwanz $d\ge d_{\min}$. Auf unseren echten Daten:
>
> | Methode | Ergebnis |
> |---|---|
> | naiver log-log-Fit (PMF) | **−1,14** ❌ |
> | MLE, $d_{\min}=5$ | 2,12 |
> | **MLE, $d_{\min}=10$** | **2,08** ✅ (Literatur: ≈ 2,1) |
>
> Ein Faktor **2** Unterschied — und die naive Methode steht in erschreckend vielen Papers.
> (Projekt 01 rechnet beides nach.)

### 1.3 Small-World, Clustering, Zentralitäten

- **Small-World**: trotz 10 000+ Knoten ist der mittlere kürzeste Pfad winzig (~3–4 Hops im
  AS-Graph). „Sechs Handschläge" — im Internet eher drei. Ursache: die Hubs.
- **Clustering-Koeffizient** $C_i = \frac{2|\{(j,k)\in E: j,k\in N(i)\}|}{d_i(d_i-1)}$ — „wie
  viele meiner Nachbarn sind untereinander verbunden?" Real gemessen: **0,297** — weit über dem
  Zufallsgraphen, typisch für reale Netze.
- **Zentralitäten** — verschiedene Antworten auf „wer ist wichtig?":
  - **Grad**: viele direkte Nachbarn (lokal).
  - **Betweenness** $g(v)=\sum_{s\neq v\neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}$: auf wie vielen
    kürzesten Pfaden liegt $v$? ⟹ **der Flaschenhals/Abhörpunkt schlechthin.** Teuer:
    $O(|V||E|)$ (Brandes).
  - **Closeness**: mittlere Distanz zu allen anderen ⟹ gut für Cache-/Server-Platzierung.
  - **Eigenvector/PageRank**: wichtig ist, wer *wichtige* Nachbarn hat (rekursiv).

### 1.4 Robustheit: der berühmteste Netzwerk-Befund

**Albert, Jeong & Barabási (2000, Nature):** Scale-free-Netze sind **extrem robust gegen
zufällige Ausfälle** und **extrem fragil gegen gezielte Angriffe auf Hubs**. Der Grund ist die
Gradverteilung selbst: Ein *zufällig* getroffener Knoten ist mit hoher Wahrscheinlichkeit ein
Blatt mit Grad 1–2 — sein Ausfall stört niemanden. Ein *gezielt* getroffener Hub reißt Tausende
Verbindungen mit.

Auf der **echten** AS-Topologie (Projekt 01 reproduziert das):

| entfernte Knoten | zufälliger Ausfall | gezielt (nach Grad) |
|---|---|---|
| 1 % | 0,989 | **0,413** |
| 5 % | 0,939 | **0,003** |
| 10 % | 0,800 | 0,001 |

*(Anteil der Knoten in der größten zusammenhängenden Komponente.)*

**5 % der Hubs entfernt — und das Internet zerfällt in Staub** (0,3 %), während 10 % zufälliger
Ausfälle es kaum jucken. Das ist zugleich eine Sicherheits-, eine Resilienz- und eine
Regulierungsaussage.

---

## 2 · Graph Representation Learning

**Das Problem:** ML will Vektoren, ein Graph ist aber kombinatorisch (und Knoten haben keine
kanonische Reihenfolge). Wie bekommt man aus Struktur brauchbare Merkmale? Drei Generationen:

### 2.1 Generation 1: handgebaute Graph-Features

Für jeden Knoten: Grad, Clustering, Zentralitäten, Nachbarschaftsstatistiken → normaler
Feature-Vektor → normales sklearn-Modell (Modul 04). **Simpel, schnell, interpretierbar** — und
eine ernstzunehmende Baseline, die man erst mal schlagen muss (Projekt 01).

Für **Kanten**(-Vorhersage) gibt es die klassischen Ähnlichkeits-Heuristiken. Mit $N(u)$ =
Nachbarschaft von $u$:

| Heuristik | Formel | Idee |
|---|---|---|
| **Common Neighbors** | $\|N(u)\cap N(v)\|$ | gemeinsame Bekannte |
| **Jaccard** | $\frac{\|N(u)\cap N(v)\|}{\|N(u)\cup N(v)\|}$ | normiert auf Nachbarschaftsgröße |
| **Adamic-Adar** | $\sum_{w\in N(u)\cap N(v)}\frac{1}{\log d_w}$ | seltene gemeinsame Nachbarn zählen mehr |
| **Preferential Attachment** | $d_u\cdot d_v$ | Hubs verbinden sich mit Hubs |

**Adamic-Adar** gilt als die stärkste dieser Heuristiken — die Gewichtung $1/\log d_w$ sagt: ein
gemeinsamer Nachbar mit Grad 3 ist ein starkes Indiz, ein gemeinsamer Nachbar mit Grad 2000
(ein Tier-1-Hub, mit dem *alle* verbunden sind) sagt fast nichts. Genau die richtige Intuition
für scale-free-Netze.

**Preferential Attachment** fällt aus der Reihe: es schaut **nur** auf die Grade und ignoriert
die Nachbarschaft komplett. Merk dir das — in Projekt 01 „gewinnt" es trotzdem, und die
Aufklärung dieses Widerspruchs ist eine der wichtigsten Lektionen des Moduls (Abschnitt 2.5).

### 2.2 Generation 2: node2vec / DeepWalk

**Idee:** „Word2Vec für Graphen" (Modul 08!). Führe **Random Walks** auf dem Graphen aus,
behandle jeden Walk als *Satz* und Knoten als *Wörter*, und trainiere Skip-Gram. Knoten, die in
ähnlichen Kontexten auftauchen, bekommen ähnliche Embeddings.

**node2vec** steuert die Walks mit zwei Parametern zwischen zwei Extremen:
- **BFS-artig** (lokal) ⟹ Embeddings kodieren **strukturelle Rolle** (Hub? Brücke? Blatt?)
- **DFS-artig** (weit weg) ⟹ Embeddings kodieren **Community-Zugehörigkeit**

**Grenze:** rein **transduktiv** und **merkmalsblind**. Jeder Knoten bekommt einen fest
gelernten Vektor; kommt ein neuer Knoten dazu, muss man **neu trainieren**. Knoten-Features
(z. B. Verkehrsstatistiken) kann node2vec gar nicht nutzen. Genau das lösen GNNs.

### 2.3 Generation 3: Graph Neural Networks

**Das Message-Passing-Framework** — der gemeinsame Nenner *aller* GNNs. In Schicht $k$ gilt für
jeden Knoten $v$:
$$\mathbf h_v^{(k)} = \underbrace{\text{UPDATE}}_{\text{was mache ich damit?}}\Big(\mathbf h_v^{(k-1)},\ \underbrace{\text{AGGREGATE}}_{\text{was sagen die Nachbarn?}}\big(\{\mathbf h_u^{(k-1)}: u\in N(v)\}\big)\Big)$$

Jede Schicht = **ein Hop** weiter. Nach $k$ Schichten „sieht" ein Knoten seine
$k$-Hop-Nachbarschaft. Da AGGREGATE über eine **Menge** läuft, muss es
**permutationsinvariant** sein (Summe, Mittel, Max) — Knoten haben keine Reihenfolge.

#### GCN (Kipf & Welling 2017) — vollständig hergeleitet

Der naive Ansatz „mittle die Nachbarn und multipliziere mit einer Gewichtsmatrix" ist
$H' = \sigma(A H W)$. Drei Probleme, drei Reparaturen:

1. **Der Knoten vergisst sich selbst.** $A$ hat Nulldiagonale ⟹ $\mathbf h_v$ geht nicht in
   die Summe ein. Reparatur: **Self-Loops**, $\tilde A = A + I$.
2. **Die Skalen explodieren.** $\tilde A H$ *summiert* die Nachbarn — ein Hub mit Grad 2312
   bekommt Aktivierungen ~1000× größer als ein Blatt mit Grad 2. Nach ein paar Schichten
   divergiert das. Reparatur: **normieren**. Zeilenweise $\tilde D^{-1}\tilde A$ = Mittelwert
   der Nachbarn.
3. **Warum dann $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$?** Die *symmetrische* Normierung
   teilt eine Kante $(u,v)$ zwischen **beiden** Endpunkten auf:
   $$\hat A_{uv}=\frac{1}{\sqrt{d_u}\sqrt{d_v}}.$$
   Das hat drei Vorteile: (a) $\hat A$ bleibt **symmetrisch** (bei $\tilde D^{-1}\tilde A$
   nicht!), (b) die Botschaft eines Hubs an ein Blatt wird durch $\sqrt{d_{\text{Hub}}}$
   gedämpft — ein Hub, der mit *allen* verbunden ist, trägt eben wenig Information (dieselbe
   Intuition wie **Adamic-Adar** und wie **IDF** aus Modul 08!), (c) die Eigenwerte von $\hat A$
   liegen in $[-1,1]$ ⟹ **stabile Tiefe**. Formal ist das eine Spektralgraph-Theorie-Approximation
   (1. Ordnung Chebyshev auf dem normalisierten Laplace $L=I-\hat A$) — daher der Name
   *spectral* GNN.

Damit lautet die **GCN-Schicht**:
$$\boxed{\;H^{(k)}=\sigma\big(\hat A\,H^{(k-1)}\,W^{(k)}\big),\qquad \hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\;}$$

Das ist **alles**. Zwei Zeilen Mathematik, ~15 Zeilen PyTorch — und $\hat A$ berechnet man
**einmal** vorab (sparse). Projekt 02 baut genau das.

#### GraphSAGE (Hamilton et al. 2017) — induktiv

GCN ist **transduktiv**: $\hat A$ enthält den ganzen Graphen, ein neuer Knoten erzwingt
Neuberechnung. **GraphSAGE** *sampelt* stattdessen eine feste Zahl Nachbarn und lernt eine
**Aggregator-Funktion**:
$$\mathbf h_v^{(k)}=\sigma\Big(W\cdot\text{CONCAT}\big(\mathbf h_v^{(k-1)},\ \text{AGG}(\{\mathbf h_u^{(k-1)}\})\big)\Big)$$
Weil die *Funktion* gelernt wird (nicht ein Vektor pro Knoten), lässt sie sich auf **nie
gesehene** Knoten/Graphen anwenden — **induktiv**. Das Nachbar-Sampling macht es zudem auf
riesigen Graphen skalierbar (Hubs würden sonst jeden Batch sprengen). Für Netze mit ständig
neuen Knoten ist das der praxisrelevante Ansatz.

#### GAT (Veličković et al. 2018) — Attention

GCN gewichtet Nachbarn **fest** nach Grad. **GAT** *lernt* die Gewichte:
$$\alpha_{vu}=\frac{\exp\big(\text{LeakyReLU}(\mathbf a^\top[W\mathbf h_v\,\|\,W\mathbf h_u])\big)}{\sum_{w\in N(v)}\exp\big(\text{LeakyReLU}(\mathbf a^\top[W\mathbf h_v\,\|\,W\mathbf h_w])\big)},\qquad
\mathbf h_v'=\sigma\Big(\sum_{u\in N(v)}\alpha_{vu}W\mathbf h_u\Big)$$
Das ist **exakt** die Attention aus Modul 09 — nur über die Nachbarschaft statt über alle
Tokens. Umgekehrt gilt: **ein Transformer ist ein GAT auf dem vollständigen Graphen** (jedes
Token ist mit jedem verbunden, Positional Encoding ersetzt die Kanten). Das ist keine Analogie,
sondern dieselbe Rechnung.

### 2.4 Over-Smoothing: warum GNNs flach bleiben

Tiefe Netze sind in Vision/NLP normal (50+ Schichten). GNNs haben meist **2–3**. Grund:
Jede Schicht mittelt über Nachbarn. Nach $k$ Schichten ist $H^{(k)}\approx\hat A^k H W\dots$ —
und $\hat A^k$ konvergiert für großes $k$ gegen die stationäre Verteilung eines Random Walks.
**Alle Knotenvektoren werden einander ähnlich**, die Information kollabiert (formal: die
Eigenwerte $<1$ sterben aus, nur der zum Eigenwert 1 überlebt). Das heißt **Over-Smoothing**.
Verschärft wird es im **Small-World**-Netz: bei mittlerem Pfad ~4 sieht ein 4-Schicht-GNN schon
*den ganzen Graphen* — jeder Knoten aggregiert dann praktisch dasselbe. Gegenmittel:
Residual-/Skip-Connections, **Jumping Knowledge**, PairNorm — oder schlicht: flach bleiben.

### 2.5 Link Prediction

**Aufgabe:** Gegeben den Graphen ohne einige Kanten — welche fehlen? **Anwendungen im Netz:**
Topologie-Inferenz (die gemessene AS-Topologie ist notorisch **unvollständig** — viele Peerings
sieht kein Route-Collector), Vorhersage künftiger Peerings, Anomalieerkennung (eine Kante, die
„nicht sein dürfte").

**Der große Vorteil:** Link Prediction braucht **keine Labels** — der Graph labelt sich selbst.
Man entfernt einen Teil der Kanten (**positive** Beispiele), zieht gleich viele nicht-existente
Knotenpaare (**negative**), und prüft, ob das Modell sie trennt.

> ### ⚠️ Zwei Fallen, die Link-Prediction-Ergebnisse regelmäßig ruinieren
> 1. **Leakage über den Graphen selbst.** Die Test-Kanten müssen **aus dem Graphen entfernt
>    sein**, bevor man Features/Embeddings berechnet. Sonst berechnet man „Common Neighbors"
>    auf einem Graphen, der die gesuchte Kante schon enthält — und misst Auswendiglernen.
>    Das ist die Graph-Variante des Leakage-Problems aus Modul 15 (3.5).
> 2. **Die Basisrate — schon wieder.** Ein Graph mit $n=10\,670$ hat ~57 Mio. mögliche Paare,
>    aber nur 22 000 Kanten ⟹ die echte Basisrate ist $\pi\approx 3{,}9\cdot10^{-4}$. Evaluiert
>    man (wie üblich) mit **1:1** positiv:negativ, misst man bei $\pi=0{,}5$ — und der **Base-Rate-
>    Fallacy aus Modul 15 (3.1)** schlägt in voller Härte zu: eine AUC von 0,95 auf balancierten
>    Daten heißt im echten Einsatz (alle Paare durchprobieren) fast nur Fehlalarme. Balanciertes
>    Sampling ist bequem — aber man muss dazusagen, dass es die Realität **nicht** abbildet.
> 3. **Der Grad-Confound.** Zieht man die negativen Paare **uniform zufällig** (der Standard!),
>    sind sie in einem scale-free-Graphen fast immer **Blatt × Blatt** — echte Kanten betreffen
>    dagegen überdurchschnittlich oft Hubs. Gemessen auf der echten AS-Topologie: Median-
>    Gradprodukt **500** (echte Kanten) vs. **4** (Zufallspaare). Jede gradbasierte Größe trennt
>    das fast trivial, **ohne irgendetwas über Struktur zu wissen**. Konsequenz (Projekt 01
>    misst es):
>
>    | Heuristik | uniform Negative | **grad-gematchte** Negative |
>    |---|---|---|
>    | Adamic-Adar | 0,724 | **0,566** ← jetzt bester |
>    | Common Neighbors | 0,719 | 0,559 |
>    | **Preferential Attachment** | **0,763** ← „Sieger" | **0,406** ← Zufallsniveau |
>
>    **PA stürzt von 0,763 auf 0,406** — sein ganzer Vorsprung war ein **Artefakt der
>    Negativ-Auswahl**, kein Signal. Und alle Verfahren fallen auf ~0,56: die Aufgabe ist
>    *viel* schwerer als die naive Evaluation suggeriert. **Nicht nur das Modell kann täuschen —
>    schon die Konstruktion der Testmenge entscheidet, was man misst.**

---

## 3 · Netzwerk-Zeitreihen & Forecasting

### 3.1 Die Aufgabe und die Struktur

Verkehr $y_t$ auf einem Link/Knoten vorhersagen — für Kapazitätsplanung, Autoscaling,
Energiesparen (Links nachts abschalten) und als **Baseline für Anomalieerkennung** („mehr
Verkehr als vorhergesagt = verdächtig"). Netzverkehr hat eine sehr freundliche Eigenschaft:
**starke, stabile Saisonalität**.
- **Tagesrhythmus** (Periode 24 h): Nachts wenig, abends Peak („Prime Time").
- **Wochenrhythmus** (Periode 7 d): Werktag ≠ Wochenende.
- **Trend**: langfristiges Wachstum.
- **Rauschen/Bursts**: Netzverkehr ist bekanntermaßen **selbstähnlich/langzeitkorreliert**
  (Leland et al. 1994) — Bursts auf *allen* Zeitskalen, nicht Poisson-glatt.

### 3.2 Baselines — und warum sie die halbe Miete sind

> **Die wichtigste Regel des Forecastings:** Schlage erst die triviale Baseline, dann rede über
> Modelle.

- **Naiv**: $\hat y_{t+1}=y_t$ („morgen wie heute").
- **Saisonal-naiv**: $\hat y_{t+h}=y_{t+h-m}$ mit Saisonlänge $m$ („nächsten Dienstag 20 Uhr wie
  letzten Dienstag 20 Uhr"). **Bei Netzverkehr ist das erschreckend stark** — jedes teure Modell
  muss sich zuerst hieran messen. Wer diese Baseline nicht berichtet, berichtet nichts.
- **Mittelwert/Drift**: Referenz nach unten.

Metriken: **MAE**, **RMSE** (bestraft Ausreißer stärker — bei burstigem Verkehr relevant),
**MAPE** (prozentual, aber explodiert bei $y_t\approx0$ ⟹ nachts unbrauchbar), und
**MASE** = MAE relativ zur naiven Baseline (**<1 = besser als naiv**; skalenfrei, deshalb der
sauberste Vergleich über verschiedene Links).

### 3.3 Klassisch: ARIMA / SARIMA (formal)

Mit dem **Backshift-Operator** $B y_t = y_{t-1}$:

- **AR(p)** — der Wert hängt von seinen $p$ Vorgängern ab:
  $\phi(B)y_t=\varepsilon_t$ mit $\phi(B)=1-\phi_1B-\dots-\phi_pB^p$.
- **MA(q)** — der Wert hängt von den letzten $q$ **Schocks** ab:
  $y_t=\theta(B)\varepsilon_t$ mit $\theta(B)=1+\theta_1B+\dots+\theta_qB^q$.
- **I(d)** — **Differenzieren** $\nabla^d=(1-B)^d$ macht die Reihe **stationär** (entfernt Trend).

**ARIMA(p,d,q)**: $\phi(B)\,(1-B)^d y_t=\theta(B)\,\varepsilon_t$.

**SARIMA(p,d,q)(P,D,Q)$_m$** ergänzt die Saison mit Periode $m$:
$$\Phi(B^m)\,\phi(B)\,(1-B^m)^D(1-B)^d\,y_t=\Theta(B^m)\,\theta(B)\,\varepsilon_t.$$
Der Term $(1-B^m)^D$ ist die **saisonale Differenz** — im Kern die saisonal-naive Baseline, in
das Modell eingebaut. Ordnungswahl klassisch über ACF/PACF oder AIC/BIC (Box-Jenkins).

*(Hier keine Code-Demo: `statsmodels` ist in dieser Umgebung nicht installiert. Die Projekte
nutzen stattdessen Lag-Features + ML — in der Praxis ohnehin oft stärker und flexibler.)*

### 3.4 ML auf Lag-Features

Der pragmatische, meist beste Weg: Zeitreihe → **Tabelle**. Features pro Zeitpunkt: Lags
($y_{t-1},y_{t-2},\dots,y_{t-m},y_{t-2m}$), gleitende Mittel/Std, und **Kalender-Features**
(Stunde, Wochentag) — Letztere am besten **zyklisch kodiert**:
$$\sin\!\Big(\frac{2\pi\,\text{Stunde}}{24}\Big),\ \cos\!\Big(\frac{2\pi\,\text{Stunde}}{24}\Big)$$
damit 23 Uhr und 0 Uhr benachbart sind (als rohe Zahl wären sie maximal weit auseinander!).
Darauf: Ridge oder Gradient Boosting (Modul 04/05). Für Sequenzen: LSTM (Modul 09) — bei
starker Saisonalität und wenig Daten aber oft **schlechter** als Ridge mit guten Lag-Features.

> **Evaluation: nur zeitbasiert.** Bei Zeitreihen ist ein zufälliger Split **fatal** — man
> trainiert auf der Zukunft und testet auf der Vergangenheit. Korrekt: **zeitlicher Split** bzw.
> **Rolling-Origin/Walk-Forward-Validation** (`TimeSeriesSplit`). Das ist dieselbe Leakage-Regel
> wie in Modul 15 (3.5) — hier nur noch offensichtlicher.

### 3.5 Die Synthese: Spatio-Temporal GNNs

Jetzt kommen beide Hälften zusammen. Verkehr auf einem Netz ist **gleichzeitig** zeitlich
korreliert (Saisonalität) **und räumlich** (benachbarte Links tragen denselben Verkehr — was
über Link A fließt, fließt gleich über B). Ein Modell pro Link ignoriert die Topologie; ein GNN
ohne Zeit ignoriert die Saisonalität.

**Spatio-Temporal GNNs** kombinieren beides. Die Datenstruktur ist ein Tensor
$X\in\mathbb R^{T\times n\times f}$ (Zeit × Knoten × Features), und die Architektur wechselt
zwischen:
- **räumlicher** Aggregation (GNN-Schicht über die Topologie) und
- **zeitlicher** Aggregation (1D-Faltung, GRU/LSTM oder Attention über $T$).

Bekannte Vertreter: **STGCN** (Graph-Conv + temporale Conv), **DCRNN** (Diffusion Convolution +
GRU; modelliert Verkehr explizit als **Diffusionsprozess** auf dem Graphen), **Graph WaveNet**
(lernt die Adjazenz gleich mit — nützlich, wenn die wahre Topologie unbekannt ist).

> **Ehrliche Einordnung:** Spatio-Temporal GNNs sind das Aushängeschild dieses Feldes — und
> gleichzeitig gilt in vielen veröffentlichten Vergleichen, dass eine **gut gemachte
> saisonal-naive oder Lag-Ridge-Baseline** erstaunlich nah herankommt. Die Topologie hilft nur,
> wenn die **räumliche Korrelation** wirklich stark ist und die zeitliche Struktur nicht ohnehin
> schon alles erklärt. **Projekt 03 misst genau das** — statt es zu behaupten.

---

## 4 · Advanced

### 4.1 Verschlüsselte Verkehrsanalyse

Modul 15 (2.1) hielt fest: Verschlüsselung schützt den **Inhalt**, nicht die **Metadaten**.
Zu Ende gedacht wird daraus **Website Fingerprinting**: Ein Beobachter (WLAN-Betreiber, ISP,
Tor-Exit-Beobachter) sieht nur **Paketgrößen, Richtungen und Timing** — und erkennt daraus,
*welche Seite* du besuchst. Denn jede Seite hat ein charakteristisches Ladeprofil (Anzahl und
Größe der Ressourcen). Merkmale sind Größen-/Richtungs-Sequenzen, Burst-Muster,
Inter-Arrival-Times; Modelle klassisch (kNN/RF) oder CNN/Transformer auf der Rohsequenz.

**Gegenmaßnahmen** kosten alle Bandbreite oder Latenz: **Padding** (auf feste Größen),
**Traffic Shaping** (konstante Rate), **Dummy-Traffic**, Multiplexing. Und: die
Laborergebnisse („95 % Genauigkeit!") sind meist **closed-world** (nur 100 Kandidatenseiten).
In der **open world** (Millionen Seiten, meist uninteressante) schlägt — natürlich — der
**Base-Rate-Fallacy** aus Modul 15 zu. Dieselbe Technik in Gut: QoE-Monitoring des Betreibers
(„läuft das Video gerade ruckelfrei?") ohne die Nutzdaten zu entschlüsseln.

### 4.2 Self-Driving Networks

Die Vision: Netze, die sich **selbst** beobachten, verstehen und optimieren — die
Konsequenz aus Modul 15+16.
- **Intent-Based Networking**: Der Operator sagt *was* er will („Videokonferenzen < 50 ms"),
  nicht *wie* (Router-Konfigurationen). Das System übersetzt, setzt um, verifiziert, korrigiert.
- **Digital Twin**: ein laufendes Simulationsmodell des echten Netzes — für „Was-wäre-wenn"
  ohne Risiko. Genau hier zahlt sich **modellbasiertes Lernen** aus (Modul 14, 4.4).
- **Closed Loop**: messen → verstehen → entscheiden → wirken → messen. Die Netz-Variante des
  Agent-Umgebung-Kreises aus Modul 13.
- **Realitäts-Check**: Netzbetrieb ist **sicherheitskritisch**. Ein Modell mit 99 % Genauigkeit,
  das 1 % der Zeit das Backbone fehlkonfiguriert, ist inakzeptabel. Deshalb sind
  **Erklärbarkeit**, **Verifikation**, konservative Fallbacks und begrenzte Autonomie hier keine
  Kür, sondern Voraussetzung. Das erklärt, warum die Vision seit ~2017 diskutiert und nur
  zögerlich eingesetzt wird.

### 4.3 Routing & Traffic Engineering als Entscheidungsproblem

Hier trifft dieses Modul den RL-Block (13/14). **Traffic Engineering**: Wie lege ich Pfade/
Gewichte, um Auslastung und Verzögerung zu minimieren? Klassisch: Optimierung (Multi-Commodity
Flow, LP) auf einer **geschätzten** Verkehrsmatrix. Mit RL: Zustand = Auslastung + Topologie
(→ **GNN als Encoder**!), Aktion = Routing-Gewichte, Belohnung = −Auslastung/Verzögerung.

**Warum GNN + RL hier zusammengehören:** Ein MLP auf einer festen Adjazenzmatrix müsste für
**jede** Topologie neu lernen. Ein GNN **generalisiert über Topologien** — genau das brauchen
Netze, die sich ständig ändern. (Vgl. **RouteNet**, das QoS-Metriken aus Topologie + Routing +
Verkehrsmatrix vorhersagt.)

**Aber:** Der Vergleich muss ehrlich gegen die **klassische Optimierung** laufen — die bei
bekanntem Modell oft **exakt optimal** ist. Das ist genau die Lektion aus dem Modul-14-Finale:
*Kennst du dein Modell, nimm Optimierung. RL ist für den Fall, dass du es nicht kennst.*

### 4.4 Skalierung und Privatsphäre

- **Skalierung:** Millionen Knoten passen nicht in einen Batch. Lösungen: Nachbar-Sampling
  (GraphSAGE), Cluster-GCN (Graph partitionieren), historische Embeddings.
- **Federated Learning:** Betreiber wollen/dürfen ihre Verkehrsdaten nicht teilen
  (Geschäftsgeheimnis, DSGVO). Lösung: Modelle statt Daten teilen. Am Rande relevant: schon die
  **Topologie** ist ein Geschäftsgeheimnis (Peering-Beziehungen).
- **Messverzerrung:** Die „echte" AS-Topologie kennt **niemand**. Route-Collectors sehen nur,
  was BGP ihnen zeigt; **Peer-to-Peer-Links zwischen kleinen AS fehlen systematisch**. Man
  trainiert also auf einem **unvollständigen, verzerrt gemessenen** Graphen — und die fehlenden
  Kanten sind nicht zufällig verteilt. Das ist bei *jeder* Aussage über Internet-Topologie
  mitzudenken (auch bei unserer Robustheits-Tabelle).

---

## 5 · Zusammenfassung / Cheat-Sheet

**Graph.** $G=(V,E)$, $A$ (sparse!), $D=\operatorname{diag}(d_i)$. AS-Level ≠ Router-Level ≠
Verkehrsgraph.

**Scale-Free.** $P(d)\propto d^{-\alpha}$, Internet-AS $\alpha\approx2{,}1$. Entsteht durch
**Preferential Attachment**. **Exponent per MLE schätzen**
($\hat\alpha=1+n[\sum\ln\frac{d_i}{d_{\min}-0{,}5}]^{-1}$), **nie** per log-log-Fit (Faktor 2 daneben!).

**Robustheit.** Zufallsausfall: harmlos (10 % → 0,80). **Hub-Angriff: fatal (5 % → 0,003).**

**Message Passing.** $\mathbf h_v^{(k)}=\text{UPDATE}(\mathbf h_v^{(k-1)},\text{AGGREGATE}(\{\mathbf h_u\}))$,
AGGREGATE **permutationsinvariant**.

**GCN.** $H^{(k)}=\sigma(\hat A H^{(k-1)}W^{(k)})$, $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$.
Self-Loops (sich selbst nicht vergessen) + symmetrische Normierung (Skalen zähmen, Hubs dämpfen,
Eigenwerte in $[-1,1]$).

**Familien.** GCN (transduktiv, feste Gewichte) · **GraphSAGE** (induktiv, Sampling) ·
**GAT** (Attention = Modul 09; Transformer = GAT auf vollständigem Graphen).

**Over-Smoothing.** $\hat A^k$ → stationär ⟹ alle Knoten gleich ⟹ GNNs bleiben **2–3 Schichten**.

**Link Prediction.** Common Neighbors · Jaccard · **Adamic-Adar** ($\sum 1/\log d_w$) ·
Pref. Attachment. Drei Fallen: **Test-Kanten VOR der Feature-Berechnung entfernen** · echte
Basisrate $\approx3{,}9\cdot10^{-4}$ statt 0,5 · **Grad-Confound** (uniform gezogene Negative sind
Blatt×Blatt ⟹ PA 0,763 → **0,406** bei grad-gematchten Negativen).

**Forecasting.** Erst **saisonal-naiv** schlagen! · Lags + zyklische Kalender-Features
($\sin/\cos$) · **MASE < 1** = besser als naiv · SARIMA
$\Phi(B^m)\phi(B)(1-B^m)^D(1-B)^d y_t=\Theta(B^m)\theta(B)\varepsilon_t$ · **zeitlich splitten!**

**Spatio-Temporal.** $X\in\mathbb R^{T\times n\times f}$; STGCN/DCRNN/Graph WaveNet =
räumliche + zeitliche Aggregation im Wechsel.

---

## 6 · Selbsttest

<details>
<summary><b>1.</b> Was heißt „scale-free", und warum ist der mittlere Grad dabei wenig aussagekräftig?</summary>

$P(d)\propto d^{-\alpha}$ — es gibt **keine typische Skala**: die meisten Knoten haben Grad 1–2,
wenige Hubs Tausende. Der Mittelwert liegt (wie bei allen schweren Rändern, vgl. Flow-Größen in
Modul 15) irgendwo dazwischen und beschreibt **niemanden**: real Median 2, Mittel 4,12,
Maximum 2312. Ursache: **Preferential Attachment** beim Wachstum.
</details>

<details>
<summary><b>2.</b> Warum ist der naive log-log-Fit für den Power-Law-Exponenten falsch?</summary>

Im Schwanz liegen pro Bin nur 0–3 Beobachtungen; deren logarithmiertes Rauschen ist stark
verzerrt, leere Bins verschwinden ganz — die Kleinste-Quadrate-Gerade wird verbogen. Korrekt ist
der **MLE** (Clauset et al. 2009) auf dem Schwanz $d\ge d_{\min}$. Real gemessen: naiv **−1,14**
vs. MLE **2,08** — Faktor 2.
</details>

<details>
<summary><b>3.</b> Erkläre, warum das Internet robust gegen Zufallsausfälle und fragil gegen Hub-Angriffe ist.</summary>

Wegen der Gradverteilung. Ein **zufällig** getroffener Knoten ist fast sicher ein Blatt (Grad
1–2) — sein Ausfall stört nichts. Ein **gezielt** getroffener Hub reißt Tausende Kanten mit.
Real: 10 % zufällig → 80 % bleiben verbunden; **5 % Hubs → 0,3 %** (Albert/Jeong/Barabási 2000).
</details>

<details>
<summary><b>4.</b> Formuliere Message Passing. Warum muss AGGREGATE permutationsinvariant sein?</summary>

$\mathbf h_v^{(k)}=\text{UPDATE}(\mathbf h_v^{(k-1)},\ \text{AGGREGATE}(\{\mathbf h_u^{(k-1)}:u\in N(v)\}))$.
AGGREGATE läuft über eine **Menge** — Nachbarn haben **keine kanonische Reihenfolge**. Wäre die
Funktion reihenfolgeabhängig, hinge die Ausgabe von der willkürlichen Knotennummerierung ab.
Daher Summe/Mittel/Max.
</details>

<details>
<summary><b>5.</b> Warum $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ — was leistet jeder Teil?</summary>

**$+I$ (Self-Loops):** sonst fällt $\mathbf h_v$ aus seiner eigenen Aktualisierung heraus — der
Knoten vergisst sich selbst. **$\tilde D^{-1/2}\cdot\tilde D^{-1/2}$:** normiert, sonst
explodieren die Skalen (Hub mit Grad 2312 vs. Blatt mit Grad 2). **Symmetrisch** statt
zeilenweise, weil (a) $\hat A$ symmetrisch bleibt, (b) die Botschaft eines Hubs gedämpft wird
(Hubs sind uninformativ — wie **Adamic-Adar**/IDF), (c) die Eigenwerte in $[-1,1]$ bleiben
⟹ stabile Tiefe.
</details>

<details>
<summary><b>6.</b> GCN vs. GraphSAGE vs. GAT — je ein Satz.</summary>

**GCN**: feste, gradbasierte Gewichte, braucht den **ganzen** Graphen ⟹ **transduktiv**.
**GraphSAGE**: **sampelt** Nachbarn und lernt eine **Aggregator-Funktion** ⟹ **induktiv**
(funktioniert auf neuen Knoten/Graphen) und skalierbar. **GAT**: **lernt** die
Nachbargewichte per **Attention** (= Modul 09, nur über die Nachbarschaft).
</details>

<details>
<summary><b>7.</b> Was ist Over-Smoothing, und warum sind GNNs deshalb flach?</summary>

Jede Schicht mittelt über Nachbarn; $\hat A^k$ konvergiert gegen die stationäre Verteilung eines
Random Walks ⟹ **alle Knotenvektoren werden gleich**, die Information kollabiert. Deshalb 2–3
Schichten. Im **Small-World**-Netz (mittlerer Pfad ~4) verschärft: ein 4-Schicht-GNN sieht schon
den ganzen Graphen.
</details>

<details>
<summary><b>8.</b> Nenne die zwei Fallen bei Link Prediction.</summary>

(1) **Leakage über den Graphen**: Test-Kanten müssen entfernt sein, **bevor** man Features/
Embeddings rechnet — sonst „sieht" Common Neighbors die gesuchte Kante schon.
(2) **Basisrate**: real $\pi\approx4\cdot10^{-4}$ (22 000 Kanten unter 57 Mio. Paaren), evaluiert
wird aber meist **1:1-balanciert** ⟹ Base-Rate-Fallacy (Modul 15): tolle AUC, im Echteinsatz
fast nur Fehlalarme.
</details>

<details>
<summary><b>9.</b> Was ist die saisonal-naive Baseline und was sagt MASE < 1?</summary>

$\hat y_{t+h}=y_{t+h-m}$ („wie letzte Woche zur selben Zeit"), $m$ = Saisonlänge. Bei Netzverkehr
**sehr stark**. **MASE** = MAE relativ zur naiven Baseline: **< 1 = besser als naiv**, ≥ 1 = das
Modell ist sein Geld nicht wert. Skalenfrei ⟹ über verschiedene Links vergleichbar.
</details>

<details>
<summary><b>10.</b> Warum Stunde/Wochentag zyklisch als sin/cos kodieren?</summary>

Als rohe Zahl wären **23 Uhr und 0 Uhr maximal weit entfernt** (Distanz 23), obwohl sie
benachbart sind. $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ bilden die Stunde auf einen **Kreis** ab —
23 und 0 liegen dann direkt nebeneinander. Zwei Komponenten sind nötig, weil eine allein nicht
eindeutig ist.
</details>

<details>
<summary><b>11.</b> Was ist ein Spatio-Temporal GNN, und wann lohnt er sich?</summary>

Ein Modell auf $X\in\mathbb R^{T\times n\times f}$, das **räumliche** Aggregation (GNN über die
Topologie) und **zeitliche** Aggregation (Conv/GRU/Attention) abwechselt (STGCN, DCRNN, Graph
WaveNet). Er lohnt sich nur, wenn die **räumliche Korrelation** wirklich Zusatzinformation
trägt — sonst erklärt die Saisonalität allein schon alles und die saisonal-naive Baseline hält mit.
</details>

<details>
<summary><b>12.</b> Warum kennt niemand die echte AS-Topologie — und was heißt das für unsere Ergebnisse?</summary>

Route-Collectors sehen nur, was BGP ihnen zeigt; **Peer-to-Peer-Links zwischen kleinen AS fehlen
systematisch** (sie werden nirgends announciert). Der gemessene Graph ist also **unvollständig
und verzerrt**, und die fehlenden Kanten sind **nicht zufällig** verteilt. Jede Aussage — auch
unsere Robustheits-Tabelle — steht unter diesem Vorbehalt.
</details>

---

## 7 · Literatur & Quellen

**Graph/Netzwerk-Grundlagen:**
- 📗 **Barabási — *Network Science*** (networksciencebook.com) — **komplett frei online**,
  hervorragend illustriert; Kap. 4 (Scale-Free), 8 (Robustheit) decken Abschnitt 1 ab.
  *Einsteigerfreundlich.* **Beste Einzelquelle für Teil 1.**
- 📄 **Faloutsos, Faloutsos & Faloutsos (1999), *On Power-Law Relationships of the Internet
  Topology*** — das Paper, das alles startete.
- 📄 **Albert, Jeong & Barabási (2000), *Error and Attack Tolerance of Complex Networks***
  (Nature) — das Robustheits-Ergebnis aus 1.4.
- 📄 **Clauset, Shalizi & Newman (2009), *Power-Law Distributions in Empirical Data*** — warum
  der log-log-Fit falsch ist und wie man es richtig macht. *Vertiefend, sehr lehrreich.*

**Graph Learning:**
- 📗 **Hamilton — *Graph Representation Learning Book*** (frei, cs.mcgill.ca/~wlh/grl_book/) —
  das Standardwerk. *Einsteiger→vertiefend.* **Beste Einzelquelle für Teil 2.**
- 📄 **Kipf & Welling (2017), *Semi-Supervised Classification with GCNs*** — das GCN-Paper.
- 📄 **Hamilton et al. (2017), *Inductive Representation Learning on Large Graphs*** (GraphSAGE).
- 📄 **Veličković et al. (2018), *Graph Attention Networks*** (GAT).
- 📄 **Grover & Leskovec (2016), *node2vec***.
- 🎥 **Stanford CS224W — *Machine Learning with Graphs*** (Leskovec; Videos + Folien frei) —
  der beste Kurs zum Thema. *Vertiefend.*
- 🌐 **PyTorch Geometric Docs** (pyg.org) — die Standardbibliothek. Wir bauen im Modul bewusst
  from scratch; für die Praxis solltest du PyG kennen.

**Zeitreihen:**
- 📗 **Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*** (otexts.com/fpp3) —
  **frei online**, das Standardwerk; Kap. 5 (Baselines, MASE), 9 (ARIMA/SARIMA).
  *Einsteigerfreundlich.* **Beste Einzelquelle für Teil 3.**
- 📄 **Leland et al. (1994), *On the Self-Similar Nature of Ethernet Traffic*** — der Klassiker:
  Netzverkehr ist **nicht** Poisson.
- 📄 **Yu et al. (2018), *STGCN***; **Li et al. (2018), *DCRNN***; **Wu et al. (2019),
  *Graph WaveNet*** — Spatio-Temporal GNNs. *Vertiefend.*

**Anwendung/Vision:**
- 📄 **Rusek et al. (2020), *RouteNet***— GNN sagt QoS aus Topologie+Routing+Verkehr vorher.
- 📄 **Feamster & Rexford (2018), *Why (and How) Networks Should Run Themselves*** —
  Self-Driving Networks. *Einsteigerfreundlich.*
- 📄 **Panchenko et al. (2016), *Website Fingerprinting at Internet Scale*** — inkl. der
  ernüchternden Open-World-Realität.

**Daten:**
- 🌐 **SNAP** (snap.stanford.edu/data) — echte Netzgraphen, u. a. die **Oregon-AS-Topologie**
  der Projekte. *Frei, kleine Downloads.*
- 🌐 **CAIDA** (caida.org) — die Referenz für Internet-Messdaten (teils Registrierung nötig).
- 🌐 **SNDlib** (sndlib.zib.de) — echte Backbone-Topologien **inkl. Verkehrsmatrizen**.

---

## Nächstes Modul

Damit ist der Netzwerk-Block (15+16) abgeschlossen. Es folgt **Modul 17 — Core XR: Principles
of Interactive Systems** und damit ein völlig neues Feld. Was du hier gelernt hast — Graphen als
Datenstruktur, GNNs, Zeitreihen, und vor allem die **Disziplin, gegen ehrliche Baselines zu
messen** — trägt weit über Netzwerke hinaus: 3D-Punktwolken (Modul 20) sind Graphen, Robotik
(21/22) ist Zeitreihen + Regelung, und Self-aware Computing (24) ist die Idee aus 4.2 in
allgemein.
