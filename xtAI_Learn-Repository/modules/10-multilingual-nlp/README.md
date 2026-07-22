# Module 10 — Multilingual NLP

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** So far (modules 08 & 09) *one* language was implicitly
> assumed — mostly English. This module asks: how do you build language technology
> that works across **many languages**, and how do you **transfer** knowledge from
> resource-rich to resource-poor languages? Core topics: language-agnostic
> **subword tokenization**, **cross-lingual representations**, **neural machine
> translation (NMT)** and **massively multilingual** models (mBERT, XLM-R, mT5)
> together with **zero-shot transfer**.

**Helpful prior knowledge:** probability & linear algebra (matrix decompositions,
orthogonality), PyTorch (modules 05/09), the basics of neural networks.

**Modules you should have done first:**
- **Module 08 (NLP 1)** — tokenization, n-grams, TF-IDF, word representations, evaluation.
- **Module 09 (NLP 2)** — embeddings, RNN/LSTM, **self-attention & the transformer**,
  pretraining (BERT vs. GPT), BPE. This module completes the **encoder-decoder
  transformer** that module 09 split into its halves (encoder for BERT, decoder for GPT).

---

## Learning objectives

After this module you can …

- explain **why** multilingual NLP is its own thing (diversity of scripts, morphology,
  word order; the resource inequality between languages);
- describe **subword tokenization** (BPE, WordPiece, unigram LM/SentencePiece) formally
  and justify why a **shared** vocabulary across languages makes sense;
- build **cross-lingual word embeddings** by **aligning** monolingual spaces —
  including the closed-form **orthogonal Procrustes solution** via the SVD;
- derive the **NMT architecture**: seq2seq → attention (Bahdanau/Luong) → the
  **encoder-decoder transformer** with **cross-attention**; training (teacher forcing,
  label smoothing) and decoding (greedy, **beam search**);
- place **massively multilingual** models (mBERT, XLM-R, mT5, multilingual NMT) and
  explain **zero-shot transfer** — including the *curse of multilinguality*;
- **evaluate** translation quality correctly (**BLEU**, chrF, COMET) and know the pitfalls.

---

## 1 · Basics — why is multilinguality its own problem?

### 1.1 The starting point

There are ~7000 living languages; NLP resources are extremely **unequally**
distributed. A handful of *high-resource* languages (English, Chinese, German, …)
have huge corpora, tools and models; the vast majority are *low-resource* — little or
no annotated data. The central promise of multilingual NLP:

> **Build *one* model that covers many languages, and transfer knowledge from
> languages with much data to languages with little** (*cross-lingual transfer*).

### 1.2 Why is it hard? Typological diversity

Languages differ systematically, which breaks naive "one model per language, same
pipeline" approaches:

- **Writing systems (scripts):** Latin, Cyrillic, Devanagari, Han, Arabic … — disjoint
  character sets. A word-based vocabulary shares *nothing* between languages.
- **Morphology:** *isolating* (Chinese, barely any inflection) vs. *fusional* (German,
  Russian) vs. *agglutinative* (Turkish, Finnish — one "word" ≈ a whole sentence). A
  Turkish word can have dozens of forms → vocabulary explosion, many *out-of-vocabulary*
  (OOV) words.
- **Word order:** SVO (English), SOV (Japanese, Turkish), VSO (Arabic); German with its
  verb bracket. Models have to align different orders.
- **Segmentation:** Chinese/Japanese/Thai write **without spaces** — even "where is a
  word?" is non-trivial.

This diversity motivates two basic ideas of the module: **(a)** a tokenization that
works *below* the word level and *language-agnostically* (section 2), and **(b)**
representations that are *aligned* across languages (section 3).

---

## 2 · Subword tokenization across languages

### 2.1 Why not words?

A word vocabulary has three deadly sins in the multilingual setting:
1. **Size:** the union of the vocabularies of many languages is huge.
2. **OOV:** every unseen word (inflection, compound, proper name) becomes `<unk>` —
   dramatically so for morphologically rich languages.
3. **No sharing:** "nation"/"national"/"nationality" are three unrelated symbols; across
   languages even more so (e.g. German *Information* vs. English *information*).

Character level (as in module 09, project 3) solves OOV, but makes sequences very long
and forces the model to learn word structure from scratch. **Subwords** are the
compromise: frequent words stay a single token, rare ones break into meaning-carrying
pieces (*un-believ-able*), and pieces are **shared between words and languages**.

### 2.2 Byte-pair encoding (BPE) — recap & precision

BPE (Sennrich et al. 2016, for NMT) learns a vocabulary *bottom-up*:

1. Initialize the vocabulary with all **single characters** (or **bytes** → *byte-level
   BPE*, then every Unicode character is representable, never OOV).
2. Count all **adjacent symbol pairs** in the corpus. Merge the **most frequent pair**
   $(a,b)$ into a new symbol $ab$ and add it to the vocabulary.
3. Repeat until the desired vocabulary size $V$ is reached.

The list of merges is deterministic; **tokenization** applies the merges in the learned
order to a new word. Result: a fixed compromise between characters (many merges needed)
and words (few but coarse units), controllable via $V$.

### 2.3 WordPiece and unigram LM

- **WordPiece** (BERT): like BPE, but instead of "most frequent pair" it merges the pair
  that most increases the **likelihood** of a unigram language model over the corpus —
  i.e. $\arg\max_{(a,b)} \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$
  (pointwise-mutual-information-like) rather than raw frequency.
- **Unigram LM** (Kudo 2018, the core of **SentencePiece**): *top-down*. Start with a
  large candidate vocabulary and a unigram model $p(\text{token})$. The probability of a
  segmentation $\mathbf{x}=(x_1,\dots,x_k)$ of a text is
  $$P(\mathbf{x})=\prod_{i=1}^{k} p(x_i),$$
  and the best segmentation is found via **Viterbi** (dynamic programming — the link to
  modules 07/08!). Train $p$ via **EM** (module 05: expectations over all segmentations),
  then iteratively **prune** the tokens whose removal lowers the total likelihood the
  least, until $|V|$ is reached. Advantage: a probabilistic model allows **subword
  regularization** (sampling different segmentations as data augmentation).

### 2.4 SentencePiece: raw text in, language-agnostic

**SentencePiece** is not a new method but an *implementation* (BPE or unigram) with an
important trick: it treats **the space as a normal character** (encoded as `▁`, U+2581)
and works directly on **raw, untokenized text**. This makes exactly the same pipeline
work for German (spaces) *and* Chinese (no spaces) — and detokenization is losslessly
reversible. That makes it the standard for multilingual models.

### 2.5 Shared vocabulary & *fertility*

If you train **one** subword vocabulary on the **concatenation** of several languages,
you get a *shared vocabulary*: common word stems, digits, punctuation and proper names
share tokens. A measure of how well a vocabulary fits a language is the **fertility** —
the average number of subword tokens per word. If a vocabulary is English-dominated,
German/Turkish has a higher fertility (more fragmentation) — a fairness/efficiency
problem of large LLMs. (Project 01 measures exactly this.)

---

## 3 · Cross-lingual word embeddings through alignment

Before end-to-end models dominated, the most elegant idea of cross-lingual NLP was:
**train word embeddings per language separately and align the spaces afterwards.**

### 3.1 The observation

Monolingual embedding spaces (Word2Vec/GloVe, module 08) have a **similar geometric
structure** across languages — the neighbourhoods of *(king, queen, man, woman)* look
alike in English and in German. Mikolov et al. (2013) concluded: there should be a
**linear mapping** $W$ that rotates the source into the target space.

### 3.2 Supervised alignment: the Procrustes problem

Given a **seed lexicon** of $n$ translation pairs, stack the source embeddings as rows
of $X\in\mathbb{R}^{n\times d}$ and the corresponding target embeddings as
$Y\in\mathbb{R}^{n\times d}$. Look for $W$ with $XW \approx Y$:
$$W^\star=\arg\min_{W}\;\lVert XW - Y\rVert_F^2 .$$

Without a constraint this is *linear regression*. What matters, though, is the
**orthogonality constraint** $W^\top W = I$ (Xing et al. 2015): $W$ should be a
**rotation/reflection**, not a distortion — that preserves distances and dot products
(and thus cosine neighbourhoods). This is the **orthogonal Procrustes problem** with a
*closed-form* solution via the **singular value decomposition**:
$$X^\top Y = U\,\Sigma\,V^\top \quad\Longrightarrow\quad W^\star = U V^\top .$$

> **Why $UV^\top$?** From $\lVert XW-Y\rVert_F^2 = \lVert XW\rVert_F^2 - 2\operatorname{tr}(W^\top X^\top Y) + \lVert Y\rVert_F^2$
> and $\lVert XW\rVert_F=\lVert X\rVert_F$ (orthogonal $W$) it remains to **maximize**
> $\operatorname{tr}(W^\top X^\top Y)=\operatorname{tr}(W^\top U\Sigma V^\top)$.
> With $Z=V^\top W^\top U$ (orthogonal) this becomes $\operatorname{tr}(Z\Sigma)=\sum_i \sigma_i Z_{ii}\le\sum_i\sigma_i$,
> maximal when $Z=I$, so $W^\top=VU^\top \Rightarrow W=UV^\top$. (All $\sigma_i\ge0$.)
> **Rule of thumb for the direction:** the matrix in the SVD is
> $(\text{source})^\top(\text{target})$, then $W$ maps the source onto the target
> ($XW\approx Y$).

Beforehand one usually normalizes the embeddings (length normalization, possibly mean
centering) so that cosine similarity is the relevant metric.

### 3.3 Retrieval, hubness and CSLS

Translating then means: map a source word ($\mathbf{x}W$) and take the nearest neighbour
in the target space (cosine). The problem is **hubness**: in high-dimensional spaces some
target vectors are "hubs" that are the nearest neighbour of *many* source words. **CSLS**
(*cross-domain similarity local scaling*, Conneau et al. 2018) corrects this by
penalizing the similarity with the average neighbourhood density:
$$\text{CSLS}(\mathbf{x},\mathbf{y}) = 2\cos(\mathbf{x},\mathbf{y}) - r_T(\mathbf{x}) - r_S(\mathbf{y}),$$
where $r_T(\mathbf{x})$ is the mean cosine similarity of $\mathbf{x}$ to its $k$ nearest
target neighbours (analogously $r_S$). It is scored via **precision@1** on a test lexicon.

### 3.4 Unsupervised alignment (outlook)

Without a seed lexicon (Conneau et al. 2018, *MUSE*): an **adversarial** network learns
$W$ so that a discriminator cannot distinguish projected source vectors from real target
vectors; then **iterative Procrustes** refines it on automatically extracted,
high-confidence pairs. Remarkably, this works *entirely without* parallel data.

---

## 4 · Neural machine translation (NMT)

The core problem: map a source sequence $\mathbf{x}=(x_1,\dots,x_S)$ onto a target
sequence $\mathbf{y}=(y_1,\dots,y_T)$ and model
$$P(\mathbf{y}\mid\mathbf{x})=\prod_{t=1}^{T} P(y_t\mid y_{<t},\mathbf{x}).$$

### 4.1 seq2seq with RNNs and the bottleneck

Sutskever et al. (2014): an **encoder** RNN reads $\mathbf{x}$ into a context vector
$\mathbf{c}$ (the last state), a **decoder** RNN generates $\mathbf{y}$ from it
autoregressively. Problem: the **entire** source sentence has to pass through *one*
vector $\mathbf{c}$ — a bottleneck that breaks for long sentences (exactly the limit we
saw with the LSTM in project 09-01).

### 4.2 Attention (Bahdanau & Luong)

The solution (Bahdanau et al. 2015): the decoder may look at **all** encoder states
$\mathbf{h}_1,\dots,\mathbf{h}_S$ at **every** step $t$ and form a *dynamic* context:
$$e_{ti}=\text{score}(\mathbf{s}_{t-1},\mathbf{h}_i),\quad
\alpha_{ti}=\frac{\exp e_{ti}}{\sum_j \exp e_{tj}},\quad
\mathbf{c}_t=\sum_{i=1}^{S}\alpha_{ti}\,\mathbf{h}_i.$$
The $\alpha_{ti}$ form a soft **alignment** between target and source words. *Bahdanau*
uses an additive score ($\mathbf{v}^\top\tanh(W[\mathbf{s};\mathbf{h}])$), *Luong* the
simpler multiplicative one ($\mathbf{s}^\top W\mathbf{h}$) — the latter is the direct
predecessor of transformer attention.

### 4.3 The encoder-decoder transformer

The transformer (Vaswani et al. 2017) replaces recurrence completely with attention. In
module 09 you built the **encoder** (bidirectional self-attention, for BERT) and the
**decoder** (causal self-attention, for GPT) separately. The full NMT architecture
**connects** both and has **three** attention types:

1. **Encoder self-attention** — *bidirectional* over the source. Every source token sees
   all others. Output: contextualized source representations
   $\mathbf{H}=(\mathbf{h}_1,\dots,\mathbf{h}_S)$.
2. **Decoder masked self-attention** — *causal* over the target sequence generated so far
   (no look into the future, as in GPT).
3. **Cross-attention** — the new heart: the **query** comes from the decoder, the **key**
   and **value** from the encoder output $\mathbf{H}$:
   $$\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})
     =\operatorname{softmax}\!\Big(\tfrac{Q_{\text{dec}}K_{\text{enc}}^\top}{\sqrt{d_k}}\Big)V_{\text{enc}}.$$
   This way every decoder step looks selectively at the relevant source tokens — the
   transformer generalization of the Bahdanau attention from 4.2.

A decoder layer is therefore: *masked self-attn → cross-attn → FFN*, each with residual +
LayerNorm. The rest (positional encoding, multi-head, FFN) is as in module 09.

### 4.4 Training and decoding

- **Teacher forcing:** in training one feeds the decoder the **true** previous target
  tokens $y_{<t}$ (not its own predictions), masks causally and computes the loss over all
  positions in parallel. Loss: cross entropy, often with **label smoothing** (target
  distribution $=(1-\varepsilon)$ on the true token, $\varepsilon$ uniform) — this dampens
  overconfidence and improves BLEU.
- **Decoding (inference):** autoregressive, token by token.
  - **Greedy:** take the most probable token at each step. Fast, but short-sighted.
  - **Beam search:** keep the $k$ best *partial* hypotheses in parallel and expand them;
    at the end pick the sequence with the highest (length-normalized) log probability.
    Almost always better than greedy.
  - **Special tokens:** `<bos>`/`<eos>` mark the start/end; generation stops at `<eos>`.

---

## 5 · Massively multilingual models & zero-shot transfer

### 5.1 One model for many language pairs

Johnson et al. (2017, *Google's Multilingual NMT*): train **one** encoder-decoder model
on **many** language pairs at once, with a **shared** subword vocabulary. A simple trick
controls the target language: you prepend a **language token** to the source, e.g. `<2de>`
for "translate into German". The astonishing result: the model can translate **zero-shot**
— language pairs it has *never* seen (e.g. Ja→Ko, although only Ja↔En and Ko↔En were
trained), because it learns a shared, language-neutral intermediate representation
(*interlingua* hypothesis).

### 5.2 Multilingual pretrained encoders

- **mBERT:** BERT (masked LM), but on the Wikipedias of 104 languages with a shared
  WordPiece vocabulary — *without* any cross-lingual supervision. Yet surprisingly
  well-aligned representations emerge.
- **XLM** (Lample & Conneau 2019): adds a *translation language modeling* (TLM) objective
  — masked LM on **concatenated translation pairs**, so the model can look *across the
  language boundary* to fill a gap.
- **XLM-R** (Conneau et al. 2020): the XLM approach, but scaled RoBERTa-style to **2.5 TB**
  of CommonCrawl in 100 languages — long the standard for cross-lingual understanding.
- **mT5 / mBART:** multilingual **encoder-decoders** (text-to-text resp. denoising) — for
  generation/translation.

### 5.3 Zero-shot cross-lingual transfer

The killer application: **fine-tune** a multilingual encoder (e.g. XLM-R) on a task **with
English labels only** (because those exist), and **evaluate directly on
German/Swahili/…** — *without* a single target-language label. Because the representations
are aligned across languages, the task-specific head transfers. This brings NLP to
languages without training data (standard benchmark: **XTREME**, **XNLI**).

### 5.4 The price: *curse of multilinguality*

For a fixed model size, quality per language first rises with more languages (positive
transfer, shared structure), but **falls** again beyond a point because the **capacity**
is diluted over too many languages (*capacity dilution*) and languages compete for
parameters (*interference*). Remedies: larger models, language-specific adapters, clever
data sampling (upsampling low-resource languages with a temperature $\tau$).

---

## 6 · Evaluation of translation

How do you measure translation quality automatically (human evaluation is expensive)?

### 6.1 BLEU

**BLEU** (Papineni et al. 2002) compares the candidate translation with one or more
**references** via **n-gram precision**. For $n=1,\dots,N$ (usually $N=4$):
$$p_n=\frac{\sum_{\text{n-grams}} \min(\text{count}_{\text{cand}},\,\text{count}_{\text{ref}})}
{\sum_{\text{n-grams}} \text{count}_{\text{cand}}}$$
(*clipped precision*: an n-gram counts at most as often as it appears in the reference —
this penalizes repetitions). Because pure precision rewards **too short** translations, a
**brevity penalty** is added:
$$\text{BP}=\begin{cases}1 & c>r\\ e^{1-r/c} & c\le r\end{cases},\qquad
\text{BLEU}=\text{BP}\cdot\exp\!\Big(\sum_{n=1}^{N} w_n \log p_n\Big),$$
with candidate length $c$, reference length $r$ and weights $w_n=1/N$. BLEU is a
**corpus** measure (the numerators/denominators are aggregated over the whole test set),
lies in $[0,1]$ (often $\times 100$) and correlates roughly with human judgement.

**Weaknesses:** superficial (word overlap, no synonyms/meaning), reference-dependent, not
comparable between languages/tokenizations (→ **sacreBLEU** standardizes the tokenization).

### 6.2 Alternatives

- **chrF:** F-score over **character** n-grams — more robust for morphologically rich
  languages (partial word matches count).
- **METEOR:** accounts for stemming/synonyms, recall-weighted.
- **COMET / BERTScore:** **learned**, embedding-based metrics — correlate markedly better
  with humans, but need a (multilingual) model.

---

## 7 · Summary / cheat sheet

| Term | Core in one sentence |
|---|---|
| **Low-/high-resource** | languages with little/much annotated data; goal: transfer between them. |
| **Subword** | tokens *below* the word level; frequent words whole, rare ones split; shared across words/languages. |
| **BPE** | bottom-up: iteratively merge the most frequent symbol pair until $\lvert V\rvert$. |
| **WordPiece** | like BPE, but merges by **likelihood gain** (PMI-like). |
| **Unigram LM / SentencePiece** | top-down probabilistic: best segmentation via Viterbi, $p$ via EM, prune the vocabulary; raw text, `▁`=space. |
| **fertility** | avg. subword tokens per word — fit vocabulary↔language. |
| **Procrustes** | orthogonal $W$ with $XW\approx Y$; solution $W=UV^\top$ from SVD$(X^\top Y)$. |
| **CSLS** | retrieval measure against *hubness*: $2\cos - r_T - r_S$. |
| **seq2seq** | encoder→context→decoder; the RNN variant has a bottleneck. |
| **Attention $\alpha_{ti}$** | soft alignment: $\mathbf{c}_t=\sum_i\alpha_{ti}\mathbf{h}_i$. |
| **Cross-attention** | decoder query ↔ encoder key/value — connects the transformer halves. |
| **Teacher forcing** | training with true $y_{<t}$; label smoothing against overconfidence. |
| **Beam search** | keep the $k$ best partial hypotheses; better than greedy. |
| **Language token `<2de>`** | controls the target language in multilingual NMT → **zero-shot**. |
| **mBERT/XLM-R/mT5** | multilingually pretrained; XLM-R = scaling; mT5 = enc-dec. |
| **Zero-shot transfer** | fine-tune on EN, test on DE — without DE labels. |
| **curse of multilinguality** | too many languages per capacity → quality falls. |
| **BLEU** | clipped n-gram precision × brevity penalty; corpus measure. |
| **chrF/COMET** | character-n-gram F / learned metric — often better than BLEU. |

**Formulas to remember:**
- Procrustes: $W=UV^\top$ with $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$.
- Attention context: $\mathbf{c}_t=\sum_i \alpha_{ti}\mathbf{h}_i,\ \alpha_{ti}=\mathrm{softmax}_i(e_{ti})$.
- BLEU: $\text{BP}\cdot\exp(\sum_n w_n\log p_n)$, $\text{BP}=\min(1,e^{1-r/c})$.

---

## 8 · Self-test

<details><summary><b>1.</b> Why does a word-based vocabulary fail on *three* fronts in the multilingual setting?</summary>

Size (the union of many language vocabularies is huge), OOV (every unseen
inflection/compound becomes `<unk>`, especially in morphologically rich languages), and
*no sharing* (related words and cross-lingual cognates are independent symbols). Subwords
address all three.
</details>

<details><summary><b>2.</b> How do BPE, WordPiece and unigram LM differ in the *criterion* by which they build the vocabulary?</summary>

BPE merges the **most frequent** symbol pair (pure frequency, bottom-up). WordPiece merges
the pair with the largest **likelihood gain** of a unigram LM (PMI-like, bottom-up).
Unigram LM starts **top-down** with a large vocabulary and **prunes** the tokens whose
removal lowers the corpus likelihood the least (probabilistic, with Viterbi segmentation
and EM).
</details>

<details><summary><b>3.</b> Why does embedding alignment require $W^\top W=I$ instead of an arbitrary $W$, and what is the solution?</summary>

Orthogonality enforces a pure **rotation/reflection** that **preserves** lengths and dot
products (i.e. cosine neighbourhoods) — otherwise a free linear mapping would distort the
geometry and destroy the semantic structure. The closed-form solution of the orthogonal
Procrustes problem is $W=UV^\top$ with $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$.
</details>

<details><summary><b>4.</b> What is *hubness* and how does CSLS mitigate it?</summary>

In high-dimensional spaces some target vectors ("hubs") become the nearest neighbour of
*many* source words, which distorts nearest-neighbour retrieval. CSLS penalizes the cosine
similarity by the local neighbourhood density of both sides ($2\cos - r_T - r_S$), so that
hubs are relatively downweighted.
</details>

<details><summary><b>5.</b> Name the three attention types in the encoder-decoder transformer and where Q, K, V come from in each.</summary>

(1) **Encoder self-attention** — Q,K,V all from the source, bidirectional. (2) **Decoder
masked self-attention** — Q,K,V from the target sequence so far, causally masked. (3)
**Cross-attention** — **Q from the decoder**, **K and V from the encoder output**;
connects the target with the source representation.
</details>

<details><summary><b>6.</b> What is teacher forcing and why does *one* forward pass suffice for the whole target sentence in training?</summary>

Teacher forcing feeds the decoder the **true** previous tokens $y_{<t}$ instead of its own
predictions. With a **causal mask** one can compute all positions in parallel: position
$t$ sees only $y_{<t}$, so a single forward pass yields the logits for *all* $t$ at once
(and thus the loss over the whole sentence).
</details>

<details><summary><b>7.</b> How does a language token `<2de>` enable *zero-shot* translation?</summary>

A single multilingual model with a shared vocabulary learns a language-neutral
intermediate representation. The prepended target-language token only controls the output
language. Because source encoding and target control are decoupled, the model can produce
combinations it never saw as a pair (e.g. Ja→Ko from Ja↔En and Ko↔En training).
</details>

<details><summary><b>8.</b> Why does pure n-gram precision reward too-short translations, and how does BLEU fix that?</summary>

A very short output containing only a few safe words has high precision (almost all of its
n-grams are in the reference), even though it is incomplete. BLEU therefore multiplies by
the **brevity penalty** $\min(1, e^{1-r/c})$, which exponentially penalizes candidates
shorter than the reference ($c<r$).
</details>

<details><summary><b>9.</b> What does the *curse of multilinguality* state and what are the remedies?</summary>

For a fixed capacity, adding languages improves quality only up to a point; afterwards it
falls because the parameters are spread over too many languages (*capacity dilution*,
*interference*). Remedies: more model capacity, language-specific adapters,
temperature-based upsampling of low-resource languages.
</details>

<details><summary><b>10.</b> Why is BLEU between two systems from different papers often not directly comparable — and what helps?</summary>

BLEU depends strongly on **tokenization**, casing and reference preparation; different
implementations produce different numbers for the same translation. **sacreBLEU**
standardizes tokenization and computation and reports a signature string, so that results
become reproducible and comparable.
</details>

---

## 9 · Literature & sources

*Legend: (free) = freely available, (beginner) = beginner-friendly, (in-depth) = advanced.*

**Textbooks / overviews**
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft) — chapters on MT &
  seq2seq. (beginner, free) online.
- Koehn, *Neural Machine Translation* (2020, Cambridge). (in-depth) — the standard work on NMT.
- Philipp Koehn, *Statistical Machine Translation* (2010) — background before the neural era.

**Key papers (all free on arXiv/ACL Anthology)**
- Sennrich, Haddow & Birch (2016): *Neural MT of Rare Words with Subword Units* — **BPE**.
- Kudo (2018): *Subword Regularization* & Kudo/Richardson (2018): *SentencePiece*.
- Mikolov et al. (2013): *Exploiting Similarities among Languages for MT* — linear mapping.
- Xing et al. (2015): *Normalized Word Embedding …* — the orthogonality constraint.
- Conneau et al. (2018): *Word Translation Without Parallel Data* (**MUSE**, CSLS).
- Bahdanau, Cho & Bengio (2015): *NMT by Jointly Learning to Align and Translate* — **attention**.
- Vaswani et al. (2017): *Attention Is All You Need* — the **transformer**.
- Johnson et al. (2017): *Google's Multilingual NMT* — **zero-shot**, language token.
- Devlin et al. (2019): *BERT* (**mBERT**); Conneau et al. (2020): *XLM-R*; Xue et al. (2021): *mT5*.
- Papineni et al. (2002): *BLEU*; Post (2018): *A Call for Clarity in Reporting BLEU* (**sacreBLEU**).

**Courses / interactive (free)**
- Stanford **CS224N** (NLP with Deep Learning) — lectures on MT, attention, transformers.
- Hugging Face **NLP Course**, chapters on tokenizers & translation. (beginner)
- Jay Alammar, *The Illustrated Transformer* / *Visualizing seq2seq with Attention* — blog. (beginner)
- **MUSE** (github.com/facebookresearch/MUSE) & **SentencePiece** (github.com/google/sentencepiece) — reference implementations.

---

## The three projects

The projects span the arc from tokenization via cross-lingual representations to full
translation — on real **German–English** data (Tatoeba), small and CPU/MPS-capable:

- **01 – basic** (`projects/01-basic/`): **Multilingual subword tokenization with
  SentencePiece.** Guided notebook: train a shared BPE/unigram vocabulary on DE+EN,
  analyse the tokenization, measure **fertility** and vocabulary sharing between the
  languages, compare with word/character tokenization. Plenty of instruction.
- **02 – medium** (`projects/02-medium/`): **Align cross-lingual embeddings (Procrustes +
  CSLS).** Python project with a test suite: monolingual embeddings per language, derive
  the orthogonal alignment via the SVD yourself, "translate" a dictionary via
  **precision@1** and **CSLS**. Little instruction.
- **03 – final** (`projects/03-final/`): **An NMT transformer from scratch (DE→EN).** No
  code given: a full encoder-decoder with **cross-attention**, teacher forcing,
  greedy/beam decoding and **BLEU** evaluation on Tatoeba. The master's-level capstone that
  brings modules 09 and 10 together.

Details, setup and reference solutions are in the `README.md` of each project folder.

---
# Modul 10 — Multilingual NLP (deutsche Fassung)

> **Worum geht es?** Bisher (Module 08 & 09) war *eine* Sprache implizit gesetzt —
> meist Englisch. Dieses Modul fragt: Wie baut man Sprachtechnologie, die über **viele
> Sprachen** funktioniert, und wie **überträgt** man Wissen von ressourcenreichen auf
> ressourcenarme Sprachen? Kernthemen: sprach-agnostische **Subword-Tokenisierung**,
> **cross-linguale Repräsentationen**, **neuronale maschinelle Übersetzung (NMT)** und
> **massiv-mehrsprachige** Modelle (mBERT, XLM-R, mT5) samt **Zero-Shot-Transfer**.

**Hilfreiche Vorkenntnisse:** Wahrscheinlichkeit & lineare Algebra (Matrixzerlegungen,
Orthogonalität), PyTorch (Module 05/09), Grundzüge neuronaler Netze.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 08 (NLP 1)** — Tokenisierung, N-Gramme, TF-IDF, Wortrepräsentationen, Evaluation.
- **Modul 09 (NLP 2)** — Embeddings, RNN/LSTM, **Self-Attention & der Transformer**,
  Vortraining (BERT vs. GPT), BPE. Dieses Modul baut den **Encoder-Decoder-Transformer**
  fertig, den 09 in seine Hälften (Encoder für BERT, Decoder für GPT) zerlegt hatte.

---

## Lernziele

Nach diesem Modul kannst du …

- erklären, **warum** mehrsprachiges NLP eigen ist (Skript-, Morphologie-,
  Wortstellungs-Vielfalt; die Ressourcen-Ungleichheit zwischen Sprachen);
- **Subword-Tokenisierung** (BPE, WordPiece, Unigram-LM/SentencePiece) formal beschreiben
  und begründen, warum ein **gemeinsames** Vokabular über Sprachen hinweg sinnvoll ist;
- **cross-linguale Wort-Embeddings** durch **Alignment** monolingualer Räume herstellen —
  inklusive der geschlossenen **orthogonalen Procrustes-Lösung** über die SVD;
- die **NMT-Architektur** herleiten: seq2seq → Attention (Bahdanau/Luong) → der
  **Encoder-Decoder-Transformer** mit **Cross-Attention**; Training (Teacher Forcing,
  Label Smoothing) und Decoding (Greedy, **Beam Search**);
- **massiv-mehrsprachige** Modelle einordnen (mBERT, XLM-R, mT5, mehrsprachige NMT) und
  **Zero-Shot-Transfer** erklären — inkl. der *curse of multilinguality*;
- Übersetzungsqualität korrekt **evaluieren** (**BLEU**, chrF, COMET) und die
  Fallstricke kennen.

---

## 1 · Grundlagen — Warum ist Mehrsprachigkeit ein eigenes Problem?

### 1.1 Die Ausgangslage

Es gibt ~7000 lebende Sprachen; NLP-Ressourcen sind extrem **ungleich** verteilt. Eine
Handvoll *high-resource*-Sprachen (Englisch, Chinesisch, Deutsch, …) besitzt riesige
Korpora, Werkzeuge und Modelle; die große Mehrheit ist *low-resource* — wenig oder keine
annotierten Daten. Das zentrale Versprechen mehrsprachigen NLPs:

> **Baue *ein* Modell, das viele Sprachen abdeckt, und übertrage Wissen von Sprachen mit
> vielen Daten auf Sprachen mit wenigen** (*cross-lingual transfer*).

### 1.2 Warum ist das schwer? Typologische Vielfalt

Sprachen unterscheiden sich systematisch, was naive „ein Modell pro Sprache, gleiche
Pipeline"-Ansätze bricht:

- **Schriftsysteme (Skripte):** Latein, Kyrillisch, Devanagari, Han, Arabisch … —
  disjunkte Zeichenmengen. Ein wortbasiertes Vokabular teilt *nichts* zwischen Sprachen.
- **Morphologie:** *isolierend* (Chinesisch, kaum Flexion) vs. *fusional* (Deutsch,
  Russisch) vs. *agglutinierend* (Türkisch, Finnisch — ein „Wort" ≈ ganzer Satz). Ein
  türkisches Wort kann Dutzende Formen haben → Vokabularexplosion, viele *out-of-vocabulary*
  (OOV)-Wörter.
- **Wortstellung:** SVO (Englisch), SOV (Japanisch, Türkisch), VSO (Arabisch); Deutsch
  mit Verbklammer. Modelle müssen unterschiedliche Reihenfolgen alignen.
- **Segmentierung:** Chinesisch/Japanisch/Thai schreiben **ohne Leerzeichen** — schon
  „Wo ist ein Wort?" ist nichttrivial.

Diese Vielfalt motiviert zwei Grundideen des Moduls: **(a)** eine Tokenisierung, die
*unter* der Wortebene und *sprach-agnostisch* arbeitet (Abschnitt 2), und **(b)**
Repräsentationen, die über Sprachen hinweg *ausgerichtet* sind (Abschnitt 3).

---

## 2 · Subword-Tokenisierung über Sprachen

### 2.1 Warum nicht Wörter?

Ein Wort-Vokabular hat drei Todsünden im mehrsprachigen Setting:
1. **Größe:** die Vereinigung der Vokabulare vieler Sprachen ist riesig.
2. **OOV:** jedes ungesehene Wort (Flexion, Kompositum, Eigenname) wird `<unk>` — bei
   morphologisch reichen Sprachen dramatisch.
3. **Kein Teilen:** „Nation"/„national"/„Nationalität" sind drei unverwandte Symbole; über
   Sprachen erst recht (z. B. dt. *Information* vs. en. *information*).

Zeichen-Level (wie in Modul 09, Projekt 3) löst OOV, macht Sequenzen aber sehr lang und
zwingt das Modell, Wortstruktur von Null zu lernen. **Subwords** sind der Kompromiss:
häufige Wörter bleiben ein Token, seltene zerfallen in bedeutungstragende Stücke
(*un-glaub-lich*), und Stücke werden **zwischen Wörtern und Sprachen geteilt**.

### 2.2 Byte-Pair Encoding (BPE) — Recap & Präzisierung

BPE (Sennrich et al. 2016, für NMT) lernt ein Vokabular *bottom-up*:

1. Initialisiere das Vokabular mit allen **Einzelzeichen** (oder **Bytes** →
   *byte-level BPE*, dann ist jedes Unicode-Zeichen darstellbar, nie OOV).
2. Zähle im Korpus alle **benachbarten Symbolpaare**. Verschmilz das **häufigste Paar**
   $(a,b)$ zu einem neuen Symbol $ab$ und füge es dem Vokabular hinzu.
3. Wiederhole, bis die gewünschte Vokabulargröße $V$ erreicht ist.

Die Liste der Merges ist deterministisch; **Tokenisierung** wendet die Merges in gelernter
Reihenfolge auf ein neues Wort an. Ergebnis: ein fester Kompromiss zwischen Zeichen (viele
Merges nötig) und Wörtern (wenige, aber grobe Einheiten), steuerbar über $V$.

### 2.3 WordPiece und Unigram-LM

- **WordPiece** (BERT): wie BPE, aber statt „häufigstes Paar" verschmilzt es das Paar, das
  die **Likelihood** eines Unigramm-Sprachmodells über dem Korpus am stärksten erhöht —
  also $\arg\max_{(a,b)} \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$
  (Pointwise Mutual Information-artig) statt reiner Häufigkeit.
- **Unigram-LM** (Kudo 2018, Kern von **SentencePiece**): *top-down*. Starte mit einem
  großen Kandidaten-Vokabular und einem Unigramm-Modell $p(\text{token})$. Die
  Wahrscheinlichkeit einer Segmentierung $\mathbf{x}=(x_1,\dots,x_k)$ eines Textes ist
  $$P(\mathbf{x})=\prod_{i=1}^{k} p(x_i),$$
  und die beste Segmentierung findet man per **Viterbi** (dynamische Programmierung — der
  Bezug zu Modul 07/08!). Trainiere $p$ per **EM** (Modul 05: Erwartungswerte über alle
  Segmentierungen), dann **beschneide** iterativ die Tokens, deren Entfernen die
  Gesamt-Likelihood am wenigsten senkt, bis $|V|$ erreicht ist. Vorteil: ein
  probabilistisches Modell erlaubt **Subword-Regularisierung** (Sampling verschiedener
  Segmentierungen als Daten-Augmentation).

### 2.4 SentencePiece: roher Text rein, sprach-agnostisch

**SentencePiece** ist keine neue Methode, sondern eine *Implementierung* (BPE oder
Unigram) mit einem wichtigen Kniff: Es behandelt **das Leerzeichen als normales Zeichen**
(kodiert als `▁`, U+2581) und arbeitet direkt auf **rohem, ungetokenisiertem Text**. Damit
funktioniert exakt dieselbe Pipeline für Deutsch (Leerzeichen) *und* Chinesisch (keine
Leerzeichen) — und die Detokenisierung ist verlustfrei umkehrbar. Das macht es zum
Standard für mehrsprachige Modelle.

### 2.5 Gemeinsames Vokabular & *fertility*

Trainiert man **ein** Subword-Vokabular auf der **Konkatenation** mehrerer Sprachen,
entsteht *shared vocabulary*: gemeinsame Wortstämme, Ziffern, Interpunktion und
Eigennamen teilen sich Tokens. Ein Maß für die Passung eines Vokabulars zu einer Sprache
ist die **fertility** — durchschnittliche Anzahl Subword-Tokens pro Wort. Ist ein
Vokabular englisch-dominiert, hat Deutsch/Türkisch eine höhere fertility (mehr
Zerstückelung) — ein Fairness-/Effizienzproblem großer LLMs. (Projekt 01 misst genau das.)

---

## 3 · Cross-linguale Wort-Embeddings durch Alignment

Bevor End-to-End-Modelle dominierten, war die eleganteste Idee des cross-lingualen NLP:
**Trainiere Wort-Embeddings pro Sprache getrennt und richte die Räume danach aus.**

### 3.1 Die Beobachtung

Monolinguale Embedding-Räume (Word2Vec/GloVe, Modul 08) haben über Sprachen hinweg eine
**ähnliche geometrische Struktur** — die Nachbarschaften von *(König, Königin, Mann,
Frau)* sehen im Englischen wie im Deutschen aus. Mikolov et al. (2013) schlossen: es sollte
eine **lineare Abbildung** $W$ geben, die den Quell- in den Zielraum dreht.

### 3.2 Supervised Alignment: das Procrustes-Problem

Gegeben ein **Seed-Lexikon** von $n$ Übersetzungspaaren, staple die Quell-Embeddings als
Zeilen von $X\in\mathbb{R}^{n\times d}$ und die zugehörigen Ziel-Embeddings als
$Y\in\mathbb{R}^{n\times d}$. Suche $W$ mit $XW \approx Y$:
$$W^\star=\arg\min_{W}\;\lVert XW - Y\rVert_F^2 .$$

Ohne Einschränkung ist das *lineare Regression*. Entscheidend ist aber die **Orthogonalitäts-
Beschränkung** $W^\top W = I$ (Xing et al. 2015): $W$ soll eine **Rotation/Spiegelung**
sein, keine Verzerrung — das erhält Abstände und Skalarprodukte (und damit Kosinus-
Nachbarschaften). Das ist das **orthogonale Procrustes-Problem** mit *geschlossener*
Lösung über die **Singulärwertzerlegung**:
$$X^\top Y = U\,\Sigma\,V^\top \quad\Longrightarrow\quad W^\star = U V^\top .$$

> **Warum $UV^\top$?** Aus $\lVert XW-Y\rVert_F^2 = \lVert XW\rVert_F^2 - 2\operatorname{tr}(W^\top X^\top Y) + \lVert Y\rVert_F^2$
> und $\lVert XW\rVert_F=\lVert X\rVert_F$ (orthogonales $W$) bleibt: **maximiere**
> $\operatorname{tr}(W^\top X^\top Y)=\operatorname{tr}(W^\top U\Sigma V^\top)$.
> Mit $Z=V^\top W^\top U$ (orthogonal) wird das zu $\operatorname{tr}(Z\Sigma)=\sum_i \sigma_i Z_{ii}\le\sum_i\sigma_i$,
> maximal wenn $Z=I$, also $W^\top=VU^\top \Rightarrow W=UV^\top$. (Alle $\sigma_i\ge0$.)
> **Merkregel für die Richtung:** die Matrix im SVD ist $(\text{Quelle})^\top(\text{Ziel})$,
> dann bildet $W$ die Quelle auf das Ziel ab ($XW\approx Y$).

Vorher normalisiert man die Embeddings üblicherweise (Längen-Normierung, ggf. Mean-Centering),
damit Kosinus-Ähnlichkeit die relevante Metrik ist.

### 3.3 Retrieval, Hubness und CSLS

Übersetzen heißt dann: bilde ein Quellwort ab ($\mathbf{x}W$) und nimm den nächsten
Nachbarn im Zielraum (Kosinus). Problem **Hubness**: in hochdimensionalen Räumen sind
manche Zielvektoren „Hubs", die zu *vielen* Quellwörtern nächster Nachbar sind. **CSLS**
(*Cross-domain Similarity Local Scaling*, Conneau et al. 2018) korrigiert das, indem es
die Ähnlichkeit um die durchschnittliche Nachbarschafts-Dichte bestraft:
$$\text{CSLS}(\mathbf{x},\mathbf{y}) = 2\cos(\mathbf{x},\mathbf{y}) - r_T(\mathbf{x}) - r_S(\mathbf{y}),$$
wobei $r_T(\mathbf{x})$ die mittlere Kosinus-Ähnlichkeit von $\mathbf{x}$ zu seinen $k$
nächsten Ziel-Nachbarn ist (analog $r_S$). Bewertet wird per **Precision@1** auf einem
Test-Lexikon.

### 3.4 Unsupervised Alignment (Ausblick)

Ohne Seed-Lexikon (Conneau et al. 2018, *MUSE*): Ein **adversariales** Netz lernt $W$ so,
dass ein Diskriminator projizierte Quell- nicht von echten Ziel-Vektoren unterscheiden
kann; danach verfeinert **iteratives Procrustes** auf automatisch extrahierten,
hochsicheren Paaren. Erstaunlich: das funktioniert *ganz ohne* Parallel-Daten.

---

## 4 · Neuronale Maschinelle Übersetzung (NMT)

Das Kernproblem: Bilde eine Quellsequenz $\mathbf{x}=(x_1,\dots,x_S)$ auf eine
Zielsequenz $\mathbf{y}=(y_1,\dots,y_T)$ ab und modelliere
$$P(\mathbf{y}\mid\mathbf{x})=\prod_{t=1}^{T} P(y_t\mid y_{<t},\mathbf{x}).$$

### 4.1 seq2seq mit RNNs und der Flaschenhals

Sutskever et al. (2014): ein **Encoder**-RNN liest $\mathbf{x}$ in einen Kontextvektor
$\mathbf{c}$ (den letzten Zustand), ein **Decoder**-RNN erzeugt $\mathbf{y}$ daraus
autoregressiv. Problem: der **ganze** Quellsatz muss durch *einen* Vektor $\mathbf{c}$ —
ein Flaschenhals, der bei langen Sätzen bricht (genau die Grenze, die wir am LSTM in
Projekt 09-01 gesehen haben).

### 4.2 Attention (Bahdanau & Luong)

Die Lösung (Bahdanau et al. 2015): Der Decoder darf in **jedem** Schritt $t$ auf **alle**
Encoder-Zustände $\mathbf{h}_1,\dots,\mathbf{h}_S$ schauen und einen *dynamischen*
Kontext bilden:
$$e_{ti}=\text{score}(\mathbf{s}_{t-1},\mathbf{h}_i),\quad
\alpha_{ti}=\frac{\exp e_{ti}}{\sum_j \exp e_{tj}},\quad
\mathbf{c}_t=\sum_{i=1}^{S}\alpha_{ti}\,\mathbf{h}_i.$$
Die $\alpha_{ti}$ bilden ein weiches **Alignment** zwischen Ziel- und Quellwörtern.
*Bahdanau* nutzt einen additiven Score ($\mathbf{v}^\top\tanh(W[\mathbf{s};\mathbf{h}])$),
*Luong* den einfacheren multiplikativen ($\mathbf{s}^\top W\mathbf{h}$) — letzterer ist der
direkte Vorläufer der Transformer-Attention.

### 4.3 Der Encoder-Decoder-Transformer

Der Transformer (Vaswani et al. 2017) ersetzt die Rekurrenz vollständig durch Attention.
In Modul 09 hast du den **Encoder** (bidirektionale Self-Attention, für BERT) und den
**Decoder** (kausale Self-Attention, für GPT) einzeln gebaut. Die vollständige
NMT-Architektur **verbindet** beide und hat **drei** Attention-Typen:

1. **Encoder-Self-Attention** — *bidirektional* über die Quelle. Jedes Quell-Token sieht
   alle anderen. Ausgabe: kontextualisierte Quell-Repräsentationen
   $\mathbf{H}=(\mathbf{h}_1,\dots,\mathbf{h}_S)$.
2. **Decoder-Masked-Self-Attention** — *kausal* über die bisher erzeugte Zielsequenz
   (kein Blick in die Zukunft, wie beim GPT).
3. **Cross-Attention** — das neue Herzstück: **Query** kommt aus dem Decoder, **Key** und
   **Value** aus dem Encoder-Output $\mathbf{H}$:
   $$\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})
     =\operatorname{softmax}\!\Big(\tfrac{Q_{\text{dec}}K_{\text{enc}}^\top}{\sqrt{d_k}}\Big)V_{\text{enc}}.$$
   So schaut jeder Decoder-Schritt gezielt auf die relevanten Quell-Tokens — die
   Transformer-Verallgemeinerung der Bahdanau-Attention aus 4.2.

Ein Decoder-Layer ist also: *Masked-Self-Attn → Cross-Attn → FFN*, jeweils mit Residual +
LayerNorm. Der Rest (Positional Encoding, Multi-Head, FFN) ist wie in Modul 09.

### 4.4 Training und Decoding

- **Teacher Forcing:** Im Training füttert man dem Decoder die **echten** vorherigen
  Zieltokens $y_{<t}$ (nicht die eigenen Vorhersagen), maskiert kausal und berechnet den
  Loss über alle Positionen parallel. Verlust: Cross-Entropy, oft mit **Label Smoothing**
  (Zielverteilung $=(1-\varepsilon)$ auf dem wahren Token, $\varepsilon$ gleichverteilt) —
  das dämpft Überkonfidenz und verbessert BLEU.
- **Decoding (Inferenz):** autoregressiv, Token für Token.
  - **Greedy:** nimm in jedem Schritt das wahrscheinlichste Token. Schnell, aber
    kurzsichtig.
  - **Beam Search:** verfolge die $k$ besten *Teil*-Hypothesen parallel und expandiere
    sie; wähle am Ende die Sequenz mit der höchsten (längen-normalisierten)
    Log-Wahrscheinlichkeit. Fast immer besser als Greedy.
  - **Spezialtokens:** `<bos>`/`<eos>` markieren Anfang/Ende; die Generierung stoppt bei
    `<eos>`.

---

## 5 · Massiv-mehrsprachige Modelle & Zero-Shot-Transfer

### 5.1 Ein Modell für viele Sprachpaare

Johnson et al. (2017, *Google's Multilingual NMT*): trainiere **ein** Encoder-Decoder-Modell
auf **vielen** Sprachpaaren gleichzeitig, mit **gemeinsamem** Subword-Vokabular. Ein
simpler Trick steuert die Zielsprache: man stellt der Quelle ein **Sprach-Token** voran,
z. B. `<2de>` für „übersetze nach Deutsch". Verblüffendes Ergebnis: das Modell kann
**Zero-Shot** übersetzen — Sprachpaare, die es *nie* gesehen hat (z. B. Ja→Ko, obwohl nur
Ja↔En und Ko↔En trainiert wurden), weil es eine geteilte, sprach-neutrale
Zwischenrepräsentation lernt (*Interlingua*-Hypothese).

### 5.2 Mehrsprachige vortrainierte Encoder

- **mBERT:** BERT (Masked-LM), aber auf der Wikipedia von 104 Sprachen mit gemeinsamem
  WordPiece-Vokabular — *ohne* jede cross-linguale Aufsicht. Trotzdem entstehen erstaunlich
  gut ausgerichtete Repräsentationen.
- **XLM** (Lample & Conneau 2019): fügt ein *Translation Language Modeling* (TLM)-Ziel
  hinzu — Masked-LM auf **konkatenierten Übersetzungspaaren**, sodass das Modell zum Füllen
  einer Lücke *über die Sprachgrenze* schauen kann.
- **XLM-R** (Conneau et al. 2020): XLM-Ansatz, aber RoBERTa-artig auf **2,5 TB**
  CommonCrawl in 100 Sprachen skaliert — lange der Standard für cross-linguales Verständnis.
- **mT5 / mBART:** mehrsprachige **Encoder-Decoder** (Text-zu-Text bzw. Denoising) — für
  Generierung/Übersetzung.

### 5.3 Zero-Shot Cross-Lingual Transfer

Die Killer-Anwendung: **Fine-tune** einen mehrsprachigen Encoder (z. B. XLM-R) auf einer
Aufgabe **nur mit englischen** gelabelten Daten (weil es die gibt), und **evaluiere direkt
auf Deutsch/Suaheli/…** — *ohne* ein einziges zielsprachliches Label. Weil die
Repräsentationen sprachübergreifend ausgerichtet sind, überträgt sich der aufgabenspezifische
Kopf. So bringt man NLP in Sprachen ohne Trainingsdaten (Standard-Benchmark: **XTREME**,
**XNLI**).

### 5.4 Der Preis: *curse of multilinguality*

Bei fester Modellgröße steigt die Qualität pro Sprache zunächst mit mehr Sprachen (positiver
Transfer, geteilte Struktur), **fällt** aber ab einem Punkt wieder, weil die **Kapazität**
auf zu viele Sprachen verdünnt wird (*capacity dilution*) und Sprachen um Parameter
konkurrieren (*interference*). Gegenmittel: größere Modelle, sprach-spezifische Adapter,
kluges Daten-Sampling (Hochsampeln von low-resource-Sprachen mit Temperatur $\tau$).

---

## 6 · Evaluation von Übersetzung

Wie misst man Übersetzungsqualität automatisch (menschliche Bewertung ist teuer)?

### 6.1 BLEU

**BLEU** (Papineni et al. 2002) vergleicht die Kandidatenübersetzung mit einer oder
mehreren **Referenzen** über **n-Gramm-Präzision**. Für $n=1,\dots,N$ (meist $N=4$):
$$p_n=\frac{\sum_{\text{n-Gramme}} \min(\text{count}_{\text{cand}},\,\text{count}_{\text{ref}})}
{\sum_{\text{n-Gramme}} \text{count}_{\text{cand}}}$$
(*clipped precision*: ein n-Gramm zählt höchstens so oft, wie es in der Referenz vorkommt —
das bestraft Wiederholungen). Weil reine Präzision zu **kurze** Übersetzungen belohnt,
kommt eine **Brevity Penalty** dazu:
$$\text{BP}=\begin{cases}1 & c>r\\ e^{1-r/c} & c\le r\end{cases},\qquad
\text{BLEU}=\text{BP}\cdot\exp\!\Big(\sum_{n=1}^{N} w_n \log p_n\Big),$$
mit Kandidatenlänge $c$, Referenzlänge $r$ und Gewichten $w_n=1/N$. BLEU ist ein
**Korpus**-Maß (die Zähler/Nenner werden über den ganzen Testsatz aggregiert), liegt in
$[0,1]$ (oft $\times 100$) und korreliert grob mit menschlichem Urteil.

**Schwächen:** oberflächlich (Wortüberlappung, keine Synonyme/Bedeutung), referenz-abhängig,
zwischen Sprachen/Tokenisierungen nicht vergleichbar (→ **sacreBLEU** standardisiert die
Tokenisierung).

### 6.2 Alternativen

- **chrF:** F-Score über **Zeichen**-n-Gramme — robuster für morphologisch reiche Sprachen
  (partielle Wortübereinstimmung zählt).
- **METEOR:** berücksichtigt Stemming/Synonyme, Recall-gewichtet.
- **COMET / BERTScore:** **gelernte**, embedding-basierte Metriken — korrelieren deutlich
  besser mit Menschen, brauchen aber ein (mehrsprachiges) Modell.

---

## 7 · Zusammenfassung / Cheat-Sheet

| Begriff | Kern in einem Satz |
|---|---|
| **Low-/High-Resource** | Sprachen mit wenig/viel annotierten Daten; Ziel: Transfer dazwischen. |
| **Subword** | Tokens *unter* Wortebene; häufige Wörter ganz, seltene zerlegt; sprach-/wortübergreifend geteilt. |
| **BPE** | Bottom-up: verschmilz iterativ das häufigste Symbolpaar bis $\lvert V\rvert$. |
| **WordPiece** | Wie BPE, aber verschmilzt nach **Likelihood-Gewinn** (PMI-artig). |
| **Unigram-LM / SentencePiece** | Top-down probabilistisch: beste Segmentierung per Viterbi, $p$ per EM, Vokabular beschneiden; roher Text, `▁`=Space. |
| **fertility** | Ø Subword-Tokens pro Wort — Passung Vokabular↔Sprache. |
| **Procrustes** | Orthogonales $W$ mit $XW\approx Y$; Lösung $W=UV^\top$ aus SVD$(X^\top Y)$. |
| **CSLS** | Retrieval-Maß gegen *Hubness*: $2\cos - r_T - r_S$. |
| **seq2seq** | Encoder→Kontext→Decoder; RNN-Variante hat Flaschenhals. |
| **Attention $\alpha_{ti}$** | Weiches Alignment: $\mathbf{c}_t=\sum_i\alpha_{ti}\mathbf{h}_i$. |
| **Cross-Attention** | Decoder-Query ↔ Encoder-Key/Value — verbindet die Transformer-Hälften. |
| **Teacher Forcing** | Training mit echten $y_{<t}$; Label Smoothing gegen Überkonfidenz. |
| **Beam Search** | $k$ beste Teil-Hypothesen verfolgen; besser als Greedy. |
| **Sprach-Token `<2de>`** | Steuert Zielsprache in mehrsprachiger NMT → **Zero-Shot**. |
| **mBERT/XLM-R/mT5** | Mehrsprachig vortrainiert; XLM-R = Skalierung; mT5 = Enc-Dec. |
| **Zero-Shot Transfer** | Fine-tune auf EN, teste auf DE — ohne DE-Labels. |
| **curse of multilinguality** | Zu viele Sprachen pro Kapazität → Qualität fällt. |
| **BLEU** | Clipped n-Gramm-Präzision × Brevity Penalty; Korpus-Maß. |
| **chrF/COMET** | Zeichen-n-Gramm-F / gelernte Metrik — oft besser als BLEU. |

**Formeln zum Merken:**
- Procrustes: $W=UV^\top$ mit $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$.
- Attention-Kontext: $\mathbf{c}_t=\sum_i \alpha_{ti}\mathbf{h}_i,\ \alpha_{ti}=\mathrm{softmax}_i(e_{ti})$.
- BLEU: $\text{BP}\cdot\exp(\sum_n w_n\log p_n)$, $\text{BP}=\min(1,e^{1-r/c})$.

---

## 8 · Selbsttest

<details><summary><b>1.</b> Warum scheitert ein wortbasiertes Vokabular im mehrsprachigen Setting an *drei* Fronten?</summary>

Größe (Vereinigung vieler Sprach-Vokabulare ist riesig), OOV (jede ungesehene Flexion/
Zusammensetzung wird `<unk>`, besonders bei morphologisch reichen Sprachen), und *kein
Teilen* (verwandte Wörter und sprachübergreifende Kognaten sind unabhängige Symbole).
Subwords adressieren alle drei.
</details>

<details><summary><b>2.</b> Worin unterscheiden sich BPE, WordPiece und Unigram-LM im *Kriterium*, nach dem sie das Vokabular bilden?</summary>

BPE verschmilzt das **häufigste** Symbolpaar (reine Frequenz, bottom-up). WordPiece
verschmilzt das Paar mit dem größten **Likelihood-Gewinn** eines Unigramm-LM (PMI-artig,
bottom-up). Unigram-LM startet **top-down** mit großem Vokabular und **beschneidet** die
Tokens, deren Entfernen die Korpus-Likelihood am wenigsten senkt (probabilistisch, mit
Viterbi-Segmentierung und EM).
</details>

<details><summary><b>3.</b> Warum verlangt man beim Embedding-Alignment $W^\top W=I$ statt beliebiges $W$, und wie lautet die Lösung?</summary>

Orthogonalität erzwingt eine reine **Rotation/Spiegelung**, die Längen und Skalarprodukte
(also Kosinus-Nachbarschaften) **erhält** — sonst würde eine freie lineare Abbildung die
Geometrie verzerren und die semantische Struktur zerstören. Die geschlossene Lösung des
orthogonalen Procrustes-Problems ist $W=UV^\top$ mit $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$.
</details>

<details><summary><b>4.</b> Was ist *Hubness* und wie mildert CSLS sie?</summary>

In hochdimensionalen Räumen werden manche Zielvektoren („Hubs") nächster Nachbar
*vieler* Quellwörter, was Nearest-Neighbor-Retrieval verfälscht. CSLS bestraft die
Kosinus-Ähnlichkeit um die lokale Nachbarschafts-Dichte beider Seiten
($2\cos - r_T - r_S$), sodass Hubs relativ abgewertet werden.
</details>

<details><summary><b>5.</b> Nenne die drei Attention-Typen im Encoder-Decoder-Transformer und woher jeweils Q, K, V kommen.</summary>

(1) **Encoder-Self-Attention** — Q,K,V alle aus der Quelle, bidirektional. (2)
**Decoder-Masked-Self-Attention** — Q,K,V aus der bisherigen Zielsequenz, kausal maskiert.
(3) **Cross-Attention** — **Q aus dem Decoder**, **K und V aus dem Encoder-Output**;
verbindet Ziel- mit Quellrepräsentation.
</details>

<details><summary><b>6.</b> Was ist Teacher Forcing und warum genügt im Training *ein* Forward-Pass für den ganzen Zielsatz?</summary>

Teacher Forcing füttert dem Decoder die **echten** vorherigen Tokens $y_{<t}$ statt seiner
eigenen Vorhersagen. Mit **kausaler Maske** kann man alle Positionen parallel berechnen:
Position $t$ sieht nur $y_{<t}$, also liefert ein einziger Forward-Pass die Logits für
*alle* $t$ gleichzeitig (und damit den Loss über den ganzen Satz).
</details>

<details><summary><b>7.</b> Wie ermöglicht ein Sprach-Token `<2de>` *Zero-Shot*-Übersetzung?</summary>

Ein einziges mehrsprachiges Modell mit gemeinsamem Vokabular lernt eine sprach-neutrale
Zwischenrepräsentation. Das vorangestellte Ziel-Sprach-Token steuert nur die Ausgabesprache.
Weil Quelle-Codierung und Zielsteuerung entkoppelt sind, kann das Modell Kombinationen
erzeugen, die es nie als Paar gesehen hat (z. B. Ja→Ko aus Ja↔En- und Ko↔En-Training).
</details>

<details><summary><b>8.</b> Warum belohnt reine n-Gramm-Präzision zu kurze Übersetzungen, und wie behebt BLEU das?</summary>

Eine sehr kurze Ausgabe, die nur ein paar sichere Wörter enthält, hat hohe Präzision (fast
alle ihre n-Gramme stehen in der Referenz), obwohl sie unvollständig ist. BLEU multipliziert
deshalb mit der **Brevity Penalty** $\min(1, e^{1-r/c})$, die Kandidaten kürzer als die
Referenz ($c<r$) exponentiell abstraft.
</details>

<details><summary><b>9.</b> Was besagt der *curse of multilinguality* und welche Gegenmittel gibt es?</summary>

Bei fester Kapazität verbessert das Hinzufügen von Sprachen die Qualität nur bis zu einem
Punkt; danach fällt sie, weil die Parameter auf zu viele Sprachen verteilt werden
(*capacity dilution*, *interference*). Gegenmittel: mehr Modellkapazität, sprach-spezifische
Adapter, temperatur-basiertes Hochsampeln von low-resource-Sprachen.
</details>

<details><summary><b>10.</b> Warum ist BLEU zwischen zwei Systemen aus verschiedenen Papers oft nicht direkt vergleichbar — und was hilft?</summary>

BLEU hängt stark von **Tokenisierung**, Groß-/Kleinschreibung und Referenz-Aufbereitung ab;
verschiedene Implementierungen liefern unterschiedliche Zahlen für dieselbe Übersetzung.
**sacreBLEU** standardisiert Tokenisierung und Berechnung und meldet einen Signatur-String,
sodass Ergebnisse reproduzierbar und vergleichbar werden.
</details>

---

## 9 · Literatur & Quellen

**Lehrbücher / Übersichten**
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft) — Kap. zu MT & seq2seq.
  *Einsteigerfreundlich, kostenlos* online. 🟢💰
- Koehn, *Neural Machine Translation* (2020, Cambridge). *Vertiefend* — das Standardwerk zu NMT.
- Philipp Koehn, *Statistical Machine Translation* (2010) — Hintergrund vor der neuronalen Ära.

**Schlüssel-Papers (alle frei auf arXiv/ACL Anthology 💰)**
- Sennrich, Haddow & Birch (2016): *Neural MT of Rare Words with Subword Units* — **BPE**. 🟢
- Kudo (2018): *Subword Regularization* & Kudo/Richardson (2018): *SentencePiece*. 🟢
- Mikolov et al. (2013): *Exploiting Similarities among Languages for MT* — lineares Mapping. 🟢
- Xing et al. (2015): *Normalized Word Embedding …* — Orthogonalitäts-Beschränkung.
- Conneau et al. (2018): *Word Translation Without Parallel Data* (**MUSE**, CSLS). 🟢
- Bahdanau, Cho & Bengio (2015): *NMT by Jointly Learning to Align and Translate* — **Attention**. 🟢
- Vaswani et al. (2017): *Attention Is All You Need* — der **Transformer**. 🟢
- Johnson et al. (2017): *Google's Multilingual NMT* — **Zero-Shot**, Sprach-Token. 🟢
- Devlin et al. (2019): *BERT* (**mBERT**); Conneau et al. (2020): *XLM-R*; Xue et al. (2021): *mT5*. 🟢
- Papineni et al. (2002): *BLEU*; Post (2018): *A Call for Clarity in Reporting BLEU* (**sacreBLEU**). 🟢

**Kurse / interaktiv (kostenlos 💰)**
- Stanford **CS224N** (NLP with Deep Learning) — Vorlesungen zu MT, Attention, Transformers. 🟢
- Hugging Face **NLP Course**, Kapitel zu Tokenizern & Übersetzung. 🟢 *einsteigerfreundlich*
- Jay Alammar, *The Illustrated Transformer* / *Visualizing seq2seq with Attention* — Blog. 🟢 *einsteigerfreundlich*
- **MUSE** (github.com/facebookresearch/MUSE) & **SentencePiece** (github.com/google/sentencepiece) — Referenz-Implementierungen. 🟢

---

## Die drei Projekte

Die Projekte spannen den Bogen von der Tokenisierung über cross-linguale Repräsentationen
zur vollständigen Übersetzung — auf echten **Deutsch–Englisch**-Daten (Tatoeba), klein und
CPU-/MPS-tauglich:

- **01 – basic** (`projects/01-basic/`): **Mehrsprachige Subword-Tokenisierung mit
  SentencePiece.** Geführtes Notebook: ein gemeinsames BPE-/Unigram-Vokabular auf DE+EN
  trainieren, Tokenisierung analysieren, **fertility** und Vokabular-Teilung zwischen den
  Sprachen messen, mit Wort-/Zeichen-Tokenisierung vergleichen. Viel Anleitung.
- **02 – medium** (`projects/02-medium/`): **Cross-linguale Embeddings alignen
  (Procrustes + CSLS).** Python-Projekt mit Testsuite: monolinguale Embeddings pro Sprache,
  orthogonales Alignment über die SVD selbst herleiten, per **Precision@1** und **CSLS**
  ein Wörterbuch „übersetzen". Wenig Anleitung.
- **03 – final** (`projects/03-final/`): **Ein NMT-Transformer von Grund auf (DE→EN).**
  Keine Code-Vorgabe: vollständiger Encoder-Decoder mit **Cross-Attention**, Teacher
  Forcing, Greedy/Beam-Decoding und **BLEU**-Evaluation auf Tatoeba. Der Master-Level-
  Abschluss, der Modul 09 und 10 zusammenführt.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
