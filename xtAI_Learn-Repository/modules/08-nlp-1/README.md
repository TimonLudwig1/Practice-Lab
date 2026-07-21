# Module 08 — Natural Language Processing 1

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

**What is this about?** This module deals with statistical, "classical" NLP —
the foundations of machine language processing *before* the transformer era
(that comes in module 09). You learn how to represent text formally, how
**language models** estimate the probability of word sequences, how to
**classify** texts, how **words become vectors** (embeddings) and how to
**label sequences** (POS tagging) and **parse sentences**. These procedures are
not merely historical: n-gram models, naive Bayes, TF-IDF and Viterbi still run
in production today, and their concepts (likelihood, smoothing, sequence
modelling) are the foundation the neural models build on.

**Helpful prior knowledge.** Probability (conditional probability, Bayes —
module 07), some linear algebra, Python. From module 07 you already know
**HMMs, filtering and Viterbi** — POS tagging is the direct application. From
modules 04/05, familiarity with classification and logistic regression helps.

**Recommended earlier modules.** Data Science 1/2, Machine Learning 1, Theory of
AI 2 (for HMM/Viterbi). Module 01 already had a naive Bayes spam filter — here
we go deeper.

**Following module.** NLP 2 (module 09): neural language models, RNN/LSTM,
seq2seq, attention, transformers. Multilingual NLP (module 10) and others build
on it.

---

## Learning objectives

After this module you should be able to

- **preprocess** text systematically (tokenization, normalization, stemming vs.
  lemmatization) and explain the statistical structure of language (**Zipf's law**);
- set up **n-gram language models**, understand the **sparsity problem** and
  solve it with **smoothing** (Laplace/add-$k$, Good-Turing, **Kneser-Ney**,
  backoff/interpolation); evaluate models with **perplexity**;
- classify texts with **multinomial naive Bayes** and **logistic
  regression / MaxEnt** and evaluate them cleanly with **precision/recall/F1**;
- understand **word representations** — from **TF-IDF** and **PPMI** through
  **Word2Vec** (skip-gram with negative sampling) to **GloVe** — and explain the
  **distributional hypothesis** behind them;
- carry out **sequence labeling**: **POS tagging** with **HMM + Viterbi**, and
  place the limits (label bias) as well as the solution (**MEMM**, **CRF**);
- understand the foundations of **syntax**: context-free grammars, the **CKY
  parser**, **PCFGs** and **dependency parsing**;
- explain *why* every procedure works — likelihood, smoothing, distributional
  semantics, dynamic programming.

---

## Part 1 — Foundations: text and language models

### 1.1 From text to tokens

Raw text is a string of characters; NLP needs **discrete units**. The pipeline:

- **Tokenization**: splitting into **tokens** (words, numbers, punctuation). It
  sounds trivial, and it is not: "don't" → `do` + `n't`? Is "New York" one token
  or two? URLs, hashtags, emoji? Common today are **subword tokenizers** (BPE,
  WordPiece, SentencePiece — the details are in modules 09/10), which split rare
  words into frequent pieces and thereby defuse the **out-of-vocabulary problem**.
- **Normalization**: lowercasing, Unicode normalization, unifying numbers and
  dates. It is context dependent — for sentiment, "GREAT" ≠ "great".
- **Stemming vs. lemmatization**: both reduce word forms to a base form.
  **Stemming** (e.g. the Porter stemmer) chops off endings by rule (`running`,
  `runs` → `run`; but `argument` → `argu` — crude). **Lemmatization** uses a
  lexicon plus morphology and delivers real base forms (`better` → `good`); it is
  more accurate but more expensive.
- **Stop words** (the, is, of …) are removed depending on the task — often useful
  for classification, harmful for language modelling.

**Zipf's law.** Word frequency follows a **power law**: if you order words by
rank $r$, their frequency is $f \propto 1/r$. The consequence: few words are
extremely frequent, but the **long tail** of rare words is huge — most word forms
are seen in training *very rarely or never*. That is the root of the **sparsity
problem** that shapes the whole of statistical NLP.

### 1.2 Language models and the chain rule

A **language model (LM)** assigns a probability $P(w_{1:n})$ to a word sequence
$w_{1:n} = w_1 \dots w_n$. Applications: speech recognition, translation,
autocompletion, text generation. Exactly, by the chain rule:
$$
P(w_{1:n}) = \prod_{i=1}^{n} P(w_i \mid w_{1:i-1}).
$$
The problem: $P(w_i \mid w_{1:i-1})$ over the *whole* history cannot be estimated
(infinitely many possible contexts). The **Markov assumption** (as in module 07):
the next context depends only on the last $k$ words. An **n-gram model**
approximates
$$
P(w_i \mid w_{1:i-1}) \approx P(w_i \mid w_{i-N+1:i-1}).
$$
For the **bigram** ($N=2$): $P(w_i\mid w_{i-1})$; the **trigram** ($N=3$):
$P(w_i\mid w_{i-2},w_{i-1})$.

**Maximum likelihood estimation (MLE):** count relative frequencies.
$$
P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})},
$$
where $C(\cdot)$ is the count in the corpus. The start and end of a sentence are
marked with the special tokens `<s>` / `</s>`.

### 1.3 The sparsity problem and smoothing

The MLE assigns probability **0** to every n-gram **unseen in training** — and
thereby makes the whole sentence impossible ($P=0$), which is fatal (perplexity
$\infty$). Because of Zipf that happens *constantly*. **Smoothing** shifts
probability mass from what has been seen to what has not.

**Laplace / add-$k$ smoothing.** Add a pseudo count $k$ (often $k=1$) to every
counter:
$$
P_{\text{Add-}k}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k\,|V|},
$$
with the vocabulary size $|V|$. Simple, but crude — it takes too much mass away
from frequent n-grams. Sensible only as a baseline.

**Good-Turing.** It estimates the mass for the unseen from the number of n-grams
seen **once** (the singletons): the rescaled counts are
$c^\ast = (c+1)\,\dfrac{N_{c+1}}{N_c}$, where $N_c$ is the number of n-grams with
frequency $c$. The total mass for unseen events is $N_1/N$.

**Backoff and interpolation.** If a trigram is unseen, use the bigram; if that is
missing too, the unigram. **Katz backoff** *falls back* to the lower order (with
discounting); **interpolation** always *mixes* all orders:
$$
P_{\text{interp}}(w_i\mid w_{i-2},w_{i-1}) = \lambda_3 P(w_i\mid w_{i-2},w_{i-1})
+ \lambda_2 P(w_i\mid w_{i-1}) + \lambda_1 P(w_i), \quad \textstyle\sum\lambda=1.
$$
The $\lambda$ are learned on a held-out set.

**Kneser-Ney (the gold standard).** Two ideas. (1) **Absolute discounting:**
subtract a fixed amount $d$ from every count and redistribute the freed mass.
(2) The clever part — the **continuation model** for the lower order: instead of
"how *frequent* is the word $w$?", Kneser-Ney asks "in how *many different
contexts* does $w$ occur?". The classical example: "Francisco" is frequent, but
almost only after "San" — as a fallback probability it should be *low*. The
(interpolated) Kneser-Ney formula for the bigram:
$$
P_{\text{KN}}(w_i\mid w_{i-1}) = \frac{\max\big(C(w_{i-1},w_i)-d,\,0\big)}{C(w_{i-1})}
+ \lambda(w_{i-1})\; P_{\text{cont}}(w_i),
$$
$$
P_{\text{cont}}(w_i) = \frac{\big|\{w' : C(w', w_i) > 0\}\big|}{\big|\{(w',w'') : C(w',w'')>0\}\big|},
\qquad
\lambda(w_{i-1}) = \frac{d}{C(w_{i-1})}\,\big|\{w : C(w_{i-1},w)>0\}\big|.
$$
$\lambda(w_{i-1})$ is the normalized discount (the mass freed by subtracting $d$).
Kneser-Ney (in its *modified* variant) was for years the best n-gram procedure.

### 1.4 Perplexity — how good is a language model?

The standard intrinsic metric. The **perplexity** of a model on a test sequence
$w_{1:n}$ is the inverse geometric mean probability per word:
$$
\mathrm{PP}(w_{1:n}) = P(w_{1:n})^{-1/n}
= \Big(\prod_{i=1}^{n} \frac{1}{P(w_i\mid w_{1:i-1})}\Big)^{1/n}
= 2^{\,-\frac{1}{n}\sum_i \log_2 P(w_i\mid \cdot)}.
$$
The interpretation: the **effective branching factor** — on average the model has
to distinguish "that many equally probable words" at every position. **Lower is
better.** Perplexity is $2$ to the power of the **cross-entropy** (in bits) — the
bridge to information theory. Important: perplexity is only comparable *within*
the same vocabulary, and an unseen word ($P=0$) makes it $\infty$ — which is why
one *must* smooth.

---

## Part 2 — Building up: classification and word representations

### 2.1 Text classification with naive Bayes

**The task:** assign a class $c$ to a document $d$ (spam/ham, sentiment, topic).
The **naive Bayes** classifier picks the class with the maximum posterior
probability (MAP):
$$
\hat c = \arg\max_c P(c\mid d) = \arg\max_c P(c)\,P(d\mid c),
$$
with the **naive independence assumption**: the words are conditionally
independent given the class. In the **multinomial** model (words as counts):
$$
\hat c = \arg\max_c \Big[\log P(c) + \sum_{i} \log P(w_i\mid c)\Big],
$$
with $P(w\mid c) = \dfrac{C(w,c)+\alpha}{\sum_{w'}\big(C(w',c)+\alpha\big)}$
(Laplace smoothing, $\alpha=1$). One computes in **log space** against underflow.
Despite the obviously false independence assumption, naive Bayes is surprisingly
strong, extremely fast and an indispensable baseline. (Module 01 had a spam
filter; here you understand the model completely.)

### 2.2 Logistic regression / MaxEnt

Naive Bayes is a **generative** model ($P(d\mid c)$); **logistic regression** is
its **discriminative** counterpart ($P(c\mid d)$ directly). It models
$$
P(c\mid d) = \mathrm{softmax}\big(\mathbf{w}_c^\top \mathbf{f}(d)\big)
= \frac{\exp(\mathbf{w}_c^\top \mathbf{f}(d))}{\sum_{c'}\exp(\mathbf{w}_{c'}^\top \mathbf{f}(d))},
$$
where $\mathbf{f}(d)$ is a **feature vector** (word counts, TF-IDF, arbitrary
features). In the NLP tradition this is called a **maximum entropy model
(MaxEnt)**. The weights are learned by minimizing the **cross-entropy** (with L2
regularization) via gradient descent (module 05). The advantage over naive Bayes:
LR can cope with **correlated, overlapping features** (e.g. word + bigram +
capitalization) without counting them twice — usually somewhat more accurate when
there is enough data.

**Evaluation.** Accuracy alone is deceptive with unbalanced classes. For one class:
$$
\text{precision} = \frac{TP}{TP+FP}, \quad
\text{recall} = \frac{TP}{TP+FN}, \quad
F_1 = \frac{2\cdot P\cdot R}{P+R}
$$
($F_1$ = the harmonic mean). With several classes one averages **macro** (every
class equal) or **micro** (every instance equal). Always on a separate **test
set** and ideally with **cross-validation**.

### 2.3 Vector representations: from TF-IDF to embeddings

**Bag of words and TF-IDF.** A document as a vector over the vocabulary. Raw
counts overweight frequent words; **TF-IDF** corrects that:
$$
\text{tf-idf}(w,d) = \underbrace{\text{tf}(w,d)}_{\text{frequency in } d}\;\cdot\;
\underbrace{\log\frac{N}{\text{df}(w)}}_{\text{inverse document frequency}},
$$
where $N$ = the number of documents and $\text{df}(w)$ = the number of documents
containing $w$. Words that occur *everywhere* (the, is) get a low weight; rare,
discriminating words a high one. Document similarity is measured with the
**cosine similarity** $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$.

**The distributional hypothesis.** *"You shall know a word by the company it
keeps"* (Firth): words with similar contexts have similar meanings. That
motivates **dense word vectors (embeddings)**, which encode semantic proximity
geometrically.

**PPMI.** A first, count-based step: the **positive pointwise mutual
information** between a word $w$ and a context $c$:
$$
\text{PMI}(w,c) = \log_2 \frac{P(w,c)}{P(w)\,P(c)}, \qquad
\text{PPMI}(w,c) = \max\big(\text{PMI}(w,c),\,0\big).
$$
A word-context matrix of PPMI values (reduced in dimension by SVD if desired)
already yields usable, dense vectors.

**Word2Vec (skip-gram with negative sampling).** Instead of counting, it is
*learned*. Skip-gram predicts the context words from a target word. With
**negative sampling** the efficient training objective becomes (for a
target-context pair $(w,c)$ and $k$ random "negative" words):
$$
\log \sigma(\mathbf{v}_c^\top \mathbf{v}_w) + \sum_{j=1}^{k}
\mathbb{E}_{c_j\sim P_n}\big[\log \sigma(-\mathbf{v}_{c_j}^\top \mathbf{v}_w)\big],
$$
with $\sigma$ = the sigmoid. The intuition: real pairs should have a high dot
product, random pairs a low one. The result is vectors in which **semantic
relations become directions** — the famous analogy
$\text{vec(king)} - \text{vec(man)} + \text{vec(woman)} \approx \text{vec(queen)}$.

**GloVe** combines both worlds: it factorizes the **global** co-occurrence matrix
and minimizes
$$
J = \sum_{i,j} f(X_{ij})\,\big(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde b_j
- \log X_{ij}\big)^2,
$$
where $X_{ij}$ is the co-occurrence count and $f$ a damping weight function.
These *static* embeddings (one vector per word type) are the precursor of the
*contextual* embeddings (BERT and company, module 09), where every context of
occurrence gets its own vector.
---

## Part 3 — Advanced: sequence labeling and syntax

### 3.1 POS tagging as a sequence problem

**Part-of-speech tagging** assigns its word class to every word (noun, verb,
adjective …). The challenge is **ambiguity**: "book" is a noun *or* a verb — the
context decides. This is a **sequence labeling** problem: find the best tag
sequence $t_{1:n}$ for the word sequence $w_{1:n}$.

**The HMM tagger.** Directly the model from module 07: the tags are the *hidden*
states, the words the *observations*. By Bayes:
$$
\hat t_{1:n} = \arg\max_{t_{1:n}} P(t_{1:n}\mid w_{1:n})
= \arg\max_{t_{1:n}} \prod_i \underbrace{P(w_i\mid t_i)}_{\text{emission}}\;
\underbrace{P(t_i\mid t_{i-1})}_{\text{transition}}.
$$
The **transition probabilities** $P(t_i\mid t_{i-1})$ and **emission
probabilities** $P(w_i\mid t_i)$ are estimated by MLE from a tagged corpus (with
smoothing, especially for unknown words). The optimal sequence is found by the
**Viterbi algorithm** (module 07) in $O(n\,|T|^2)$ by dynamic programming:
$$
v_t(j) = \max_{i}\; v_{t-1}(i)\; P(t_j\mid t_i)\; P(w_t\mid t_j),
$$
with back pointers for reconstructing the path. HMM taggers reach about 95–96 %
accuracy — a strong, transparent baseline.

### 3.2 The limit of the HMM: label bias, MEMM and CRF

An HMM is **generative** and can only use limited features (the word itself). But
one would like **rich features** (the suffix "-ing", capitalization, the
surrounding words). Two discriminative successors:

- **MEMM (maximum entropy Markov model):** it models $P(t_i\mid t_{i-1}, w_i,
  \dots)$ directly with a MaxEnt classifier per position. The problem: the
  **label bias problem** — states with few successor states "prefer" their
  transitions, because normalization is local per step.
- **CRF (conditional random field):** the solution. A **linear-chain CRF**
  normalizes **globally** over the whole sequence:
  $$
  P(t_{1:n}\mid w_{1:n}) = \frac{1}{Z(w)}\exp\Big(\sum_{i}\sum_{k}\theta_k\,
  f_k(t_{i-1}, t_i, w, i)\Big),
  $$
  with feature functions $f_k$, weights $\theta_k$ and the global partition
  function $Z(w)$. CRFs avoid the label bias, permit arbitrary features and were
  the standard for sequence labeling (NER, chunking) before the neural models.
  Decoding is again done with Viterbi, and training by gradient descent over the
  (convex) log-likelihood.

### 3.3 Syntax: constituents, CFG and the CKY parser

**Syntactic parsing** uncovers the grammatical structure of a sentence. A
**context-free grammar (CFG)** (known from module 06 / theory) consists of
terminals (words), non-terminals (phrases such as NP, VP), a start symbol (S) and
**production rules** ($S\to NP\ VP$, $NP\to Det\ N$ …).

**The CKY algorithm.** It decides and constructs parses for grammars in **Chomsky
normal form** (every rule $A\to BC$ or $A\to w$) by **dynamic programming** over a
triangular table: `chart[i][j]` contains all non-terminals that can generate the
subspan $w_{i:j}$. One fills it bottom-up by span length; the runtime is
$O(n^3\,|G|)$. CKY is the syntactic counterpart of Viterbi — dynamic programming
over structures.

**PCFG.** A **probabilistic CFG** gives every rule a probability (the $\sum$ over
all rules with the same left-hand side $=1$). The **probabilistic CKY** then finds
the **most probable** parse (useful for resolving **ambiguity**, e.g. PP
attachment: "I saw the man with the telescope"). The rule probabilities are
estimated from a **treebank** (e.g. the Penn Treebank).

### 3.4 Dependency parsing — an outlook

Instead of constituents (phrases), **dependency parsing** models direct **binary
relations** between words (head → dependent, labelled with `nsubj`, `dobj`,
`amod` …). The resulting **dependency tree** is often more practical for
extracting meaning and is the standard of the **Universal Dependencies** project
(a cross-lingual annotation convention — relevant for module 10). Two paradigms:
**transition-based** (a classifier chooses shift/reduce actions, linear in $n$)
and **graph-based** (find the maximum spanning tree over all possible edges).
Neural dependency parsers (module 09) dominate today.

---

## Summary / cheat sheet

**Language models**

| Notion | Core |
|---|---|
| N-gram | $P(w_i\mid w_{i-N+1:i-1})$; MLE $=\frac{C(w_{i-1},w_i)}{C(w_{i-1})}$ |
| Add-$k$ | $\frac{C+k}{C(w_{i-1})+k|V|}$ |
| Interpolation | $\sum_n \lambda_n P_n$, $\sum\lambda=1$; the $\lambda$ on held-out data |
| Kneser-Ney | absolute discounting $d$ + continuation $P_{\text{cont}}(w)=\frac{|\{w':C(w',w)>0\}|}{\#\text{bigram types}}$ |
| Perplexity | $PP = P(w_{1:n})^{-1/n} = 2^{H}$; lower = better; $\infty$ when $P=0$ |

**Classification and vectors**

| Notion | Core |
|---|---|
| Naive Bayes | $\hat c=\arg\max_c \log P(c)+\sum_i\log P(w_i\mid c)$; generative |
| Logistic regression | $P(c\mid d)=\mathrm{softmax}(\mathbf w_c^\top\mathbf f)$; discriminative, correlated features are fine |
| $F_1$ | $\frac{2PR}{P+R}$; macro/micro averaging |
| TF-IDF | $\text{tf}\cdot\log\frac{N}{\text{df}}$; similarity via cosine |
| PPMI | $\max(\log_2\frac{P(w,c)}{P(w)P(c)},0)$ |
| Word2Vec (SGNS) | $\log\sigma(\mathbf v_c^\top\mathbf v_w)+\sum_j\log\sigma(-\mathbf v_{c_j}^\top\mathbf v_w)$ |
| GloVe | $\sum f(X_{ij})(\mathbf w_i^\top\tilde{\mathbf w}_j+b_i+\tilde b_j-\log X_{ij})^2$ |

**Sequence and syntax**

| Notion | Core |
|---|---|
| HMM tagger | $\arg\max_t\prod_i P(w_i\mid t_i)P(t_i\mid t_{i-1})$; Viterbi $O(n|T|^2)$ |
| CRF | globally normalized, $\frac{1}{Z(w)}\exp(\sum\theta_k f_k)$; no label bias |
| CKY | a DP parser for CNF, $O(n^3|G|)$; PCFG → the most probable parse |
| Dependency | head→dependent relations; transition-/graph-based |

---

## Self-test

<details><summary><b>1. Why does the MLE of an n-gram model without smoothing make almost every test sentence impossible?</b></summary>

Because of Zipf, every real test sentence contains, with high probability, an
n-gram that never occurred in training. Its MLE is $0$, and since $P(w_{1:n})$ is
a *product*, the whole sentence probability becomes $0$ (perplexity $\infty$).
Smoothing shifts some mass onto unseen events so that nothing is exactly $0$.
</details>

<details><summary><b>2. What is the central idea of Kneser-Ney compared with add-$k$?</b></summary>

Two things. (1) **Absolute discounting:** a fixed amount $d$ is subtracted from
every count (instead of proportionally as with add-$k$), which empirically
matches the Good-Turing rescaling well. (2) The actual clever part — the
**continuation probability** for the lower order: a word is only a probable
fallback if it appears in *many different contexts*, not merely often.
"Francisco" is frequent, but almost only after "San" → low as a unigram fallback.
Add-$k$ has neither of these ideas and smooths bluntly.
</details>

<details><summary><b>3. Interpret perplexity. Why is lower better?</b></summary>

Perplexity is $PP=P(w_{1:n})^{-1/n}=2^{H}$ with $H$ = the cross-entropy per word.
It is the **effective branching factor**: the number of equally probable
alternatives the model wavers between at every position on average. A good model
is *less surprised* by the actual words, gives them a higher probability →
a smaller $H$ → a smaller perplexity. A model that predicted the test text
perfectly would have $PP=1$.
</details>

<details><summary><b>4. Naive Bayes makes an obviously false assumption. Why does it work anyway?</b></summary>

The conditional independence of the words given the class is factually false
(words correlate strongly). But for the *classification decision* only which class
gets the highest score matters — and the errors in the estimated probabilities
often cancel far enough that the *ranking* of the classes stays correct.
Furthermore NB is extremely data efficient (few parameters, no overfitting on
small data) and fast. Hence: a weak estimate of probabilities, but often a strong
classification.
</details>

<details><summary><b>5. Generative vs. discriminative: naive Bayes vs. logistic regression.</b></summary>

**Generative** (naive Bayes) models $P(d\mid c)$ and $P(c)$, that is, the joint
distribution — it "explains" how the data arise. **Discriminative** (logistic
regression) models $P(c\mid d)$ directly, only the decision boundary. The
consequences: NB learns faster / with less data and is robust, but it makes the
independence assumption; LR can use **correlated, overlapping features** without
double counting and is usually more accurate given enough data. Both are linear
classifiers in log space.
</details>

<details><summary><b>6. Why does TF-IDF weight rare words higher? What is the logarithm for?</b></summary>

The **inverse document frequency** $\log\frac{N}{\text{df}(w)}$ is large when $w$
occurs in *few* documents — such words are discriminative. Words in *all*
documents (the, is) have $\text{df}\approx N$, so $\log 1=0$, and they are faded
out. The logarithm damps: the jump from df=1 to df=2 counts far more than from
df=1000 to 1001 — it prevents extremely rare words from dominating the vectors.
</details>

<details><summary><b>7. Explain the distributional hypothesis and how Word2Vec implements it.</b></summary>

The distributional hypothesis: words that appear in similar contexts have similar
meanings ("know a word by the company it keeps"). Word2Vec (skip-gram) implements
it by learning vectors such that a word predicts its actual context words well (a
high dot product for real pairs) and random words badly (negative sampling).
Words with similar contexts thus get similar vectors; semantic relations become
consistent directions in the space (analogies).
</details>

<details><summary><b>8. How does an HMM tag parts of speech, and what does Viterbi do in it?</b></summary>

The HMM treats the tags as hidden states: it looks for
$\arg\max_t\prod_i P(w_i\mid t_i)P(t_i\mid t_{i-1})$ from emission and transition
probabilities (estimated by MLE from a treebank). **Viterbi** solves this
maximization over all exponentially many tag sequences exactly, by dynamic
programming in $O(n|T|^2)$: $v_t(j)=\max_i v_{t-1}(i)P(t_j\mid t_i)P(w_t\mid t_j)$,
with back pointers for reconstructing the best sequence.
</details>

<details><summary><b>9. What is the label bias problem, and how does a CRF solve it?</b></summary>

An MEMM normalizes the transition probabilities **locally, per position**. States
with *few* possible successor states distribute their entire mass over those few
— regardless of how well the observation fits. Such transitions are thereby
artificially favoured (*label bias*). A **CRF** instead normalizes **globally**
over the entire sequence (one partition function $Z(w)$ for the whole sentence),
so that local observations weight the overall decision correctly — the bias
disappears.
</details>

<details><summary><b>10. Why is CKY $O(n^3)$, and what does the PCFG variant achieve?</b></summary>

CKY fills a triangular table over all **spans** $w_{i:j}$: there are $O(n^2)$
spans, and for each it tries all $O(n)$ **split points** — together $O(n^3)$
(times the grammar size). For every span it stores which non-terminals can
generate it. The **probabilistic** CKY variant stores, instead of
"possible/not", the **probability** of the best subtree (the product of the rule
and subtree probabilities) and thereby finds the *most probable* parse — the
standard method for resolving syntactic ambiguity.
</details>

---

## Literature and sources

**Textbooks**
- **Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. (draft)** — *the*
  standard reference of NLP. Chapters on n-grams, naive Bayes, logistic
  regression, vector semantics, POS tagging (HMM), sequence labeling (CRF),
  CFG/CKY. **Free** as a PDF at `web.stanford.edu/~jurafsky/slp3`. *The primary
  source — beginner friendly and complete.*
- **Manning & Schütze, *Foundations of Statistical NLP*, MIT Press** — the classic
  of statistical NLP, deeper on smoothing and grammars. *Advanced.*
- **Eisenstein, *Introduction to Natural Language Processing*, MIT Press** — a
  modern, mathematically clean presentation. **The draft is free** online. *Advanced.*

**Freely available courses and materials** (free)
- **Stanford CS124 / CS224N (the pre-neural parts)** — videos and slides on
  n-grams, naive Bayes, TF-IDF. *Beginner friendly.*
- **The NLTK book** (`nltk.org/book`) — a practical introduction with Python
  (tokenization, tagging, parsing). *Beginner friendly, hands-on.*
- **The scikit-learn documentation on "Working with Text Data"** — TF-IDF, naive
  Bayes, a classification pipeline. *Directly practical.*
- **Universal Dependencies** (`universaldependencies.org`) — freely available
  tagged corpora in many languages (the basis for the POS project). *Practical.*

**Interactive / visualizations** (free)
- **The TensorFlow Embedding Projector** (`projector.tensorflow.org`) — explore
  Word2Vec/GloVe embeddings in 3D (neighbours, analogies). *Very beginner friendly.*
- **Jurafsky & Martin** contains excellent worked examples of Viterbi and CKY.

**Classical papers** (free, advanced)
- Chen & Goodman (1999): *An Empirical Study of Smoothing Techniques* — the
  definitive study of smoothing (Kneser-Ney).
- Mikolov et al. (2013): *Efficient Estimation of Word Representations* &
  *Distributed Representations* — Word2Vec.
- Pennington et al. (2014): *GloVe: Global Vectors for Word Representation*.
- Lafferty, McCallum & Pereira (2001): *Conditional Random Fields*.

---

## The three projects

The three projects cover the three parts of the module — language models,
classification, sequence labeling — with real text data and increasing amounts of
your own work:

- **01 – basic** (`projects/01-basic/`): **an n-gram language model.** A guided
  notebook: tokenization, unigram/bigram/trigram with add-$k$ smoothing and
  interpolation, **perplexity** on held-out data, and **text generation** by
  sampling. A real corpus (a public domain book). Plenty of guidance.
- **02 – medium** (`projects/02-medium/`): **text classification.** A Python
  project: implement **multinomial naive Bayes by hand**, compare it with
  **TF-IDF + logistic regression** (scikit-learn), evaluate cleanly with
  precision/recall/F1. Real data (20 newsgroups). Little guidance.
- **03 – final** (`projects/03-final/`): **an HMM POS tagger with Viterbi.** No
  given code: estimate the emission/transition probabilities (with smoothing for
  unknown words), **Viterbi from scratch**, evaluated on real **Universal
  Dependencies** data (about 91 % tag accuracy on the noisy EWT web text; unknown
  words are the bottleneck — WSJ-like corpora reach about 95–96 %). Master's
  level, a direct application of module 07.

Details, setup and reference solutions are in the `README.md` of each project folder.

---
---

# Modul 08 — Natural Language Processing 1 (deutsche Fassung)

**Worum geht es?** Dieses Modul behandelt die **statistische, „klassische" NLP**
— die Grundlagen der maschinellen Sprachverarbeitung *vor* der Transformer-Ära
(die kommt in Modul 09). Du lernst, wie man Text formal repräsentiert, wie
**Sprachmodelle** die Wahrscheinlichkeit von Wortfolgen schätzen, wie man Texte
**klassifiziert**, wie **Wörter zu Vektoren** werden (Embeddings) und wie man
**Sequenzen labelt** (POS-Tagging) und **Sätze parst**. Diese Verfahren sind nicht
bloß historisch: N-Gramm-Modelle, Naive Bayes, TF-IDF und Viterbi laufen bis heute
produktiv, und ihre Konzepte (Likelihood, Glättung, Sequenzmodellierung) sind das
Fundament, auf dem die neuronalen Modelle aufsetzen.

**Hilfreiche Vorkenntnisse.** Wahrscheinlichkeitsrechnung (bedingte
Wahrscheinlichkeit, Bayes — Modul 07), etwas lineare Algebra, Python. Aus Modul 07
kennst du bereits **HMMs, Filtering und Viterbi** — POS-Tagging ist die direkte
Anwendung. Aus Modul 04/05 hilft Vertrautheit mit Klassifikation und logistischer
Regression.

**Empfohlene Vormodule.** Data Science 1/2, Machine Learning 1, Theorie der KI 2
(für HMM/Viterbi). Modul 01 hatte bereits einen Naive-Bayes-Spamfilter — hier
gehen wir tiefer.

**Folgemodul.** NLP 2 (Modul 09): neuronale Sprachmodelle, RNN/LSTM, seq2seq,
Attention, Transformer. Multilingual NLP (Modul 10) und weitere bauen darauf auf.

---

## Lernziele

Nach diesem Modul solltest du in der Lage sein,

- Text systematisch **vorzuverarbeiten** (Tokenisierung, Normalisierung, Stemming
  vs. Lemmatisierung) und die statistische Struktur von Sprache (**Zipf-Gesetz**) zu erklären;
- **N-Gramm-Sprachmodelle** aufzustellen, das **Sparsity-Problem** zu verstehen und
  mit **Glättung** (Laplace/Add-$k$, Good-Turing, **Kneser-Ney**, Backoff/Interpolation)
  zu lösen; Modelle mit **Perplexität** zu bewerten;
- Texte mit **multinomialem Naive Bayes** und **logistischer Regression / MaxEnt**
  zu klassifizieren und mit **Precision/Recall/F1** sauber zu evaluieren;
- **Wortrepräsentationen** zu verstehen — von **TF-IDF** und **PPMI** über
  **Word2Vec** (Skip-Gram mit Negative Sampling) bis **GloVe** — und die
  **distributionelle Hypothese** dahinter zu erklären;
- **Sequenz-Labeling** durchzuführen: **POS-Tagging** mit **HMM + Viterbi**, und die
  Grenzen (label bias) sowie die Lösung (**MEMM**, **CRF**) einzuordnen;
- die Grundlagen der **Syntax** zu verstehen: kontextfreie Grammatiken, der
  **CKY-Parser**, **PCFGs** und **Dependency Parsing**;
- zu erklären, *warum* jedes Verfahren funktioniert — Likelihood, Glättung,
  distributionelle Semantik, dynamische Programmierung.

---

## Teil 1 — Grundlagen: Text und Sprachmodelle

### 1.1 Von Text zu Tokens

Rohtext ist eine Zeichenkette; NLP braucht **diskrete Einheiten**. Die Pipeline:

- **Tokenisierung**: Zerlegung in **Tokens** (Wörter, Zahlen, Satzzeichen).
  Klingt trivial, ist es nicht: „don't" → `do` + `n't`? „New York" ein oder zwei
  Tokens? URLs, Hashtags, Emojis? Heute üblich sind **Subword-Tokenizer** (BPE,
  WordPiece, SentencePiece — Details in Modul 09/10), die seltene Wörter in
  häufige Teilstücke zerlegen und so das **Out-of-Vocabulary-Problem** entschärfen.
- **Normalisierung**: Kleinschreibung, Unicode-Normalisierung, Zahlen/Datumsangaben
  vereinheitlichen. Kontextabhängig — bei Sentiment ist „GREAT" ≠ „great".
- **Stemming vs. Lemmatisierung**: Beides führt Wortformen auf eine Grundform
  zurück. **Stemming** (z. B. Porter-Stemmer) schneidet regelbasiert Endungen ab
  (`running`, `runs` → `run`; aber `argument` → `argu` — grob). **Lemmatisierung**
  nutzt ein Lexikon + Morphologie und liefert echte Grundformen (`better` → `good`),
  ist genauer, aber teurer.
- **Stoppwörter** (the, is, of …) entfernt man je nach Aufgabe — bei Klassifikation
  oft nützlich, bei Sprachmodellierung schädlich.

**Zipf-Gesetz.** Die Worthäufigkeit folgt einer **Potenzverteilung**: Ordnet man
Wörter nach Rang $r$, so ist ihre Häufigkeit $f \propto 1/r$. Konsequenz: Wenige
Wörter sind extrem häufig, aber der **lange Schwanz** seltener Wörter ist riesig —
die meisten Wortformen sieht man im Training *sehr selten oder nie*. Das ist die
Wurzel des **Sparsity-Problems**, das die ganze statistische NLP prägt.

### 1.2 Sprachmodelle und die Kettenregel

Ein **Sprachmodell (LM)** weist einer Wortfolge $w_{1:n} = w_1 \dots w_n$ eine
Wahrscheinlichkeit $P(w_{1:n})$ zu. Anwendungen: Spracherkennung, Übersetzung,
Autovervollständigung, Textgenerierung. Per Kettenregel exakt:
$$
P(w_{1:n}) = \prod_{i=1}^{n} P(w_i \mid w_{1:i-1}).
$$
Das Problem: $P(w_i \mid w_{1:i-1})$ über die *ganze* Vorgeschichte lässt sich nicht
schätzen (unendlich viele mögliche Kontexte). **Markov-Annahme** (wie in Modul 07):
Der nächste Kontext hängt nur von den letzten $k$ Wörtern ab. Ein
**N-Gramm-Modell** approximiert
$$
P(w_i \mid w_{1:i-1}) \approx P(w_i \mid w_{i-N+1:i-1}).
$$
Für das **Bigramm** ($N=2$): $P(w_i\mid w_{i-1})$; **Trigramm** ($N=3$):
$P(w_i\mid w_{i-2},w_{i-1})$.

**Maximum-Likelihood-Schätzung (MLE):** Zähle relative Häufigkeiten.
$$
P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})},
$$
wobei $C(\cdot)$ die Zählung im Korpus ist. Satzanfang/-ende markiert man mit
speziellen Tokens `<s>` / `</s>`.

### 1.3 Das Sparsity-Problem und Glättung (Smoothing)

Die MLE weist jedem **im Training ungesehenen** N-Gramm die Wahrscheinlichkeit
**0** zu — und macht damit den ganzen Satz unmöglich ($P=0$), was fatal ist
(Perplexität $\infty$). Wegen Zipf passiert das *ständig*. **Glättung** verschiebt
Wahrscheinlichkeitsmasse von Gesehenem zu Ungesehenem.

**Laplace / Add-$k$-Glättung.** Addiere eine Pseudozählung $k$ (oft $k=1$) zu jedem
Zähler:
$$
P_{\text{Add-}k}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k\,|V|},
$$
mit Vokabulargröße $|V|$. Einfach, aber grob — es nimmt häufigen N-Grammen zu viel
Masse weg. Nur als Baseline sinnvoll.

**Good-Turing.** Schätzt die Masse für Ungesehenes aus der Zahl der **einmal**
gesehenen N-Gramme (Singletons): Die reskalierten Zählungen sind
$c^\ast = (c+1)\,\dfrac{N_{c+1}}{N_c}$, wobei $N_c$ die Zahl der N-Gramme mit
Häufigkeit $c$ ist. Die Gesamtmasse für ungesehene Ereignisse ist $N_1/N$.

**Backoff und Interpolation.** Wenn ein Trigramm ungesehen ist, nutze das Bigramm;
wenn auch das fehlt, das Unigramm. **Katz-Backoff** *fällt* auf niedrigere Ordnung
zurück (mit Discounting); **Interpolation** *mischt* immer alle Ordnungen:
$$
P_{\text{interp}}(w_i\mid w_{i-2},w_{i-1}) = \lambda_3 P(w_i\mid w_{i-2},w_{i-1})
+ \lambda_2 P(w_i\mid w_{i-1}) + \lambda_1 P(w_i), \quad \textstyle\sum\lambda=1.
$$
Die $\lambda$ lernt man auf einem Held-out-Set.

**Kneser-Ney (der Goldstandard).** Zwei Ideen. (1) **Absolutes Discounting:** ziehe
einen festen Betrag $d$ von jeder Zählung ab und verteile die freigewordene Masse.
(2) Der Clou — das **Continuation-Modell** für die niedrigere Ordnung: Statt „wie
*häufig* ist Wort $w$?" fragt Kneser-Ney „in wie *vielen verschiedenen Kontexten*
kommt $w$ vor?". Klassisches Beispiel: „Francisco" ist häufig, aber fast nur nach
„San" — als Rückfall-Wahrscheinlichkeit sollte es *niedrig* sein. Die
(interpolierte) Kneser-Ney-Formel fürs Bigramm:
$$
P_{\text{KN}}(w_i\mid w_{i-1}) = \frac{\max\big(C(w_{i-1},w_i)-d,\,0\big)}{C(w_{i-1})}
+ \lambda(w_{i-1})\; P_{\text{cont}}(w_i),
$$
$$
P_{\text{cont}}(w_i) = \frac{\big|\{w' : C(w', w_i) > 0\}\big|}{\big|\{(w',w'') : C(w',w'')>0\}\big|},
\qquad
\lambda(w_{i-1}) = \frac{d}{C(w_{i-1})}\,\big|\{w : C(w_{i-1},w)>0\}\big|.
$$
$\lambda(w_{i-1})$ ist der normalisierte Discount (Masse, die durch das Abziehen von
$d$ frei wurde). Kneser-Ney (in der *modifizierten* Variante) war jahrelang das
beste N-Gramm-Verfahren.

### 1.4 Perplexität — wie gut ist ein Sprachmodell?

Die intrinsische Standardmetrik. Die **Perplexität** eines Modells auf einer
Testfolge $w_{1:n}$ ist die inverse geometrische Mittel-Wahrscheinlichkeit pro Wort:
$$
\mathrm{PP}(w_{1:n}) = P(w_{1:n})^{-1/n}
= \Big(\prod_{i=1}^{n} \frac{1}{P(w_i\mid w_{1:i-1})}\Big)^{1/n}
= 2^{\,-\frac{1}{n}\sum_i \log_2 P(w_i\mid \cdot)}.
$$
Interpretation: der **effektive Verzweigungsfaktor** — im Schnitt „so viele
gleichwahrscheinliche Wörter" muss das Modell an jeder Position unterscheiden.
**Niedriger ist besser.** Perplexität ist $2$ hoch der **Kreuzentropie** (in Bit)
— die Brücke zur Informationstheorie. Wichtig: Perplexität ist nur *innerhalb*
desselben Vokabulars vergleichbar, und ein ungesehenes Wort ($P=0$) macht sie
$\infty$ — deshalb *muss* man glätten.

---

## Teil 2 — Aufbau: Klassifikation und Wortrepräsentationen

### 2.1 Textklassifikation mit Naive Bayes

**Aufgabe:** Ordne einem Dokument $d$ eine Klasse $c$ zu (Spam/Ham, Sentiment,
Thema). Der **naive Bayes**-Klassifikator wählt die Klasse mit maximaler
Posterior-Wahrscheinlichkeit (MAP):
$$
\hat c = \arg\max_c P(c\mid d) = \arg\max_c P(c)\,P(d\mid c),
$$
mit der **naiven Unabhängigkeitsannahme**: die Wörter sind gegeben die Klasse
bedingt unabhängig. Im **multinomialen** Modell (Wörter als Zählungen):
$$
\hat c = \arg\max_c \Big[\log P(c) + \sum_{i} \log P(w_i\mid c)\Big],
$$
mit $P(w\mid c) = \dfrac{C(w,c)+\alpha}{\sum_{w'}\big(C(w',c)+\alpha\big)}$
(Laplace-Glättung, $\alpha=1$). Man rechnet im **Log-Raum** gegen Underflow. Trotz
der offensichtlich falschen Unabhängigkeitsannahme ist Naive Bayes erstaunlich
stark, extrem schnell und eine unverzichtbare Baseline. (Modul 01 hatte einen
Spamfilter; hier verstehst du das Modell vollständig.)

### 2.2 Logistische Regression / MaxEnt

Naive Bayes ist ein **generatives** Modell ($P(d\mid c)$); die **logistische
Regression** ist das **diskriminative** Gegenstück ($P(c\mid d)$ direkt). Sie
modelliert
$$
P(c\mid d) = \mathrm{softmax}\big(\mathbf{w}_c^\top \mathbf{f}(d)\big)
= \frac{\exp(\mathbf{w}_c^\top \mathbf{f}(d))}{\sum_{c'}\exp(\mathbf{w}_{c'}^\top \mathbf{f}(d))},
$$
wobei $\mathbf{f}(d)$ ein **Merkmalsvektor** ist (Wortzählungen, TF-IDF, beliebige
Features). In der NLP-Tradition heißt das **Maximum-Entropy-Modell (MaxEnt)**. Die
Gewichte lernt man durch Minimierung der **Kreuzentropie** (mit L2-Regularisierung)
per Gradientenabstieg (Modul 05). Vorteil gegenüber Naive Bayes: LR kann
**korrelierte, überlappende Features** verkraften (z. B. Wort + Bigramm +
Groß-/Kleinschreibung), ohne sie doppelt zu zählen — meist etwas genauer, wenn
genug Daten da sind.

**Evaluation.** Accuracy allein täuscht bei unbalancierten Klassen. Für eine Klasse:
$$
\text{Precision} = \frac{TP}{TP+FP}, \quad
\text{Recall} = \frac{TP}{TP+FN}, \quad
F_1 = \frac{2\cdot P\cdot R}{P+R}
$$
($F_1$ = harmonisches Mittel). Bei mehreren Klassen mittelt man **macro** (jede
Klasse gleich) oder **micro** (jede Instanz gleich). Immer auf einem separaten
**Testset** und idealerweise mit **Kreuzvalidierung**.

### 2.3 Vektorrepräsentationen: von TF-IDF zu Embeddings

**Bag-of-Words & TF-IDF.** Ein Dokument als Vektor über dem Vokabular. Rohe
Zählungen übergewichten häufige Wörter; **TF-IDF** korrigiert das:
$$
\text{tf-idf}(w,d) = \underbrace{\text{tf}(w,d)}_{\text{Häufigkeit in } d}\;\cdot\;
\underbrace{\log\frac{N}{\text{df}(w)}}_{\text{inverse Dokumenthäufigkeit}},
$$
wobei $N$ = Zahl der Dokumente, $\text{df}(w)$ = Zahl der Dokumente mit $w$. Wörter,
die *überall* vorkommen (the, is), bekommen niedriges Gewicht; seltene,
unterscheidende Wörter hohes. Dokumentähnlichkeit misst man mit der
**Kosinus-Ähnlichkeit** $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$.

**Die distributionelle Hypothese.** *„You shall know a word by the company it
keeps"* (Firth): Wörter mit ähnlichen Kontexten haben ähnliche Bedeutung. Das
motiviert **dichte Wortvektoren (Embeddings)**, die semantische Nähe geometrisch
kodieren.

**PPMI.** Ein erster, zählbasierter Schritt: die **positive punktweise
Transinformation** zwischen Wort $w$ und Kontext $c$:
$$
\text{PMI}(w,c) = \log_2 \frac{P(w,c)}{P(w)\,P(c)}, \qquad
\text{PPMI}(w,c) = \max\big(\text{PMI}(w,c),\,0\big).
$$
Eine Wort-Kontext-Matrix aus PPMI-Werten (ggf. per SVD dimensionsreduziert)
liefert bereits brauchbare, dichte Vektoren.

**Word2Vec (Skip-Gram mit Negative Sampling).** Statt zu zählen, wird *gelernt*.
Skip-Gram sagt aus einem Zielwort seine Kontextwörter voraus. Mit **Negative
Sampling** wird das effiziente Trainingsziel (für ein Ziel-Kontext-Paar $(w,c)$
und $k$ zufällige „Negativ"-Wörter):
$$
\log \sigma(\mathbf{v}_c^\top \mathbf{v}_w) + \sum_{j=1}^{k}
\mathbb{E}_{c_j\sim P_n}\big[\log \sigma(-\mathbf{v}_{c_j}^\top \mathbf{v}_w)\big],
$$
mit $\sigma$ = Sigmoid. Intuition: echte Paare sollen hohes Skalarprodukt haben,
zufällige Paare niedriges. Ergebnis sind Vektoren, in denen **semantische
Beziehungen zu Richtungen** werden — die berühmte Analogie
$\text{vec(König)} - \text{vec(Mann)} + \text{vec(Frau)} \approx \text{vec(Königin)}$.

**GloVe** kombiniert beide Welten: Es faktorisiert die **globale**
Ko-Okkurrenz-Matrix und minimiert
$$
J = \sum_{i,j} f(X_{ij})\,\big(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde b_j
- \log X_{ij}\big)^2,
$$
wobei $X_{ij}$ die Ko-Okkurrenz-Zählung und $f$ eine dämpfende Gewichtsfunktion ist.
Diese *statischen* Embeddings (ein Vektor pro Worttyp) sind der Vorläufer der
*kontextuellen* Embeddings (BERT & Co., Modul 09), bei denen jeder Vorkommens-Kontext
einen eigenen Vektor bekommt.

---

## Teil 3 — Advanced: Sequenz-Labeling und Syntax

### 3.1 POS-Tagging als Sequenzproblem

**Part-of-Speech-Tagging** ordnet jedem Wort seine Wortart zu (Noun, Verb, Adj …).
Die Herausforderung ist **Mehrdeutigkeit**: „book" ist Nomen *oder* Verb — der
Kontext entscheidet. Das ist ein **Sequenz-Labeling**-Problem: Finde die beste
Tag-Folge $t_{1:n}$ für die Wortfolge $w_{1:n}$.

**HMM-Tagger.** Direkt das Modell aus Modul 07: Die Tags sind die *versteckten*
Zustände, die Wörter die *Beobachtungen*. Nach Bayes:
$$
\hat t_{1:n} = \arg\max_{t_{1:n}} P(t_{1:n}\mid w_{1:n})
= \arg\max_{t_{1:n}} \prod_i \underbrace{P(w_i\mid t_i)}_{\text{Emission}}\;
\underbrace{P(t_i\mid t_{i-1})}_{\text{Transition}}.
$$
Die **Übergangswahrscheinlichkeiten** $P(t_i\mid t_{i-1})$ und **Emissions­wahrscheinlichkeiten**
$P(w_i\mid t_i)$ schätzt man per MLE aus einem getaggten Korpus (mit Glättung,
besonders für unbekannte Wörter). Die optimale Folge findet der **Viterbi-Algorithmus**
(Modul 07) in $O(n\,|T|^2)$ per dynamischer Programmierung:
$$
v_t(j) = \max_{i}\; v_{t-1}(i)\; P(t_j\mid t_i)\; P(w_t\mid t_j),
$$
mit Rückzeigern zur Pfadrekonstruktion. HMM-Tagger erreichen ~95–96 % Genauigkeit —
eine starke, transparente Baseline.

### 3.2 Die Grenze des HMM: Label Bias, MEMM und CRF

Ein HMM ist **generativ** und kann nur begrenzte Merkmale nutzen (das Wort selbst).
Man möchte aber **reiche Features** (Suffix „-ing", Großschreibung, umgebende
Wörter). Zwei diskriminative Nachfolger:

- **MEMM (Maximum-Entropy-Markov-Modell):** modelliert $P(t_i\mid t_{i-1}, w_i,
  \dots)$ direkt mit einem MaxEnt-Klassifikator pro Position. Problem: das
  **Label-Bias-Problem** — Zustände mit wenigen Folgezuständen „bevorzugen" ihre
  Übergänge, weil pro Schritt lokal normalisiert wird.
- **CRF (Conditional Random Field):** die Lösung. Ein **linear-chain CRF**
  normalisiert **global** über die ganze Sequenz:
  $$
  P(t_{1:n}\mid w_{1:n}) = \frac{1}{Z(w)}\exp\Big(\sum_{i}\sum_{k}\theta_k\,
  f_k(t_{i-1}, t_i, w, i)\Big),
  $$
  mit Merkmalsfunktionen $f_k$, Gewichten $\theta_k$ und der globalen
  Partitionsfunktion $Z(w)$. CRFs vermeiden den Label Bias, erlauben beliebige
  Features und waren vor den neuronalen Modellen der Standard fürs Sequenz-Labeling
  (NER, Chunking). Dekodiert wird wieder mit Viterbi, trainiert per
  Gradientenabstieg über die (konvexe) log-Likelihood.

### 3.3 Syntax: Konstituenten, CFG und der CKY-Parser

**Syntaktisches Parsing** deckt die grammatische Struktur eines Satzes auf. Eine
**kontextfreie Grammatik (CFG)** (aus Modul 06 / Theorie bekannt) besteht aus
Terminalen (Wörter), Nichtterminalen (Phrasen wie NP, VP), einem Startsymbol (S)
und **Produktionsregeln** ($S\to NP\ VP$, $NP\to Det\ N$ …).

**Der CKY-Algorithmus.** Entscheidet und konstruiert Parses für Grammatiken in
**Chomsky-Normalform** (jede Regel $A\to BC$ oder $A\to w$) per **dynamischer
Programmierung** über eine Dreieckstabelle: `chart[i][j]` enthält alle
Nichtterminale, die die Teilspanne $w_{i:j}$ erzeugen können. Man füllt sie
bottom-up nach Spannenlänge; Laufzeit $O(n^3\,|G|)$. CKY ist das syntaktische
Gegenstück zu Viterbi — dynamische Programmierung über Strukturen.

**PCFG.** Eine **probabilistische CFG** gibt jeder Regel eine Wahrscheinlichkeit
($\sum$ über alle Regeln mit gleicher linker Seite $=1$). Der **probabilistische
CKY** findet dann den **wahrscheinlichsten** Parse (nützlich zur Auflösung von
**Ambiguität**, z. B. PP-Attachment: „I saw the man with the telescope"). Die
Regelwahrscheinlichkeiten schätzt man aus einer **Treebank** (z. B. Penn Treebank).

### 3.4 Dependency Parsing — Ausblick

Statt Konstituenten (Phrasen) modelliert **Dependency Parsing** direkte **binäre
Relationen** zwischen Wörtern (Kopf → Dependent, etikettiert mit `nsubj`, `dobj`,
`amod` …). Der resultierende **Dependenzbaum** ist oft praktischer für
Bedeutungsextraktion und der Standard des **Universal-Dependencies**-Projekts (eine
sprachübergreifende Annotationskonvention — relevant für Modul 10). Zwei
Paradigmen: **transition-based** (ein Klassifikator wählt Shift/Reduce-Aktionen,
linear in $n$) und **graph-based** (finde den Maximum Spanning Tree über alle
möglichen Kanten). Neuronale Dependency-Parser (Modul 09) dominieren heute.

---

## Zusammenfassung / Cheat-Sheet

**Sprachmodelle**

| Begriff | Kern |
|---|---|
| N-Gramm | $P(w_i\mid w_{i-N+1:i-1})$; MLE $=\frac{C(w_{i-1},w_i)}{C(w_{i-1})}$ |
| Add-$k$ | $\frac{C+k}{C(w_{i-1})+k|V|}$ |
| Interpolation | $\sum_n \lambda_n P_n$, $\sum\lambda=1$; $\lambda$ auf Held-out |
| Kneser-Ney | absolutes Discounting $d$ + Continuation $P_{\text{cont}}(w)=\frac{|\{w':C(w',w)>0\}|}{\#\text{Bigrammtypen}}$ |
| Perplexität | $PP = P(w_{1:n})^{-1/n} = 2^{H}$; niedriger = besser; $\infty$ bei $P=0$ |

**Klassifikation & Vektoren**

| Begriff | Kern |
|---|---|
| Naive Bayes | $\hat c=\arg\max_c \log P(c)+\sum_i\log P(w_i\mid c)$; generativ |
| Log. Regression | $P(c\mid d)=\mathrm{softmax}(\mathbf w_c^\top\mathbf f)$; diskriminativ, korrelierte Features ok |
| $F_1$ | $\frac{2PR}{P+R}$; macro/micro-Mittelung |
| TF-IDF | $\text{tf}\cdot\log\frac{N}{\text{df}}$; Ähnlichkeit via Kosinus |
| PPMI | $\max(\log_2\frac{P(w,c)}{P(w)P(c)},0)$ |
| Word2Vec (SGNS) | $\log\sigma(\mathbf v_c^\top\mathbf v_w)+\sum_j\log\sigma(-\mathbf v_{c_j}^\top\mathbf v_w)$ |
| GloVe | $\sum f(X_{ij})(\mathbf w_i^\top\tilde{\mathbf w}_j+b_i+\tilde b_j-\log X_{ij})^2$ |

**Sequenz & Syntax**

| Begriff | Kern |
|---|---|
| HMM-Tagger | $\arg\max_t\prod_i P(w_i\mid t_i)P(t_i\mid t_{i-1})$; Viterbi $O(n|T|^2)$ |
| CRF | global normalisiert, $\frac{1}{Z(w)}\exp(\sum\theta_k f_k)$; kein Label Bias |
| CKY | DP-Parser für CNF, $O(n^3|G|)$; PCFG → wahrscheinlichster Parse |
| Dependency | Kopf→Dependent-Relationen; transition-/graph-based |

---

## Selbsttest

<details><summary><b>1. Warum macht die MLE eines N-Gramm-Modells ohne Glättung fast jeden Testsatz unmöglich?</b></summary>

Wegen Zipf enthält jeder reale Testsatz mit hoher Wahrscheinlichkeit ein N-Gramm,
das im Training nie vorkam. Dessen MLE ist $0$, und da $P(w_{1:n})$ ein *Produkt*
ist, wird die ganze Satzwahrscheinlichkeit $0$ (Perplexität $\infty$). Glättung
verschiebt etwas Masse auf ungesehene Ereignisse, sodass nichts exakt $0$ ist.
</details>

<details><summary><b>2. Was ist die zentrale Idee von Kneser-Ney gegenüber Add-$k$?</b></summary>

Zwei Dinge. (1) **Absolutes Discounting:** ein fester Betrag $d$ wird von jeder
Zählung abgezogen (statt proportional wie bei Add-$k$), was empirisch die
Good-Turing-Reskalierung gut trifft. (2) Der eigentliche Clou — die
**Continuation-Wahrscheinlichkeit** für die niedrigere Ordnung: Ein Wort ist als
Rückfall nur dann wahrscheinlich, wenn es in *vielen verschiedenen Kontexten*
auftaucht, nicht bloß oft. „Francisco" ist häufig, aber fast nur nach „San" → als
Unigramm-Rückfall niedrig. Add-$k$ hat keine dieser Ideen und glättet plump.
</details>

<details><summary><b>3. Interpretiere Perplexität. Warum ist niedriger besser?</b></summary>

Perplexität ist $PP=P(w_{1:n})^{-1/n}=2^{H}$ mit $H$ = Kreuzentropie pro Wort. Sie
ist der **effektive Verzweigungsfaktor**: die Zahl gleichwahrscheinlicher
Alternativen, zwischen denen das Modell an jeder Position im Mittel schwankt. Ein
gutes Modell ist von den echten Wörtern *weniger überrascht*, gibt ihnen höhere
Wahrscheinlichkeit → kleineres $H$ → kleinere Perplexität. Ein Modell, das den
Testtext perfekt vorhersagt, hätte $PP=1$.
</details>

<details><summary><b>4. Naive Bayes trifft eine offensichtlich falsche Annahme. Warum funktioniert es trotzdem?</b></summary>

Die bedingte Unabhängigkeit der Wörter gegeben die Klasse ist faktisch falsch
(Wörter korrelieren stark). Aber für die *Klassifikationsentscheidung* zählt nur,
welche Klasse den höchsten Score bekommt — und die Fehler in den geschätzten
Wahrscheinlichkeiten heben sich oft so weit auf, dass die *Rangfolge* der Klassen
korrekt bleibt. Zudem ist NB extrem dateneffizient (wenige Parameter, kein
Overfitting bei kleinen Daten) und schnell. Daher: schwache Wahrscheinlichkeits­schätzung,
aber oft starke Klassifikation.
</details>

<details><summary><b>5. Generativ vs. diskriminativ: Naive Bayes vs. logistische Regression.</b></summary>

**Generativ** (Naive Bayes) modelliert $P(d\mid c)$ und $P(c)$, also die
gemeinsame Verteilung — es „erklärt", wie Daten entstehen. **Diskriminativ**
(logistische Regression) modelliert $P(c\mid d)$ direkt, nur die Entscheidungsgrenze.
Folgen: NB lernt schneller/mit weniger Daten und ist robust, macht aber die
Unabhängigkeitsannahme; LR kann **korrelierte, überlappende Features** ohne
Doppelzählung nutzen und ist bei genug Daten meist genauer. Beide sind lineare
Klassifikatoren im Log-Raum.
</details>

<details><summary><b>6. Warum gewichtet TF-IDF seltene Wörter höher? Wozu der Logarithmus?</b></summary>

Die **inverse Dokumenthäufigkeit** $\log\frac{N}{\text{df}(w)}$ ist groß, wenn $w$
in *wenigen* Dokumenten vorkommt — solche Wörter sind unterscheidungskräftig.
Wörter in *allen* Dokumenten (the, is) haben $\text{df}\approx N$, also $\log 1=0$,
und werden ausgeblendet. Der Logarithmus dämpft: der Sprung von df=1 auf df=2 zählt
viel mehr als von df=1000 auf 1001 — er verhindert, dass extrem seltene Wörter die
Vektoren dominieren.
</details>

<details><summary><b>7. Erkläre die distributionelle Hypothese und wie Word2Vec sie umsetzt.</b></summary>

Distributionelle Hypothese: Wörter, die in ähnlichen Kontexten auftreten, haben
ähnliche Bedeutung („know a word by the company it keeps"). Word2Vec (Skip-Gram)
setzt das um, indem es Vektoren so lernt, dass ein Wort seine tatsächlichen
Kontextwörter gut vorhersagt (hohes Skalarprodukt für echte Paare) und zufällige
Wörter schlecht (Negative Sampling). Wörter mit ähnlichen Kontexten bekommen so
ähnliche Vektoren; semantische Beziehungen werden zu konsistenten Richtungen im
Raum (Analogien).
</details>

<details><summary><b>8. Wie taggt ein HMM Wortarten, und was macht Viterbi dabei?</b></summary>

Das HMM behandelt Tags als versteckte Zustände: Es sucht
$\arg\max_t\prod_i P(w_i\mid t_i)P(t_i\mid t_{i-1})$ aus Emissions- und
Übergangswahrscheinlichkeiten (per MLE aus einer Treebank geschätzt). Diese
Maximierung über alle exponentiell vielen Tag-Folgen löst **Viterbi** exakt per
dynamischer Programmierung in $O(n|T|^2)$: $v_t(j)=\max_i v_{t-1}(i)P(t_j\mid t_i)P(w_t\mid t_j)$,
mit Rückzeigern zur Rekonstruktion der besten Folge.
</details>

<details><summary><b>9. Was ist das Label-Bias-Problem, und wie löst ein CRF es?</b></summary>

Ein MEMM normalisiert die Übergangswahrscheinlichkeiten **pro Position lokal**.
Zustände mit *wenigen* möglichen Folgezuständen verteilen ihre gesamte Masse auf
diese wenigen — unabhängig davon, wie gut die Beobachtung passt. Dadurch werden
solche Übergänge künstlich bevorzugt (*label bias*). Ein **CRF** normalisiert
stattdessen **global** über die gesamte Sequenz (eine Partitionsfunktion $Z(w)$
für den ganzen Satz), sodass lokale Beobachtungen die Gesamtentscheidung korrekt
gewichten — der Bias verschwindet.
</details>

<details><summary><b>10. Warum ist CKY $O(n^3)$, und was leistet die PCFG-Variante?</b></summary>

CKY füllt eine Dreieckstabelle über alle **Spannen** $w_{i:j}$: es gibt $O(n^2)$
Spannen, und für jede probiert es alle $O(n)$ **Teilungspunkte** — zusammen
$O(n^3)$ (mal Grammatikgröße). Für jede Spanne speichert es, welche Nichtterminale
sie erzeugen. Die **probabilistische** CKY-Variante speichert statt „möglich/nicht"
die **Wahrscheinlichkeit** des besten Teilbaums (Produkt der Regel- und
Teilbaum-Wahrscheinlichkeiten) und findet so den *wahrscheinlichsten* Parse — die
Standardmethode zur Auflösung syntaktischer Ambiguität.
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **Jurafsky & Martin, *Speech and Language Processing*, 3. Aufl. (Entwurf)** — *das*
  Standardwerk der NLP. Kapitel zu N-Grammen, Naive Bayes, Logistischer Regression,
  Vektorsemantik, POS-Tagging (HMM), Sequenz-Labeling (CRF), CFG/CKY. **Kostenlos**
  als PDF unter `web.stanford.edu/~jurafsky/slp3`. *Die primäre Quelle — einsteigerfreundlich und vollständig.*
- **Manning & Schütze, *Foundations of Statistical NLP*, MIT Press** — der Klassiker
  der statistischen NLP, tiefer bei Glättung und Grammatiken. *Vertiefend.*
- **Eisenstein, *Introduction to Natural Language Processing*, MIT Press** —
  moderne, mathematisch saubere Darstellung. **Entwurf kostenlos** online. *Vertiefend.*

**Frei verfügbare Kurse & Materialien** (kostenlos)
- **Stanford CS124 / CS224N (Prä-Neural-Teile)** — Videos und Folien zu N-Grammen,
  Naive Bayes, TF-IDF. *Einsteigerfreundlich.*
- **NLTK Book** (`nltk.org/book`) — praktische Einführung mit Python (Tokenisierung,
  Tagging, Parsing). *Einsteigerfreundlich, hands-on.*
- **scikit-learn-Dokumentation zu „Working with Text Data"** — TF-IDF, Naive Bayes,
  Klassifikations-Pipeline. *Direkt praktisch.*
- **Universal Dependencies** (`universaldependencies.org`) — frei verfügbare
  getaggte Korpora in vielen Sprachen (Basis fürs POS-Projekt). *Praktisch.*

**Interaktiv / Visualisierungen** (kostenlos)
- **TensorFlow Embedding Projector** (`projector.tensorflow.org`) — Word2Vec/GloVe-Embeddings
  in 3D erkunden (Nachbarn, Analogien). *Sehr einsteigerfreundlich.*
- **Jurafsky & Martin** enthält exzellente durchgerechnete Beispiele zu Viterbi und CKY.

**Klassische Papers** (kostenlos, vertiefend)
- Chen & Goodman (1999): *An Empirical Study of Smoothing Techniques* — die
  definitive Glättungs-Studie (Kneser-Ney).
- Mikolov et al. (2013): *Efficient Estimation of Word Representations* & *Distributed
  Representations* — Word2Vec.
- Pennington et al. (2014): *GloVe: Global Vectors for Word Representation*.
- Lafferty, McCallum & Pereira (2001): *Conditional Random Fields*.

---

## Die drei Projekte

Die drei Projekte decken die drei Modulteile ab — Sprachmodelle, Klassifikation,
Sequenz-Labeling — mit echten Textdaten und steigender Eigenleistung:

- **01 – basic** (`projects/01-basic/`): **Ein N-Gramm-Sprachmodell.** Geführtes
  Notebook: Tokenisierung, Unigramm/Bigramm/Trigramm mit Add-$k$-Glättung und
  Interpolation, **Perplexität** auf Held-out, und **Textgenerierung** durch
  Sampling. Echter Korpus (Public-Domain-Buch). Viel Anleitung.
- **02 – medium** (`projects/02-medium/`): **Textklassifikation.** Python-Projekt:
  **multinomialer Naive Bayes von Hand** implementieren, mit **TF-IDF +
  logistischer Regression** (scikit-learn) vergleichen, sauber mit Precision/Recall/F1
  evaluieren. Echte Daten (20 Newsgroups). Wenig Anleitung.
- **03 – final** (`projects/03-final/`): **Ein HMM-POS-Tagger mit Viterbi.** Keine
  Code-Vorgabe: Emissions-/Übergangswahrscheinlichkeiten schätzen (mit Glättung für
  unbekannte Wörter), **Viterbi von Grund auf**, auf echten **Universal-Dependencies**-Daten
  evaluieren (~91 % Tag-Genauigkeit auf dem verrauschten EWT-Webtext; unbekannte
  Wörter sind der Flaschenhals — WSJ-artige Korpora erreichen ~95–96 %).
  Master-Niveau, direkte Anwendung von Modul 07.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
