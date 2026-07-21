# Module 09 — Natural Language Processing 2

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

**What is this about?** This module is the jump from statistical NLP (module 08)
to **neural** language processing — and with it to the technology behind modern
translators, chatbots and large language models (LLMs). The thread running
through it: how does a neural network represent and process **sequences**? We go
from **recurrent networks (RNN/LSTM)** via **sequence-to-sequence with
attention** to the **transformer** — the architecture that dominates *everything*
today — and close with **pretraining and transfer** (BERT, GPT) and an outlook on
LLMs.

The transformer is **derived in full** here: self-attention, multi-head,
positional encoding, the entire encoder/decoder block. Once you have worked
through this module you will understand *why* "Attention is all you need" turned
the NLP world upside down.

**Helpful prior knowledge.** Neural networks and backpropagation (module 05 —
mandatory), NLP foundations (module 08: embeddings, language models, perplexity,
tokenization). Linear algebra (matrix multiplication, softmax), PyTorch basics
(from module 05).

**Recommended preceding modules.** Machine Learning 2 (module 05), NLP 1 (module 08).

**Follow-up modules.** Multilingual NLP (module 10), Computer Vision (vision
transformer), and practically every modern AI module falls back on transformers.

---

## Learning objectives

After this module you should be able to

- explain **neural language models** (feedforward NNLM → RNN) and justify their
  advantage over n-grams;
- understand **RNN, LSTM and GRU** including their equations, analyse
  **backpropagation through time (BPTT)** and the **vanishing/exploding gradient
  problem** (connecting back to module 05), and explain how **gating** solves it;
- derive the **sequence-to-sequence architecture** (encoder-decoder), its
  **information bottleneck** and the fix through **attention** (Bahdanau/Luong);
- describe the **transformer completely**: **scaled dot-product attention**,
  **multi-head attention**, **positional encoding**, residual/LayerNorm/FFN,
  masked decoder attention — including the question *why* one scales by $\sqrt{d_k}$;
- explain **subword tokenization** (BPE/WordPiece/SentencePiece);
- distinguish **pretraining & transfer**: **BERT** (masked LM, bidirectional)
  vs. **GPT** (autoregressive), **fine-tuning** vs. **in-context learning**;
- **evaluate** sequence models correctly (perplexity, **BLEU**, **ROUGE**);
- place the broad lines of modern LLMs (scaling laws, prompting, RAG, RLHF).

---

## Part 1 — Basics: from words to recurrent networks

### 1.1 Neural language models

Module 08 showed: n-gram models suffer from **sparsity** (they never see most
contexts) and from **missing generalization** (they do not know that "cat" and
"dog" are similar). **Neural language models** solve both through **dense
embeddings**: similar words have similar vectors, so the model behaves similarly
in similar contexts — *automatic* smoothing in vector space.

The first **feedforward NNLM** (Bengio et al. 2003) predicts $P(w_t\mid w_{t-n+1:t-1})$:
it looks up the embeddings of the $n{-}1$ context words, concatenates them, sends
them through a hidden layer and a softmax over the vocabulary. It still shares the
Markov restriction with n-grams (fixed context window), but generalizes far better
thanks to embeddings. The next step removes the window restriction: **recurrence**.

### 1.2 Recurrent neural networks (RNN)

An **RNN** processes a sequence element by element and carries a **hidden state**
$\mathbf{h}_t$ that summarizes the whole past so far — in principle *unbounded*
context. The recurrence:
$$
\mathbf{h}_t = \tanh\big(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h\big),
\qquad
\mathbf{y}_t = \mathrm{softmax}\big(W_{hy}\mathbf{h}_t + \mathbf{b}_y\big).
$$
Crucially: **the same weights** $W_{hh}, W_{xh}$ are reused at *every* time step
(parameter sharing) — the network can process sequences of arbitrary length. As a
**language model**, $\mathbf{y}_t$ predicts the next word; training uses cross
entropy over all positions.

**Backpropagation through time (BPTT).** One "unrolls the RNN" into a deep network
(one layer per time step with shared weights) and applies ordinary backpropagation
(module 05). The gradient of $W_{hh}$ sums contributions from *all* time steps.

### 1.3 The vanishing/exploding gradient problem

Here the analysis from module 05 hits us with full force. The gradient across many
time steps contains a **product of Jacobian matrices**:
$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}
= \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}}
= \prod_{i=k+1}^{t} \mathrm{diag}\big(\tanh'(\cdot)\big)\, W_{hh}^\top.
$$
Over $t-k$ steps this product becomes either **exponentially small** (*vanishing*,
if the largest singular values are $<1$) or **exponentially large** (*exploding*,
if $>1$). Consequence: the RNN **cannot learn long dependencies** — the signal from
the start of the sentence has "faded away" by the end. *Exploding gradients* are
treated pragmatically with **gradient clipping**; *vanishing gradients* need an
**architectural** solution — gating.

### 1.4 LSTM and GRU — gating against forgetting

The **long short-term memory (LSTM)** introduces a separate **cell state**
$\mathbf{c}_t$ that carries information **additively** across many steps like a
"conveyor belt" (instead of overwriting it at every step). Three **gates**
(sigmoid valves in $[0,1]$) control the flow:
$$
\begin{aligned}
\mathbf{f}_t &= \sigma\big(W_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\big) & \text{(forget gate: what to forget?)}\\
\mathbf{i}_t &= \sigma\big(W_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\big) & \text{(input gate: what to take in?)}\\
\mathbf{o}_t &= \sigma\big(W_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\big) & \text{(output gate: what to emit?)}\\
\tilde{\mathbf{c}}_t &= \tanh\big(W_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\big) & \text{(candidate)}\\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t & \text{(cell update)}\\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) & \text{(hidden state)}
\end{aligned}
$$
**Why this helps against vanishing gradients:** the path $\mathbf{c}_{t-1}\to\mathbf{c}_t$
is essentially **additive** with multiplier $\mathbf{f}_t$. If the forget gate is
close to 1, the gradient flows backwards almost undamped — a "constant error
carousel". The network *learns* when to keep or discard information.

The **gated recurrent unit (GRU)** is a leaner variant with only two gates
(reset $\mathbf{r}_t$, update $\mathbf{z}_t$) and no separate cell state:
$$
\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot
\tanh\big(W[\mathbf{r}_t\odot\mathbf{h}_{t-1}, \mathbf{x}_t]\big).
$$
GRUs have fewer parameters and are often similarly good. **Bidirectional** RNNs
read the sequence forwards *and* backwards and concatenate both states — useful
when the whole sentence is available (e.g. tagging), not for causal generation.

---

## Part 2 — Building up: sequence-to-sequence and attention

### 2.1 Encoder-decoder (seq2seq)

Many tasks map a sequence onto **another** sequence of variable length:
translation, summarization, question→answer. The **seq2seq** architecture
(Sutskever et al. 2014) uses two RNNs:
- The **encoder** reads the input $x_{1:n}$ and compresses it into a **context
  vector** $\mathbf{c}$ (usually the last hidden state).
- The **decoder** is an RNN language model that — conditioned on $\mathbf{c}$ —
  generates the output $y_{1:m}$ word by word (autoregressively: every generated
  word becomes the input for the next step).

Training uses **teacher forcing** (the decoder receives the *true* previous word,
not its own); at inference time one searches for a high-probability output with
**beam search**.

**The bottleneck.** The *entire* source sentence has to pass through **one single**
fixed-size vector $\mathbf{c}$ — for long sentences information is lost and
translation quality collapses with length. That was the central problem attention
solved.

### 2.2 Attention — the breakthrough

**Idea (Bahdanau et al. 2015):** instead of memorizing everything in one vector,
the decoder may **look back at all encoder states** at *every* output step and
select the relevant ones in a weighted fashion. For the decoder state $\mathbf{s}_t$
and the encoder states $\mathbf{h}_1,\dots,\mathbf{h}_n$:
$$
e_{ti} = \text{score}(\mathbf{s}_t, \mathbf{h}_i), \qquad
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j}\exp(e_{tj})}, \qquad
\mathbf{c}_t = \sum_{i} \alpha_{ti}\,\mathbf{h}_i.
$$
The **attention weights** $\alpha_{ti}$ (a softmax over the score values) say *how
strongly* output position $t$ looks at input position $i$; the **context vector**
$\mathbf{c}_t$ is their weighted sum. Two common score functions:
- **Additive (Bahdanau):** $\text{score} = \mathbf{v}^\top\tanh(W_1\mathbf{s}_t + W_2\mathbf{h}_i)$,
- **Multiplicative (Luong):** $\text{score} = \mathbf{s}_t^\top W\,\mathbf{h}_i$ (or simply $\mathbf{s}_t^\top\mathbf{h}_i$).

Attention removed the bottleneck, made long sentences tractable **and** delivered
interpretability (the $\alpha$ show the word alignment). The next, radical step:
**drop the recurrence entirely** and use *only* attention.

---

## Part 3 — Advanced: the transformer

The **transformer** (Vaswani et al. 2017, *"Attention Is All You Need"*) replaces
recurrence completely with **self-attention**. Advantages: **full
parallelizability** (all positions at once, no sequential processing as in the RNN)
and **direct connections** between arbitrarily distant positions (no vanishing over
distance). That is what made training on gigantic amounts of data possible in the
first place — the foundation of all of today's LLMs.

### 3.1 Self-attention (scaled dot-product attention)

The core. From its input vector, every position produces three vectors through
learned linear projections: **query** $\mathbf{q}$, **key** $\mathbf{k}$, **value**
$\mathbf{v}$. Intuition via the database analogy: a position "asks" (query) for
what it is looking for; it compares that with the "keys" of all positions; and it
retrieves a weighted mixture of the "values". Stacked into matrices
$Q, K, V \in \mathbb{R}^{n\times d_k}$:
$$
\boxed{\;\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\;}
$$
Step by step: $QK^\top$ is the $n\times n$ matrix of all **query-key similarities**
(dot products); the softmax (row-wise) turns it into **attention weights**;
multiplying by $V$ mixes the values accordingly. Every position thereby becomes a
weighted summary of *all* positions — context without recurrence.

**Why the scaling by $\sqrt{d_k}$?** If the components of $\mathbf{q}$ and
$\mathbf{k}$ are independent with variance 1, their dot product
$\mathbf{q}^\top\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has **variance $d_k$**. For
large $d_k$ the logits therefore become very large and the softmax enters
saturation regions with **vanishing gradients**. Dividing by $\sqrt{d_k}$
normalizes the variance back to 1 and keeps the softmax in a well-trainable range.

### 3.2 Multi-head attention

A single attention "head" averages information and can emphasize only *one* kind of
relationship. **Multi-head attention** runs $h$ attention operations **in parallel**
in different learned subspaces — different heads can capture different relationships
(e.g. syntactic dependency, coreference, positional proximity):
$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O,
\quad
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V).
$$
Every head projects into dimension $d_k = d_{\text{model}}/h$, so that the total
cost equals that of a single head of full width.

### 3.3 Positional encoding

Self-attention is **permutation-equivariant** — it does not "see" the order of the
words (in contrast to the RNN, which knows it implicitly through the processing
order). One therefore adds **positional information** to the embeddings. The
original paper uses a **sinusoidal** encoding:
$$
\mathrm{PE}(pos, 2i) = \sin\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big),
\qquad
\mathrm{PE}(pos, 2i+1) = \cos\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big).
$$
Every dimension corresponds to a sine wave of a different frequency; the pattern
allows the model to infer **relative positions** (because $\mathrm{PE}(pos{+}k)$ is
a linear function of $\mathrm{PE}(pos)$). Modern models often use **learned** or
**rotary (RoPE)** positional encodings.

### 3.4 The complete transformer block

An **encoder layer** stacks two sublayers, each with a **residual connection**
(module 05) and **layer normalization**:
$$
\begin{aligned}
\mathbf{z} &= \mathrm{LayerNorm}\big(\mathbf{x} + \mathrm{MultiHead}(\mathbf{x},\mathbf{x},\mathbf{x})\big) & \text{(self-attention + res + norm)}\\
\mathbf{y} &= \mathrm{LayerNorm}\big(\mathbf{z} + \mathrm{FFN}(\mathbf{z})\big) & \text{(feedforward + res + norm)}
\end{aligned}
$$
The **position-wise feedforward network** is a two-layer MLP applied identically to
every position: $\mathrm{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$
(inner dimension typically $4\times d_{\text{model}}$). The **residual connections**
secure the gradient flow through many layers, **LayerNorm** stabilizes the
activations. One stacks $N$ (e.g. 6, 12, 96 …) such layers.

The **decoder layer** has three sublayers: (1) **masked** self-attention (every
position may look only at itself and *earlier* positions — otherwise the model
would "peek into the future" during autoregressive generation; the masking sets the
forbidden logits to $-\infty$ before the softmax), (2) **cross-attention** onto the
encoder output (queries from the decoder, keys/values from the encoder — this is the
seq2seq attention from part 2, now in transformer form), (3) the FFN.

**Complexity.** Self-attention costs $O(n^2 \cdot d)$ (the $n\times n$ matrix) —
quadratic in sequence length, but fully parallel. For very long sequences this is
the bottleneck (→ "efficient transformers", an active research field).

### 3.5 Subword tokenization

Transformers work on **subword units**, not on whole words — this solves the
out-of-vocabulary problem and keeps the vocabulary small. **Byte-pair encoding
(BPE)** starts with individual characters and iteratively merges the most frequent
adjacent symbol pair into a new token until a target size is reached. Frequent words
thereby become *one* token, rare ones are decomposed into known pieces
(`unhappiness` → `un` + `happi` + `ness`). **WordPiece** (BERT) and **SentencePiece**
(language-independent, directly on raw text — important for module 10) are variants
of the same idea.

### 3.6 Pretraining and transfer learning

The actual paradigm shift: **pretrain** on huge unlabelled amounts of text
(self-supervised), then **transfer** to concrete tasks. Two families:

- **BERT** (Devlin et al. 2018) — an **encoder-only**, **bidirectional** transformer.
  Pretraining via **masked language modeling (MLM)**: 15 % of the tokens are masked
  and the model predicts them from the *two-sided* context (plus, originally, next
  sentence prediction). BERT delivers strong **contextual embeddings** and is
  adapted via **fine-tuning** (a small head on top, the whole network retrained) for
  classification, NER, QA.
- **GPT** (Radford et al.) — a **decoder-only**, **autoregressive** transformer.
  Pretraining as a classical language model (predict the next token, left context
  only). GPT models **generate** text and, from sufficient size on, show
  **in-context learning**: they solve new tasks purely from a prompt description and
  examples, **without any weight change** — the core of modern "prompting".

The difference between **fine-tuning and in-context learning** is central: the
former changes the weights for *one* task, the latter uses the same frozen model for
*arbitrary* tasks via the prompt.

### 3.7 Evaluating generative models

- **Perplexity** (module 08) measures language model quality intrinsically.
- **BLEU** (translation) measures the **n-gram overlap** with references — modified
  n-gram precision (clipped against repetitions) times a **brevity penalty**
  (a penalty for overly short outputs): $\mathrm{BLEU} = \mathrm{BP}\cdot\exp\big(\sum_{n=1}^{4} w_n \log p_n\big)$.
- **ROUGE** (summarization) measures the **recall** of n-grams / longest common
  subsequences analogously.
- These metrics correlate only to a limited degree with human quality judgements;
  in addition one uses human evaluation and model-based metrics (BERTScore).

### 3.8 Outlook: large language models (LLMs)

Scaling decoder-only transformers to billions of parameters and trillions of tokens
produces **LLMs** with qualitatively new abilities. The keywords:

- **Scaling laws** (Kaplan; Chinchilla): the loss falls predictably as a power law
  in model size, data and compute — performance becomes *plannable*.
- **Emergence & in-context learning**: abilities (arithmetic, reasoning) that small
  models do not have appear beyond a certain size.
- **Instruction tuning & RLHF**: after pretraining, models are aligned with human
  preferences via fine-tuning on instructions and **reinforcement learning from
  human feedback** (connecting to modules 07/13).
- **Retrieval-augmented generation (RAG)**: external knowledge is fetched at runtime
  (embedding search) to improve recency and factual accuracy.

These topics are continued by the practice of AI application development;
theoretically they all stand on the transformer foundation of this module.

---

## Summary / cheat sheet

**Recurrent models**

| Term | Core |
|---|---|
| RNN | $\mathbf h_t=\tanh(W_{hh}\mathbf h_{t-1}+W_{xh}\mathbf x_t+\mathbf b)$; shared weights |
| BPTT | unroll + backprop; gradient = product of Jacobian matrices |
| Vanish./explod. | $\prod \mathrm{diag}(\tanh')W_{hh}^\top$ → exp. small/large; clipping + gating |
| LSTM | $\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t$; additive path rescues the gradient |
| GRU | $\mathbf h_t=(1-\mathbf z_t)\odot\mathbf h_{t-1}+\mathbf z_t\odot\tilde{\mathbf h}_t$ |

**seq2seq & attention**

| Term | Core |
|---|---|
| seq2seq | encoder→context vector→decoder; bottleneck for long sentences |
| Attention | $\alpha_{ti}=\mathrm{softmax}(e_{ti})$, $\mathbf c_t=\sum_i\alpha_{ti}\mathbf h_i$ |
| Score | additive $\mathbf v^\top\tanh(W_1\mathbf s+W_2\mathbf h)$ / multiplicative $\mathbf s^\top W\mathbf h$ |

**Transformer**

| Term | Core |
|---|---|
| Self-attention | $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ |
| $\sqrt{d_k}$ | dot product has variance $d_k$ → normalize, do not saturate the softmax |
| Multi-head | $\mathrm{Concat}(\mathrm{head}_i)W^O$; $d_k=d_{\text{model}}/h$ |
| Positional enc. | $\sin/\cos$ with geometric frequencies; encodes (relative) position |
| Encoder block | LayerNorm(x+MHA) → LayerNorm(z+FFN); FFN=$\max(0,xW_1)W_2$ |
| Decoder block | masked self-att. + cross-att. + FFN |
| Complexity | $O(n^2 d)$, fully parallel |

**Pretraining & evaluation**

| Term | Core |
|---|---|
| BPE/WordPiece | subword tokenization; solves OOV |
| BERT | encoder-only, bidirectional, masked LM; fine-tuning |
| GPT | decoder-only, autoregressive; in-context learning |
| BLEU | $\mathrm{BP}\cdot\exp(\sum w_n\log p_n)$ (precision, translation) |
| ROUGE | recall of n-grams/LCS (summarization) |

---

## Self-test

<details><summary><b>1. Why do neural language models generalize better than n-grams?</b></summary>

N-grams treat words as atomic symbols — "cat" and "dog" are completely
independent, and a context unseen in training has probability 0 (patched only by
smoothing). Neural LMs map words onto **dense embeddings**; similar words lie close
together, so the model reacts similarly in similar contexts — a *learned,
continuous* smoothing. In addition an RNN can in principle use **unbounded**
context instead of a fixed n-gram window.
</details>

<details><summary><b>2. Explain the vanishing gradient problem in the RNN formally. How does the LSTM solve it?</b></summary>

The gradient across time contains $\frac{\partial\mathbf h_t}{\partial\mathbf h_k}=\prod_{i}\mathrm{diag}(\tanh')W_{hh}^\top$
— a product of $t-k$ matrices. If their singular values are $<1$, the product
shrinks exponentially (vanishing); at $>1$ it explodes. Long dependencies thereby
become unlearnable. The **LSTM** introduces a cell state with an **additive** update
$\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t$.
If the forget gate is close to 1, the backward path is
$\partial\mathbf c_t/\partial\mathbf c_{t-1}\approx\mathbf f_t\approx 1$ — the
gradient flows almost undamped ("constant error carousel"). The network learns when
to conserve information.
</details>

<details><summary><b>3. What is the bottleneck in classical seq2seq, and how does attention remove it?</b></summary>

The encoder has to squeeze the *entire* source sentence into **one** fixed-size
vector; for long sentences information is lost and quality collapses with length.
**Attention** lets the decoder look back at *all* encoder states at every step and
mix the relevant ones in a weighted way ($\mathbf c_t=\sum_i\alpha_{ti}\mathbf h_i$).
There is thus no single choke point any more, long sentences work, and the
$\alpha_{ti}$ deliver an interpretable word alignment.
</details>

<details><summary><b>4. Write down scaled dot-product attention and explain every factor.</b></summary>

$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. $QK^\top$ computes,
for every query, the similarity (dot product) with all keys → an $n\times n$ score
matrix. $/\sqrt{d_k}$ normalizes the variance of the dot products (which would
otherwise be $d_k$) so that the softmax does not saturate. The row-wise **softmax**
turns this into attention weights (summing to 1). Multiplying by $V$ yields, for
every position, the weighted sum of the value vectors — its contextualized
representation.
</details>

<details><summary><b>5. Why the scaling by $\sqrt{d_k}$?</b></summary>

For independent components with variance 1 the dot product $\mathbf q^\top\mathbf k=\sum_{i=1}^{d_k}q_ik_i$
has expectation 0 and **variance $d_k$**. For large $d_k$ the logits therefore grow
large in magnitude, the softmax concentrates almost entirely on a single element and
enters a region with **almost zero gradient** — training stalls. Dividing by
$\sqrt{d_k}$ brings the variance back to 1 and keeps the softmax "soft" and
trainable.
</details>

<details><summary><b>6. Why multi-head attention instead of one large head?</b></summary>

A single head computes *one* weighted mixture and thereby averages different
possible relationships together. **Several heads** in separate subspaces can capture
**different** kinds of relationship simultaneously (e.g. one head for syntactic
dependency, one for coreference, one for positional proximity). Since every head
projects into $d_k=d_{\text{model}}/h$, the total cost stays constant — one gains
expressiveness "for free".
</details>

<details><summary><b>7. Why does the transformer need positional encoding but the RNN does not?</b></summary>

Self-attention is **permutation-equivariant**: if one swaps the input positions, only
the outputs get swapped accordingly — the operation itself does not "know" any
order. Without additional information "dog bites man" would equal "man bites dog".
An RNN, by contrast, knows the order *implicitly* through its sequential processing.
The transformer therefore adds **positional encodings** (e.g. sinusoidal) to the
embeddings in order to inject absolute/relative positions.
</details>

<details><summary><b>8. Why is the self-attention in the decoder masked?</b></summary>

The decoder generates **autoregressively** — position $t$ should depend only on what
has already been produced ($1..t$), not on the future. Without a mask, self-attention
would also look at later positions and "see the answer" during training (label
leakage), so that the model fails at inference time. The **causal mask** sets the
logits towards future positions to $-\infty$ *before* the softmax, so that their
weight becomes 0.
</details>

<details><summary><b>9. BERT vs. GPT: architecture, pretraining, typical use.</b></summary>

**BERT** is **encoder-only** and **bidirectional**; pretrained via **masked LM**
(predict masked tokens from two-sided context). Ideal for *understanding* tasks
(classification, NER, QA) via **fine-tuning**. **GPT** is **decoder-only** and
**autoregressive** (left context only); pretrained as a classical language model.
Ideal for *generation* and shows **in-context learning** (tasks via prompt without
weight change). In short: BERT reads bidirectionally and is fine-tuned; GPT
generates causally and is prompted.
</details>

<details><summary><b>10. Why is the $O(n^2)$ complexity of self-attention both a curse and a blessing?</b></summary>

**Blessing:** every position is directly connected to every other one in *one* step
(no vanishing over distance as in the RNN), and all positions are computed **in
parallel** — this enables efficient training on huge data/hardware. **Curse:** the
$n\times n$ attention matrix costs memory and compute **quadratically** in sequence
length $n$; for very long contexts (books, genomes) this becomes the bottleneck —
hence the research on "efficient/linear transformers" (sparse attention, linear
attention, FlashAttention).
</details>

---

## Literature & sources

**Textbooks & lecture notes**
- **Jurafsky & Martin, *Speech and Language Processing*, 3rd ed.** — chapters on
  RNN/LSTM, seq2seq/attention, transformer, contextual embeddings/fine-tuning.
  **Free** at `web.stanford.edu/~jurafsky/slp3`. *Primary source, beginner-friendly.*
- **Goodfellow, Bengio & Courville, *Deep Learning*** — ch. 10 (sequence models,
  RNN/LSTM). **Free** at `deeplearningbook.org`. *In-depth.*
- **Zhang et al., *Dive into Deep Learning* (d2l.ai)** — attention & transformer with
  runnable code. **Free.** *Very practical.*

**Freely available courses** (free)
- **Stanford CS224N "NLP with Deep Learning"** — the authoritative lecture; videos,
  slides, assignments on RNN, attention, transformer, pretraining. *Highly recommended.*
- **Hugging Face Course** (`huggingface.co/learn`) — using and fine-tuning
  transformers in practice. *Beginner-friendly, hands-on.*
- **Andrej Karpathy, "Let's build GPT" / "Zero to Hero"** (YouTube + `nanoGPT`) —
  building a transformer from scratch in PyTorch. *Outstanding for understanding.*

**Interactive / visualizations** (free)
- **"The Illustrated Transformer"** (Jay Alammar, `jalammar.github.io`) — the best
  visual explanation of self-attention and the transformer. *Required reading,
  beginner-friendly.*
- **"The Annotated Transformer"** (Harvard NLP) — the original paper line by line
  with PyTorch code. *In-depth.*
- **Transformer/attention visualizers** (`bertviz`, `poloclub.github.io/transformer-explainer`). *Very illustrative.*

**Key papers** (free, in-depth)
- Hochreiter & Schmidhuber (1997): *Long Short-Term Memory*.
- Bahdanau, Cho & Bengio (2015): *Neural Machine Translation by Jointly Learning to Align and Translate* — attention.
- Vaswani et al. (2017): *Attention Is All You Need* — the transformer.
- Devlin et al. (2018): *BERT*; Radford et al. (2018/19): *GPT / Language Models are Unsupervised Multitask Learners*.
- Brown et al. (2020): *Language Models are Few-Shot Learners* (GPT-3, in-context learning).

---

## The three projects

The three projects lead from the recurrent model via attention to the transformer —
with PyTorch (from module 05) and deliberately **small, CPU/MPS-capable** models:

- **01 – basic** (`projects/01-basic/`): **Sentiment classification with an LSTM.**
  Guided notebook: embeddings + LSTM + classification head in PyTorch, on real
  reviews; training, evaluation, comparison with a bag-of-words baseline. Plenty of
  instruction.
- **02 – medium** (`projects/02-medium/`): **Self-attention by hand + a mini
  transformer encoder.** Python/PyTorch project: implement scaled dot-product and
  multi-head attention yourself, check them against PyTorch's reference, build a
  small transformer encoder block and train it for classification. Little instruction.
- **03 – final** (`projects/03-final/`): **A character-level GPT from scratch.** No
  code given: a decoder-only transformer (token/positional embeddings, masked
  multi-head self-attention, blocks with residual/LayerNorm/FFN), trained
  autoregressively on a text corpus, text generation via sampling. Master's level —
  "build GPT".

Details, setup and reference solutions are in the `README.md` of each project folder.

---
# Modul 09 — Natural Language Processing 2 (deutsche Fassung)

**Worum geht es?** Dieses Modul ist der Sprung von der statistischen NLP (Modul 08)
zur **neuronalen** Sprachverarbeitung — und damit zu der Technologie, die hinter
modernen Übersetzern, Chatbots und großen Sprachmodellen (LLMs) steckt. Der rote
Faden: Wie repräsentiert und verarbeitet ein neuronales Netz **Sequenzen**? Wir
gehen von **rekurrenten Netzen (RNN/LSTM)** über **Sequenz-zu-Sequenz mit
Attention** zum **Transformer** — der Architektur, die heute *alles* dominiert —
und schließen mit **Vortraining und Transfer** (BERT, GPT) und dem Ausblick auf LLMs.

Der Transformer wird hier **vollständig hergeleitet**: Self-Attention, Multi-Head,
Positional Encoding, der ganze Encoder-/Decoder-Block. Wenn du dieses Modul
durchhast, verstehst du, *warum* „Attention is all you need" die NLP-Welt umgekrempelt hat.

**Hilfreiche Vorkenntnisse.** Neuronale Netze und Backpropagation (Modul 05 —
zwingend), NLP-Grundlagen (Modul 08: Embeddings, Sprachmodelle, Perplexität,
Tokenisierung). Lineare Algebra (Matrixmultiplikation, Softmax), PyTorch-Grundlagen
(aus Modul 05).

**Empfohlene Vormodule.** Machine Learning 2 (Modul 05), NLP 1 (Modul 08).

**Folgemodule.** Multilingual NLP (Modul 10), Computer Vision (Vision Transformer),
und praktisch jedes moderne KI-Modul greift auf Transformer zurück.

---

## Lernziele

Nach diesem Modul solltest du in der Lage sein,

- **neuronale Sprachmodelle** zu erklären (Feedforward-NNLM → RNN) und den Vorteil
  gegenüber N-Grammen zu begründen;
- **RNN, LSTM und GRU** samt Gleichungen zu verstehen, **Backpropagation Through
  Time (BPTT)** und das **Vanishing-/Exploding-Gradient-Problem** (Anknüpfung Modul 05)
  zu analysieren und zu erklären, wie **Gating** es löst;
- die **Sequenz-zu-Sequenz-Architektur** (Encoder-Decoder), ihren **Informations­flaschenhals**
  und dessen Lösung durch **Attention** (Bahdanau/Luong) herzuleiten;
- den **Transformer vollständig** zu beschreiben: **skalierte Skalarprodukt-Attention**,
  **Multi-Head-Attention**, **Positional Encoding**, Residual/LayerNorm/FFN, maskierte
  Decoder-Attention — inklusive der Frage *warum* durch $\sqrt{d_k}$ skaliert wird;
- **Subword-Tokenisierung** (BPE/WordPiece/SentencePiece) zu erklären;
- **Vortraining & Transfer** zu unterscheiden: **BERT** (maskiertes LM, bidirektional)
  vs. **GPT** (autoregressiv), **Fine-Tuning** vs. **In-Context-Learning**;
- Sequenzmodelle korrekt zu **evaluieren** (Perplexität, **BLEU**, **ROUGE**);
- die großen Linien moderner LLMs (Scaling Laws, Prompting, RAG, RLHF) einzuordnen.

---

## Teil 1 — Grundlagen: Von Wörtern zu rekurrenten Netzen

### 1.1 Neuronale Sprachmodelle

Modul 08 hat gezeigt: N-Gramm-Modelle leiden unter **Sparsity** (sie sehen die
meisten Kontexte nie) und **fehlender Generalisierung** (sie wissen nicht, dass
„Katze" und „Hund" ähnlich sind). **Neuronale Sprachmodelle** lösen beides über
**dichte Embeddings**: Ähnliche Wörter haben ähnliche Vektoren, also verhält sich
das Modell in ähnlichen Kontexten ähnlich — *automatische* Glättung im Vektorraum.

Das erste **Feedforward-NNLM** (Bengio et al. 2003) sagt $P(w_t\mid w_{t-n+1:t-1})$
voraus: Es schlägt die Embeddings der $n{-}1$ Kontextwörter nach, verkettet sie,
schickt sie durch eine verborgene Schicht und eine Softmax über das Vokabular. Es
teilt sich noch die Markov-Beschränkung mit N-Grammen (fixes Kontextfenster), aber
generalisiert dank Embeddings weit besser. Der nächste Schritt hebt die
Fenster-Beschränkung auf: **Rekurrenz**.

### 1.2 Rekurrente neuronale Netze (RNN)

Ein **RNN** verarbeitet eine Sequenz Element für Element und trägt einen
**versteckten Zustand** $\mathbf{h}_t$ mit, der die gesamte bisherige Vergangenheit
zusammenfasst — im Prinzip *unbegrenzter* Kontext. Die Rekurrenz:
$$
\mathbf{h}_t = \tanh\big(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h\big),
\qquad
\mathbf{y}_t = \mathrm{softmax}\big(W_{hy}\mathbf{h}_t + \mathbf{b}_y\big).
$$
Entscheidend: **dieselben Gewichte** $W_{hh}, W_{xh}$ werden an *jedem* Zeitschritt
wiederverwendet (parameter sharing) — das Netz kann Sequenzen beliebiger Länge
verarbeiten. Als **Sprachmodell** sagt $\mathbf{y}_t$ das nächste Wort voraus;
trainiert wird mit Kreuzentropie über alle Positionen.

**Backpropagation Through Time (BPTT).** Man „rollt das RNN aus" zu einem tiefen
Netz (ein Layer pro Zeitschritt mit geteilten Gewichten) und wendet normale
Backpropagation (Modul 05) an. Der Gradient von $W_{hh}$ summiert Beiträge aus
*allen* Zeitschritten.

### 1.3 Das Vanishing-/Exploding-Gradient-Problem

Hier trifft uns die Analyse aus Modul 05 mit voller Wucht. Der Gradient über viele
Zeitschritte enthält ein **Produkt von Jacobi-Matrizen**:
$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}
= \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}}
= \prod_{i=k+1}^{t} \mathrm{diag}\big(\tanh'(\cdot)\big)\, W_{hh}^\top.
$$
Über $t-k$ Schritte wird dieses Produkt entweder **exponentiell klein**
(*vanishing*, wenn die größten Singulärwerte $<1$) oder **exponentiell groß**
(*exploding*, wenn $>1$). Folge: Das RNN kann **lange Abhängigkeiten nicht lernen**
— das Signal aus dem Satzanfang ist am Ende „verhallt". *Exploding gradients*
behandelt man pragmatisch mit **Gradient Clipping**; *vanishing gradients* brauchen
eine **architektonische** Lösung — das Gating.

### 1.4 LSTM und GRU — Gating gegen das Vergessen

Das **Long Short-Term Memory (LSTM)** führt einen separaten **Zellzustand**
$\mathbf{c}_t$ ein, der wie ein „Fließband" Information über viele Schritte
**additiv** trägt (statt sie bei jedem Schritt zu überschreiben). Drei **Gates**
(Sigmoid-Ventile in $[0,1]$) steuern den Fluss:
$$
\begin{aligned}
\mathbf{f}_t &= \sigma\big(W_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\big) & \text{(Forget-Gate: was vergessen?)}\\
\mathbf{i}_t &= \sigma\big(W_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\big) & \text{(Input-Gate: was aufnehmen?)}\\
\mathbf{o}_t &= \sigma\big(W_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\big) & \text{(Output-Gate: was ausgeben?)}\\
\tilde{\mathbf{c}}_t &= \tanh\big(W_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\big) & \text{(Kandidat)}\\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t & \text{(Zellupdate)}\\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) & \text{(versteckter Zustand)}
\end{aligned}
$$
**Warum das gegen Vanishing Gradients hilft:** Der Pfad $\mathbf{c}_{t-1}\to\mathbf{c}_t$
ist im Kern **additiv** mit Multiplikator $\mathbf{f}_t$. Ist das Forget-Gate nahe 1,
fließt der Gradient nahezu ungedämpft rückwärts — ein „konstanter Fehlerkarussell".
Das Netz *lernt*, wann es Information behalten oder verwerfen soll.

Die **Gated Recurrent Unit (GRU)** ist eine schlankere Variante mit nur zwei Gates
(Reset $\mathbf{r}_t$, Update $\mathbf{z}_t$), ohne separaten Zellzustand:
$$
\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot
\tanh\big(W[\mathbf{r}_t\odot\mathbf{h}_{t-1}, \mathbf{x}_t]\big).
$$
GRUs haben weniger Parameter, sind oft ähnlich gut. **Bidirektionale** RNNs lesen
die Sequenz vorwärts *und* rückwärts und verketten beide Zustände — nützlich, wenn
der ganze Satz vorliegt (z. B. Tagging), nicht bei kausaler Generierung.

---

## Teil 2 — Aufbau: Sequenz-zu-Sequenz und Attention

### 2.1 Encoder-Decoder (seq2seq)

Viele Aufgaben bilden eine Sequenz auf eine **andere** Sequenz variabler Länge ab:
Übersetzung, Zusammenfassung, Frage→Antwort. Die **seq2seq**-Architektur
(Sutskever et al. 2014) nutzt zwei RNNs:
- Der **Encoder** liest die Eingabe $x_{1:n}$ und komprimiert sie in einen
  **Kontextvektor** $\mathbf{c}$ (meist den letzten versteckten Zustand).
- Der **Decoder** ist ein RNN-Sprachmodell, das — auf $\mathbf{c}$ konditioniert —
  die Ausgabe $y_{1:m}$ Wort für Wort generiert (autoregressiv: jedes erzeugte Wort
  wird Eingabe für den nächsten Schritt).

Trainiert wird mit **Teacher Forcing** (der Decoder bekommt das *echte* vorige Wort,
nicht sein eigenes); bei der Inferenz sucht man mit **Beam Search** eine
hochwahrscheinliche Ausgabe.

**Der Flaschenhals.** Der *gesamte* Quellsatz muss durch **einen einzigen** Vektor
$\mathbf{c}$ fester Größe — bei langen Sätzen geht Information verloren, die
Übersetzungsqualität bricht mit der Länge ein. Das war das zentrale Problem, das
Attention löste.

### 2.2 Attention — der Durchbruch

**Idee (Bahdanau et al. 2015):** Statt sich alles in einen Vektor zu merken, darf
der Decoder bei *jedem* Ausgabeschritt **auf alle Encoder-Zustände zurückblicken**
und die relevanten gewichtet auswählen. Für den Decoder-Zustand $\mathbf{s}_t$ und
die Encoder-Zustände $\mathbf{h}_1,\dots,\mathbf{h}_n$:
$$
e_{ti} = \text{score}(\mathbf{s}_t, \mathbf{h}_i), \qquad
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j}\exp(e_{tj})}, \qquad
\mathbf{c}_t = \sum_{i} \alpha_{ti}\,\mathbf{h}_i.
$$
Die **Attention-Gewichte** $\alpha_{ti}$ (eine Softmax über die Score-Werte) sagen,
*wie stark* Ausgabeposition $t$ auf Eingabeposition $i$ schaut; der **Kontextvektor**
$\mathbf{c}_t$ ist ihre gewichtete Summe. Zwei gängige Score-Funktionen:
- **Additiv (Bahdanau):** $\text{score} = \mathbf{v}^\top\tanh(W_1\mathbf{s}_t + W_2\mathbf{h}_i)$,
- **Multiplikativ (Luong):** $\text{score} = \mathbf{s}_t^\top W\,\mathbf{h}_i$ (oder einfach $\mathbf{s}_t^\top\mathbf{h}_i$).

Attention beseitigte den Flaschenhals, machte lange Sätze handhabbar **und** lieferte
Interpretierbarkeit (die $\alpha$ zeigen die Wort-Ausrichtung). Der nächste,
radikale Schritt: **die Rekurrenz ganz weglassen** und *nur* Attention nutzen.

---

## Teil 3 — Advanced: Der Transformer

Der **Transformer** (Vaswani et al. 2017, *„Attention Is All You Need"*) ersetzt
Rekurrenz vollständig durch **Self-Attention**. Vorteile: **volle
Parallelisierbarkeit** (alle Positionen gleichzeitig, kein sequenzielles Abarbeiten
wie beim RNN) und **direkte Verbindungen** zwischen beliebig weit entfernten
Positionen (kein Vanishing über die Distanz). Das machte das Training auf
gigantischen Datenmengen erst möglich — die Grundlage aller heutigen LLMs.

### 3.1 Self-Attention (Scaled Dot-Product Attention)

Der Kern. Jede Position erzeugt aus ihrem Eingabevektor drei Vektoren durch gelernte
lineare Projektionen: **Query** $\mathbf{q}$, **Key** $\mathbf{k}$, **Value**
$\mathbf{v}$. Intuition der Datenbank-Analogie: Eine Position „fragt" (Query), was
sie sucht; sie vergleicht das mit den „Schlüsseln" (Keys) aller Positionen; und holt
sich eine gewichtete Mischung der „Werte" (Values). Gestapelt zu Matrizen
$Q, K, V \in \mathbb{R}^{n\times d_k}$:
$$
\boxed{\;\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\;}
$$
Schritt für Schritt: $QK^\top$ ist die $n\times n$-Matrix aller
**Query-Key-Ähnlichkeiten** (Skalarprodukte); die Softmax (zeilenweise) macht daraus
**Attention-Gewichte**; die Multiplikation mit $V$ mischt die Werte entsprechend.
Jede Position wird so zu einer gewichteten Zusammenfassung *aller* Positionen —
Kontext ohne Rekurrenz.

**Warum die Skalierung durch $\sqrt{d_k}$?** Sind die Komponenten von $\mathbf{q}$
und $\mathbf{k}$ unabhängig mit Varianz 1, so hat ihr Skalarprodukt $\mathbf{q}^\top\mathbf{k}
= \sum_{i=1}^{d_k} q_i k_i$ **Varianz $d_k$**. Für großes $d_k$ werden die Logits
also sehr groß, die Softmax gerät in Sättigungsbereiche mit **verschwindenden
Gradienten**. Teilen durch $\sqrt{d_k}$ normiert die Varianz zurück auf 1 und hält
die Softmax im gut-trainierbaren Bereich.

### 3.2 Multi-Head-Attention

Ein einzelner Attention-„Kopf" mittelt Information und kann nur *eine* Art von
Beziehung betonen. **Multi-Head-Attention** führt $h$ Attention-Operationen
**parallel** in verschiedenen gelernten Unterräumen aus — verschiedene Köpfe können
verschiedene Beziehungen einfangen (z. B. syntaktische Abhängigkeit, Koreferenz,
Positionsnähe):
$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O,
\quad
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V).
$$
Jeder Kopf projiziert in Dimension $d_k = d_{\text{model}}/h$, sodass der
Gesamtaufwand dem eines einzelnen Kopfes voller Breite entspricht.

### 3.3 Positional Encoding

Self-Attention ist **permutations-äquivariant** — sie „sieht" die Reihenfolge der
Wörter nicht (im Gegensatz zum RNN, das sie implizit durch die Verarbeitungsfolge
kennt). Deshalb addiert man **Positionsinformation** zu den Embeddings. Die
Original-Arbeit nutzt **sinusförmige** Kodierung:
$$
\mathrm{PE}(pos, 2i) = \sin\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big),
\qquad
\mathrm{PE}(pos, 2i+1) = \cos\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big).
$$
Jede Dimension entspricht einer Sinuswelle anderer Frequenz; das Muster erlaubt dem
Modell, **relative Positionen** zu erschließen (weil $\mathrm{PE}(pos{+}k)$ eine
lineare Funktion von $\mathrm{PE}(pos)$ ist). Moderne Modelle nutzen oft **gelernte**
oder **rotatorische (RoPE)** Positionskodierungen.

### 3.4 Der vollständige Transformer-Block

Ein **Encoder-Layer** stapelt zwei Sublayer, jeder mit **Residualverbindung** (Modul 05)
und **Layer-Normalisierung**:
$$
\begin{aligned}
\mathbf{z} &= \mathrm{LayerNorm}\big(\mathbf{x} + \mathrm{MultiHead}(\mathbf{x},\mathbf{x},\mathbf{x})\big) & \text{(Self-Attention + Res + Norm)}\\
\mathbf{y} &= \mathrm{LayerNorm}\big(\mathbf{z} + \mathrm{FFN}(\mathbf{z})\big) & \text{(Feedforward + Res + Norm)}
\end{aligned}
$$
Das **positionsweise Feedforward-Netz** ist ein Zwei-Schicht-MLP, das auf jede
Position gleich angewandt wird: $\mathrm{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$
(innere Dimension typisch $4\times d_{\text{model}}$). Die **Residualverbindungen**
sichern den Gradientenfluss durch viele Layer, **LayerNorm** stabilisiert die
Aktivierungen. Man stapelt $N$ (z. B. 6, 12, 96 …) solcher Layer.

Der **Decoder-Layer** hat drei Sublayer: (1) **maskierte** Self-Attention (jede
Position darf nur auf sich und *frühere* Positionen schauen — sonst würde das Modell
beim autoregressiven Generieren „in die Zukunft spicken"; die Maskierung setzt die
verbotenen Logits auf $-\infty$ vor der Softmax), (2) **Cross-Attention** auf den
Encoder-Ausgang (Queries vom Decoder, Keys/Values vom Encoder — das ist die
seq2seq-Attention aus Teil 2, jetzt in Transformer-Form), (3) das FFN.

**Komplexität.** Self-Attention kostet $O(n^2 \cdot d)$ (die $n\times n$-Matrix) —
quadratisch in der Sequenzlänge, aber vollständig parallel. Für sehr lange Sequenzen
ist das der Engpass (→ „efficient Transformers", ein aktives Forschungsfeld).

### 3.5 Subword-Tokenisierung

Transformer arbeiten auf **Subword-Einheiten**, nicht auf ganzen Wörtern — das löst
das Out-of-Vocabulary-Problem und hält das Vokabular klein. **Byte-Pair Encoding
(BPE)** startet mit einzelnen Zeichen und verschmilzt iterativ das häufigste
benachbarte Symbolpaar zu einem neuen Token, bis eine Zielgröße erreicht ist. So
werden häufige Wörter *ein* Token, seltene in bekannte Teilstücke zerlegt
(`unhappiness` → `un` + `happi` + `ness`). **WordPiece** (BERT) und **SentencePiece**
(sprach­unabhängig, direkt auf Rohtext — wichtig für Modul 10) sind Varianten
derselben Idee.

### 3.6 Vortraining und Transfer Learning

Der eigentliche Paradigmenwechsel: **Vortrainieren** auf riesigen unmarkierten
Textmengen (selbstüberwacht), dann **Übertragen** auf konkrete Aufgaben. Zwei
Familien:

- **BERT** (Devlin et al. 2018) — ein **Encoder-only**, **bidirektionaler** Transformer.
  Vortraining per **Masked Language Modeling (MLM)**: 15 % der Tokens werden maskiert,
  das Modell sagt sie aus dem *beidseitigen* Kontext voraus (plus ursprünglich Next
  Sentence Prediction). BERT liefert starke **kontextuelle Embeddings** und wird per
  **Fine-Tuning** (ein kleiner Kopf obendrauf, ganzes Netz nachtrainiert) für
  Klassifikation, NER, QA angepasst.
- **GPT** (Radford et al.) — ein **Decoder-only**, **autoregressiver** Transformer.
  Vortraining als klassisches Sprachmodell (nächstes Token vorhersagen, nur
  Linkskontext). GPT-Modelle **generieren** Text und zeigen ab genügender Größe
  **In-Context-Learning**: Sie lösen neue Aufgaben allein aus einer Prompt-Beschreibung
  und Beispielen, **ohne Gewichtsänderung** — der Kern des modernen „Prompting".

Der Unterschied **Fine-Tuning vs. In-Context-Learning** ist zentral: Ersteres ändert
die Gewichte für *eine* Aufgabe, Letzteres nutzt dasselbe eingefrorene Modell für
*beliebige* Aufgaben per Prompt.

### 3.7 Evaluation generativer Modelle

- **Perplexität** (Modul 08) misst intrinsisch die Sprachmodell-Güte.
- **BLEU** (Übersetzung) misst die **n-Gramm-Überlappung** mit Referenzen — modifizierte
  n-Gramm-Präzision (gegen Wiederholungen geklippt) mal einer **Brevity Penalty**
  (Strafe für zu kurze Ausgaben): $\mathrm{BLEU} = \mathrm{BP}\cdot\exp\big(\sum_{n=1}^{4} w_n \log p_n\big)$.
- **ROUGE** (Zusammenfassung) misst analog **Recall** der n-Gramme/längsten
  gemeinsamen Teilfolgen.
- Diese Metriken korrelieren nur begrenzt mit menschlicher Qualität; ergänzend nutzt
  man menschliche Bewertung und modellbasierte Metriken (BERTScore).

### 3.8 Ausblick: Große Sprachmodelle (LLMs)

Skaliert man Decoder-only-Transformer auf Milliarden Parameter und Billionen Tokens,
entstehen **LLMs** mit qualitativ neuen Fähigkeiten. Die Stichworte:

- **Scaling Laws** (Kaplan; Chinchilla): Verlust fällt vorhersagbar als Potenzgesetz
  in Modellgröße, Daten und Rechenzeit — Leistung wird *planbar*.
- **Emergenz & In-Context-Learning**: Fähigkeiten (Arithmetik, Reasoning), die kleine
  Modelle nicht haben, tauchen ab einer Größe auf.
- **Instruction Tuning & RLHF**: Nach dem Vortraining richtet man Modelle per
  Feinabstimmung auf Anweisungen und **Reinforcement Learning from Human Feedback**
  (Anknüpfung Modul 07/13) an menschlichen Präferenzen aus.
- **Retrieval-Augmented Generation (RAG)**: externes Wissen wird zur Laufzeit
  hinzugeholt (Embedding-Suche), um Aktualität und Faktentreue zu verbessern.

Diese Themen führt die Praxis der KI-Anwendungsentwicklung fort; theoretisch stehen
sie alle auf dem Transformer-Fundament dieses Moduls.

---

## Zusammenfassung / Cheat-Sheet

**Rekurrente Modelle**

| Begriff | Kern |
|---|---|
| RNN | $\mathbf h_t=\tanh(W_{hh}\mathbf h_{t-1}+W_{xh}\mathbf x_t+\mathbf b)$; geteilte Gewichte |
| BPTT | ausrollen + Backprop; Gradient = Produkt von Jacobi-Matrizen |
| Vanish./Explod. | $\prod \mathrm{diag}(\tanh')W_{hh}^\top$ → exp. klein/groß; Clipping + Gating |
| LSTM | $\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t$; additiver Pfad rettet Gradient |
| GRU | $\mathbf h_t=(1-\mathbf z_t)\odot\mathbf h_{t-1}+\mathbf z_t\odot\tilde{\mathbf h}_t$ |

**seq2seq & Attention**

| Begriff | Kern |
|---|---|
| seq2seq | Encoder→Kontextvektor→Decoder; Flaschenhals bei langen Sätzen |
| Attention | $\alpha_{ti}=\mathrm{softmax}(e_{ti})$, $\mathbf c_t=\sum_i\alpha_{ti}\mathbf h_i$ |
| Score | additiv $\mathbf v^\top\tanh(W_1\mathbf s+W_2\mathbf h)$ / multiplikativ $\mathbf s^\top W\mathbf h$ |

**Transformer**

| Begriff | Kern |
|---|---|
| Self-Attention | $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ |
| $\sqrt{d_k}$ | Skalarprodukt hat Varianz $d_k$ → normieren, Softmax nicht sättigen |
| Multi-Head | $\mathrm{Concat}(\mathrm{head}_i)W^O$; $d_k=d_{\text{model}}/h$ |
| Positional Enc. | $\sin/\cos$ mit geom. Frequenzen; kodiert (relative) Position |
| Encoder-Block | LayerNorm(x+MHA) → LayerNorm(z+FFN); FFN=$\max(0,xW_1)W_2$ |
| Decoder-Block | maskierte Self-Att. + Cross-Att. + FFN |
| Komplexität | $O(n^2 d)$, voll parallel |

**Vortraining & Evaluation**

| Begriff | Kern |
|---|---|
| BPE/WordPiece | Subword-Tokenisierung; löst OOV |
| BERT | Encoder-only, bidirektional, Masked LM; Fine-Tuning |
| GPT | Decoder-only, autoregressiv; In-Context-Learning |
| BLEU | $\mathrm{BP}\cdot\exp(\sum w_n\log p_n)$ (Präzision, Übersetzung) |
| ROUGE | Recall der n-Gramme/LCS (Zusammenfassung) |

---

## Selbsttest

<details><summary><b>1. Warum generalisieren neuronale Sprachmodelle besser als N-Gramme?</b></summary>

N-Gramme behandeln Wörter als atomare Symbole — „Katze" und „Hund" sind völlig
unabhängig, und ein im Training ungesehener Kontext hat Wahrscheinlichkeit 0 (nur
durch Glättung geflickt). Neuronale LMs bilden Wörter auf **dichte Embeddings** ab;
ähnliche Wörter liegen nah beieinander, sodass das Modell in ähnlichen Kontexten
ähnlich reagiert — eine *gelernte, kontinuierliche* Glättung. Zudem kann ein RNN
prinzipiell **unbegrenzten** Kontext nutzen statt eines festen N-Gramm-Fensters.
</details>

<details><summary><b>2. Erkläre das Vanishing-Gradient-Problem im RNN formal. Wie löst das LSTM es?</b></summary>

Der Gradient über die Zeit enthält $\frac{\partial\mathbf h_t}{\partial\mathbf h_k}=\prod_{i}\mathrm{diag}(\tanh')W_{hh}^\top$
— ein Produkt von $t-k$ Matrizen. Sind deren Singulärwerte $<1$, schrumpft das
Produkt exponentiell (vanishing), bei $>1$ explodiert es. Lange Abhängigkeiten
werden dadurch unlernbar. Das **LSTM** führt einen Zellzustand mit **additivem**
Update $\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t$
ein. Steht das Forget-Gate nahe 1, ist der Rückwärts-Pfad $\partial\mathbf c_t/\partial\mathbf c_{t-1}\approx\mathbf f_t\approx 1$
— der Gradient fließt nahezu ungedämpft („constant error carousel"). Das Netz lernt,
wann es Information konserviert.
</details>

<details><summary><b>3. Was ist der Flaschenhals in klassischem seq2seq, und wie behebt Attention ihn?</b></summary>

Der Encoder muss den *ganzen* Quellsatz in **einen** Vektor fester Größe pressen;
bei langen Sätzen geht Information verloren, die Qualität bricht mit der Länge ein.
**Attention** lässt den Decoder bei jedem Schritt auf *alle* Encoder-Zustände
zurückblicken und die relevanten gewichtet mischen ($\mathbf c_t=\sum_i\alpha_{ti}\mathbf h_i$).
So gibt es keinen einzelnen Engpass mehr, lange Sätze funktionieren, und die
$\alpha_{ti}$ liefern eine interpretierbare Wort-Ausrichtung.
</details>

<details><summary><b>4. Schreibe die Scaled-Dot-Product-Attention hin und erkläre jeden Faktor.</b></summary>

$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. $QK^\top$ berechnet
für jede Query die Ähnlichkeit (Skalarprodukt) mit allen Keys → $n\times n$-Score-Matrix.
$/\sqrt{d_k}$ normiert die Varianz der Skalarprodukte (die sonst $d_k$ ist), damit die
Softmax nicht sättigt. Die zeilenweise **Softmax** macht daraus Attention-Gewichte
(Summe 1). Die Multiplikation mit $V$ liefert für jede Position die gewichtete Summe
der Value-Vektoren — ihre kontextualisierte Repräsentation.
</details>

<details><summary><b>5. Warum die Skalierung mit $\sqrt{d_k}$?</b></summary>

Bei unabhängigen Komponenten mit Varianz 1 hat das Skalarprodukt $\mathbf q^\top\mathbf k=\sum_{i=1}^{d_k}q_ik_i$
den Erwartungswert 0 und **Varianz $d_k$**. Für großes $d_k$ werden die Logits also
betragsmäßig groß, die Softmax konzentriert sich fast vollständig auf ein Element und
gerät in einen Bereich mit **fast null Gradient** — Training stockt. Teilen durch
$\sqrt{d_k}$ bringt die Varianz zurück auf 1 und hält die Softmax „weich" und trainierbar.
</details>

<details><summary><b>6. Wozu Multi-Head-Attention statt eines großen Kopfes?</b></summary>

Ein einzelner Kopf berechnet *eine* gewichtete Mischung und mittelt damit
verschiedene mögliche Beziehungen zusammen. **Mehrere Köpfe** in getrennten
Unterräumen können **verschiedene** Beziehungsarten gleichzeitig einfangen (z. B. ein
Kopf für syntaktische Abhängigkeit, einer für Koreferenz, einer für Positionsnähe).
Da jeder Kopf in $d_k=d_{\text{model}}/h$ projiziert, bleibt der Gesamtaufwand
konstant — man gewinnt Ausdrucksstärke „gratis".
</details>

<details><summary><b>7. Warum braucht der Transformer Positional Encoding, das RNN aber nicht?</b></summary>

Self-Attention ist **permutations-äquivariant**: Vertauscht man die Eingabepositionen,
vertauschen sich nur die Ausgaben entsprechend — die Operation selbst „kennt" keine
Reihenfolge. Ohne zusätzliche Information wäre „Hund beißt Mann" = „Mann beißt Hund".
Ein RNN kennt die Reihenfolge dagegen *implizit* durch seine sequenzielle
Verarbeitung. Der Transformer addiert deshalb **Positional Encodings** (z. B.
sinusförmig) zu den Embeddings, um absolute/relative Positionen einzuspeisen.
</details>

<details><summary><b>8. Warum ist die Self-Attention im Decoder maskiert?</b></summary>

Der Decoder generiert **autoregressiv** — Position $t$ soll nur vom bereits
Erzeugten ($1..t$) abhängen, nicht von der Zukunft. Ohne Maske würde die
Self-Attention auch auf spätere Positionen schauen und beim Training „die Antwort
sehen" (Label-Leakage), sodass das Modell zur Inferenzzeit versagt. Die **kausale
Maske** setzt die Logits zu zukünftigen Positionen auf $-\infty$ *vor* der Softmax,
sodass deren Gewicht 0 wird.
</details>

<details><summary><b>9. BERT vs. GPT: Architektur, Vortraining, typische Nutzung.</b></summary>

**BERT** ist **Encoder-only** und **bidirektional**; vortrainiert per **Masked LM**
(maskierte Tokens aus beidseitigem Kontext vorhersagen). Ideal für *Verstehens*-Aufgaben
(Klassifikation, NER, QA) via **Fine-Tuning**. **GPT** ist **Decoder-only** und
**autoregressiv** (nur Linkskontext); vortrainiert als klassisches Sprachmodell.
Ideal für *Generierung* und zeigt **In-Context-Learning** (Aufgaben per Prompt ohne
Gewichtsänderung). Kurz: BERT liest bidirektional und wird feinabgestimmt; GPT
generiert kausal und wird geprompt.
</details>

<details><summary><b>10. Warum ist die $O(n^2)$-Komplexität der Self-Attention Fluch und Segen?</b></summary>

**Segen:** Jede Position ist mit jeder anderen in *einem* Schritt direkt verbunden
(kein Vanishing über Distanz wie im RNN), und alle Positionen werden **parallel**
berechnet — das ermöglicht effizientes Training auf riesigen Daten/Hardware. **Fluch:**
Die $n\times n$-Attention-Matrix kostet Speicher und Rechenzeit **quadratisch** in der
Sequenzlänge $n$; für sehr lange Kontexte (Bücher, Genome) wird das der Engpass —
daher die Forschung an „efficient/linear Transformers" (Sparse-, Linear-Attention, FlashAttention).
</details>

---

## Literatur & Quellen

**Lehrbücher & Skripte**
- **Jurafsky & Martin, *Speech and Language Processing*, 3. Aufl.** — Kapitel zu RNN/LSTM,
  seq2seq/Attention, Transformer, kontextuellen Embeddings/Fine-Tuning. **Kostenlos**
  unter `web.stanford.edu/~jurafsky/slp3`. *Primäre Quelle, einsteigerfreundlich.*
- **Goodfellow, Bengio & Courville, *Deep Learning*** — Kap. 10 (Sequenzmodelle, RNN/LSTM).
  **Kostenlos** unter `deeplearningbook.org`. *Vertiefend.*
- **Zhang et al., *Dive into Deep Learning* (d2l.ai)** — Attention & Transformer mit
  lauffähigem Code. **Kostenlos.** *Sehr praktisch.*

**Frei verfügbare Kurse** (kostenlos)
- **Stanford CS224N „NLP with Deep Learning"** — die maßgebliche Vorlesung; Videos,
  Folien, Aufgaben zu RNN, Attention, Transformer, Pretraining. *Sehr empfohlen.*
- **Hugging Face Course** (`huggingface.co/learn`) — Transformer praktisch nutzen und
  fine-tunen. *Einsteigerfreundlich, hands-on.*
- **Andrej Karpathy, „Let's build GPT" / „Zero to Hero"** (YouTube + `nanoGPT`) — einen
  Transformer von Grund auf in PyTorch bauen. *Herausragend zum Verstehen.*

**Interaktiv / Visualisierungen** (kostenlos)
- **„The Illustrated Transformer"** (Jay Alammar, `jalammar.github.io`) — die beste
  visuelle Erklärung von Self-Attention und dem Transformer. *Pflichtlektüre, einsteigerfreundlich.*
- **„The Annotated Transformer"** (Harvard NLP) — das Original-Paper Zeile für Zeile
  mit PyTorch-Code. *Vertiefend.*
- **Transformer-/Attention-Visualizer** (`bertviz`, `poloclub.github.io/transformer-explainer`). *Sehr anschaulich.*

**Schlüssel-Papers** (kostenlos, vertiefend)
- Hochreiter & Schmidhuber (1997): *Long Short-Term Memory*.
- Bahdanau, Cho & Bengio (2015): *Neural Machine Translation by Jointly Learning to Align and Translate* — Attention.
- Vaswani et al. (2017): *Attention Is All You Need* — der Transformer.
- Devlin et al. (2018): *BERT*; Radford et al. (2018/19): *GPT / Language Models are Unsupervised Multitask Learners*.
- Brown et al. (2020): *Language Models are Few-Shot Learners* (GPT-3, In-Context-Learning).

---

## Die drei Projekte

Die drei Projekte führen vom rekurrenten Modell über Attention zum Transformer — mit
PyTorch (aus Modul 05) und bewusst **kleinen, CPU-/MPS-tauglichen** Modellen:

- **01 – basic** (`projects/01-basic/`): **Sentiment-Klassifikation mit einem LSTM.**
  Geführtes Notebook: Embeddings + LSTM + Klassifikationskopf in PyTorch, auf echten
  Filmkritiken; Training, Evaluation, Vergleich mit einer Bag-of-Words-Baseline. Viel Anleitung.
- **02 – medium** (`projects/02-medium/`): **Self-Attention von Hand + ein
  Mini-Transformer-Encoder.** Python/PyTorch-Projekt: Scaled-Dot-Product- und
  Multi-Head-Attention selbst implementieren, gegen PyTorchs Referenz prüfen, einen
  kleinen Transformer-Encoder-Block bauen und für Klassifikation trainieren. Wenig Anleitung.
- **03 – final** (`projects/03-final/`): **Ein Zeichen-Level-GPT von Grund auf.** Keine
  Code-Vorgabe: ein Decoder-only-Transformer (Token/Positional Embeddings, maskierte
  Multi-Head-Self-Attention, Blöcke mit Residual/LayerNorm/FFN), autoregressiv auf einem
  Textkorpus trainiert, Textgenerierung per Sampling. Master-Niveau — „build GPT".

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
