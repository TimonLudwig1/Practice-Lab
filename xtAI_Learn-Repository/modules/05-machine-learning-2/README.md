# Module 05 — Machine Learning 2

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

**What this is about:** This module leads from the classical ML methods of module 04 to the two big topic blocks of modern machine learning: **deep neural networks** (multilayer perceptrons, backpropagation, optimization, regularization, CNNs) and **unsupervised learning** (clustering, mixture models with EM, dimensionality reduction). All central methods are derived in full — not merely used.

**Helpful prior knowledge:** Linear algebra (matrices, eigenvalues), calculus (partial derivatives, chain rule), probability (densities, expectation, maximum likelihood). Modules 02–04 should have been worked through; in particular, much of this builds on logistic regression and on the bias-variance understanding from module 04.

---

## Learning objectives

After this module you can:

- define a multilayer perceptron (MLP) formally and **derive backpropagation completely** (including matrix form and complexity analysis) — and implement it by hand.
- derive loss functions from the **maximum likelihood principle** (MSE corresponds to Gaussian noise, cross-entropy to a categorical distribution).
- explain the most important optimizers (SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam/AdamW) with their exact update equations, including the bias correction in Adam.
- derive and justify initialization (Xavier/He) via variance propagation.
- describe regularization techniques (L2/weight decay, dropout, early stopping, batch normalization, data augmentation) with mathematical precision and know when each one bites.
- define convolutional neural networks formally (convolution, padding/stride arithmetic, receptive field, parameter count) and state the backprop pass through a convolutional layer.
- understand k-means as alternating minimization (including the monotonicity proof), **derive EM for Gaussian mixture models completely** (ELBO, E-step, M-step) and explain the connection between k-means and GMM.
- define hierarchical clustering (linkage criteria, Ward) and DBSCAN (density notions) precisely and validate cluster solutions (silhouette, ARI, BIC).
- derive PCA in two ways (variance maximization via Lagrange; minimization of the reconstruction error) and state the SVD connection; explain t-SNE including its cost function and gradient.
- place the advanced topics: universal approximation, vanishing/exploding gradients (Jacobian analysis), residual connections, double descent, autoencoders and **the VAE with a complete ELBO derivation**.

---

## Part 1 — Foundations (basics)

### 1.1 From the linear model to the neural network

Module 04 ended with models of the form

$$\hat{y} = f(\mathbf{w}^\top \phi(\mathbf{x}) + b),$$

where the feature map $\phi$ was chosen **by hand** (polynomials, interactions, TF-IDF, …). The central idea of neural networks: **learn $\phi$ as well.** A neural network is a composition of parameterized, differentiable functions whose parameters are optimized jointly by gradient methods.

**Why nonlinearity is mandatory:** If you compose only affine maps, then
$W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$ —
the result is affine again. Only a nonlinear **activation function** between the layers creates real expressive power. The classical counterexample for a single perceptron is XOR: the four points $(0,0)\mapsto 0$, $(1,1)\mapsto 0$, $(0,1)\mapsto 1$, $(1,0)\mapsto 1$ are not linearly separable; an MLP with one hidden layer of two neurons solves XOR exactly.

### 1.2 The multilayer perceptron (MLP), formally

An MLP with $L$ layers is the function $f_\theta : \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$,

$$
\mathbf{a}^{(0)} = \mathbf{x}, \qquad
\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad
\mathbf{a}^{(\ell)} = \sigma^{(\ell)}\!\big(\mathbf{z}^{(\ell)}\big), \qquad \ell = 1,\dots,L,
$$

with weight matrices $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$, bias vectors $\mathbf{b}^{(\ell)} \in \mathbb{R}^{d_\ell}$ and elementwise activations $\sigma^{(\ell)}$. The quantities are called: $\mathbf{z}^{(\ell)}$ the **pre-activation**, $\mathbf{a}^{(\ell)}$ the **activation**, $\theta = \{W^{(\ell)}, \mathbf{b}^{(\ell)}\}_{\ell=1}^L$ the parameters. Number of parameters: $\sum_{\ell=1}^L d_\ell (d_{\ell-1} + 1)$.

**Common activation functions** (with derivatives — we need those in 1.5):

| Name | $\sigma(z)$ | $\sigma'(z)$ | Properties |
|---|---|---|---|
| Sigmoid | $\dfrac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | output in $(0,1)$; saturates: $\sigma' \le 1/4$, and $\approx 0$ for large $\lvert z\rvert$ |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | zero-centred; saturates as well |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | no saturation for $z>0$; permanent "dying ReLU" possible for $z<0$ |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \approx 0{.}01$ | $\mathbb{1}[z>0] + \alpha\,\mathbb{1}[z\le 0]$ | fixes the dying ReLU |
| GELU | $z\,\Phi(z)$ ($\Phi$ = standard normal CDF) | $\Phi(z) + z\,\varphi(z)$ | smooth; the standard in transformers |
| Softmax (output) | $\mathrm{softmax}(\mathbf{z})_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$ | Jacobian: $\dfrac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$ | turns logits into a probability vector |

ReLU is not differentiable at $z=0$; in practice one picks an element of the subgradient (usually $0$). That is theoretically clean via **subgradient methods** and practically irrelevant, because $z=0$ has measure zero.

### 1.3 Loss functions from maximum likelihood

Loss functions do not fall from the sky — they follow from a probabilistic model of the data. Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ be i.i.d. We model $p_\theta(y \mid \mathbf{x})$ and maximize the log-likelihood, equivalently: we minimize the **negative log-likelihood (NLL)**

$$\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(y_i \mid \mathbf{x}_i).$$

**Case 1 — regression with Gaussian noise.** Assume $y = f_\theta(\mathbf{x}) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, i.e. $p_\theta(y\mid\mathbf{x}) = \mathcal{N}(y;\, f_\theta(\mathbf{x}), \sigma^2)$. Then

$$
-\log p_\theta(y_i \mid \mathbf{x}_i)
= \frac{(y_i - f_\theta(\mathbf{x}_i))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2),
$$

and since the second term does not depend on $\theta$, minimizing the NLL is **equivalent to minimizing the mean squared error (MSE)**. Laplace noise would analogously give the MAE — the choice of loss is an assumption about the noise.

**Case 2 — classification with $K$ classes.** The network outputs logits $\mathbf{z} \in \mathbb{R}^K$, and $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$ models the categorical distribution $p_\theta(y = k \mid \mathbf{x}) = p_k$. The NLL is the **cross-entropy**

$$\mathcal{L}_i = -\log p_{y_i} = -\sum_{k=1}^K \mathbb{1}[y_i = k] \log p_k.$$

**An important calculation (softmax + cross-entropy):** The gradient with respect to the logits is remarkably simple. With the one-hot vector $\mathbf{y}$:

$$
\frac{\partial \mathcal{L}_i}{\partial z_j}
= -\sum_k y_k \frac{1}{p_k}\frac{\partial p_k}{\partial z_j}
= -\sum_k y_k \frac{1}{p_k} p_k(\delta_{kj} - p_j)
= -\big(y_j - p_j \sum_k y_k\big)
= p_j - y_j,
$$

so in vector form $\nabla_{\mathbf{z}} \mathcal{L}_i = \mathbf{p} - \mathbf{y}$ — "prediction minus truth". The gradient of logistic regression in module 04 had exactly the same form; the MLP only replaces the input of the last layer by learned features.

**A note on numerics:** Softmax and log are never computed separately. One uses the **log-sum-exp trick**: $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$, to avoid overflow (in PyTorch, `nn.CrossEntropyLoss` does this internally and expects raw logits).

### 1.4 Gradient descent

The learning problem is $\min_\theta \mathcal{L}(\theta)$ for a non-convex, differentiable function. **Gradient descent (GD)** iterates

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

with **learning rate** $\eta > 0$. The justification: the first-order Taylor expansion $\mathcal{L}(\theta + \Delta) \approx \mathcal{L}(\theta) + \nabla\mathcal{L}^\top \Delta$ is minimized under the constraint $\lVert\Delta\rVert \le \varepsilon$ by $\Delta \propto -\nabla\mathcal{L}$ — the negative gradient is the direction of steepest descent (with respect to the Euclidean norm).

For $L$-smooth functions (i.e. $\lVert \nabla\mathcal{L}(\theta) - \nabla\mathcal{L}(\theta')\rVert \le L_s \lVert \theta - \theta' \rVert$), $\eta \le 1/L_s$ guarantees monotone descent, because smoothness implies the **descent inequality**

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta\Big(1 - \frac{L_s \eta}{2}\Big)\lVert\nabla\mathcal{L}(\theta_t)\rVert^2 .$$

For non-convex problems, GD therefore converges to stationary points ($\nabla\mathcal{L}=0$), not necessarily to global minima. Empirically, for overparameterized networks almost all local minima that are reached are good — more on that in part 3.6.

### 1.5 Backpropagation — the complete derivation

Backpropagation is **not a learning method of its own**, but an algorithm that computes $\nabla_\theta \mathcal{L}$ efficiently for layered functions: a systematic application of the multidimensional chain rule with a clever order of evaluation (a special case of **reverse-mode automatic differentiation**).

**Setup.** Consider a single training example with loss $\mathcal{L} = \ell(\mathbf{a}^{(L)}, \mathbf{y})$ and the forward equations from 1.2. Define the **error signal** of layer $\ell$:

$$\boldsymbol{\delta}^{(\ell)} \;:=\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}} \in \mathbb{R}^{d_\ell}.$$

**Step 1 — output layer.** By the chain rule through the elementwise activation:

$$\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \odot \sigma'^{(L)}\big(\mathbf{z}^{(L)}\big),$$

where $\odot$ is the elementwise (Hadamard) product. For softmax + cross-entropy this collapses, as shown in 1.3, to $\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}$ (here the "activation" softmax is not elementwise; one computes directly with the Jacobian, and that is exactly why the result is so simple).

**Step 2 — recursion over the layers.** $\mathcal{L}$ depends on $\mathbf{z}^{(\ell)}$ only through $\mathbf{z}^{(\ell+1)} = W^{(\ell+1)}\sigma(\mathbf{z}^{(\ell)}) + \mathbf{b}^{(\ell+1)}$. The chain rule in components:

$$
\delta_i^{(\ell)}
= \sum_j \frac{\partial \mathcal{L}}{\partial z_j^{(\ell+1)}} \frac{\partial z_j^{(\ell+1)}}{\partial z_i^{(\ell)}}
= \sum_j \delta_j^{(\ell+1)} \, W_{ji}^{(\ell+1)} \, \sigma'\big(z_i^{(\ell)}\big),
$$

because $z_j^{(\ell+1)} = \sum_i W_{ji}^{(\ell+1)} \sigma(z_i^{(\ell)}) + b_j^{(\ell+1)}$. In matrix form:

$$\boxed{\;\boldsymbol{\delta}^{(\ell)} = \Big( {W^{(\ell+1)}}^\top \boldsymbol{\delta}^{(\ell+1)} \Big) \odot \sigma'\big(\mathbf{z}^{(\ell)}\big)\;}$$

The error signal therefore flows **backwards** through the transposed weight matrices — hence the name.

**Step 3 — parameter gradients.** $\mathcal{L}$ depends on $W_{ij}^{(\ell)}$ only through $z_i^{(\ell)}$, and $\partial z_i^{(\ell)} / \partial W_{ij}^{(\ell)} = a_j^{(\ell-1)}$. Hence

$$
\frac{\partial \mathcal{L}}{\partial W_{ij}^{(\ell)}} = \delta_i^{(\ell)} \, a_j^{(\ell-1)}
\quad\Longleftrightarrow\quad
\boxed{\;\nabla_{W^{(\ell)}} \mathcal{L} = \boldsymbol{\delta}^{(\ell)} {\mathbf{a}^{(\ell-1)}}^\top, \qquad \nabla_{\mathbf{b}^{(\ell)}} \mathcal{L} = \boldsymbol{\delta}^{(\ell)}\;}
$$

— an outer product: "error of the layer times input of the layer".

**The algorithm** (for a minibatch one averages the gradients of the examples; in matrix notation with the batch matrix $A^{(\ell)} \in \mathbb{R}^{d_\ell \times B}$ the outer products become matrix products):

1. **Forward pass:** compute and **store** $\mathbf{z}^{(\ell)}, \mathbf{a}^{(\ell)}$ for all $\ell$.
2. **Backward pass:** $\boldsymbol{\delta}^{(L)}$ from step 1; then for $\ell = L-1, \dots, 1$ the recursion from step 2, computing the gradients from step 3 along the way.
3. **Update:** one step of the chosen optimizer (part 2.1).

**Complexity.** Forward and backward both cost $O\big(\sum_\ell d_\ell d_{\ell-1}\big)$ per example — backprop computes the *complete* gradient with respect to *all* parameters for the price of roughly **two forward passes**. For comparison: numerical differentiation would need $O(\lvert\theta\rvert)$ forward passes. The price is memory: all intermediate activations have to be kept for the backward pass, $O\big(B \sum_\ell d_\ell\big)$.

**Why reverse mode?** Forward-mode AD propagates directional derivatives $\partial(\cdot)/\partial\theta_j$ forwards — one pass per parameter, good for *few inputs, many outputs*. Reverse mode propagates $\partial\mathcal{L}/\partial(\cdot)$ backwards — one pass for *all* parameters, good for *many inputs, one scalar output*. Training is exactly the second case.

### 1.6 A worked numerical example (one complete backprop step)

Network: 2 inputs → 2 hidden neurons (sigmoid) → 1 output (sigmoid), loss binary cross-entropy $\mathcal{L} = -[y\log\hat y + (1-y)\log(1-\hat y)]$.

Parameters: $W^{(1)} = \begin{pmatrix} 0.5 & -0.3 \\ 0.8 & 0.2 \end{pmatrix}$, $\mathbf{b}^{(1)} = \mathbf{0}$, $W^{(2)} = (1.0,\; -1.0)$, $b^{(2)} = 0$. Input $\mathbf{x} = (1, 2)^\top$, target $y = 1$.

*Forward:*
- $\mathbf{z}^{(1)} = (0.5 - 0.6,\; 0.8 + 0.4)^\top = (-0.1,\; 1.2)^\top$
- $\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)}) \approx (0.4750,\; 0.7685)^\top$
- $z^{(2)} = 1.0\cdot 0.4750 - 1.0 \cdot 0.7685 = -0.2935$, $\hat y = \sigma(-0.2935) \approx 0.4271$
- $\mathcal{L} = -\log 0.4271 \approx 0.8508$

*Backward:*
- For sigmoid + BCE we have (the same calculation as softmax+CE): $\delta^{(2)} = \hat y - y = -0.5729$.
- $\nabla_{W^{(2)}}\mathcal{L} = \delta^{(2)} {\mathbf{a}^{(1)}}^\top \approx (-0.2721,\; -0.4403)$, $\nabla_{b^{(2)}}\mathcal{L} = -0.5729$.
- $\boldsymbol{\delta}^{(1)} = \big({W^{(2)}}^\top \delta^{(2)}\big) \odot \mathbf{a}^{(1)}\odot(1-\mathbf{a}^{(1)}) \approx \begin{pmatrix}-0.5729 \\ +0.5729\end{pmatrix} \odot \begin{pmatrix}0.2494 \\ 0.1779\end{pmatrix} \approx \begin{pmatrix}-0.1429 \\ 0.1019\end{pmatrix}$
- $\nabla_{W^{(1)}}\mathcal{L} = \boldsymbol{\delta}^{(1)}\mathbf{x}^\top \approx \begin{pmatrix} -0.1429 & -0.2858 \\ 0.1019 & 0.2038 \end{pmatrix}$

*Update* with $\eta = 0.5$: for example $W^{(2)} \leftarrow (1.0,\,-1.0) - 0.5\,(-0.2721,\,-0.4403) = (1.1361,\; -0.7799)$. Another forward pass would give $\hat y \approx 0.55$ — the prediction moves towards $y=1$. This is exactly the calculation you implement by hand in **project 01** and verify by gradient checking.

**Gradient checking:** Compare the analytical gradient with the **central difference** $\frac{\partial\mathcal{L}}{\partial\theta_j} \approx \frac{\mathcal{L}(\theta + \epsilon\mathbf{e}_j) - \mathcal{L}(\theta - \epsilon\mathbf{e}_j)}{2\epsilon}$ (error $O(\epsilon^2)$, whereas the one-sided difference only reaches $O(\epsilon)$), $\epsilon \approx 10^{-5}$, and measure the relative deviation $\frac{\lVert g_{\text{ana}} - g_{\text{num}}\rVert}{\lVert g_{\text{ana}}\rVert + \lVert g_{\text{num}}\rVert} \lesssim 10^{-7}$ (in float64).
---

## Part 2 — Building up (intermediate)

### 2.1 Stochastic optimization: from SGD to AdamW

**Minibatch SGD.** The full gradient costs $O(n)$ per step. Instead one draws a minibatch $\mathcal{B}_t$ ($|\mathcal{B}_t| = B \ll n$) and uses

$$\theta_{t+1} = \theta_t - \eta_t \, \mathbf{g}_t, \qquad \mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla_\theta \mathcal{L}_i(\theta_t).$$

$\mathbf{g}_t$ is an **unbiased estimator** of the full gradient: $\mathbb{E}[\mathbf{g}_t] = \nabla\mathcal{L}(\theta_t)$, with covariance $\propto 1/B$. The noise is not only a saving in cost — it helps to escape saddle points and acts as an implicit regularizer (part 3.6). The classical convergence result (Robbins & Monro, 1951): for convex objectives SGD converges if $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (e.g. $\eta_t \propto 1/t$); with a constant learning rate SGD only converges into a **noise ball** around the optimum whose radius grows with $\eta$ and with the gradient variance.

**Momentum (Polyak, "heavy ball").** Carry an exponentially weighted velocity along:

$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \mathbf{g}_t, \qquad \theta_{t+1} = \theta_t - \eta\, \mathbf{v}_{t+1}, \qquad \mu \in [0,1),\ \text{typically } 0.9.$$

Unrolled: $\mathbf{v}_{t+1} = \sum_{s=0}^{t} \mu^{s}\, \mathbf{g}_{t-s}$ — a moving average of the gradients with an effective horizon of $\approx 1/(1-\mu)$. The effect: in directions of consistent gradients (the flat direction of a valley) the step accumulates up to a factor of $\frac{1}{1-\mu}$; in oscillating directions (the steep valley walls) the contributions average out. For quadratic objectives with condition number $\kappa$, momentum improves the convergence rate from $O(\kappa)$ to $O(\sqrt{\kappa})$ iterations.

**Nesterov accelerated gradient (NAG):** like momentum, but the gradient is evaluated at the **look-ahead** point: $\mathbf{v}_{t+1} = \mu\mathbf{v}_t + \nabla\mathcal{L}(\theta_t - \eta\mu\mathbf{v}_t)$, $\theta_{t+1} = \theta_t - \eta\mathbf{v}_{t+1}$ — a correction that brakes overshooting earlier.

**AdaGrad** (Duchi et al., 2011) scales the learning rate per parameter with the history of squared gradients:

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^{\odot 2}, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \odot \mathbf{g}_t.$$

Rarely active parameters (small accumulated squares) get large steps — good for sparse features. The problem: $\mathbf{s}_t$ grows monotonically, so the effective learning rate dies.

**RMSProp** (Hinton) replaces the sum by an exponential average: $\mathbf{s}_t = \rho\, \mathbf{s}_{t-1} + (1-\rho)\, \mathbf{g}_t^{\odot 2}$ (typically $\rho = 0.99$), with the same update as AdaGrad — the learning rate no longer dies.

**Adam** (Kingma & Ba, 2015) combines momentum (1st moment) and RMSProp (2nd moment) with a **bias correction**:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t, &\qquad \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t},\\[2pt]
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^{\odot 2}, &\qquad \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t},\\[2pt]
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}. & &
\end{aligned}
$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

*Derivation of the bias correction:* with $\mathbf{m}_0 = \mathbf{0}$, unrolling gives $\mathbf{m}_t = (1-\beta_1)\sum_{s=1}^{t} \beta_1^{t-s}\, \mathbf{g}_s$. Under the (idealizing) assumption of stationary gradients $\mathbb{E}[\mathbf{g}_s] = \mathbf{g}$ it follows that $\mathbb{E}[\mathbf{m}_t] = (1-\beta_1)\,\mathbf{g}\sum_{s=1}^{t}\beta_1^{t-s} = \mathbf{g}\,(1 - \beta_1^t)$ (a geometric sum). The estimator is therefore biased towards zero by the factor $(1-\beta_1^t)$ — massively so at the very beginning ($t$ small; $\beta_2^t \approx 1$ makes it even worse for the 2nd moment). Dividing by $(1-\beta^t)$ makes it unbiased.

**AdamW** (Loshchilov & Hutter, 2019): with Adam, L2 regularization is **not** equivalent to weight decay (see 2.3) — the L2 gradient is divided by $\sqrt{\hat{\mathbf{v}}_t}$ as well and is thereby weakened for parameters with large gradients. AdamW decouples them: $\theta_{t+1} = \theta_t - \eta\big(\hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) + \lambda \theta_t\big)$. Today the standard optimizer.

**Learning rate schedules.** A constant $\eta$ is rarely optimal. Common choices: **step decay** ($\eta$ times $0.1$ every $k$ epochs), **cosine annealing** $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1 + \cos(\pi t/T))$, and **warmup** (linearly from 0 to $\eta_{\max}$ over the first steps — this stabilizes Adam early on, when $\hat{\mathbf{v}}_t$ is still unreliable).

### 2.2 Initialization: Xavier and He, derived

Initializing all weights with 0 is fatal: all neurons of a layer receive identical gradients and stay identical forever (the **symmetry problem**). But the scale of random initialization is critical too.

**Variance propagation.** Consider $z_i = \sum_{j=1}^{d_{\text{in}}} W_{ij} a_j$ with independent, centred $W_{ij}$ (variance $\sigma_W^2$) and activations $a_j$ (variance $\sigma_a^2$, centred). Then

$$\operatorname{Var}(z_i) = \sum_{j=1}^{d_{\text{in}}} \operatorname{Var}(W_{ij} a_j) = d_{\text{in}}\, \sigma_W^2 \sigma_a^2.$$

So that the signal variance neither explodes nor vanishes across the layers, one demands $d_{\text{in}} \sigma_W^2 = 1$. The same calculation for the backward-flowing error signal (through $W^\top$) demands $d_{\text{out}} \sigma_W^2 = 1$. **Xavier/Glorot initialization** compromises:

$$\sigma_W^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}} \qquad \text{(for tanh-like activations that are linear around 0)}.$$

**He initialization** for ReLU: ReLU zeroes half of the inputs. For $z$ symmetric around 0 we have $\mathbb{E}[\mathrm{ReLU}(z)^2] = \tfrac12 \mathbb{E}[z^2]$, so the variance halves per layer and one compensates with a factor of 2:

$$\sigma_W^2 = \frac{2}{d_{\text{in}}}.$$

Rule of thumb: **He for the ReLU family, Xavier for tanh/sigmoid** — PyTorch layers ship sensible defaults, but you should know why.

### 2.3 Regularization

Deep networks often have more parameters than data points — controlling capacity decides generalization.

**L2 regularization / weight decay.** Objective: $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\lVert\theta\rVert_2^2$. The GD step becomes

$$\theta_{t+1} = \theta_t - \eta\big(\nabla\mathcal{L}(\theta_t) + \lambda\theta_t\big) = (1 - \eta\lambda)\,\theta_t - \eta\nabla\mathcal{L}(\theta_t),$$

i.e. the weights are shrunk multiplicatively at every step ("decay"). For **plain SGD** the L2 loss term and weight decay are therefore identical; for **adaptive methods (Adam) they are not** — hence AdamW. In Bayesian terms, L2 corresponds to a Gaussian prior $\theta \sim \mathcal{N}(0, \lambda^{-1} I)$ and the minimization to the MAP estimate.

**Dropout** (Srivastava et al., 2014). During training every activation is zeroed independently with probability $p$, using a Bernoulli mask $\mathbf{m} \sim \mathrm{Bernoulli}(1-p)^{d}$:

$$\tilde{\mathbf{a}} = \frac{\mathbf{m} \odot \mathbf{a}}{1-p} \qquad \text{("inverted dropout")}.$$

The division by $1-p$ keeps the expectation constant: $\mathbb{E}[\tilde a_i] = \frac{(1-p)\,a_i}{1-p} = a_i$ — which is why at inference time one can simply use the full network without a mask. Interpretations: (i) training an implicit **ensemble** of $2^d$ subnetworks with shared weights whose predictions are approximately averaged at test time; (ii) preventing **co-adaptation** — no neuron can rely on one particular other one; (iii) for linear regression, dropout is equivalent to a data-dependent L2 penalty.

**Early stopping.** Watch the validation loss and stop (or keep the best checkpoint) when it has not improved for $k$ epochs ("patience"). For linear models with squared loss, early stopping is provably closely related to L2 regularization: GD first learns the directions with large eigenvalues of the data covariance; stopping early cuts off the small eigendirections — exactly like ridge with $\lambda \approx 1/(\eta t)$.

**Data augmentation** encodes invariances (images: mirroring, translation, cropping; audio: time shifting), effectively enlarges the data set and is often the single strongest measure — it acts on the data side, not on the parameter side.

### 2.4 Batch normalization

**Definition** (Ioffe & Szegedy, 2015). For every feature dimension $k$ of a minibatch $\{z_{i,k}\}_{i=1}^B$:

$$
\mu_k = \frac{1}{B}\sum_{i=1}^B z_{i,k}, \qquad
\sigma_k^2 = \frac{1}{B}\sum_{i=1}^B (z_{i,k}-\mu_k)^2, \qquad
\hat z_{i,k} = \frac{z_{i,k} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}, \qquad
y_{i,k} = \gamma_k \hat z_{i,k} + \beta_k,
$$

with **learnable** scale/shift parameters $\gamma_k, \beta_k$ (the normalization must not restrict expressive power — with $\gamma_k = \sigma_k$, $\beta_k = \mu_k$ it would be the identity). **At inference time** there is no batch: one uses the running averages $\bar\mu_k, \bar\sigma_k^2$ that were tracked during training (in PyTorch this is why `model.eval()` before evaluation is mandatory!).

**Why it helps:** The original explanation ("internal covariate shift") is considered outdated; the view accepted today (Santurkar et al., 2018): batch norm **smooths the loss landscape** (smaller Lipschitz constants of the loss and of the gradient), permits larger learning rates and makes training robust against the scale of the initialization, because the output is invariant to a rescaling of the previous layer's weights. A side effect: the batch noise in $\mu_k, \sigma_k^2$ regularizes slightly. With small batches or sequence models one uses **layer normalization** instead (normalization over the feature dimension per example — batch independent, the standard in transformers).

### 2.5 Convolutional neural networks (CNNs)

Fully connected layers on images waste parameters (a $224{\times}224{\times}3$ image → a first layer with 1000 neurons would have 150 million weights) and ignore the spatial structure. CNNs build in two **inductive biases**: **locality** (pixels correlate with their neighbours) and **translation equivariance** (an edge detector is the same everywhere → **weight sharing**).

**The convolutional layer, formally.** Input $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$, kernel $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$, output

$$
Y_{c',\,i,\,j} \;=\; b_{c'} + \sum_{c=1}^{C_{\text{in}}} \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} K_{c',c,u,v}\; X_{c,\; i\cdot s + u - p,\; j\cdot s + v - p}
$$

with stride $s$ and padding $p$ (indices outside: 0). Strictly speaking this is a **cross-correlation** (a convolution would flip the kernel); since the kernels are learned, the difference is irrelevant, and all frameworks implement it this way.

**Output size:** $H_{\text{out}} = \left\lfloor \dfrac{H + 2p - k_h}{s} \right\rfloor + 1$ (analogously $W_{\text{out}}$). "Same" padding: $p = (k-1)/2$ with $s=1$ preserves the resolution.

**Number of parameters:** $C_{\text{out}}(C_{\text{in}} k_h k_w + 1)$ — independent of $H, W$. Example: a $3{\times}3$ conv from 64 to 128 channels: $128\cdot(64\cdot 9 + 1) = 73\,856$ parameters, no matter whether the image is $32^2$ or $1024^2$.

**Receptive field:** the set of input pixels that influence one output neuron. With $L$ layers of kernel size $k$ and stride 1 it grows linearly: $r_L = 1 + L(k-1)$. Pooling and stride enlarge it multiplicatively — in general $r_L = r_{L-1} + (k_L - 1)\prod_{\ell<L} s_\ell$. Deep networks therefore only see "the whole image" in their late layers.

**Pooling:** max pooling $Y_{c,i,j} = \max_{(u,v) \in \text{window}} X_{c, si+u, sj+v}$ makes representations locally translation **invariant** and reduces the resolution. Modern architectures often replace it by strided convolutions; at the end of the network one usually finds **global average pooling** (the mean over all positions per channel) instead of huge dense layers.

**Backprop through the convolution** (the key statements, derivable exactly as in 1.5 by differentiating the summation formula):
- $\dfrac{\partial \mathcal{L}}{\partial K_{c',c,u,v}} = \sum_{i,j} \delta_{c',i,j}\, X_{c,\, si+u-p,\, sj+v-p}$ — a **cross-correlation between the input and the error signal** (weight sharing shows up as the sum over all positions).
- $\dfrac{\partial \mathcal{L}}{\partial X}$ is a **"full" convolution of the error signal with the kernel rotated by 180 degrees** (transposed convolution).

**A typical architecture** (as in project 02): several blocks `[Conv → BatchNorm → ReLU] ×2 → MaxPool` with a growing number of channels (e.g. 32→64→128), then global average pooling → linear → softmax. Historical milestones: LeNet-5 (1998), AlexNet (2012, ReLU+dropout+GPU), VGG (2014, only 3×3), ResNet (2015, part 3.3).

### 2.6 Clustering I: k-means

From here on: **unsupervised learning** — data $\{\mathbf{x}_i\}_{i=1}^n$ without labels; the goal is structure: groups (clustering), compact representations (dimensionality reduction), densities (mixture models).

**Objective function.** Find centres $\boldsymbol{\mu}_1,\dots,\boldsymbol{\mu}_K$ and assignments $r_{ik} \in \{0,1\}$ ($\sum_k r_{ik}=1$) that minimize the **within-cluster scatter** (inertia/distortion):

$$J(\{r_{ik}\}, \{\boldsymbol{\mu}_k\}) = \sum_{i=1}^n \sum_{k=1}^K r_{ik}\, \lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2 \;\to\; \min.$$

The exact problem is NP-hard (already for $K=2$ in general dimension). **Lloyd's algorithm** is an alternating minimization:

1. **Assignment step** (with the $\boldsymbol{\mu}_k$ fixed): $r_{ik} = 1$ for $k = \arg\min_j \lVert\mathbf{x}_i - \boldsymbol{\mu}_j\rVert^2$ — pointwise optimal, since every $\mathbf{x}_i$ is assigned independently to its nearest centre.
2. **Update step** (with the $r_{ik}$ fixed): $\nabla_{\boldsymbol{\mu}_k} J = -2\sum_i r_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k) \overset{!}{=} 0 \Rightarrow \boldsymbol{\mu}_k = \dfrac{\sum_i r_{ik}\mathbf{x}_i}{\sum_i r_{ik}}$ — the **centroid** of the cluster (the objective is convex-quadratic in $\boldsymbol{\mu}_k$, so this is the global minimum of that substep).

**Monotonicity and convergence:** neither step can ever increase $J$; $J \ge 0$ is bounded below; there are only finitely many partitions — so the algorithm converges in finitely many steps. But only to a **local** optimum, depending on the start. In practice: several restarts (`n_init`), initialization with **k-means++** (Arthur & Vassilvitskii, 2007): the first centre uniformly, every further one with probability $\propto D(\mathbf{x})^2$ (the squared distance to the nearest already chosen centre) — this yields an $O(\log K)$ approximation of the optimal distortion in expectation.

**Choosing $K$:** (i) the **elbow method** — $J$ falls monotonically in $K$; one looks for the kink (heuristic, often ambiguous). (ii) The **silhouette coefficient** per point:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} \in [-1, 1],$$

with $a(i)$ = the mean distance to points of its own cluster, $b(i)$ = the smallest mean distance to a foreign cluster. A mean close to 1 means compact and well separated; around 0 means overlap; negative means probably misassigned. (iii) Model based: BIC over GMMs (2.7).

**Limitations:** k-means implicitly assumes isotropic, equally large, convex clusters (Voronoi cells!) and is scale sensitive → standardize beforehand; non-convex shapes (rings, moons) fail fundamentally → DBSCAN/spectral clustering.

### 2.7 Mixture models and the EM algorithm — the complete derivation

**Gaussian mixture model (GMM).** Probabilistic clustering: the data come from $K$ Gaussian components,

$$p(\mathbf{x} \mid \theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma_k), \qquad \pi_k \ge 0,\ \sum_k \pi_k = 1,$$

equivalently with a **latent variable** $z_i \in \{1,\dots,K\}$: $p(z_i = k) = \pi_k$, $\;\mathbf{x}_i \mid z_i = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$.

**The problem.** The log-likelihood of the observed data

$$\log p(X \mid \theta) = \sum_{i=1}^n \log \underbrace{\sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)}_{\text{a sum inside the log!}}$$

has no closed-form maximum because of the sum **inside** the logarithm. The **EM algorithm** (Dempster, Laird & Rubin, 1977) solves this iteratively.

**The lower bound (ELBO).** For an arbitrary distribution $q_i(k)$ over the latent variable, Jensen's inequality ($\log$ is concave) gives:

$$
\log p(\mathbf{x}_i \mid \theta)
= \log \sum_k q_i(k)\, \frac{p(\mathbf{x}_i, z_i = k \mid \theta)}{q_i(k)}
\;\ge\; \sum_k q_i(k) \log \frac{p(\mathbf{x}_i, z_i = k \mid \theta)}{q_i(k)}
\;=:\; \mathcal{F}(q_i, \theta).
$$

More precisely: $\log p(\mathbf{x}_i \mid \theta) = \mathcal{F}(q_i, \theta) + \mathrm{KL}\big(q_i \,\Vert\, p(z_i \mid \mathbf{x}_i, \theta)\big)$, where $\mathrm{KL}(q\Vert p) = \sum_k q(k)\log\frac{q(k)}{p(k)} \ge 0$ is the Kullback-Leibler divergence (with equality iff $q = p$). One verifies this decomposition directly by substituting $p(\mathbf{x}_i, z_i) = p(z_i \mid \mathbf{x}_i)\,p(\mathbf{x}_i)$. EM is **coordinate ascent on $\mathcal{F}$**:

**E-step** — maximize $\mathcal{F}$ over $q$ with $\theta^{(t)}$ fixed: since the KL term is the only difference to the (fixed) log-likelihood, the optimum is $q_i = p(z_i \mid \mathbf{x}_i, \theta^{(t)})$, the **posterior distribution**. By Bayes:

$$
\gamma_{ik} := p(z_i = k \mid \mathbf{x}_i, \theta^{(t)}) = \frac{\pi_k\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)}{\sum_{j} \pi_j\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \Sigma_j)}
\qquad \text{("responsibilities" — soft assignments)}.
$$

**M-step** — maximize $\mathcal{F}$ over $\theta$ with $q$ fixed. The $q$-entropy term is constant in $\theta$, what remains is the **expected complete log-likelihood**

$$
Q(\theta) = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \Big[ \log \pi_k - \tfrac{1}{2}\log\lvert 2\pi\Sigma_k\rvert - \tfrac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top \Sigma_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) \Big].
$$

Here the log stands **inside** the sum — solvable in closed form. Setting the derivatives to zero (for $\pi$ with a Lagrange multiplier for $\sum_k \pi_k = 1$) gives, with $N_k := \sum_i \gamma_{ik}$ (the effective cluster size):

$$
\boxed{\;
\pi_k^{\text{new}} = \frac{N_k}{n}, \qquad
\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\, \mathbf{x}_i, \qquad
\Sigma_k^{\text{new}} = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\, (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^\top
\;}
$$

— weighted means/covariances with the responsibilities as weights. *(The calculation for $\boldsymbol{\mu}_k$: $\nabla_{\boldsymbol{\mu}_k} Q = \sum_i \gamma_{ik}\, \Sigma_k^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_k) = 0$; for $\pi_k$: $\partial/\partial\pi_k \big[\sum_{i,k}\gamma_{ik}\log\pi_k + \lambda(1{-}\sum_k\pi_k)\big] = N_k/\pi_k - \lambda = 0$, and summing over $k$ yields $\lambda = n$.)*

**The monotonicity guarantee:** $\log p(X\mid\theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t)}) = \log p(X \mid \theta^{(t)})$ — the last equality holds because the E-step sets the KL to 0. The likelihood therefore increases monotonically; EM converges to a stationary point (a local maximum or a saddle). As with k-means: restarts are needed; initialization is often done with k-means.

**k-means as a limiting case:** fix $\Sigma_k = \epsilon I$ and let $\epsilon \to 0$: the responsibilities $\gamma_{ik}$ become hard ($\to$ the indicator of the nearest centre), the M-step becomes the centroid — EM degenerates exactly into Lloyd's algorithm. A GMM is "soft k-means" with learned, anisotropic cluster shapes and mixing weights.

**Covariance types** (a bias-variance trade-off, in sklearn `covariance_type`): `spherical` ($\sigma_k^2 I$) ⊂ `diag` ⊂ `tied` (one shared $\Sigma$) ⊂ `full` — with $d$ dimensions, `full` costs $d(d+1)/2$ parameters per component.

**Model selection with information criteria:** $\mathrm{BIC} = -2\log \hat{L} + m \log n$, $\mathrm{AIC} = -2\log \hat{L} + 2m$ ($m$ = the number of parameters). Smaller is better; BIC penalizes complexity more strongly and is the usual recommendation for choosing $K$. **Careful, degeneracy:** the likelihood is unbounded (a component can collapse onto a single point, $\lvert\Sigma_k\rvert \to 0$) — the practical remedy: regularizing the covariance (`reg_covar`), minimum cluster sizes.

### 2.8 Clustering II: hierarchical and DBSCAN

**Agglomerative (hierarchical) clustering.** Start with $n$ singleton clusters; repeatedly merge the pair with the smallest distance $d(A, B)$; the result is a **dendrogram** (a binary tree of all merges) from which cutting at an arbitrary height gives any number of clusters. The choice of $d(A,B)$ (the **linkage**) determines the character:

| Linkage | $d(A,B)$ | Behaviour |
|---|---|---|
| Single | $\min_{a \in A, b \in B} \lVert a - b\rVert$ | chaining; finds elongated structures |
| Complete | $\max_{a \in A, b \in B} \lVert a - b\rVert$ | compact, equally sized clusters; sensitive to outliers |
| Average (UPGMA) | $\frac{1}{\lvert A\rvert \lvert B\rvert}\sum_{a,b} \lVert a - b\rVert$ | a compromise |
| **Ward** | Merge the pair with the minimal increase in within-cluster variance: $\Delta J(A,B) = \frac{\lvert A\rvert\,\lvert B\rvert}{\lvert A\rvert + \lvert B\rvert}\, \lVert \bar{\mathbf{a}} - \bar{\mathbf{b}} \rVert^2$ | k-means-like, spherical clusters; usually the best default |

After a merge $A \cup B$, all linkages can be updated efficiently by the **Lance-Williams recursion**: $d(A{\cup}B, C) = \alpha_A d(A,C) + \alpha_B d(B,C) + \beta\, d(A,B) + \gamma\, \lvert d(A,C) - d(B,C)\rvert$ with linkage-specific coefficients. Complexity in general $O(n^2 \log n)$ time and $O(n^2)$ memory — prohibitive for large $n$.

**DBSCAN** (Ester et al., 1996) — density based, with two parameters $\varepsilon$ (radius) and $\textit{minPts}$:

- $\mathbf{p}$ is a **core point** if $\lvert N_\varepsilon(\mathbf{p}) \rvert \ge \textit{minPts}$ (including $\mathbf{p}$ itself), with $N_\varepsilon(\mathbf{p}) = \{\mathbf{q} : \lVert\mathbf{p}-\mathbf{q}\rVert \le \varepsilon\}$.
- $\mathbf{q}$ is **directly density-reachable** from $\mathbf{p}$ if $\mathbf{q} \in N_\varepsilon(\mathbf{p})$ and $\mathbf{p}$ is a core point. **Density-reachable** is the transitive closure of that (a chain of core points). Two points are **density-connected** if both are density-reachable from a common core point.
- A **cluster** is a maximal set of density-connected points. Points in no cluster are **noise** (label $-1$). Non-core points at the edge of a cluster are called **border points**.

Properties: it finds **arbitrarily shaped** clusters, determines the number of clusters itself and has an explicit notion of noise — but: a global $\varepsilon$ fails for clusters of strongly differing density (successors: OPTICS, HDBSCAN), and in high dimensions $\varepsilon$-balls lose meaning through the curse of dimensionality. **Choosing the parameters:** $\textit{minPts} \approx 2d$ as a rule of thumb; for $\varepsilon$ the **k-distance plot** ($k = \textit{minPts}-1$): the sorted distance of every point to its $k$-th nearest neighbour; take $\varepsilon$ at the "knee".

**External cluster validation** (when a reference partition exists, e.g. in benchmarks): the **Rand index** measures the share of consistently treated pairs of points (together/together or separate/separate). The **adjusted Rand index (ARI)** corrects for the value expected by chance:

$$\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \big[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\big] / \binom{n}{2}}{\frac{1}{2}\big[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\big] - \big[\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}\big]/\binom{n}{2}},$$

with the contingency table $n_{ij}$ and the marginals $a_i, b_j$. ARI $= 1$: identical partitions; $\approx 0$: chance level; ARI is permutation invariant (cluster labels are arbitrary, after all). An alternative: **normalized mutual information** $\mathrm{NMI}(U,V) = \frac{I(U;V)}{\sqrt{H(U)H(V)}}$.

### 2.9 Dimensionality reduction I: PCA — two derivations

**The goal:** project centred data $X \in \mathbb{R}^{n \times d}$ (column means 0) onto a $q$-dimensional subspace that preserves "as much information as possible". Let $S = \frac{1}{n} X^\top X$ be the empirical covariance matrix.

**Derivation 1 — variance maximization.** Look for the direction $\mathbf{w}$, $\lVert\mathbf{w}\rVert = 1$, that maximizes the variance of the projection:

$$\max_{\mathbf{w}} \ \operatorname{Var}(X\mathbf{w}) = \mathbf{w}^\top S\, \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^\top\mathbf{w} = 1.$$

Lagrange: $\Lambda(\mathbf{w}, \lambda) = \mathbf{w}^\top S \mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w} - 1)$, $\;\nabla_{\mathbf{w}} \Lambda = 2S\mathbf{w} - 2\lambda\mathbf{w} = 0 \Rightarrow S\mathbf{w} = \lambda\mathbf{w}$ — an **eigenvalue problem**. The variance attained is $\mathbf{w}^\top S\mathbf{w} = \lambda$, so one picks the **largest** eigenvalue; the $q$ principal components are the eigenvectors of the $q$ largest eigenvalues $\lambda_1 \ge \dots \ge \lambda_q$ (successively, with an orthogonality constraint). **Explained variance:** the share $\sum_{j\le q}\lambda_j / \sum_{j\le d}\lambda_j$ (the standard criterion for choosing $q$, e.g. 90–95 %).

**Derivation 2 — minimal reconstruction error.** Look for the orthonormal basis $W \in \mathbb{R}^{d\times q}$ ($W^\top W = I_q$) that minimizes the squared reconstruction error:

$$\min_W \frac{1}{n}\sum_{i=1}^n \lVert \mathbf{x}_i - W W^\top \mathbf{x}_i \rVert^2 = \underbrace{\operatorname{tr}(S)}_{\text{const.}} - \operatorname{tr}\big(W^\top S\, W\big),$$

(Pythagoras: $\lVert\mathbf{x}\rVert^2 = \lVert W^\top\mathbf{x}\rVert^2 + \lVert\mathbf{x} - WW^\top\mathbf{x}\rVert^2$). Minimizing the error = maximizing $\operatorname{tr}(W^\top S W)$ = the same top-$q$ eigenvectors. **Maximizing the variance and minimizing the reconstruction error are exactly the same problem.** The minimal error is $\sum_{j > q} \lambda_j$.

**The SVD connection.** With the singular value decomposition $X = U \Sigma V^\top$ we have $S = \frac{1}{n} V \Sigma^2 V^\top$: the right singular vectors $V$ are the principal directions, $\lambda_j = \sigma_j^2 / n$. Numerically one **always computes PCA via the SVD of $X$** (more stable than forming $X^\top X$, whose condition number is squared). The projections ("scores") are $XV = U\Sigma$.

**In practice:** centring beforehand is mandatory; with features in different units, standardize (then $S$ is the correlation matrix). PCA is linear and variance oriented — it can merge clusters that are separated along directions of small variance.

### 2.10 Dimensionality reduction II: t-SNE (and UMAP)

**t-SNE** (van der Maaten & Hinton, 2008) is a **nonlinear** embedding for visualization (usually $q = 2$) that preserves neighbourhoods rather than distances.

**High-dimensional similarities:** conditional Gaussian neighbour probabilities

$$p_{j\mid i} = \frac{\exp\big(-\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2 / 2\sigma_i^2\big)}{\sum_{k \ne i} \exp\big(-\lVert\mathbf{x}_i - \mathbf{x}_k\rVert^2 / 2\sigma_i^2\big)}, \qquad p_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2n}.$$

The bandwidth $\sigma_i$ is chosen **per point** so that the effective number of neighbours matches a global parameter: $\mathrm{Perp}(P_i) = 2^{H(P_i)}$ with the Shannon entropy $H(P_i) = -\sum_j p_{j\mid i}\log_2 p_{j\mid i}$ — the **perplexity** (typically 5–50); found by a binary search over $\sigma_i$.

**Low-dimensional similarities:** a **Student t-distribution with 1 degree of freedom** (Cauchy):

$$q_{ij} = \frac{\big(1 + \lVert\mathbf{y}_i - \mathbf{y}_j\rVert^2\big)^{-1}}{\sum_{k \ne l} \big(1 + \lVert\mathbf{y}_k - \mathbf{y}_l\rVert^2\big)^{-1}}.$$

The heavy tails are the key trick: they allow moderate distances in high dimensions to be mapped to **large** distances in 2D — this fixes the **crowding problem** (in 2D there is simply too little room for all the intermediate neighbours) and produces the characteristically separated clusters.

**Cost function and gradient:** minimize $\mathrm{KL}(P \Vert Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$ by gradient methods;

$$\frac{\partial\, \mathrm{KL}}{\partial \mathbf{y}_i} = 4\sum_{j \neq i} (p_{ij} - q_{ij})\,\big(1 + \lVert\mathbf{y}_i - \mathbf{y}_j\rVert^2\big)^{-1} (\mathbf{y}_i - \mathbf{y}_j)$$

— attractive where $p_{ij} > q_{ij}$, repulsive otherwise. The KL is asymmetric: **separating neighbours is expensive, putting strangers together is cheap** — t-SNE preserves local structure, not global structure.

**Traps of interpretation (exam relevant):** cluster sizes and the distances *between* clusters in t-SNE plots are not interpretable; the result depends on the perplexity and on the seed; t-SNE is not a mapping (new points cannot simply be projected). **UMAP** (McInnes et al., 2018) is the graph-based successor: faster, preserves global structure somewhat better, with similar caveats.
---

## Part 3 — Advanced

### 3.1 Universal approximation — what the theorem says (and what it does not)

**Theorem (Cybenko 1989; Hornik 1991).** Let $\sigma$ be continuous and non-polynomial (e.g. sigmoid, ReLU). Then the set of one-hidden-layer networks $\;f(\mathbf{x}) = \sum_{j=1}^m c_j\, \sigma(\mathbf{w}_j^\top \mathbf{x} + b_j)\;$ is **dense in $C(K)$** for every compact $K \subset \mathbb{R}^d$: for every continuous function $g$ and every $\varepsilon > 0$ there exist an $m$ and parameters with $\sup_{\mathbf{x} \in K} \lvert f(\mathbf{x}) - g(\mathbf{x})\rvert < \varepsilon$.

**What it does not say:** (i) nothing about the **size** $m$ — in the worst case $m$ grows exponentially in $d$; (ii) nothing about whether SGD **finds** those parameters (approximation error, optimization error and generalization error are three separate questions); (iii) nothing about data outside $K$. **Why depth?** There are families of functions that deep networks represent with $\mathrm{poly}(d)$ neurons while shallow ones need $\exp(d)$ neurons (depth separation results, e.g. Telgarsky 2016; the intuition: depth allows compositionality and the reuse of intermediate features; a ReLU network with $L$ layers and width $w$ can realize $O(w^L)$ linear regions, a shallow one only $O(w)$... more precisely $O(w^d)$, but the exponential advantage in $L$ remains).

### 3.2 Vanishing and exploding gradients — the Jacobian analysis

From the backprop recursion it follows for the gradient in layer $\ell$ of an $L$-layer network:

$$\boldsymbol{\delta}^{(\ell)} = \Bigg(\prod_{m=\ell+1}^{L} \mathrm{diag}\big(\sigma'(\mathbf{z}^{(m-1)})\big)\, {W^{(m)}}^\top \Bigg)\, \boldsymbol{\delta}^{(L)} \quad\text{(reading the order appropriately)},$$

a **product of $L - \ell$ Jacobians**. The norm of this product is controlled by the singular values: if the typical largest singular values of the layer Jacobians are $s < 1$, the signal shrinks like $s^{L-\ell}$ → **vanishing gradients** (the early layers learn nothing); if they are $> 1$, it grows exponentially → **exploding gradients** (divergence, NaNs). Sigmoid makes the problem structurally worse: $\sigma' \le 1/4$ multiplies into every layer.

**Countermeasures at a glance:** ReLU-like activations ($\sigma' = 1$ in the active region), He/Xavier initialization (singular values start around 1), BatchNorm/LayerNorm (rescaling per layer), **gradient clipping** against explosion ($\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, c/\lVert\mathbf{g}\rVert)$), and above all:

### 3.3 Residual connections (ResNet)

**The idea** (He et al., 2015): instead of $\mathbf{a}^{(\ell+1)} = F(\mathbf{a}^{(\ell)})$, learn the **residual**:

$$\mathbf{a}^{(\ell+1)} = \mathbf{a}^{(\ell)} + F\big(\mathbf{a}^{(\ell)}\big).$$

**Gradient analysis:** $\dfrac{\partial \mathbf{a}^{(\ell+1)}}{\partial \mathbf{a}^{(\ell)}} = I + \dfrac{\partial F}{\partial \mathbf{a}^{(\ell)}}$, hence over many blocks

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \prod_{m=\ell}^{L-1}\Big(I + \frac{\partial F_m}{\partial \mathbf{a}^{(m)}}\Big) = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}}\Big(I + \text{mixed terms}\Big).$$

The identity part guarantees an **undisturbed gradient path** all the way into the first layer — the product can no longer vanish as a whole. Furthermore: an additional block can represent the identity by letting $F \to 0$ (easy to learn, all weights → 0), i.e. deeper networks are never *worse* representable than shallower ones — and that was exactly what was empirically violated before ResNet (56-layer networks worse than 20-layer ones, **on the training data** — an optimization problem, not an overfitting problem). ResNets made networks with hundreds of layers trainable; skip connections are the standard in practically every architecture today (including transformers).

### 3.4 Autoencoders

An **autoencoder** learns a compression: encoder $\mathbf{h} = f_{\text{enc}}(\mathbf{x}) \in \mathbb{R}^q$ ($q \ll d$, the "bottleneck"), decoder $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{h})$, objective $\min \sum_i \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert^2$. **The connection to PCA:** if the encoder and decoder are linear, the optimal solution is exactly the PCA subspace (the bottleneck spans the top-$q$ eigenspace, though not necessarily orthonormally). Nonlinear autoencoders learn curved manifolds. Variants: the **denoising AE** (reconstruct $\mathbf{x}$ from a noisy $\tilde{\mathbf{x}}$ — this forces robust features), the **sparse AE** (L1 on $\mathbf{h}$).

### 3.5 Variational autoencoder (VAE) — the ELBO, in full

The VAE (Kingma & Welling, 2014) is a **generative** latent variable model: $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$, $\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})$ (a decoder network). The likelihood $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$ is intractable — the same structure as with the GMM (2.7), only with a continuous $\mathbf{z}$ and neural networks. And the same solution: a variational distribution $q_\phi(\mathbf{z} \mid \mathbf{x})$ (an encoder network, typically $\mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$) and the **evidence lower bound**:

$$
\log p_\theta(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big] - \mathrm{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\Vert\, p(\mathbf{z})\big)}_{=\;\mathrm{ELBO}(\theta, \phi;\, \mathbf{x})} + \underbrace{\mathrm{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\Vert\, p_\theta(\mathbf{z}\mid\mathbf{x})\big)}_{\ge\, 0}.
$$

*(Verify it as in 2.7: pull $\log p_\theta(\mathbf{x})$ out, since it does not depend on $\mathbf{z}$.)* One maximizes the ELBO jointly over $\theta$ **and** $\phi$: the reconstruction term wants good decoding, the KL term keeps the posterior approximation close to the prior (a regularization of the latent space — which is why one can sample from $\mathcal{N}(\mathbf{0}, I)$ and decode: a *generator*). So instead of an E-step (impossible, since $q$ is a network) the VAE performs **amortized variational inference** with gradient steps.

**The reparameterization trick.** $\nabla_\phi \mathbb{E}_{q_\phi}[\cdot]$ cannot be estimated directly by sampling (the distribution depends on $\phi$). The solution: write the sample as a deterministic function of $\phi$ and of external noise,

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I),$$

then $\mathbb{E}_{q_\phi}[g(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\varepsilon}}[g(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon})]$ and the gradient flows by backprop through $\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi$. **The closed-form KL** for two diagonal Gaussians (verifiable by integration):

$$\mathrm{KL}\big(\mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2)) \,\Vert\, \mathcal{N}(\mathbf{0}, I)\big) = \frac{1}{2}\sum_{j=1}^q \big(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\big).$$

### 3.6 Generalization in deep learning: double descent and implicit regularization

The classical learning curve (module 04) is U-shaped: more capacity → less bias, more variance → eventually overfitting. Deep networks break this picture: modern networks reach **zero training error** (they can even memorize random labels — Zhang et al., 2017) and generalize *anyway*. Empirically one observes **double descent** (Belkin et al., 2019): the test error rises up to the **interpolation threshold** (capacity ≈ just enough to fit the data exactly — there the model is maximally "forced" and often at its worst) and **falls again afterwards** as capacity keeps growing. An explanatory approach: in the overparameterized regime there are many interpolating solutions, and the optimizer systematically picks "smooth" ones — **implicit regularization**: GD on linear regression converges (when started at 0) to the **minimum-norm solution** among all interpolants; SGD noise additionally favours flat minima, which are more robust against parameter perturbations (and hence against distribution shift). The consequence for practice: "more parameters = overfitting" is dead as a rule of thumb; validation curves beat counting parameters.

### 3.7 Outlook: self-supervised learning

Between "supervised" and "unsupervised" lies the paradigm that dominates today: **self-supervised learning** generates learning signals from the data themselves — for example contrastive learning (SimCLR): two augmentations of the same image should get similar representations, different images dissimilar ones, using the **InfoNCE loss** $\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \ne i} \exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$ (similarity = cosine, temperature $\tau$). Masked modelling (BERT, MAE) is the counterpart: mask parts of the input and reconstruct them. The NLP modules (08–10) build on this centrally.

---

## Summary / cheat sheet

**Neural networks:**

| Concept | Key formula |
|---|---|
| Forward | $\mathbf{z}^{(\ell)} = W^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$, $\mathbf{a}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})$ |
| Softmax+CE gradient | $\nabla_{\mathbf{z}}\mathcal{L} = \mathbf{p} - \mathbf{y}$ |
| Backprop recursion | $\boldsymbol{\delta}^{(\ell)} = ({W^{(\ell+1)}}^\top \boldsymbol{\delta}^{(\ell+1)}) \odot \sigma'(\mathbf{z}^{(\ell)})$ |
| Gradients | $\nabla_{W^{(\ell)}}\mathcal{L} = \boldsymbol{\delta}^{(\ell)}{\mathbf{a}^{(\ell-1)}}^\top$, $\nabla_{\mathbf{b}^{(\ell)}}\mathcal{L} = \boldsymbol{\delta}^{(\ell)}$ |
| SGD+momentum | $\mathbf{v} \leftarrow \mu\mathbf{v} + \mathbf{g}$; $\theta \leftarrow \theta - \eta\mathbf{v}$ |
| Adam | $\hat{\mathbf{m}} = \frac{\mathbf{m}}{1-\beta_1^t}$, $\hat{\mathbf{v}} = \frac{\mathbf{v}}{1-\beta_2^t}$, $\theta \leftarrow \theta - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}$ |
| He init (ReLU) | $\sigma_W^2 = 2/d_{\text{in}}$; Xavier: $2/(d_{\text{in}}+d_{\text{out}})$ |
| Weight decay | $\theta \leftarrow (1-\eta\lambda)\theta - \eta\nabla\mathcal{L}$; decouple it with Adam (AdamW) |
| Dropout (training) | $\tilde{\mathbf{a}} = \mathbf{m}\odot\mathbf{a}/(1-p)$, $\mathbf{m}\sim\mathrm{Bern}(1-p)$ |
| BatchNorm | $\hat z = \frac{z - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}$, $y = \gamma\hat z + \beta$; inference: running averages |
| Conv output | $H_{\text{out}} = \lfloor (H + 2p - k)/s \rfloor + 1$; parameters: $C_{\text{out}}(C_{\text{in}}k_hk_w + 1)$ |
| ResNet | $\mathbf{a}^{(\ell+1)} = \mathbf{a}^{(\ell)} + F(\mathbf{a}^{(\ell)})$; Jacobian $I + \partial F$ |

**Unsupervised learning:**

| Concept | Key formula / key statement |
|---|---|
| k-means objective | $J = \sum_{i,k} r_{ik}\lVert\mathbf{x}_i - \boldsymbol{\mu}_k\rVert^2$; Lloyd = alternating minimization, monotone, local |
| GMM | $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\Sigma_k)$ |
| E-step | $\gamma_{ik} = \frac{\pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k,\Sigma_k)}{\sum_j \pi_j\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_j,\Sigma_j)}$ |
| M-step | $\pi_k = \frac{N_k}{n}$, $\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_i \gamma_{ik}\mathbf{x}_i$, $\Sigma_k = \frac{1}{N_k}\sum_i \gamma_{ik}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\cdot)^\top$ |
| EM guarantee | the likelihood increases monotonically (ELBO + KL decomposition); k-means = GMM with $\Sigma_k = \epsilon I$, $\epsilon\to 0$ |
| Silhouette | $s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$ |
| BIC | $-2\log\hat L + m\log n$ (smaller = better) |
| DBSCAN | core point: $\lvert N_\varepsilon\rvert \ge \textit{minPts}$; cluster = a maximal density-connected set |
| Ward | $\Delta J = \frac{\lvert A\rvert\lvert B\rvert}{\lvert A\rvert+\lvert B\rvert}\lVert\bar{\mathbf{a}} - \bar{\mathbf{b}}\rVert^2$ |
| PCA | $S\mathbf{w} = \lambda\mathbf{w}$; max. variance ≡ min. reconstruction; compute via SVD |
| t-SNE | $\min \mathrm{KL}(P\Vert Q)$, $Q$ with Student t(1); do not interpret distances between clusters! |
| VAE ELBO | $\mathbb{E}_q[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - \mathrm{KL}(q_\phi(\mathbf{z}\mid\mathbf{x})\Vert p(\mathbf{z}))$; reparameterization $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\varepsilon}$ |

---

## Self-test

Answer the questions yourself first, then unfold the answers.

<details><summary><b>1. Why is an MLP without activation functions equivalent to a linear model? Show it.</b></summary>

A composition of affine maps is affine: $W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$. Inductively this holds for arbitrarily many layers — the network collapses to *one* affine map and therefore cannot represent XOR, for instance. Only nonlinear $\sigma$ between the layers give expressive power.
</details>

<details><summary><b>2. Derive the gradient of softmax + cross-entropy with respect to the logits. Why is the result so pleasant, numerically and didactically?</b></summary>

With $p_k = \mathrm{softmax}(\mathbf{z})_k$ and one-hot $\mathbf{y}$: $\frac{\partial\mathcal{L}}{\partial z_j} = -\sum_k y_k \frac{1}{p_k} p_k(\delta_{kj} - p_j) = p_j - y_j$, hence $\nabla_{\mathbf{z}}\mathcal{L} = \mathbf{p} - \mathbf{y}$. Pleasant because (i) no Jacobian is needed explicitly, (ii) the gradient is bounded (no $1/p$ explosion — the saturation of the softmax and the log cancel exactly), (iii) the form "prediction minus target" makes the generalization of linear/logistic regression visible.
</details>

<details><summary><b>3. Explain the backprop recursion $\boldsymbol{\delta}^{(\ell)} = ({W^{(\ell+1)}}^\top\boldsymbol{\delta}^{(\ell+1)})\odot\sigma'(\mathbf{z}^{(\ell)})$ in words. Why does backprop cost only about 2 forward passes?</b></summary>

The error of a layer is the back-projected error of the following layer (through $W^\top$ — every neuron collects the errors of all neurons it contributes to, weighted by the connecting weights), damped by the local sensitivity $\sigma'$. The cost: the backward pass consists of the same matrix-vector products as the forward pass (only transposed) plus outer products for the gradients — asymptotically the same FLOPs, hence the factor of about 2. Naive numerical differentiation would need one forward pass *per parameter*.
</details>

<details><summary><b>4. Why is L2 regularization not the same as weight decay with Adam, but is the same with SGD?</b></summary>

With SGD: $\theta \leftarrow \theta - \eta(\nabla\mathcal{L} + \lambda\theta) = (1-\eta\lambda)\theta - \eta\nabla\mathcal{L}$ — the L2 gradient acts exactly as a multiplicative shrinkage. With Adam the total gradient (including $\lambda\theta$) is divided by $\sqrt{\hat{\mathbf{v}}}+\epsilon$: parameters with large gradients are regularized *more weakly* — the penalty is no longer proportional to $\theta$. AdamW decouples the decay term from the adaptive part: $\theta \leftarrow \theta - \eta(\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}}+\epsilon) + \lambda\theta)$.
</details>

<details><summary><b>5. Derive the He initialization. Why a factor of 2 instead of 1?</b></summary>

$\operatorname{Var}(z_i) = d_{\text{in}}\sigma_W^2\,\mathbb{E}[a_j^2]$ for independent centred weights. With ReLU, $a = \max(0, z)$ with $z$ symmetric around 0, so $\mathbb{E}[a^2] = \frac{1}{2}\mathbb{E}[z^2]$ — ReLU "deletes" half of the signal power. For the variance to stay constant across layers: $d_{\text{in}}\sigma_W^2 \cdot \frac{1}{2} = 1 \Rightarrow \sigma_W^2 = 2/d_{\text{in}}$.
</details>

<details><summary><b>6. Why do you have to call <code>model.eval()</code> in PyTorch before evaluating? Name the two mechanisms affected, with reasons.</b></summary>

(i) **Dropout**: during training activations are randomly zeroed (and rescaled by $1/(1-p)$); at inference time one wants the full deterministic network. (ii) **BatchNorm**: during training the layer normalizes with the batch statistics $\mu_B, \sigma_B^2$; at inference time the running averages must be used — otherwise the prediction for one image depends on the other images in the batch (and with batch size 1, $\sigma_B^2 = 0$).
</details>

<details><summary><b>7. Prove that Lloyd's algorithm terminates. Why is the result nevertheless not globally optimal?</b></summary>

Assignment step: every point moves at most to a closer centre → $J$ falls or stays equal. Update step: the centroid minimizes $\sum_i r_{ik}\lVert\mathbf{x}_i - \boldsymbol{\mu}\rVert^2$ (convex-quadratic in $\boldsymbol{\mu}$, gradient zero at the mean) → $J$ falls or stays equal. $J \ge 0$ and there are finitely many partitions; since $J$ falls monotonically, no partition can be visited twice (given a strict decrease) → termination in finitely many steps. It is not globally optimal because the procedure only improves locally, coordinate by coordinate — the exact problem is NP-hard; different starts give different local minima (hence k-means++ and `n_init`).
</details>

<details><summary><b>8. Sketch the EM derivation for GMMs: where does the lower bound come from, what do the E- and M-step do, and why does the likelihood increase monotonically?</b></summary>

Jensen (the log is concave) gives, for every $q$: $\log p(\mathbf{x}\mid\theta) \ge \mathbb{E}_q[\log\frac{p(\mathbf{x},z\mid\theta)}{q(z)}] = \mathcal{F}(q,\theta)$, with the gap $\mathrm{KL}(q \Vert p(z\mid\mathbf{x},\theta))$. E-step: $q = $ the posterior (the responsibilities $\gamma_{ik}$) → KL $= 0$, so the bound touches the likelihood. M-step: maximize $Q(\theta) = \sum_{i,k}\gamma_{ik}\log p(\mathbf{x}_i, k\mid\theta)$ in closed form ($\pi_k = N_k/n$, weighted means/covariances). Monotonicity: $\log p(\theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t)}) = \log p(\theta^{(t)})$.
</details>

<details><summary><b>9. Why does t-SNE use a Student t-distribution instead of a Gaussian in the low-dimensional space?</b></summary>

The crowding problem: in 2D there is not enough "room" to faithfully map all the intermediate distances of the high-dimensional space (the volume of a ball grows with the dimension; many roughly equidistant neighbours do not fit into a plane). The heavy tails of the t-distribution allow moderately distant points to lie *far* apart in the plot and still keep the appropriate similarity $q_{ij}$ — otherwise all points would be squeezed into the centre. A side effect: exaggeratedly clear cluster separation, which is why distances between clusters are not interpretable.
</details>

<details><summary><b>10. A fellow student says: "The network has zero training error, so it has overfitted and generalizes badly." Give a differentiated response.</b></summary>

Not necessarily. Modern overparameterized networks interpolate the training data routinely and still generalize well (double descent: beyond the interpolation threshold the test error *falls* again). The reason: among the many interpolating solutions, (S)GD implicitly picks regularized ones (a minimum-norm character, flat minima). What matters is the *validation* error, not the training error. That said: zero training error *plus a rising* validation error over the epochs is classical overfitting → early stopping.
</details>

---

## Literature and sources

**Textbooks:**
- I. Goodfellow, Y. Bengio, A. Courville: *Deep Learning* (MIT Press, 2016) — ch. 6 (MLP/backprop), 7 (regularization), 8 (optimization), 9 (CNNs), 14 (autoencoders). **Free online: deeplearningbook.org.** *(advanced, the standard reference)*
- C. Bishop: *Pattern Recognition and Machine Learning* (Springer, 2006) — ch. 9 (mixture models and EM, exactly our derivation), 12 (PCA, probabilistic PCA). *(advanced; PDF free from Microsoft Research)*
- K. Murphy: *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022) — a modern complete treatment. **Free online: probml.github.io.** *(intermediate to advanced)*
- A. Géron: *Hands-On Machine Learning* (O'Reilly, 3rd ed. 2022) — ch. 9 (clustering/GMM), 10–14 (neural nets in practice). *(beginner friendly)*

**Courses and lectures (free):**
- Stanford **CS231n** — CNNs for Visual Recognition (cs231n.github.io): the notes on backprop, initialization and CNNs are excellent. *(beginner friendly to intermediate)*
- **3Blue1Brown**, the neural networks series (YouTube) — the best visualization of backprop. *(beginner friendly)*
- Andrej Karpathy: *Neural Networks: Zero to Hero* (YouTube) — backprop from scratch in code, matching project 01 exactly. *(beginner friendly, highly recommended)*

**Key papers (all free on arXiv):**
- Kingma & Ba (2015): *Adam*. — Loshchilov & Hutter (2019): *Decoupled Weight Decay (AdamW)*.
- Srivastava et al. (2014): *Dropout*. — Ioffe & Szegedy (2015): *Batch Normalization*; alongside it Santurkar et al. (2018): *How Does BN Help Optimization?*
- He et al. (2015): *Delving Deep into Rectifiers* (He init) and *Deep Residual Learning* (ResNet).
- Dempster, Laird & Rubin (1977): *Maximum Likelihood from Incomplete Data via the EM Algorithm* (JRSS-B).
- Arthur & Vassilvitskii (2007): *k-means++*. — Ester et al. (1996): *DBSCAN* (KDD).
- van der Maaten & Hinton (2008): *Visualizing Data using t-SNE* (JMLR, free). — McInnes et al. (2018): *UMAP*.
- Kingma & Welling (2014): *Auto-Encoding Variational Bayes* (VAE).
- Zhang et al. (2017): *Understanding Deep Learning Requires Rethinking Generalization*. — Belkin et al. (2019): *Reconciling Modern ML Practice and the Bias-Variance Trade-off* (double descent).

**Interactive:**
- **playground.tensorflow.org** — train MLPs in the browser (activations, regularization, live). *(beginner friendly, free)*
- distill.pub: *How to Use t-SNE Effectively* — required reading before any t-SNE interpretation. *(free)*
- **poloclub.github.io/cnn-explainer** — the CNN forward pass, interactively. *(free)*

---

## The three projects

| Project | Topic | Format | Data |
|---|---|---|---|
| [01-basic](projects/01-basic/) | MLP + backprop **by hand** (NumPy only), gradient checking | Notebook | synthetic (`make_moons`) |
| [02-medium](projects/02-medium/) | CNN image classification in PyTorch, optimizer/regularization ablation | Notebook | Fashion-MNIST (real) |
| [03-final](projects/03-final/) | Customer segmentation: k-means vs. GMM/EM vs. DBSCAN vs. Ward, PCA/t-SNE, BIC and validation | Notebook | UCI Wholesale Customers (real) |

---
---

# Modul 05 — Machine Learning 2 (deutsche Fassung)

**Worum es geht:** Dieses Modul führt von den klassischen ML-Verfahren aus Modul 04 zu den beiden großen Themenblöcken des modernen maschinellen Lernens: **tiefe neuronale Netze** (Multilayer-Perzeptrons, Backpropagation, Optimierung, Regularisierung, CNNs) und **unüberwachtes Lernen** (Clustering, Mixture Models mit EM, Dimensionsreduktion). Alle zentralen Verfahren werden vollständig hergeleitet — nicht nur benutzt.

**Hilfreiche Vorkenntnisse:** Lineare Algebra (Matrizen, Eigenwerte), Analysis (partielle Ableitungen, Kettenregel), Wahrscheinlichkeitsrechnung (Dichten, Erwartungswert, Maximum Likelihood). Module 02–04 sollten durchgearbeitet sein; insbesondere baut vieles auf logistischer Regression und dem Bias-Varianz-Verständnis aus Modul 04 auf.

---

## Lernziele

Nach diesem Modul kannst du:

- ein Multilayer-Perzeptron (MLP) formal definieren und **Backpropagation vollständig herleiten** (inkl. Matrixform und Komplexitätsanalyse) — und von Hand implementieren.
- Verlustfunktionen aus dem **Maximum-Likelihood-Prinzip** ableiten (MSE ↔ Gauß-Rauschen, Cross-Entropy ↔ kategoriale Verteilung).
- die wichtigsten Optimierer (SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam/AdamW) mit ihren exakten Update-Gleichungen erklären, inkl. Bias-Korrektur bei Adam.
- Initialisierung (Xavier/He) über Varianz-Propagation herleiten und begründen.
- Regularisierungstechniken (L2/Weight Decay, Dropout, Early Stopping, Batch Normalization, Datenaugmentierung) mathematisch präzise beschreiben und wissen, wann welche greift.
- Convolutional Neural Networks formal definieren (Faltung, Padding/Stride-Arithmetik, rezeptives Feld, Parameterzählung) und den Backprop-Durchgang durch eine Faltungsschicht angeben.
- k-Means als alternierende Minimierung verstehen (inkl. Monotonie-Beweis), **EM für Gaussian Mixture Models vollständig herleiten** (ELBO, E-Step, M-Step) und die Verbindung k-Means ↔ GMM erklären.
- hierarchisches Clustering (Linkage-Kriterien, Ward) und DBSCAN (Dichtebegriffe) präzise definieren und Clusterlösungen validieren (Silhouette, ARI, BIC).
- PCA auf zwei Wegen herleiten (Varianzmaximierung via Lagrange; Minimierung des Rekonstruktionsfehlers) und die SVD-Verbindung angeben; t-SNE inkl. Kostenfunktion und Gradient erklären.
- Advanced-Themen einordnen: Universal Approximation, Vanishing/Exploding Gradients (Jacobian-Analyse), Residual Connections, Double Descent, Autoencoder und **VAE mit vollständiger ELBO-Herleitung**.

---

## Teil 1 — Grundlagen (Basics)

### 1.1 Vom linearen Modell zum neuronalen Netz

Modul 04 endete bei Modellen der Form

$$\hat{y} = f(\mathbf{w}^\top \phi(\mathbf{x}) + b),$$

wobei die Feature-Abbildung $\phi$ **von Hand** gewählt wurde (Polynome, Interaktionen, TF-IDF, …). Die zentrale Idee neuronaler Netze: **lerne $\phi$ mit.** Ein neuronales Netz ist eine Verkettung parametrisierter, differenzierbarer Funktionen, deren Parameter gemeinsam per Gradientenverfahren optimiert werden.

**Warum Nichtlinearität zwingend ist:** Verkettet man nur affine Abbildungen, gilt
$W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$ —
das Ergebnis ist wieder affin. Erst eine nichtlineare **Aktivierungsfunktion** zwischen den Schichten erzeugt echte Ausdrucksmächtigkeit. Das klassische Gegenbeispiel für ein einzelnes Perzeptron ist XOR: Die vier Punkte $(0,0)\mapsto 0$, $(1,1)\mapsto 0$, $(0,1)\mapsto 1$, $(1,0)\mapsto 1$ sind nicht linear separierbar; ein MLP mit einer verborgenen Schicht aus zwei Neuronen löst XOR exakt.

### 1.2 Das Multilayer-Perzeptron (MLP), formal

Ein MLP mit $L$ Schichten ist die Funktion $f_\theta : \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$,

$$
\mathbf{a}^{(0)} = \mathbf{x}, \qquad
\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad
\mathbf{a}^{(\ell)} = \sigma^{(\ell)}\!\big(\mathbf{z}^{(\ell)}\big), \qquad \ell = 1,\dots,L,
$$

mit Gewichtsmatrizen $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$, Bias-Vektoren $\mathbf{b}^{(\ell)} \in \mathbb{R}^{d_\ell}$ und elementweisen Aktivierungen $\sigma^{(\ell)}$. Die Größen heißen: $\mathbf{z}^{(\ell)}$ **Präaktivierung**, $\mathbf{a}^{(\ell)}$ **Aktivierung**, $\theta = \{W^{(\ell)}, \mathbf{b}^{(\ell)}\}_{\ell=1}^L$ die Parameter. Parameterzahl: $\sum_{\ell=1}^L d_\ell (d_{\ell-1} + 1)$.

**Gängige Aktivierungsfunktionen** (mit Ableitungen — die brauchen wir in 1.5):

| Name | $\sigma(z)$ | $\sigma'(z)$ | Eigenschaften |
|---|---|---|---|
| Sigmoid | $\dfrac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | Ausgabe in $(0,1)$; sättigt: $\sigma' \le 1/4$, für $\lvert z\rvert$ groß $\approx 0$ |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | nullzentriert; sättigt ebenfalls |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | keine Sättigung für $z>0$; „dying ReLU“ bei $z<0$ dauerhaft möglich |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \approx 0{,}01$ | $\mathbb{1}[z>0] + \alpha\,\mathbb{1}[z\le 0]$ | behebt dying ReLU |
| GELU | $z\,\Phi(z)$ ($\Phi$ = Standardnormal-CDF) | $\Phi(z) + z\,\varphi(z)$ | glatt; Standard in Transformern |
| Softmax (Ausgabe) | $\mathrm{softmax}(\mathbf{z})_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$ | Jacobian: $\dfrac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$ | wandelt Logits in Wahrscheinlichkeitsvektor |

ReLU ist in $z=0$ nicht differenzierbar; in der Praxis wählt man ein Element des Subgradienten (üblich: $0$). Das ist theoretisch sauber über **Subgradienten-Methoden** begründbar und praktisch irrelevant, weil $z=0$ Maß null hat.

### 1.3 Verlustfunktionen aus Maximum Likelihood

Verlustfunktionen fallen nicht vom Himmel — sie folgen aus einem probabilistischen Modell der Daten. Sei $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ i.i.d. Wir modellieren $p_\theta(y \mid \mathbf{x})$ und maximieren die Log-Likelihood, äquivalent: minimieren die **negative Log-Likelihood (NLL)**

$$\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(y_i \mid \mathbf{x}_i).$$

**Fall 1 — Regression mit Gauß-Rauschen.** Annahme $y = f_\theta(\mathbf{x}) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, also $p_\theta(y\mid\mathbf{x}) = \mathcal{N}(y;\, f_\theta(\mathbf{x}), \sigma^2)$. Dann

$$
-\log p_\theta(y_i \mid \mathbf{x}_i)
= \frac{(y_i - f_\theta(\mathbf{x}_i))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2),
$$

und da der zweite Term nicht von $\theta$ abhängt, ist NLL-Minimierung **äquivalent zur Minimierung des mittleren quadratischen Fehlers (MSE)**. Laplace-Rauschen ergäbe analog den MAE — die Wahl der Loss ist eine Rauschannahme.

**Fall 2 — Klassifikation mit $K$ Klassen.** Das Netz gibt Logits $\mathbf{z} \in \mathbb{R}^K$ aus, $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$ modelliert die kategoriale Verteilung $p_\theta(y = k \mid \mathbf{x}) = p_k$. Die NLL ist die **Cross-Entropy**

$$\mathcal{L}_i = -\log p_{y_i} = -\sum_{k=1}^K \mathbb{1}[y_i = k] \log p_k.$$

**Wichtige Rechnung (Softmax + Cross-Entropy):** Der Gradient bezüglich der Logits ist bemerkenswert einfach. Mit One-Hot-Vektor $\mathbf{y}$:

$$
\frac{\partial \mathcal{L}_i}{\partial z_j}
= -\sum_k y_k \frac{1}{p_k}\frac{\partial p_k}{\partial z_j}
= -\sum_k y_k \frac{1}{p_k} p_k(\delta_{kj} - p_j)
= -\big(y_j - p_j \sum_k y_k\big)
= p_j - y_j,
$$

also vektoriell $\nabla_{\mathbf{z}} \mathcal{L}_i = \mathbf{p} - \mathbf{y}$ — „Vorhersage minus Wahrheit“. Genau dieselbe Form hatte der Gradient der logistischen Regression in Modul 04; das MLP ersetzt nur die Eingabe der letzten Schicht durch gelernte Features.

**Numerik-Hinweis:** Softmax und Log werden nie getrennt berechnet. Man nutzt den **Log-Sum-Exp-Trick**: $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ mit $m = \max_j z_j$, um Überlauf zu vermeiden (in PyTorch erledigt `nn.CrossEntropyLoss` das intern und erwartet rohe Logits).

### 1.4 Gradientenabstieg (Gradient Descent)

Das Lernproblem ist $\min_\theta \mathcal{L}(\theta)$ für eine nichtkonvexe, differenzierbare Funktion. **Gradient Descent (GD)** iteriert

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

mit **Lernrate** $\eta > 0$. Begründung: Die Taylor-Entwicklung erster Ordnung $\mathcal{L}(\theta + \Delta) \approx \mathcal{L}(\theta) + \nabla\mathcal{L}^\top \Delta$ wird unter der Nebenbedingung $\lVert\Delta\rVert \le \varepsilon$ durch $\Delta \propto -\nabla\mathcal{L}$ minimiert — der negative Gradient ist die Richtung des steilsten Abstiegs (bzgl. der euklidischen Norm).

Für $L$-glatte Funktionen (d. h. $\lVert \nabla\mathcal{L}(\theta) - \nabla\mathcal{L}(\theta')\rVert \le L_s \lVert \theta - \theta' \rVert$) garantiert $\eta \le 1/L_s$ monotonen Abstieg, denn aus der Glattheit folgt die **Descent-Ungleichung**

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta\Big(1 - \frac{L_s \eta}{2}\Big)\lVert\nabla\mathcal{L}(\theta_t)\rVert^2 .$$

Bei nichtkonvexen Problemen konvergiert GD damit gegen stationäre Punkte ($\nabla\mathcal{L}=0$), nicht notwendig globale Minima. Empirisch sind bei überparametrisierten Netzen fast alle erreichten lokalen Minima gut — mehr dazu in Teil 3.6.

### 1.5 Backpropagation — die vollständige Herleitung

Backpropagation ist **kein eigenes Lernverfahren**, sondern ein Algorithmus, der $\nabla_\theta \mathcal{L}$ für geschichtete Funktionen effizient berechnet: eine systematische Anwendung der mehrdimensionalen Kettenregel mit geschickter Auswertungsreihenfolge (Spezialfall des **Reverse-Mode Automatic Differentiation**).

**Setup.** Betrachte ein einzelnes Trainingsbeispiel mit Verlust $\mathcal{L} = \ell(\mathbf{a}^{(L)}, \mathbf{y})$ und die Vorwärtsgleichungen aus 1.2. Definiere das **Fehlersignal** der Schicht $\ell$:

$$\boldsymbol{\delta}^{(\ell)} \;:=\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}} \in \mathbb{R}^{d_\ell}.$$

**Schritt 1 — Ausgabeschicht.** Nach Kettenregel durch die elementweise Aktivierung:

$$\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \odot \sigma'^{(L)}\big(\mathbf{z}^{(L)}\big),$$

wobei $\odot$ das elementweise (Hadamard-)Produkt ist. Für Softmax + Cross-Entropy kollabiert das wie in 1.3 gezeigt zu $\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}$ (hier ist die „Aktivierung“ Softmax nicht elementweise; man rechnet direkt mit dem Jacobian, und genau deshalb ist das Ergebnis so einfach).

**Schritt 2 — Rekursion über die Schichten.** $\mathcal{L}$ hängt von $\mathbf{z}^{(\ell)}$ nur über $\mathbf{z}^{(\ell+1)} = W^{(\ell+1)}\sigma(\mathbf{z}^{(\ell)}) + \mathbf{b}^{(\ell+1)}$ ab. Die Kettenregel in Komponenten:

$$
\delta_i^{(\ell)}
= \sum_j \frac{\partial \mathcal{L}}{\partial z_j^{(\ell+1)}} \frac{\partial z_j^{(\ell+1)}}{\partial z_i^{(\ell)}}
= \sum_j \delta_j^{(\ell+1)} \, W_{ji}^{(\ell+1)} \, \sigma'\big(z_i^{(\ell)}\big),
$$

denn $z_j^{(\ell+1)} = \sum_i W_{ji}^{(\ell+1)} \sigma(z_i^{(\ell)}) + b_j^{(\ell+1)}$. In Matrixform:

$$\boxed{\;\boldsymbol{\delta}^{(\ell)} = \Big( {W^{(\ell+1)}}^\top \boldsymbol{\delta}^{(\ell+1)} \Big) \odot \sigma'\big(\mathbf{z}^{(\ell)}\big)\;}$$

Das Fehlersignal fließt also **rückwärts** durch die transponierten Gewichtsmatrizen — daher der Name.

**Schritt 3 — Parametergradienten.** $\mathcal{L}$ hängt von $W_{ij}^{(\ell)}$ nur über $z_i^{(\ell)}$ ab, und $\partial z_i^{(\ell)} / \partial W_{ij}^{(\ell)} = a_j^{(\ell-1)}$. Also

$$
\frac{\partial \mathcal{L}}{\partial W_{ij}^{(\ell)}} = \delta_i^{(\ell)} \, a_j^{(\ell-1)}
\quad\Longleftrightarrow\quad
\boxed{\;\nabla_{W^{(\ell)}} \mathcal{L} = \boldsymbol{\delta}^{(\ell)} {\mathbf{a}^{(\ell-1)}}^\top, \qquad \nabla_{\mathbf{b}^{(\ell)}} \mathcal{L} = \boldsymbol{\delta}^{(\ell)}\;}
$$

— ein äußeres Produkt: „Fehler der Schicht mal Eingang der Schicht“.

**Der Algorithmus** (für einen Minibatch mittelt man die Gradienten der Beispiele; in Matrixschreibweise mit Batch-Matrix $A^{(\ell)} \in \mathbb{R}^{d_\ell \times B}$ werden die äußeren Produkte zu Matrixprodukten):

1. **Forward Pass:** berechne und **speichere** $\mathbf{z}^{(\ell)}, \mathbf{a}^{(\ell)}$ für alle $\ell$.
2. **Backward Pass:** $\boldsymbol{\delta}^{(L)}$ aus Schritt 1; dann für $\ell = L-1, \dots, 1$ die Rekursion aus Schritt 2; nebenbei die Gradienten aus Schritt 3.
3. **Update:** ein Schritt des gewählten Optimierers (Teil 2.1).

**Komplexität.** Forward und Backward kosten beide $O\big(\sum_\ell d_\ell d_{\ell-1}\big)$ pro Beispiel — Backprop berechnet den *vollständigen* Gradienten bzgl. *aller* Parameter zum Preis von grob **zwei Forward-Passes**. Zum Vergleich: numerische Differenziation bräuchte $O(\lvert\theta\rvert)$ Forward-Passes. Der Preis ist Speicher: alle Zwischenaktivierungen müssen für den Backward Pass gehalten werden, $O\big(B \sum_\ell d_\ell\big)$.

**Warum Reverse Mode?** Forward-Mode AD propagiert Richtungsableitungen $\partial(\cdot)/\partial\theta_j$ vorwärts — pro Parameter ein Durchlauf, gut bei *wenigen Eingängen, vielen Ausgängen*. Reverse Mode propagiert $\partial\mathcal{L}/\partial(\cdot)$ rückwärts — ein Durchlauf für *alle* Parameter, gut bei *vielen Eingängen, einem skalaren Ausgang*. Training ist genau der zweite Fall.

### 1.6 Durchgerechnetes Zahlenbeispiel (ein kompletter Backprop-Schritt)

Netz: 2 Eingänge → 2 verborgene Neuronen (Sigmoid) → 1 Ausgang (Sigmoid), Verlust binäre Cross-Entropy $\mathcal{L} = -[y\log\hat y + (1-y)\log(1-\hat y)]$.

Parameter: $W^{(1)} = \begin{pmatrix} 0.5 & -0.3 \\ 0.8 & 0.2 \end{pmatrix}$, $\mathbf{b}^{(1)} = \mathbf{0}$, $W^{(2)} = (1.0,\; -1.0)$, $b^{(2)} = 0$. Eingabe $\mathbf{x} = (1, 2)^\top$, Ziel $y = 1$.

*Forward:*
- $\mathbf{z}^{(1)} = (0.5 - 0.6,\; 0.8 + 0.4)^\top = (-0.1,\; 1.2)^\top$
- $\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)}) \approx (0.4750,\; 0.7685)^\top$
- $z^{(2)} = 1.0\cdot 0.4750 - 1.0 \cdot 0.7685 = -0.2935$, $\hat y = \sigma(-0.2935) \approx 0.4271$
- $\mathcal{L} = -\log 0.4271 \approx 0.8508$

*Backward:*
- Für Sigmoid + BCE gilt (gleiche Rechnung wie Softmax+CE): $\delta^{(2)} = \hat y - y = -0.5729$.
- $\nabla_{W^{(2)}}\mathcal{L} = \delta^{(2)} {\mathbf{a}^{(1)}}^\top \approx (-0.2721,\; -0.4403)$, $\nabla_{b^{(2)}}\mathcal{L} = -0.5729$.
- $\boldsymbol{\delta}^{(1)} = \big({W^{(2)}}^\top \delta^{(2)}\big) \odot \mathbf{a}^{(1)}\odot(1-\mathbf{a}^{(1)}) \approx \begin{pmatrix}-0.5729 \\ +0.5729\end{pmatrix} \odot \begin{pmatrix}0.2494 \\ 0.1779\end{pmatrix} \approx \begin{pmatrix}-0.1429 \\ 0.1019\end{pmatrix}$
- $\nabla_{W^{(1)}}\mathcal{L} = \boldsymbol{\delta}^{(1)}\mathbf{x}^\top \approx \begin{pmatrix} -0.1429 & -0.2858 \\ 0.1019 & 0.2038 \end{pmatrix}$

*Update* mit $\eta = 0.5$: z. B. $W^{(2)} \leftarrow (1.0,\,-1.0) - 0.5\,(-0.2721,\,-0.4403) = (1.1361,\; -0.7799)$. Ein erneuter Forward Pass ergäbe $\hat y \approx 0.55$ — die Vorhersage bewegt sich Richtung $y=1$. Genau diese Rechnung implementierst du in **Projekt 01** von Hand und verifizierst sie per Gradient Checking.

**Gradient Checking:** Vergleiche den analytischen Gradienten mit der **zentralen Differenz** $\frac{\partial\mathcal{L}}{\partial\theta_j} \approx \frac{\mathcal{L}(\theta + \epsilon\mathbf{e}_j) - \mathcal{L}(\theta - \epsilon\mathbf{e}_j)}{2\epsilon}$ (Fehler $O(\epsilon^2)$, während die einseitige Differenz nur $O(\epsilon)$ erreicht), $\epsilon \approx 10^{-5}$, und miss die relative Abweichung $\frac{\lVert g_{\text{ana}} - g_{\text{num}}\rVert}{\lVert g_{\text{ana}}\rVert + \lVert g_{\text{num}}\rVert} \lesssim 10^{-7}$ (in float64).

---

## Teil 2 — Aufbau (Intermediate)

### 2.1 Stochastische Optimierung: von SGD bis AdamW

**Minibatch-SGD.** Der volle Gradient kostet $O(n)$ pro Schritt. Stattdessen zieht man einen Minibatch $\mathcal{B}_t$ ($|\mathcal{B}_t| = B \ll n$) und nutzt

$$\theta_{t+1} = \theta_t - \eta_t \, \mathbf{g}_t, \qquad \mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla_\theta \mathcal{L}_i(\theta_t).$$

$\mathbf{g}_t$ ist ein **erwartungstreuer Schätzer** des vollen Gradienten: $\mathbb{E}[\mathbf{g}_t] = \nabla\mathcal{L}(\theta_t)$, mit Kovarianz $\propto 1/B$. Das Rauschen ist nicht nur Kostenersparnis — es hilft, Sattelpunkte zu verlassen, und wirkt implizit regularisierend (Teil 3.6). Klassisches Konvergenzresultat (Robbins & Monro, 1951): Für konvexe Ziele konvergiert SGD, wenn $\sum_t \eta_t = \infty$ und $\sum_t \eta_t^2 < \infty$ (z. B. $\eta_t \propto 1/t$); bei konstanter Lernrate konvergiert SGD nur in eine **Rauschumgebung** des Optimums, deren Radius mit $\eta$ und der Gradientenvarianz wächst.

**Momentum (Polyak, „Heavy Ball“).** Führe eine exponentiell gewichtete Geschwindigkeit mit:

$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \mathbf{g}_t, \qquad \theta_{t+1} = \theta_t - \eta\, \mathbf{v}_{t+1}, \qquad \mu \in [0,1),\ \text{typisch } 0.9.$$

Ausgerollt: $\mathbf{v}_{t+1} = \sum_{s=0}^{t} \mu^{s}\, \mathbf{g}_{t-s}$ — ein gleitender Mittelwert der Gradienten mit effektivem Horizont $\approx 1/(1-\mu)$. Wirkung: In Richtungen konsistenten Gradienten (flache Talrichtung) akkumuliert sich der Schritt bis auf das $\frac{1}{1-\mu}$-fache; in oszillierenden Richtungen (steile Talwände) mitteln sich die Beiträge weg. Für quadratische Ziele mit Konditionszahl $\kappa$ verbessert Momentum die Konvergenzrate von $O(\kappa)$ auf $O(\sqrt{\kappa})$ Iterationen.

**Nesterov Accelerated Gradient (NAG):** wie Momentum, aber der Gradient wird am **vorausgeschauten** Punkt ausgewertet: $\mathbf{v}_{t+1} = \mu\mathbf{v}_t + \nabla\mathcal{L}(\theta_t - \eta\mu\mathbf{v}_t)$, $\theta_{t+1} = \theta_t - \eta\mathbf{v}_{t+1}$ — eine Korrektur, die Überschießen früher bremst.

**AdaGrad** (Duchi et al., 2011) skaliert die Lernrate pro Parameter mit der Historie der quadrierten Gradienten:

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^{\odot 2}, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \odot \mathbf{g}_t.$$

Selten aktive Parameter (kleine akkumulierte Quadrate) bekommen große Schritte — gut für sparse Features. Problem: $\mathbf{s}_t$ wächst monoton, die effektive Lernrate stirbt.

**RMSProp** (Hinton) ersetzt die Summe durch einen exponentiellen Mittelwert: $\mathbf{s}_t = \rho\, \mathbf{s}_{t-1} + (1-\rho)\, \mathbf{g}_t^{\odot 2}$ (typisch $\rho = 0.99$), Update wie AdaGrad — die Lernrate stirbt nicht mehr.

**Adam** (Kingma & Ba, 2015) kombiniert Momentum (1. Moment) und RMSProp (2. Moment) mit **Bias-Korrektur**:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t, &\qquad \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t},\\[2pt]
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^{\odot 2}, &\qquad \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t},\\[2pt]
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}. & &
\end{aligned}
$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

*Herleitung der Bias-Korrektur:* Mit $\mathbf{m}_0 = \mathbf{0}$ gilt ausgerollt $\mathbf{m}_t = (1-\beta_1)\sum_{s=1}^{t} \beta_1^{t-s}\, \mathbf{g}_s$. Unter der (idealisierenden) Annahme stationärer Gradienten $\mathbb{E}[\mathbf{g}_s] = \mathbf{g}$ folgt $\mathbb{E}[\mathbf{m}_t] = (1-\beta_1)\,\mathbf{g}\sum_{s=1}^{t}\beta_1^{t-s} = \mathbf{g}\,(1 - \beta_1^t)$ (geometrische Summe). Der Schätzer ist also um den Faktor $(1-\beta_1^t)$ zur Null hin verzerrt — gerade am Anfang massiv ($t$ klein, $\beta_2^t \approx 1$ macht es beim 2. Moment noch schlimmer). Division durch $(1-\beta^t)$ macht ihn erwartungstreu.

**AdamW** (Loshchilov & Hutter, 2019): Bei Adam ist L2-Regularisierung **nicht** äquivalent zu Weight Decay (siehe 2.3) — der L2-Gradient wird mit durch $\sqrt{\hat{\mathbf{v}}_t}$ geteilt und dadurch für Parameter mit großen Gradienten abgeschwächt. AdamW entkoppelt: $\theta_{t+1} = \theta_t - \eta\big(\hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) + \lambda \theta_t\big)$. Heute der Standard-Optimierer.

**Lernraten-Schedules.** Konstantes $\eta$ ist selten optimal. Üblich: **Step Decay** ($\eta$ alle $k$ Epochen mal $0.1$), **Cosine Annealing** $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1 + \cos(\pi t/T))$, und **Warmup** (linear von 0 auf $\eta_{\max}$ über die ersten Schritte — stabilisiert Adam früh, wenn $\hat{\mathbf{v}}_t$ noch unzuverlässig ist).

### 2.2 Initialisierung: Xavier und He, hergeleitet

Alle Gewichte mit 0 zu initialisieren ist fatal: Alle Neuronen einer Schicht erhalten identische Gradienten und bleiben für immer identisch (**Symmetrieproblem**). Aber auch die Skala zufälliger Initialisierung ist kritisch.

**Varianz-Propagation.** Betrachte $z_i = \sum_{j=1}^{d_{\text{in}}} W_{ij} a_j$ mit unabhängigen, zentrierten $W_{ij}$ (Varianz $\sigma_W^2$) und Aktivierungen $a_j$ (Varianz $\sigma_a^2$, zentriert). Dann

$$\operatorname{Var}(z_i) = \sum_{j=1}^{d_{\text{in}}} \operatorname{Var}(W_{ij} a_j) = d_{\text{in}}\, \sigma_W^2 \sigma_a^2.$$

Damit die Signalvarianz über die Schichten weder explodiert noch verschwindet, fordert man $d_{\text{in}} \sigma_W^2 = 1$. Dieselbe Rechnung für das rückwärtsfließende Fehlersignal (durch $W^\top$) fordert $d_{\text{out}} \sigma_W^2 = 1$. **Xavier/Glorot-Initialisierung** kompromittiert:

$$\sigma_W^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}} \qquad \text{(für tanh-artige, um 0 lineare Aktivierungen)}.$$

**He-Initialisierung** für ReLU: ReLU nullt die Hälfte der Eingänge. Für $z$ symmetrisch um 0 gilt $\mathbb{E}[\mathrm{ReLU}(z)^2] = \tfrac12 \mathbb{E}[z^2]$, die Varianz halbiert sich pro Schicht, also kompensiert man mit Faktor 2:

$$\sigma_W^2 = \frac{2}{d_{\text{in}}}.$$

Faustregel: **He für ReLU-Familie, Xavier für tanh/sigmoid** — PyTorch-Layer bringen sinnvolle Defaults mit, aber du solltest wissen, warum.

### 2.3 Regularisierung

Tiefe Netze haben oft mehr Parameter als Datenpunkte — Kontrolle der Kapazität entscheidet über Generalisierung.

**L2-Regularisierung / Weight Decay.** Ziel: $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\lVert\theta\rVert_2^2$. Der GD-Schritt wird

$$\theta_{t+1} = \theta_t - \eta\big(\nabla\mathcal{L}(\theta_t) + \lambda\theta_t\big) = (1 - \eta\lambda)\,\theta_t - \eta\nabla\mathcal{L}(\theta_t),$$

d. h. die Gewichte werden pro Schritt multiplikativ geschrumpft („Decay“). Für **plain SGD** sind L2-Loss-Term und Weight Decay also identisch; für **adaptive Verfahren (Adam) nicht** — daher AdamW. Bayesianisch entspricht L2 einem Gauß-Prior $\theta \sim \mathcal{N}(0, \lambda^{-1} I)$ und die Minimierung dem MAP-Schätzer.

**Dropout** (Srivastava et al., 2014). Im Training wird jede Aktivierung unabhängig mit Wahrscheinlichkeit $p$ genullt, mit einer Bernoulli-Maske $\mathbf{m} \sim \mathrm{Bernoulli}(1-p)^{d}$:

$$\tilde{\mathbf{a}} = \frac{\mathbf{m} \odot \mathbf{a}}{1-p} \qquad \text{(„inverted dropout“)}.$$

Die Division durch $1-p$ hält den Erwartungswert konstant: $\mathbb{E}[\tilde a_i] = \frac{(1-p)\,a_i}{1-p} = a_i$ — deshalb kann man zur Inferenz einfach das volle Netz ohne Maske nutzen. Interpretationen: (i) Training eines impliziten **Ensembles** aus $2^d$ geteilten Subnetzen, deren Vorhersagen zur Testzeit approximativ gemittelt werden; (ii) Verhindern von **Co-Adaptation** — kein Neuron kann sich auf ein bestimmtes anderes verlassen; (iii) für lineare Regression ist Dropout äquivalent zu einer datenabhängigen L2-Strafe.

**Early Stopping.** Beobachte den Validierungsverlust und stoppe (bzw. behalte den besten Checkpoint), wenn er sich $k$ Epochen („Patience“) nicht verbessert. Für lineare Modelle mit quadratischem Verlust ist Early Stopping nachweislich eng mit L2-Regularisierung verwandt: GD lernt zuerst die Richtungen großer Eigenwerte der Datenkovarianz; frühes Stoppen kappt die kleinen Eigenrichtungen — genau wie Ridge mit $\lambda \approx 1/(\eta t)$.

**Datenaugmentierung** kodiert Invarianzen (Bild: Spiegelung, Verschiebung, Ausschnitt; Audio: Zeitverschiebung), erweitert effektiv den Datensatz und ist oft die stärkste einzelne Maßnahme — sie wirkt auf die Daten-, nicht die Parameterseite.

### 2.4 Batch Normalization

**Definition** (Ioffe & Szegedy, 2015). Für jede Feature-Dimension $k$ eines Minibatches $\{z_{i,k}\}_{i=1}^B$:

$$
\mu_k = \frac{1}{B}\sum_{i=1}^B z_{i,k}, \qquad
\sigma_k^2 = \frac{1}{B}\sum_{i=1}^B (z_{i,k}-\mu_k)^2, \qquad
\hat z_{i,k} = \frac{z_{i,k} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}, \qquad
y_{i,k} = \gamma_k \hat z_{i,k} + \beta_k,
$$

mit **lernbaren** Skalen-/Shift-Parametern $\gamma_k, \beta_k$ (die Normalisierung soll Ausdrucksmächtigkeit nicht einschränken — mit $\gamma_k = \sigma_k$, $\beta_k = \mu_k$ wäre sie die Identität). **Zur Inferenz** gibt es keinen Batch: Man nutzt während des Trainings mitgeführte gleitende Mittel $\bar\mu_k, \bar\sigma_k^2$ (in PyTorch: deshalb ist `model.eval()` vor der Evaluation zwingend!).

**Warum es hilft:** Die ursprüngliche Erklärung („Internal Covariate Shift“) gilt als überholt; die heute akzeptierte Sicht (Santurkar et al., 2018): BatchNorm **glättet die Verlustlandschaft** (kleinere Lipschitz-Konstanten von Verlust und Gradient), erlaubt größere Lernraten und macht das Training robust gegen die Initialisierungsskala, weil die Ausgabe invariant gegen Skalierung der Vorschicht-Gewichte ist. Nebenwirkung: das Batch-Rauschen in $\mu_k, \sigma_k^2$ regularisiert leicht. Bei kleinen Batches oder Sequenzmodellen nutzt man stattdessen **Layer Normalization** (Normalisierung über die Feature-Dimension pro Beispiel — batchunabhängig, Standard in Transformern).

### 2.5 Convolutional Neural Networks (CNNs)

Vollverbundene Schichten auf Bildern verschwenden Parameter (ein $224{\times}224{\times}3$-Bild → erste Schicht mit 1000 Neuronen hätte 150 Mio. Gewichte) und ignorieren die räumliche Struktur. CNNs bauen zwei **Induktive Biases** ein: **Lokalität** (Pixel korrelieren mit Nachbarn) und **Translationsäquivarianz** (ein Kantendetektor ist überall derselbe → **Weight Sharing**).

**Die Faltungsschicht, formal.** Eingabe $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$, Kernel $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$, Ausgabe

$$
Y_{c',\,i,\,j} \;=\; b_{c'} + \sum_{c=1}^{C_{\text{in}}} \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} K_{c',c,u,v}\; X_{c,\; i\cdot s + u - p,\; j\cdot s + v - p}
$$

mit Stride $s$ und Padding $p$ (Indizes außerhalb: 0). Streng genommen ist das eine **Kreuzkorrelation** (Faltung würde den Kernel spiegeln); da die Kernel gelernt werden, ist der Unterschied irrelevant, und alle Frameworks implementieren es so.

**Output-Größe:** $H_{\text{out}} = \left\lfloor \dfrac{H + 2p - k_h}{s} \right\rfloor + 1$ (analog $W_{\text{out}}$). „Same“-Padding: $p = (k-1)/2$ bei $s=1$ hält die Auflösung.

**Parameterzahl:** $C_{\text{out}}(C_{\text{in}} k_h k_w + 1)$ — unabhängig von $H, W$. Beispiel: $3{\times}3$-Conv von 64 auf 128 Kanäle: $128\cdot(64\cdot 9 + 1) = 73\,856$ Parameter, egal ob das Bild $32^2$ oder $1024^2$ groß ist.

**Rezeptives Feld:** die Menge der Eingabepixel, die ein Ausgabeneuron beeinflusst. Bei $L$ Schichten mit Kernel $k$ und Stride 1 wächst es linear: $r_L = 1 + L(k-1)$. Pooling/Stride vergrößern es multiplikativ — allgemein $r_L = r_{L-1} + (k_L - 1)\prod_{\ell<L} s_\ell$. Tiefe Netze sehen also erst in späten Schichten „das ganze Bild“.

**Pooling:** Max-Pooling $Y_{c,i,j} = \max_{(u,v) \in \text{Fenster}} X_{c, si+u, sj+v}$ macht Repräsentationen lokal translations**invariant** und reduziert die Auflösung. Moderne Architekturen ersetzen es oft durch Strided Convolutions; am Netzende steht heute meist **Global Average Pooling** (Mittel über alle Positionen pro Kanal) statt riesiger Dense-Schichten.

**Backprop durch die Faltung** (Kernaussagen, herleitbar exakt wie in 1.5 durch Ableiten der Summenformel):
- $\dfrac{\partial \mathcal{L}}{\partial K_{c',c,u,v}} = \sum_{i,j} \delta_{c',i,j}\, X_{c,\, si+u-p,\, sj+v-p}$ — eine **Kreuzkorrelation zwischen Eingabe und Fehlersignal** (das Weight Sharing erscheint als Summe über alle Positionen).
- $\dfrac{\partial \mathcal{L}}{\partial X}$ ist eine **„volle“ Faltung des Fehlersignals mit dem um 180° gespiegelten Kernel** (transponierte Faltung).

**Typische Architektur** (wie in Projekt 02): mehrere Blöcke `[Conv → BatchNorm → ReLU] ×2 → MaxPool` mit wachsender Kanalzahl (z. B. 32→64→128), dann Global Average Pooling → Linear → Softmax. Historische Meilensteine: LeNet-5 (1998), AlexNet (2012, ReLU+Dropout+GPU), VGG (2014, nur 3×3), ResNet (2015, Teil 3.3).

### 2.6 Clustering I: k-Means

Ab hier: **unüberwachtes Lernen** — Daten $\{\mathbf{x}_i\}_{i=1}^n$ ohne Labels; Ziel ist Struktur: Gruppen (Clustering), kompakte Darstellungen (Dimensionsreduktion), Dichten (Mixture Models).

**Zielfunktion.** Finde Zentren $\boldsymbol{\mu}_1,\dots,\boldsymbol{\mu}_K$ und Zuordnungen $r_{ik} \in \{0,1\}$ ($\sum_k r_{ik}=1$), die die **Within-Cluster-Streuung** (Inertia/Distortion) minimieren:

$$J(\{r_{ik}\}, \{\boldsymbol{\mu}_k\}) = \sum_{i=1}^n \sum_{k=1}^K r_{ik}\, \lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2 \;\to\; \min.$$

Das exakte Problem ist NP-schwer (schon für $K=2$ in allgemeiner Dimension). **Lloyd's Algorithmus** ist eine alternierende Minimierung:

1. **Assignment-Schritt** (bei festen $\boldsymbol{\mu}_k$): $r_{ik} = 1$ für $k = \arg\min_j \lVert\mathbf{x}_i - \boldsymbol{\mu}_j\rVert^2$ — punktweise optimal, da jedes $\mathbf{x}_i$ unabhängig seinem nächsten Zentrum zugeordnet wird.
2. **Update-Schritt** (bei festen $r_{ik}$): $\nabla_{\boldsymbol{\mu}_k} J = -2\sum_i r_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k) \overset{!}{=} 0 \Rightarrow \boldsymbol{\mu}_k = \dfrac{\sum_i r_{ik}\mathbf{x}_i}{\sum_i r_{ik}}$ — der **Schwerpunkt** des Clusters (das Ziel ist in $\boldsymbol{\mu}_k$ konvex-quadratisch, also ist das das globale Minimum dieses Teilschritts).

**Monotonie & Konvergenz:** Beide Schritte können $J$ nie erhöhen; $J \ge 0$ ist nach unten beschränkt; es gibt nur endlich viele Partitionen — also konvergiert der Algorithmus in endlich vielen Schritten. Aber nur zu einem **lokalen** Optimum, abhängig vom Start. Praxis: mehrere Restarts (`n_init`), Initialisierung mit **k-Means++** (Arthur & Vassilvitskii, 2007): erstes Zentrum uniform, jedes weitere mit Wahrscheinlichkeit $\propto D(\mathbf{x})^2$ (quadrierter Abstand zum nächsten bereits gewählten Zentrum) — liefert in Erwartung eine $O(\log K)$-Approximation der optimalen Distortion.

**Wahl von $K$:** (i) **Elbow-Methode** — $J$ fällt monoton in $K$; man sucht den Knick (heuristisch, oft uneindeutig). (ii) **Silhouette-Koeffizient** pro Punkt:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} \in [-1, 1],$$

mit $a(i)$ = mittlere Distanz zu Punkten des eigenen Clusters, $b(i)$ = kleinste mittlere Distanz zu einem Fremdcluster. Mittelwert nahe 1 = kompakt und gut getrennt; um 0 = Überlappung; negativ = wahrscheinlich falsch zugeordnet. (iii) Modellbasiert: BIC über GMMs (2.7).

**Grenzen:** k-Means setzt implizit isotrope, gleich große, konvexe Cluster voraus (Voronoi-Zellen!) und ist skalenempfindlich → vorher standardisieren; nicht-konvexe Formen (Ringe, Monde) scheitern grundsätzlich → DBSCAN/Spectral Clustering.

### 2.7 Mixture Models und der EM-Algorithmus — vollständige Herleitung

**Gaussian Mixture Model (GMM).** Probabilistisches Clustering: Die Daten stammen aus $K$ Gauß-Komponenten,

$$p(\mathbf{x} \mid \theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma_k), \qquad \pi_k \ge 0,\ \sum_k \pi_k = 1,$$

äquivalent mit **latenter Variable** $z_i \in \{1,\dots,K\}$: $p(z_i = k) = \pi_k$, $\;\mathbf{x}_i \mid z_i = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$.

**Das Problem.** Die Log-Likelihood der beobachteten Daten

$$\log p(X \mid \theta) = \sum_{i=1}^n \log \underbrace{\sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)}_{\text{Summe im Log!}}$$

hat wegen der Summe **innerhalb** des Logarithmus keine geschlossene Maximalstelle. Der **EM-Algorithmus** (Dempster, Laird & Rubin, 1977) löst das iterativ.

**Die untere Schranke (ELBO).** Für eine beliebige Verteilung $q_i(k)$ über die latente Variable gilt mit der Jensen-Ungleichung ($\log$ ist konkav):

$$
\log p(\mathbf{x}_i \mid \theta)
= \log \sum_k q_i(k)\, \frac{p(\mathbf{x}_i, z_i = k \mid \theta)}{q_i(k)}
\;\ge\; \sum_k q_i(k) \log \frac{p(\mathbf{x}_i, z_i = k \mid \theta)}{q_i(k)}
\;=:\; \mathcal{F}(q_i, \theta).
$$

Exakter: $\log p(\mathbf{x}_i \mid \theta) = \mathcal{F}(q_i, \theta) + \mathrm{KL}\big(q_i \,\Vert\, p(z_i \mid \mathbf{x}_i, \theta)\big)$, wobei $\mathrm{KL}(q\Vert p) = \sum_k q(k)\log\frac{q(k)}{p(k)} \ge 0$ die Kullback-Leibler-Divergenz ist (Gleichheit gdw. $q = p$). Diese Zerlegung rechnet man direkt nach, indem man $p(\mathbf{x}_i, z_i) = p(z_i \mid \mathbf{x}_i)\,p(\mathbf{x}_i)$ einsetzt. EM ist **Koordinatenaufstieg auf $\mathcal{F}$**:

**E-Step** — maximiere $\mathcal{F}$ über $q$ bei festem $\theta^{(t)}$: Da die KL der einzige Unterschied zur (festen) Log-Likelihood ist, ist das Optimum $q_i = p(z_i \mid \mathbf{x}_i, \theta^{(t)})$, die **Posterior-Verteilung**. Nach Bayes:

$$
\gamma_{ik} := p(z_i = k \mid \mathbf{x}_i, \theta^{(t)}) = \frac{\pi_k\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)}{\sum_{j} \pi_j\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \Sigma_j)}
\qquad \text{(„Responsibilities“ — Soft-Zuordnungen)}.
$$

**M-Step** — maximiere $\mathcal{F}$ über $\theta$ bei festem $q$. Der $q$-Entropie-Term ist konstant in $\theta$, es bleibt die **erwartete vollständige Log-Likelihood**

$$
Q(\theta) = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \Big[ \log \pi_k - \tfrac{1}{2}\log\lvert 2\pi\Sigma_k\rvert - \tfrac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top \Sigma_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) \Big].
$$

Hier steht der Log **innerhalb** der Summe — geschlossen lösbar. Nullsetzen der Ableitungen (für $\pi$ mit Lagrange-Multiplikator für $\sum_k \pi_k = 1$) ergibt mit $N_k := \sum_i \gamma_{ik}$ (effektive Clustergröße):

$$
\boxed{\;
\pi_k^{\text{neu}} = \frac{N_k}{n}, \qquad
\boldsymbol{\mu}_k^{\text{neu}} = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\, \mathbf{x}_i, \qquad
\Sigma_k^{\text{neu}} = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\, (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{neu}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{neu}})^\top
\;}
$$

— gewichtete Mittelwerte/Kovarianzen mit den Responsibilities als Gewichten. *(Rechenweg für $\boldsymbol{\mu}_k$: $\nabla_{\boldsymbol{\mu}_k} Q = \sum_i \gamma_{ik}\, \Sigma_k^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_k) = 0$; für $\pi_k$: $\partial/\partial\pi_k \big[\sum_{i,k}\gamma_{ik}\log\pi_k + \lambda(1{-}\sum_k\pi_k)\big] = N_k/\pi_k - \lambda = 0$, Summieren über $k$ liefert $\lambda = n$.)*

**Monotonie-Garantie:** $\log p(X\mid\theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t)}) = \log p(X \mid \theta^{(t)})$ — die letzte Gleichheit, weil der E-Step die KL auf 0 setzt. Die Likelihood steigt also monoton; EM konvergiert gegen einen stationären Punkt (lokales Maximum oder Sattel). Wie k-Means: Restarts nötig; Initialisierung oft mit k-Means.

**k-Means als Grenzfall:** Setze $\Sigma_k = \epsilon I$ fix und $\epsilon \to 0$: Die Responsibilities $\gamma_{ik}$ werden hart ($\to$ Indikator des nächsten Zentrums), der M-Step wird der Schwerpunkt — EM degeneriert exakt zu Lloyd's Algorithmus. GMM = „soft k-Means“ mit gelernten, anisotropen Clusterformen und Mischgewichten.

**Kovarianz-Typen** (Bias-Varianz-Abwägung, in sklearn `covariance_type`): `spherical` ($\sigma_k^2 I$) ⊂ `diag` ⊂ `tied` (ein gemeinsames $\Sigma$) ⊂ `full` — mit $d$ Dimensionen kostet `full` pro Komponente $d(d+1)/2$ Parameter.

**Modellwahl mit Informationskriterien:** $\mathrm{BIC} = -2\log \hat{L} + m \log n$, $\mathrm{AIC} = -2\log \hat{L} + 2m$ ($m$ = Parameterzahl). Kleiner = besser; BIC bestraft Komplexität stärker und ist für die Wahl von $K$ die übliche Empfehlung. **Achtung Degenerierung:** Die Likelihood ist unbeschränkt (eine Komponente kann auf einen Punkt kollabieren, $\lvert\Sigma_k\rvert \to 0$) — Praxislösung: Regularisierung der Kovarianz (`reg_covar`), Mindestclustergrößen.

### 2.8 Clustering II: Hierarchisch und DBSCAN

**Agglomeratives (hierarchisches) Clustering.** Starte mit $n$ Singleton-Clustern; verschmelze wiederholt das Paar mit der kleinsten Distanz $d(A, B)$; das Ergebnis ist ein **Dendrogramm** (Binärbaum aller Verschmelzungen), aus dem man durch Abschneiden auf beliebiger Höhe jede Clusterzahl erhält. Die Wahl von $d(A,B)$ (**Linkage**) bestimmt den Charakter:

| Linkage | $d(A,B)$ | Verhalten |
|---|---|---|
| Single | $\min_{a \in A, b \in B} \lVert a - b\rVert$ | kettenbildend („chaining“); findet längliche Strukturen |
| Complete | $\max_{a \in A, b \in B} \lVert a - b\rVert$ | kompakte, gleich große Cluster; ausreißerempfindlich |
| Average (UPGMA) | $\frac{1}{\lvert A\rvert \lvert B\rvert}\sum_{a,b} \lVert a - b\rVert$ | Kompromiss |
| **Ward** | Verschmelze das Paar mit minimalem Zuwachs der Within-Cluster-Varianz: $\Delta J(A,B) = \frac{\lvert A\rvert\,\lvert B\rvert}{\lvert A\rvert + \lvert B\rvert}\, \lVert \bar{\mathbf{a}} - \bar{\mathbf{b}} \rVert^2$ | k-Means-artige, kugelige Cluster; meist der beste Default |

Alle Linkages lassen sich nach einer Verschmelzung $A \cup B$ effizient per **Lance-Williams-Rekursion** aktualisieren: $d(A{\cup}B, C) = \alpha_A d(A,C) + \alpha_B d(B,C) + \beta\, d(A,B) + \gamma\, \lvert d(A,C) - d(B,C)\rvert$ mit linkage-spezifischen Koeffizienten. Komplexität allgemein $O(n^2 \log n)$ Zeit, $O(n^2)$ Speicher — für große $n$ prohibitiv.

**DBSCAN** (Ester et al., 1996) — dichtebasiert, mit zwei Parametern $\varepsilon$ (Radius) und $\textit{minPts}$:

- $\mathbf{p}$ ist **Kernpunkt**, wenn $\lvert N_\varepsilon(\mathbf{p}) \rvert \ge \textit{minPts}$ (inkl. $\mathbf{p}$ selbst), mit $N_\varepsilon(\mathbf{p}) = \{\mathbf{q} : \lVert\mathbf{p}-\mathbf{q}\rVert \le \varepsilon\}$.
- $\mathbf{q}$ ist von $\mathbf{p}$ **direkt dichte-erreichbar**, wenn $\mathbf{q} \in N_\varepsilon(\mathbf{p})$ und $\mathbf{p}$ Kernpunkt ist. **Dichte-erreichbar** = transitive Hülle davon (Kette von Kernpunkten). Zwei Punkte sind **dichte-verbunden**, wenn beide von einem gemeinsamen Kernpunkt aus dichte-erreichbar sind.
- Ein **Cluster** ist eine maximale Menge dichte-verbundener Punkte. Punkte in keinem Cluster sind **Noise** (Label $-1$). Nicht-Kernpunkte am Clusterrand heißen **Randpunkte**.

Eigenschaften: findet **beliebig geformte** Cluster, bestimmt die Clusterzahl selbst, hat ein explizites Rauschkonzept — aber: ein globales $\varepsilon$ scheitert bei Clustern stark unterschiedlicher Dichte (Nachfolger: OPTICS, HDBSCAN), und in hohen Dimensionen verlieren $\varepsilon$-Kugeln durch den Fluch der Dimensionalität an Bedeutung. **Parameterwahl:** $\textit{minPts} \approx 2d$ als Faustregel; für $\varepsilon$ den **k-Distanz-Plot** ($k = \textit{minPts}-1$): sortierte Distanz jedes Punkts zu seinem $k$-nächsten Nachbarn; $\varepsilon$ am „Knie“.

**Externe Cluster-Validierung** (wenn eine Referenzpartition existiert, z. B. in Benchmarks): Der **Rand Index** misst den Anteil konsistent behandelter Punktpaare (zusammen/zusammen oder getrennt/getrennt). Der **Adjusted Rand Index (ARI)** korrigiert um den Zufallserwartungswert:

$$\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \big[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\big] / \binom{n}{2}}{\frac{1}{2}\big[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\big] - \big[\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}\big]/\binom{n}{2}},$$

mit Kontingenztafel $n_{ij}$, Randsummen $a_i, b_j$. ARI $= 1$: identische Partitionen; $\approx 0$: Zufallsniveau; ARI ist permutationsinvariant (Clusterlabels sind ja beliebig). Alternative: **Normalized Mutual Information** $\mathrm{NMI}(U,V) = \frac{I(U;V)}{\sqrt{H(U)H(V)}}$.

### 2.9 Dimensionsreduktion I: PCA — zwei Herleitungen

**Ziel:** Projiziere zentrierte Daten $X \in \mathbb{R}^{n \times d}$ (Spaltenmittel 0) auf einen $q$-dimensionalen Unterraum, der „möglichst viel Information“ erhält. Sei $S = \frac{1}{n} X^\top X$ die empirische Kovarianzmatrix.

**Herleitung 1 — Varianzmaximierung.** Suche die Richtung $\mathbf{w}$, $\lVert\mathbf{w}\rVert = 1$, die die Varianz der Projektion maximiert:

$$\max_{\mathbf{w}} \ \operatorname{Var}(X\mathbf{w}) = \mathbf{w}^\top S\, \mathbf{w} \quad \text{u.d.N.} \quad \mathbf{w}^\top\mathbf{w} = 1.$$

Lagrange: $\Lambda(\mathbf{w}, \lambda) = \mathbf{w}^\top S \mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w} - 1)$, $\;\nabla_{\mathbf{w}} \Lambda = 2S\mathbf{w} - 2\lambda\mathbf{w} = 0 \Rightarrow S\mathbf{w} = \lambda\mathbf{w}$ — ein **Eigenwertproblem**. Die erreichte Varianz ist $\mathbf{w}^\top S\mathbf{w} = \lambda$, also wählt man den **größten** Eigenwert; die $q$ Hauptkomponenten sind die Eigenvektoren zu den $q$ größten Eigenwerten $\lambda_1 \ge \dots \ge \lambda_q$ (sukzessive mit Orthogonalitätsnebenbedingung). **Erklärte Varianz:** Anteil $\sum_{j\le q}\lambda_j / \sum_{j\le d}\lambda_j$ (Standardkriterium für die Wahl von $q$, z. B. 90–95 %).

**Herleitung 2 — minimaler Rekonstruktionsfehler.** Suche die orthonormale Basis $W \in \mathbb{R}^{d\times q}$ ($W^\top W = I_q$), die den quadratischen Rekonstruktionsfehler minimiert:

$$\min_W \frac{1}{n}\sum_{i=1}^n \lVert \mathbf{x}_i - W W^\top \mathbf{x}_i \rVert^2 = \underbrace{\operatorname{tr}(S)}_{\text{konst.}} - \operatorname{tr}\big(W^\top S\, W\big),$$

(Pythagoras: $\lVert\mathbf{x}\rVert^2 = \lVert W^\top\mathbf{x}\rVert^2 + \lVert\mathbf{x} - WW^\top\mathbf{x}\rVert^2$). Minimierung des Fehlers = Maximierung von $\operatorname{tr}(W^\top S W)$ = dieselben Top-$q$-Eigenvektoren. **Varianz maximieren und Rekonstruktionsfehler minimieren sind exakt dasselbe Problem.** Der minimale Fehler ist $\sum_{j > q} \lambda_j$.

**SVD-Verbindung.** Mit der Singulärwertzerlegung $X = U \Sigma V^\top$ gilt $S = \frac{1}{n} V \Sigma^2 V^\top$: Die rechten Singulärvektoren $V$ sind die Hauptrichtungen, $\lambda_j = \sigma_j^2 / n$. Numerisch berechnet man PCA **immer über die SVD von $X$** (stabiler als $X^\top X$ zu bilden, dessen Konditionszahl quadriert ist). Die Projektionen („Scores“) sind $XV = U\Sigma$.

**Praxis:** Vorher zwingend zentrieren; bei Features unterschiedlicher Einheiten standardisieren (dann ist $S$ die Korrelationsmatrix). PCA ist linear und varianzorientiert — Cluster, die in Richtungen kleiner Varianz getrennt sind, kann sie verschmelzen.

### 2.10 Dimensionsreduktion II: t-SNE (und UMAP)

**t-SNE** (van der Maaten & Hinton, 2008) ist eine **nichtlineare** Einbettung für Visualisierung (meist $q = 2$), die Nachbarschaften statt Distanzen erhält.

**Hochdimensionale Ähnlichkeiten:** bedingte Gauß-Nachbarwahrscheinlichkeiten

$$p_{j\mid i} = \frac{\exp\big(-\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2 / 2\sigma_i^2\big)}{\sum_{k \ne i} \exp\big(-\lVert\mathbf{x}_i - \mathbf{x}_k\rVert^2 / 2\sigma_i^2\big)}, \qquad p_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2n}.$$

Die Bandbreite $\sigma_i$ wird **pro Punkt** so gewählt, dass die effektive Nachbarzahl einem globalen Parameter entspricht: $\mathrm{Perp}(P_i) = 2^{H(P_i)}$ mit Shannon-Entropie $H(P_i) = -\sum_j p_{j\mid i}\log_2 p_{j\mid i}$ — die **Perplexität** (typisch 5–50); Binärsuche über $\sigma_i$.

**Niedrigdimensionale Ähnlichkeiten:** eine **Student-t-Verteilung mit 1 Freiheitsgrad** (Cauchy):

$$q_{ij} = \frac{\big(1 + \lVert\mathbf{y}_i - \mathbf{y}_j\rVert^2\big)^{-1}}{\sum_{k \ne l} \big(1 + \lVert\mathbf{y}_k - \mathbf{y}_l\rVert^2\big)^{-1}}.$$

Die schweren Ränder sind der Kernkniff: Sie erlauben, moderate Distanzen im Hochdimensionalen auf **große** Distanzen in 2D abzubilden — das behebt das **Crowding-Problem** (in 2D ist schlicht zu wenig Platz für alle mittleren Nachbarn) und erzeugt die charakteristisch getrennten Cluster.

**Kostenfunktion & Gradient:** Minimiere $\mathrm{KL}(P \Vert Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$ per Gradientenverfahren;

$$\frac{\partial\, \mathrm{KL}}{\partial \mathbf{y}_i} = 4\sum_{j \neq i} (p_{ij} - q_{ij})\,\big(1 + \lVert\mathbf{y}_i - \mathbf{y}_j\rVert^2\big)^{-1} (\mathbf{y}_i - \mathbf{y}_j)$$

— anziehend, wo $p_{ij} > q_{ij}$, abstoßend sonst. Die KL ist asymmetrisch: **Nachbarn zu trennen ist teuer, Fremde zusammenzulegen billig** — t-SNE erhält lokale Struktur, nicht globale.

**Interpretationsfallen (prüfungsrelevant):** Clustergrößen und Abstände **zwischen** Clustern in t-SNE-Plots sind nicht interpretierbar; das Ergebnis hängt von Perplexität und Seed ab; t-SNE ist keine Abbildung (neue Punkte können nicht einfach projiziert werden). **UMAP** (McInnes et al., 2018) ist der graphbasierte Nachfolger: schneller, erhält globale Struktur etwas besser, ähnliche Vorsichtsregeln.

---

## Teil 3 — Advanced

### 3.1 Universal Approximation — was das Theorem sagt (und was nicht)

**Satz (Cybenko 1989; Hornik 1991).** Sei $\sigma$ stetig und nicht-polynomial (z. B. Sigmoid, ReLU). Dann ist die Menge der Ein-Hidden-Layer-Netze $\;f(\mathbf{x}) = \sum_{j=1}^m c_j\, \sigma(\mathbf{w}_j^\top \mathbf{x} + b_j)\;$ **dicht in $C(K)$** für jedes Kompaktum $K \subset \mathbb{R}^d$: Zu jeder stetigen Funktion $g$ und jedem $\varepsilon > 0$ existiert ein $m$ und Parameter mit $\sup_{\mathbf{x} \in K} \lvert f(\mathbf{x}) - g(\mathbf{x})\rvert < \varepsilon$.

**Was es nicht sagt:** (i) nichts über die **Größe** $m$ — im Worst Case wächst $m$ exponentiell in $d$; (ii) nichts darüber, ob SGD diese Parameter **findet** (Approximations- vs. Optimierungs- vs. Generalisierungsfehler sind drei getrennte Fragen); (iii) nichts über Daten außerhalb von $K$. **Warum Tiefe?** Es gibt Funktionenfamilien, die tiefe Netze mit $\mathrm{poly}(d)$ Neuronen darstellen, flache aber nur mit $\exp(d)$ Neuronen (Depth-Separation-Resultate, z. B. Telgarsky 2016; Intuition: Tiefe erlaubt Kompositionalität und Wiederverwendung von Zwischenfeatures; ein ReLU-Netz mit $L$ Schichten und Breite $w$ kann $O(w^L)$ lineare Regionen realisieren, ein flaches nur $O(w)$... genauer $O(w^d)$, aber der exponentielle Vorteil in $L$ bleibt).

### 3.2 Vanishing & Exploding Gradients — die Jacobian-Analyse

Aus der Backprop-Rekursion folgt für den Gradienten in Schicht $\ell$ eines $L$-Schichten-Netzes:

$$\boldsymbol{\delta}^{(\ell)} = \Bigg(\prod_{m=\ell+1}^{L} \mathrm{diag}\big(\sigma'(\mathbf{z}^{(m-1)})\big)\, {W^{(m)}}^\top \Bigg)\, \boldsymbol{\delta}^{(L)} \quad\text{(Reihenfolge geeignet gelesen)},$$

ein **Produkt von $L - \ell$ Jacobians**. Die Norm dieses Produkts wird durch die Singulärwerte kontrolliert: Sind die typischen größten Singulärwerte der Schicht-Jacobians $s < 1$, schrumpft das Signal wie $s^{L-\ell}$ → **Vanishing Gradients** (frühe Schichten lernen nichts); sind sie $> 1$, wächst es exponentiell → **Exploding Gradients** (Divergenz, NaNs). Sigmoid verschärft das Problem strukturell: $\sigma' \le 1/4$ multipliziert in jeder Schicht hinein.

**Gegenmaßnahmen im Überblick:** ReLU-artige Aktivierungen ($\sigma' = 1$ im aktiven Bereich), He/Xavier-Initialisierung (Singulärwerte um 1 starten), BatchNorm/LayerNorm (Reskalierung pro Schicht), **Gradient Clipping** gegen Explosion ($\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, c/\lVert\mathbf{g}\rVert)$), und vor allem:

### 3.3 Residual Connections (ResNet)

**Idee** (He et al., 2015): Statt $\mathbf{a}^{(\ell+1)} = F(\mathbf{a}^{(\ell)})$ lerne das **Residuum**:

$$\mathbf{a}^{(\ell+1)} = \mathbf{a}^{(\ell)} + F\big(\mathbf{a}^{(\ell)}\big).$$

**Gradientenanalyse:** $\dfrac{\partial \mathbf{a}^{(\ell+1)}}{\partial \mathbf{a}^{(\ell)}} = I + \dfrac{\partial F}{\partial \mathbf{a}^{(\ell)}}$, also über viele Blöcke

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \prod_{m=\ell}^{L-1}\Big(I + \frac{\partial F_m}{\partial \mathbf{a}^{(m)}}\Big) = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}}\Big(I + \text{gemischte Terme}\Big).$$

Der Identitätsanteil garantiert einen **ungestörten Gradientenpfad** bis in die erste Schicht — das Produkt kann nicht mehr als Ganzes verschwinden. Außerdem: Ein zusätzlicher Block kann die Identität darstellen, indem $F \to 0$ (leicht lernbar, alle Gewichte → 0), d. h. tiefere Netze sind nie *schlechter* darstellbar als flachere — genau das war vor ResNet empirisch verletzt (56-Layer-Netze schlechter als 20-Layer, **auf den Trainingsdaten** — ein Optimierungs-, kein Overfitting-Problem). ResNets machten Netze mit Hunderten Schichten trainierbar; Skip Connections sind heute in praktisch jeder Architektur (auch Transformern) Standard.

### 3.4 Autoencoder

Ein **Autoencoder** lernt eine Kompression: Encoder $\mathbf{h} = f_{\text{enc}}(\mathbf{x}) \in \mathbb{R}^q$ ($q \ll d$, „Bottleneck“), Decoder $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{h})$, Ziel $\min \sum_i \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert^2$. **Verbindung zu PCA:** Sind Encoder und Decoder linear, ist die optimale Lösung genau der PCA-Unterraum (der Bottleneck spannt den Top-$q$-Eigenraum auf, wenn auch nicht notwendig orthonormal). Nichtlineare Autoencoder lernen gekrümmte Mannigfaltigkeiten. Varianten: **Denoising AE** (rekonstruiere $\mathbf{x}$ aus verrauschtem $\tilde{\mathbf{x}}$ — erzwingt robuste Features), **Sparse AE** (L1 auf $\mathbf{h}$).

### 3.5 Variational Autoencoder (VAE) — die ELBO, vollständig

Der VAE (Kingma & Welling, 2014) ist ein **generatives** latentes Variablenmodell: $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$, $\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})$ (Decoder-Netz). Die Likelihood $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$ ist intraktabel — dieselbe Struktur wie beim GMM (2.7), nur mit kontinuierlichem $\mathbf{z}$ und neuronalen Netzen. Und dieselbe Lösung: eine Variationsverteilung $q_\phi(\mathbf{z} \mid \mathbf{x})$ (Encoder-Netz, typisch $\mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$) und die **Evidence Lower Bound**:

$$
\log p_\theta(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big] - \mathrm{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\Vert\, p(\mathbf{z})\big)}_{=\;\mathrm{ELBO}(\theta, \phi;\, \mathbf{x})} + \underbrace{\mathrm{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\Vert\, p_\theta(\mathbf{z}\mid\mathbf{x})\big)}_{\ge\, 0}.
$$

*(Nachrechnen wie in 2.7: $\log p_\theta(\mathbf{x})$ herausziehen, da es nicht von $\mathbf{z}$ abhängt.)* Man maximiert die ELBO gemeinsam über $\theta$ **und** $\phi$: der Rekonstruktionsterm will gute Dekodierung, der KL-Term hält die Posterior-Approximation nahe am Prior (Regularisierung des Latentraums — deshalb kann man aus $\mathcal{N}(\mathbf{0}, I)$ sampeln und dekodieren: ein *Generator*). Statt E-Step (unmöglich, da $q$ ein Netz ist) macht der VAE also **amortisierte Variationsinferenz** mit Gradientenschritten.

**Reparametrisierungstrick.** $\nabla_\phi \mathbb{E}_{q_\phi}[\cdot]$ kann nicht direkt durch Sampling geschätzt werden (die Verteilung hängt von $\phi$ ab). Lösung: Schreibe das Sample als deterministische Funktion von $\phi$ und externem Rauschen,

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I),$$

dann ist $\mathbb{E}_{q_\phi}[g(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\varepsilon}}[g(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon})]$ und der Gradient fließt per Backprop durch $\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi$. **Geschlossene KL** für zwei Diagonal-Gauß (nachrechenbar durch Integration):

$$\mathrm{KL}\big(\mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2)) \,\Vert\, \mathcal{N}(\mathbf{0}, I)\big) = \frac{1}{2}\sum_{j=1}^q \big(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\big).$$

### 3.6 Generalisierung im Deep Learning: Double Descent & implizite Regularisierung

Die klassische Lernkurve (Modul 04) ist U-förmig: mehr Kapazität → weniger Bias, mehr Varianz → irgendwann Overfitting. Tiefe Netze brechen dieses Bild: Moderne Netze erreichen **null Trainingsfehler** (sie können sogar Zufallslabels auswendig lernen — Zhang et al., 2017) und generalisieren *trotzdem*. Empirisch zeigt sich **Double Descent** (Belkin et al., 2019): Der Testfehler steigt bis zur **Interpolationsschwelle** (Kapazität ≈ gerade genug, um die Daten exakt zu fitten — dort ist das Modell maximal „gezwungen“ und oft am schlechtesten) und **fällt danach wieder**, wenn die Kapazität weiter wächst. Erklärungsansatz: Im überparametrisierten Regime gibt es viele interpolierende Lösungen, und der Optimierer wählt systematisch „glatte“ aus — **implizite Regularisierung**: GD auf linearer Regression konvergiert (bei Start in 0) zur **Minimum-Norm-Lösung** unter allen Interpolierenden; SGD-Rauschen bevorzugt zusätzlich flache Minima, die gegen Parameterstörungen (und damit Verteilungsverschiebungen) robuster sind. Konsequenz für die Praxis: „Mehr Parameter = Overfitting“ ist als Faustregel tot; Validierungskurven schlagen Parameterzählen.

### 3.7 Ausblick: Selbstüberwachtes Lernen

Zwischen „überwacht“ und „unüberwacht“ liegt das heute dominante Paradigma: **Self-Supervised Learning** erzeugt Lernsignale aus den Daten selbst — z. B. kontrastives Lernen (SimCLR): zwei Augmentierungen desselben Bildes sollen ähnliche Repräsentationen bekommen, verschiedene Bilder unähnliche, mit der **InfoNCE-Loss** $\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \ne i} \exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$ (Ähnlichkeit = Kosinus, Temperatur $\tau$). Masked Modeling (BERT, MAE) ist das Pendant: Teile der Eingabe maskieren und rekonstruieren. Die NLP-Module (08–10) bauen zentral darauf auf.

---

## Zusammenfassung / Cheat-Sheet

**Neuronale Netze:**

| Konzept | Kernformel |
|---|---|
| Forward | $\mathbf{z}^{(\ell)} = W^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$, $\mathbf{a}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})$ |
| Softmax+CE-Gradient | $\nabla_{\mathbf{z}}\mathcal{L} = \mathbf{p} - \mathbf{y}$ |
| Backprop-Rekursion | $\boldsymbol{\delta}^{(\ell)} = ({W^{(\ell+1)}}^\top \boldsymbol{\delta}^{(\ell+1)}) \odot \sigma'(\mathbf{z}^{(\ell)})$ |
| Gradienten | $\nabla_{W^{(\ell)}}\mathcal{L} = \boldsymbol{\delta}^{(\ell)}{\mathbf{a}^{(\ell-1)}}^\top$, $\nabla_{\mathbf{b}^{(\ell)}}\mathcal{L} = \boldsymbol{\delta}^{(\ell)}$ |
| SGD+Momentum | $\mathbf{v} \leftarrow \mu\mathbf{v} + \mathbf{g}$; $\theta \leftarrow \theta - \eta\mathbf{v}$ |
| Adam | $\hat{\mathbf{m}} = \frac{\mathbf{m}}{1-\beta_1^t}$, $\hat{\mathbf{v}} = \frac{\mathbf{v}}{1-\beta_2^t}$, $\theta \leftarrow \theta - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}$ |
| He-Init (ReLU) | $\sigma_W^2 = 2/d_{\text{in}}$; Xavier: $2/(d_{\text{in}}+d_{\text{out}})$ |
| Weight Decay | $\theta \leftarrow (1-\eta\lambda)\theta - \eta\nabla\mathcal{L}$; bei Adam entkoppeln (AdamW) |
| Dropout (Training) | $\tilde{\mathbf{a}} = \mathbf{m}\odot\mathbf{a}/(1-p)$, $\mathbf{m}\sim\mathrm{Bern}(1-p)$ |
| BatchNorm | $\hat z = \frac{z - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}$, $y = \gamma\hat z + \beta$; Inferenz: gleitende Mittel |
| Conv-Output | $H_{\text{out}} = \lfloor (H + 2p - k)/s \rfloor + 1$; Parameter: $C_{\text{out}}(C_{\text{in}}k_hk_w + 1)$ |
| ResNet | $\mathbf{a}^{(\ell+1)} = \mathbf{a}^{(\ell)} + F(\mathbf{a}^{(\ell)})$; Jacobian $I + \partial F$ |

**Unüberwachtes Lernen:**

| Konzept | Kernformel / Kernaussage |
|---|---|
| k-Means-Ziel | $J = \sum_{i,k} r_{ik}\lVert\mathbf{x}_i - \boldsymbol{\mu}_k\rVert^2$; Lloyd = alternierende Minimierung, monoton, lokal |
| GMM | $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\Sigma_k)$ |
| E-Step | $\gamma_{ik} = \frac{\pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k,\Sigma_k)}{\sum_j \pi_j\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_j,\Sigma_j)}$ |
| M-Step | $\pi_k = \frac{N_k}{n}$, $\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_i \gamma_{ik}\mathbf{x}_i$, $\Sigma_k = \frac{1}{N_k}\sum_i \gamma_{ik}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\cdot)^\top$ |
| EM-Garantie | Likelihood steigt monoton (ELBO + KL-Zerlegung); k-Means = GMM mit $\Sigma_k = \epsilon I$, $\epsilon\to 0$ |
| Silhouette | $s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$ |
| BIC | $-2\log\hat L + m\log n$ (kleiner = besser) |
| DBSCAN | Kernpunkt: $\lvert N_\varepsilon\rvert \ge \textit{minPts}$; Cluster = maximale dichte-verbundene Menge |
| Ward | $\Delta J = \frac{\lvert A\rvert\lvert B\rvert}{\lvert A\rvert+\lvert B\rvert}\lVert\bar{\mathbf{a}} - \bar{\mathbf{b}}\rVert^2$ |
| PCA | $S\mathbf{w} = \lambda\mathbf{w}$; Varianzmax. ≡ Rekonstruktionsmin.; via SVD rechnen |
| t-SNE | $\min \mathrm{KL}(P\Vert Q)$, $Q$ mit Student-t(1); Abstände zwischen Clustern nicht interpretieren! |
| VAE-ELBO | $\mathbb{E}_q[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - \mathrm{KL}(q_\phi(\mathbf{z}\mid\mathbf{x})\Vert p(\mathbf{z}))$; Reparametrisierung $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\varepsilon}$ |

---

## Selbsttest

Beantworte die Fragen erst selbst, klappe dann die Antworten auf.

<details><summary><b>1. Warum ist ein MLP ohne Aktivierungsfunktionen äquivalent zu einem linearen Modell? Zeige es.</b></summary>

Verkettung affiner Abbildungen ist affin: $W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$. Induktiv gilt das für beliebig viele Schichten — das Netz kollabiert zu *einer* affinen Abbildung, kann also z. B. XOR nicht darstellen. Erst nichtlineare $\sigma$ zwischen den Schichten geben Ausdrucksmächtigkeit.
</details>

<details><summary><b>2. Leite den Gradienten von Softmax + Cross-Entropy bzgl. der Logits her. Warum ist das Ergebnis numerisch und didaktisch so angenehm?</b></summary>

Mit $p_k = \mathrm{softmax}(\mathbf{z})_k$, One-Hot $\mathbf{y}$: $\frac{\partial\mathcal{L}}{\partial z_j} = -\sum_k y_k \frac{1}{p_k} p_k(\delta_{kj} - p_j) = p_j - y_j$, also $\nabla_{\mathbf{z}}\mathcal{L} = \mathbf{p} - \mathbf{y}$. Angenehm, weil (i) kein Jacobian explizit gebraucht wird, (ii) der Gradient beschränkt ist (keine $1/p$-Explosion — die Sättigung von Softmax und der Log heben sich exakt auf), (iii) die Form „Vorhersage minus Ziel“ die Verallgemeinerung der linearen/logistischen Regression sichtbar macht.
</details>

<details><summary><b>3. Erkläre die Backprop-Rekursion $\boldsymbol{\delta}^{(\ell)} = ({W^{(\ell+1)}}^\top\boldsymbol{\delta}^{(\ell+1)})\odot\sigma'(\mathbf{z}^{(\ell)})$ in Worten. Warum kostet Backprop nur ~2 Forward-Passes?</b></summary>

Der Fehler einer Schicht ist der rückprojizierte Fehler der Folgeschicht (durch $W^\top$ — jedes Neuron sammelt die Fehler aller Neuronen, zu denen es beiträgt, gewichtet mit den Verbindungsgewichten), gedämpft durch die lokale Empfindlichkeit $\sigma'$. Kosten: Der Backward Pass besteht aus denselben Matrix-Vektor-Produkten wie der Forward Pass (nur transponiert) plus äußeren Produkten für die Gradienten — asymptotisch gleiche FLOPs, daher Faktor ~2. Naive numerische Differenziation bräuchte einen Forward Pass *pro Parameter*.
</details>

<details><summary><b>4. Warum ist L2-Regularisierung bei Adam nicht dasselbe wie Weight Decay, bei SGD aber schon?</b></summary>

Bei SGD: $\theta \leftarrow \theta - \eta(\nabla\mathcal{L} + \lambda\theta) = (1-\eta\lambda)\theta - \eta\nabla\mathcal{L}$ — der L2-Gradient wirkt exakt als multiplikatives Schrumpfen. Bei Adam wird der Gesamtgradient (inkl. $\lambda\theta$) durch $\sqrt{\hat{\mathbf{v}}}+\epsilon$ geteilt: Parameter mit großen Gradienten werden *schwächer* regularisiert — die Strafe ist nicht mehr proportional zu $\theta$. AdamW entkoppelt den Decay-Term vom adaptiven Teil: $\theta \leftarrow \theta - \eta(\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}}+\epsilon) + \lambda\theta)$.
</details>

<details><summary><b>5. Leite die He-Initialisierung her. Warum Faktor 2 statt 1?</b></summary>

$\operatorname{Var}(z_i) = d_{\text{in}}\sigma_W^2\,\mathbb{E}[a_j^2]$ für unabhängige zentrierte Gewichte. Bei ReLU ist $a = \max(0, z)$ mit $z$ symmetrisch um 0, also $\mathbb{E}[a^2] = \frac{1}{2}\mathbb{E}[z^2]$ — ReLU „löscht“ die halbe Signalleistung. Damit die Varianz über Schichten konstant bleibt: $d_{\text{in}}\sigma_W^2 \cdot \frac{1}{2} = 1 \Rightarrow \sigma_W^2 = 2/d_{\text{in}}$.
</details>

<details><summary><b>6. Warum muss man in PyTorch vor der Evaluation <code>model.eval()</code> aufrufen? Nenne die zwei betroffenen Mechanismen mit Begründung.</b></summary>

(i) **Dropout**: Im Training werden Aktivierungen zufällig genullt (und mit $1/(1-p)$ reskaliert); zur Inferenz will man das volle deterministische Netz. (ii) **BatchNorm**: Im Training normalisiert die Schicht mit Batch-Statistiken $\mu_B, \sigma_B^2$; zur Inferenz müssen die gleitenden Mittel verwendet werden — sonst hängt die Vorhersage eines Bildes von den anderen Bildern im Batch ab (und bei Batchgröße 1 wäre $\sigma_B^2 = 0$).
</details>

<details><summary><b>7. Beweise, dass Lloyd's Algorithmus terminiert. Warum ist das Ergebnis trotzdem nicht global optimal?</b></summary>

Assignment-Schritt: Jeder Punkt wechselt höchstens zu einem näheren Zentrum → $J$ sinkt oder bleibt gleich. Update-Schritt: Der Schwerpunkt minimiert $\sum_i r_{ik}\lVert\mathbf{x}_i - \boldsymbol{\mu}\rVert^2$ (konvex-quadratisch in $\boldsymbol{\mu}$, Gradient null beim Mittelwert) → $J$ sinkt oder bleibt gleich. $J \ge 0$ und es gibt endlich viele Partitionen; da $J$ monoton fällt, kann keine Partition zweimal besucht werden (bei strikter Senkung) → Terminierung in endlich vielen Schritten. Global optimal ist es nicht, weil das Verfahren nur lokal koordinatenweise verbessert — das exakte Problem ist NP-schwer; verschiedene Starts liefern verschiedene lokale Minima (darum k-Means++ und `n_init`).
</details>

<details><summary><b>8. Skizziere die EM-Herleitung für GMMs: Woher kommt die untere Schranke, was tun E- und M-Step, und warum steigt die Likelihood monoton?</b></summary>

Jensen (Log ist konkav) liefert für jedes $q$: $\log p(\mathbf{x}\mid\theta) \ge \mathbb{E}_q[\log\frac{p(\mathbf{x},z\mid\theta)}{q(z)}] = \mathcal{F}(q,\theta)$, mit Lücke $\mathrm{KL}(q \Vert p(z\mid\mathbf{x},\theta))$. E-Step: $q = $ Posterior (Responsibilities $\gamma_{ik}$) → KL $= 0$, Schranke berührt die Likelihood. M-Step: maximiere $Q(\theta) = \sum_{i,k}\gamma_{ik}\log p(\mathbf{x}_i, k\mid\theta)$ geschlossen ($\pi_k = N_k/n$, gewichtete Mittel/Kovarianzen). Monotonie: $\log p(\theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t)}) = \log p(\theta^{(t)})$.
</details>

<details><summary><b>9. Warum benutzt t-SNE im Niedrigdimensionalen eine Student-t- statt einer Gauß-Verteilung?</b></summary>

Crowding-Problem: In 2D gibt es nicht genug „Platz“, um alle mittleren Distanzen des Hochdimensionalen treu abzubilden (das Volumen einer Kugel wächst mit der Dimension; viele annähernd äquidistante Nachbarn passen nicht in eine Ebene). Die schweren Ränder der t-Verteilung erlauben, dass moderat entfernte Punkte im Plot *weit* auseinander liegen und trotzdem die passende Ähnlichkeit $q_{ij}$ behalten — sonst würden alle Punkte ins Zentrum gequetscht. Nebeneffekt: übertrieben klare Clustertrennung, weshalb Abstände zwischen Clustern nicht interpretierbar sind.
</details>

<details><summary><b>10. Ein Kommilitone sagt: „Das Netz hat null Trainingsfehler, es hat also overfittet und generalisiert schlecht.“ Nimm differenziert Stellung.</b></summary>

Nicht zwingend. Moderne überparametrisierte Netze interpolieren die Trainingsdaten routinemäßig und generalisieren trotzdem gut (Double Descent: hinter der Interpolationsschwelle *fällt* der Testfehler wieder). Grund: Unter den vielen interpolierenden Lösungen wählt (S)GD implizit regularisierte aus (Minimum-Norm-Charakter, flache Minima). Entscheidend ist der *Validierungsfehler*, nicht der Trainingsfehler. Allerdings: Null Trainingsfehler *plus steigender* Validierungsfehler über die Epochen ist klassisches Overfitting → Early Stopping.
</details>

---

## Literatur & Quellen

**Lehrbücher:**
- I. Goodfellow, Y. Bengio, A. Courville: *Deep Learning* (MIT Press, 2016) — Kap. 6 (MLP/Backprop), 7 (Regularisierung), 8 (Optimierung), 9 (CNNs), 14 (Autoencoder). **Frei online: deeplearningbook.org.** *(vertiefend, das Standardwerk)*
- C. Bishop: *Pattern Recognition and Machine Learning* (Springer, 2006) — Kap. 9 (Mixture Models & EM, exakt unsere Herleitung), 12 (PCA, probabilistische PCA). *(vertiefend; PDF frei von Microsoft Research)*
- K. Murphy: *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022) — moderne Gesamtdarstellung. **Frei online: probml.github.io.** *(mittel bis vertiefend)*
- A. Géron: *Hands-On Machine Learning* (O'Reilly, 3. Aufl. 2022) — Kap. 9 (Clustering/GMM), 10–14 (Neural Nets praktisch). *(einsteigerfreundlich)*

**Kurse & Vorlesungen (frei):**
- Stanford **CS231n** — CNNs for Visual Recognition (cs231n.github.io): die Notes zu Backprop, Initialisierung und CNNs sind hervorragend. *(einsteigerfreundlich bis mittel)*
- **3Blue1Brown**, Neural-Networks-Serie (YouTube) — die beste Visualisierung von Backprop. *(einsteigerfreundlich)*
- Andrej Karpathy: *Neural Networks: Zero to Hero* (YouTube) — Backprop from scratch in Code, passt exakt zu Projekt 01. *(einsteigerfreundlich, sehr empfehlenswert)*

**Schlüssel-Papers (alle frei auf arXiv):**
- Kingma & Ba (2015): *Adam*. — Loshchilov & Hutter (2019): *Decoupled Weight Decay (AdamW)*.
- Srivastava et al. (2014): *Dropout*. — Ioffe & Szegedy (2015): *Batch Normalization*; dazu Santurkar et al. (2018): *How Does BN Help Optimization?*
- He et al. (2015): *Delving Deep into Rectifiers* (He-Init) und *Deep Residual Learning* (ResNet).
- Dempster, Laird & Rubin (1977): *Maximum Likelihood from Incomplete Data via the EM Algorithm* (JRSS-B).
- Arthur & Vassilvitskii (2007): *k-means++*. — Ester et al. (1996): *DBSCAN* (KDD).
- van der Maaten & Hinton (2008): *Visualizing Data using t-SNE* (JMLR, frei). — McInnes et al. (2018): *UMAP*.
- Kingma & Welling (2014): *Auto-Encoding Variational Bayes* (VAE).
- Zhang et al. (2017): *Understanding Deep Learning Requires Rethinking Generalization*. — Belkin et al. (2019): *Reconciling Modern ML Practice and the Bias-Variance Trade-off* (Double Descent).

**Interaktiv:**
- **playground.tensorflow.org** — MLPs im Browser trainieren (Aktivierungen, Regularisierung live). *(einsteigerfreundlich, frei)*
- distill.pub: *How to Use t-SNE Effectively* — Pflichtlektüre vor jeder t-SNE-Interpretation. *(frei)*
- **poloclub.github.io/cnn-explainer** — CNN-Forward-Pass interaktiv. *(frei)*

---

## Die drei Projekte

| Projekt | Thema | Format | Daten |
|---|---|---|---|
| [01-basic](projects/01-basic/) | MLP + Backprop **von Hand** (nur NumPy), Gradient Checking | Notebook | synthetisch (`make_moons`) |
| [02-medium](projects/02-medium/) | CNN-Bildklassifikation in PyTorch, Optimierer-/Regularisierungs-Ablation | Notebook | Fashion-MNIST (echt) |
| [03-final](projects/03-final/) | Kundensegmentierung: k-Means vs. GMM/EM vs. DBSCAN vs. Ward, PCA/t-SNE, BIC & Validierung | Notebook | UCI Wholesale Customers (echt) |
