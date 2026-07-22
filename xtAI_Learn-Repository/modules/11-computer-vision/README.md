# Module 11 — Computer Vision

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** How do you teach a computer to **understand images** —
> to recognize objects, classify scenes, segment regions? We start with the
> **convolution** (the basic operation of image processing), build up to
> **convolutional neural networks (CNNs)** and then devote a large part to
> **pretrained models** and **transfer learning** — the approach with which one
> builds strong image models today *without* large training of one's own and
> *without* a GPU cluster. That is exactly what we use in the projects. But so that
> you *understand* what is happening there, this script explains both in detail: how
> you would train a network **from scratch** *and* how and why pretrained models
> work.

**Helpful prior knowledge:** linear algebra (matrices/tensors), the basics of neural
networks and gradient descent, PyTorch basics.

**Modules you should have done first:**
- **Module 04/05 (Machine Learning 1/2)** — classification, training/validation, and
  in **05-P02** you already trained a **CNN from scratch** on Fashion-MNIST. This
  module deepens CNNs and adds the way that dominates today: **transfer learning**.
- **Module 09/10** — attention/transformer (for the **vision transformer** section).

---

## Learning objectives

After this module you can …

- explain the **convolution** as an operation and compute it by hand (kernel, stride,
  padding, receptive field) — and apply classical filters (Gaussian, Sobel);
- name the building blocks of a **CNN** (conv, pooling, activation, BatchNorm, FC) and
  explain *why* CNNs work well for images (local connectivity, weight sharing,
  hierarchy);
- **train a CNN from scratch** — describe the complete pipeline (data, augmentation,
  loss, optimization), including *why it needs a lot of data and compute*;
- **explain what a pretrained model is**, why the learned features are transferable,
  and apply the **three usage modes** (direct inference, feature extraction,
  fine-tuning) **precisely and practically** — including the exact torchvision API;
- place classical **architectures** (LeNet → AlexNet → VGG → ResNet) and understand the
  **vision transformer**;
- distinguish the central **CV tasks** (classification, object detection,
  segmentation) and name their metrics (accuracy, IoU, mAP).

---

## 1 · Basics — image, convolution, filter

### 1.1 What is an image for the computer?

A digital image is a **grid of pixels**. A grayscale image of size $H\times W$ is a
matrix of brightness values (usually $0$–$255$ or normalized to $[0,1]$). A color image
has **three channels** (red, green, blue) and is a **tensor** of shape $3\times H\times W$
(in PyTorch: `(C, H, W)`, in a batch `(N, C, H, W)`). Everything that follows operates on
these tensors.

### 1.2 The convolution

The **convolution** slides a small **kernel** (filter) $K$ of size $k\times k$ over the
image $I$ and computes a weighted local average at every position:
$$(I * K)(i,j)=\sum_{u=-a}^{a}\sum_{v=-a}^{a} I(i+u,\,j+v)\,K(u,v),\qquad a=\tfrac{k-1}{2}.$$
(In practice / in deep learning this is technically a **cross-correlation** — the kernel is
not flipped —, but everyone calls it "convolution".)

**Important terms:**
- **Padding** $p$: fill the border with zeros so that the output does not shrink. Without
  padding you lose $a$ pixels at each border.
- **Stride** $s$: the step size. $s=2$ skips every other position → half the resolution.
- **Output size:** $\displaystyle H_{\text{out}}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1$
  (analogously for the width).
- **Receptive field:** the image region that influences one output unit. It *grows* with
  the network depth — deep layers "see" larger image regions.

**A small worked example** — the vertical **Sobel kernel** (edge detector) on a 3×3 patch:
$$K_x=\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix},\quad
I=\begin{pmatrix}10&10&90\\10&10&90\\10&10&90\end{pmatrix}
\;\Rightarrow\; (I*K_x)_{\text{center}} = (-1\!\cdot\!10 + 1\!\cdot\!90)+(-2\!\cdot\!10+2\!\cdot\!90)+(-1\!\cdot\!10+1\!\cdot\!90)=320.$$
The large positive value indicates a **vertical edge** (dark on the left, bright on the
right). In a homogeneous region the response would be $0$. That is exactly how a filter
"finds" structure.

### 1.3 Classical filters (hand-designed)

- **Gaussian blur** (smoothing/noise reduction): a kernel with Gaussian weights, summing
  to $1$.
- **Sobel $K_x, K_y$**: derivatives in x/y; the **gradient magnitude**
  $G=\sqrt{G_x^2+G_y^2}$ highlights **edges** (the basis of the Canny detector).
- **Sharpening**: $\begin{smallmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{smallmatrix}$ emphasizes
  contrasts.

Historically such filters were designed **by hand** and feature descriptors like **SIFT**
and **HOG** were built on top. The central idea of CNNs is: **learn the filters from
data** instead of designing them. (Project 01 shows exactly this bridge: hand-designed
filters vs. the first filters *learned* by a network — they look strikingly similar.)

---

## 2 · Convolutional neural networks (CNNs)

### 2.1 Why not just an MLP?

A fully connected network (module 05) on a $224\times224\times3$ image would have
$\sim150{,}000$ inputs per neuron in the first layer — **billions** of weights, no use of
the 2D structure, no translation invariance. CNNs solve this through three principles:
1. **Local connectivity:** a neuron sees only a small window (the kernel).
2. **Weight sharing:** *the same* kernel slides over the whole image → few parameters, and
   a feature is recognized **everywhere** (translation equivariance).
3. **Hierarchy:** stack many layers → early ones learn edges/colors, middle ones
   textures/parts, late ones whole objects.

### 2.2 Building blocks

- **Conv layer:** $C_{\text{out}}$ filters of shape $C_{\text{in}}\times k\times k$;
  produces $C_{\text{out}}$ **feature maps**. Parameters: $C_{\text{out}}(C_{\text{in}}k^2+1)$.
- **Activation:** usually **ReLU** ($\max(0,x)$) — nonlinear, otherwise the stack would be
  linear.
- **Pooling:** **max pooling** $2\times2$ halves the resolution, makes the network more
  robust to small shifts and enlarges the receptive field. (Alternatively a strided conv.)
- **Batch normalization:** normalizes the activations per mini-batch (mean 0, variance 1,
  plus a learnable scale/shift) → faster, more stable training.
- **Fully connected (FC) head:** at the end the feature maps are "flattened" into a vector
  (or globally averaged, *global average pooling*) and mapped to the class logits.
  **Softmax + cross entropy** as in any classification.

A typical CNN is therefore: `[Conv → BN → ReLU → Pool] × N → GlobalAvgPool → FC → Softmax`.

### 2.3 Training a CNN **from scratch** (the "without pretrained" way)

Exactly what you do when you use **no** pretrained model — the full pipeline:

1. **Data:** many **labelled** images. Split into train/val/test. Preprocess (bring to a
   fixed size, to a tensor, **normalize** to mean 0 / variance 1 per channel).
2. **Data augmentation:** create artificial variety — random flipping, cropping, rotating,
   color jitter. This acts like "more data" and **reduces overfitting** (important because
   CNNs have many parameters).
3. **Model:** define the architecture from 2.2 (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`,
   `nn.MaxPool2d`, `nn.Linear`).
4. **Loss & optimization:** `CrossEntropyLoss`; optimizer **SGD with momentum** or
   **Adam**; a **learning-rate schedule** (e.g. step-wise decay). Train over many
   **epochs**, monitoring the **validation accuracy**.
5. **Regularization:** weight decay, dropout, early stopping against overfitting.

> **The catch — and why we usually do *not* do it this way:** for the early layers to
> learn good, general filters, a CNN needs **very many** labelled images (ImageNet: 1.2
> million images, 1000 classes) and **a lot of compute** (hours to days on GPUs). For most
> real tasks you have neither the amount of data nor the hardware. **This is exactly where
> transfer learning comes in** (section 3). — In **module 05-P02** you trained a small CNN
> from scratch; that only worked because Fashion-MNIST is tiny and simple. For real,
> high-resolution images the from-scratch way is impractical on a laptop.

### 2.4 Classical architectures (a short lineage)

- **LeNet-5** (LeCun, 1998): the first successful CNN (digit recognition).
- **AlexNet** (2012): won ImageNet, triggered the deep-learning wave — ReLU, dropout, GPU
  training.
- **VGG** (2014): very uniform, only $3\times3$ convs, very deep → showed that **depth**
  matters (but expensive).
- **ResNet** (2015): introduced **residual connections** — $y=\mathcal{F}(x)+x$. The
  "shortcut" lets gradients flow unimpeded through very deep networks and solves the
  **degradation/vanishing-gradient problem**. This made networks with 50–150+ layers
  trainable. ResNet is the workhorse to this day (and our default backbone).
- **Efficient networks** (MobileNet, EfficientNet): optimized for few parameters/FLOPs →
  they run even on CPU/phone. That is why we use **MobileNetV3** (small, fast) and
  **ResNet18** (small, classical) in the projects.

---

## 3 · Pretrained models & transfer learning  (the key part)

The most important practical part of this module.

### 3.1 What **is** a pretrained model?

A **pretrained model** is a neural network whose **weights have already been learned** —
typically by training on a **huge**, general dataset (almost always **ImageNet**: 1.2
million images, 1000 object classes). Someone (e.g. the PyTorch team) did the expensive
training work **once**; the result is the learned **parameters** (a file with millions of
numbers). When you load the model with these weights, you get a network that can already
**see** — it has learned usable visual features.

**Why is this useful (the core of transfer learning)?** The **early and middle layers** of
a CNN trained on ImageNet learn **general** features — edges, colors, textures, shapes,
object parts — that are **not** ImageNet-specific but useful for **almost any** image task.
Only the **last** layer (the classification head) is specialized to the 1000 ImageNet
classes. The idea: **keep the learned features, replace/adjust only the task-specific
part.** This transfers knowledge from the large source task to your own, often
data-scarce target task — with a fraction of the data and compute.

> This is the same principle as **pretraining in NLP** (module 09: BERT/GPT; module 10:
> XLM-R): first pretrain generally on huge data, then specialize to the concrete task.

### 3.2 The three usage modes — **exactly how you do it**

Assume we have a pretrained backbone (e.g. `resnet18`).

**(A) Direct inference** — use the model *unchanged* for its original task (ImageNet
classification). No training. Application: "what is in this image?" among the 1000 ImageNet
classes.

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1     # a concrete set of weights
model = resnet18(weights=weights)            # loads the architecture WITH learned weights
model.eval()                                 # inference mode (freeze BatchNorm/dropout)

preprocess = weights.transforms()            # EXACTLY the preprocessing it was trained with
#   -> resize/center-crop to 224, to tensor, normalize with ImageNet mean/std
x = preprocess(pil_image).unsqueeze(0)       # (1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)                        # (1, 1000)
pred = logits.argmax(1).item()
label = weights.meta["categories"][pred]     # class name
```

> **Critical:** a pretrained model **must** be fed with **exactly the preprocessing** it
> was trained with — in particular the **normalization** with the ImageNet mean
> `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. `weights.transforms()` provides
> this pipeline ready-made. Wrong normalization → poor results.

**(B) Feature extraction** ("freeze" the backbone) — use the network as a **fixed feature
extractor**. You remove the ImageNet head, send your own images through the **frozen**
network and obtain a **feature vector** (embedding) per image. On top of that you train a
**new, small classifier** (e.g. logistic regression or an FC layer) for *your* classes.
Very **cheap** (only forward passes, no backprop through the backbone) — exactly what we
use in project 02.

```python
model = resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False                  # freeze: no gradients
model.fc = torch.nn.Identity()               # remove the classification head -> gives 512-dim features
model.eval()
with torch.no_grad():
    features = model(preprocess_batch)       # (N, 512) embeddings
# train your own classifier on top (LogReg / a small nn.Linear)
```

**(C) Fine-tuning** — **keep training** the backbone (fully or partially). You replace the
head with a new one for your own classes and keep training the network with a **small
learning rate**, so that the pretrained weights are only *gently* adapted to the new task.
Often you freeze the early layers (those are already good) and only adjust the later ones.
Usually yields the **highest accuracy**, but costs more compute than (B).

```python
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)   # new head
# optional: freeze the early layers; only train model.layer4 + model.fc
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)       # SMALL learning rate!
# a normal training loop (no teacher forcing; standard classification)
```

**Rule of thumb — which mode?**

| Situation | Recommendation |
|---|---|
| Task = ImageNet classes, no own labels | **(A)** direct inference |
| Few own data, target similar to ImageNet | **(B)** feature extraction (fast, robust) |
| Medium/many own data, different domain, max. accuracy | **(C)** fine-tuning |
| Very much data + compute + exotic domain | possibly **from scratch** (section 2.3) |

### 3.3 From scratch vs. pretrained — the comparison

| | From scratch | Pretrained (transfer) |
|---|---|---|
| Data needed | very high (10⁴–10⁶+) | low (10²–10³ often suffice) |
| Compute | hours–days (GPU) | minutes (often CPU) |
| Starting point | random weights | learned, general features |
| Typical accuracy with little data | low (overfits) | high |

Project 02 shows this quantitatively: on **satellite images** (EuroSAT — a completely
different domain from ImageNet!) a frozen ImageNet backbone raises the accuracy from
**~0.41** (a classifier on raw pixels) to **~0.94** (a classifier on pretrained features) —
*without* training the backbone. That is the power of transferable features.

---

## 4 · Beyond classification — the CV tasks

- **Image classification:** one label per image ("cat"). Metric: **accuracy** (top-1/top-5).
- **Object detection:** *several* objects with **bounding boxes** + class. Families:
  **R-CNN → Fast/Faster R-CNN** (region proposals + classification) and **single-stage**
  detectors **YOLO/SSD** (fast, one pass). Core metrics: **IoU** (*intersection over union*
  of two boxes) and **mAP** (mean average precision over classes/IoU thresholds).
- **Semantic segmentation:** *every pixel* gets a class (road/car/sky), but instances are
  not separated. Architectures: **FCN**, **U-Net** (encoder-decoder with skip
  connections), **DeepLab**.
- **Instance segmentation:** pixel masks **per object instance** — **Mask R-CNN**.

All of these models exist in torchvision **pretrained** (`torchvision.models.detection`,
`...segmentation`) and are used analogously to section 3.

---

## 5 · Vision transformer (ViT) — attention for images

The **vision transformer** (Dosovitskiy et al. 2021) carries the transformer from
modules 09/10 over to images — **without** convolution:

1. Split the image into fixed **patches** (e.g. $16\times16$). Each patch is linearly
   embedded into a vector → a **sequence of "image tokens"** (patches = tokens, exactly the
   tokenization analogy from NLP).
2. Add **positional embeddings** (patches have a position) and a learnable **[CLS] token**.
3. Send the sequence through a **standard transformer encoder** (self-attention as in
   module 09). The [CLS] token output goes into a classification head.

**Trade-off:** ViTs have fewer built-in image priors (*inductive bias*) than CNNs (no
locality/translation equivariance "for free") and therefore need **even more data** or
strong pretraining — which is why they are almost always used **pretrained**. Modern
hybrids (ConvNeXt, Swin) combine both worlds. ViTs too are available pretrained in
torchvision and are used as in section 3.

---

## 6 · Summary / cheat sheet

| Term | Core in one sentence |
|---|---|
| **Image tensor** | `(C,H,W)` resp. `(N,C,H,W)`; values normalized per channel. |
| **Convolution** | a kernel slides over the image, a weighted local sum; padding/stride control the size. |
| **Output size** | $\lfloor (H+2p-k)/s\rfloor+1$. |
| **Receptive field** | the image region that influences one output; grows with depth. |
| **Sobel / Gaussian** | hand-designed edge / smoothing filters; CNNs *learn* such filters. |
| **CNN principles** | local connectivity, weight sharing, feature hierarchy. |
| **Pooling** | reduce the resolution, increase robustness + receptive field. |
| **BatchNorm** | normalize activations per batch → more stable training. |
| **ResNet / residual** | $y=\mathcal F(x)+x$; the shortcut enables very deep networks. |
| **Pretrained model** | a network with general weights learned on huge data (ImageNet). |
| **Transfer (A/B/C)** | inference / feature extraction (freeze the backbone) / fine-tuning (continue with a small LR). |
| **`weights.transforms()`** | provides the exact matching preprocessing (incl. ImageNet normalization). |
| **Detection** | boxes+classes; R-CNN vs. YOLO/SSD; metrics IoU, mAP. |
| **Segmentation** | per-pixel class (FCN/U-Net) resp. per-instance (Mask R-CNN). |
| **ViT** | image = a patch sequence → transformer encoder; needs a lot of pretraining. |

**Formulas to remember:** convolution $(I*K)(i,j)=\sum_{u,v}I(i+u,j+v)K(u,v)$; output size
$\lfloor(H+2p-k)/s\rfloor+1$; residual $y=\mathcal F(x)+x$;
IoU $=\frac{|A\cap B|}{|A\cup B|}$.

---

## 7 · Self-test

<details><summary><b>1.</b> A 5×5 kernel, stride 2, padding 2 on a 64×64 image — how large is the output?</summary>

$\lfloor(64+2\cdot2-5)/2\rfloor+1=\lfloor63/2\rfloor+1=31+1=32$. So $32\times32$.
</details>

<details><summary><b>2.</b> Name the three principles with which CNNs avoid the parameter explosion of an MLP.</summary>

Local connectivity (a neuron sees only a small window), weight sharing (one kernel slides
over the whole image → few parameters + translation equivariance) and feature hierarchy
(stacked layers from edges to objects).
</details>

<details><summary><b>3.</b> What is a pretrained model, and *why* are its features transferable?</summary>

A network whose weights have already been learned on a large, general dataset (ImageNet).
Its early/middle layers encode **general** visual features (edges, textures, parts) that
are not dataset-specific but useful for almost any image task — only the last head is
specialized to the source classes. Therefore one can keep the features and only adjust the
head.
</details>

<details><summary><b>4.</b> Describe the three usage modes of a pretrained model and when to choose which.</summary>

(A) **Direct inference** — unchanged for the original task (ImageNet classes), no training.
(B) **Feature extraction** — freeze the backbone, remove the head, produce embeddings, train
a new small classifier; with little data/a similar domain, very cheap. (C) **Fine-tuning** —
set a new head and keep training the network with a small learning rate; with more data/a
different domain for maximum accuracy.
</details>

<details><summary><b>5.</b> Why is the *normalization* critical when using a pretrained model, and how do you get the right one?</summary>

The network has learned to expect inputs in a certain statistic (ImageNet mean/std per
channel). If you feed differently scaled/normalized images, the activations do not match the
learned weights → poor results. `weights.transforms()` provides the exact matching
preprocessing (resize/crop + exactly this normalization).
</details>

<details><summary><b>6.</b> Why can't you simply make a CNN arbitrarily deep, and how do ResNets solve it?</summary>

Very deep networks suffer from vanishing/degrading gradients — training gets *worse*, not
just slower. **Residual connections** $y=\mathcal F(x)+x$ give the gradient an unimpeded
"shortcut" path, so that very deep networks (50–150+ layers) become trainable.
</details>

<details><summary><b>7.</b> Why does training *from scratch* need so much more data than transfer learning?</summary>

From scratch starts with random weights — the network has to learn *all* features (from
edges to objects) itself from the labels, which requires many examples, otherwise it
overfits. Transfer learning starts with already learned, general features and only has to
learn the task-specific mapping — that works with few examples.
</details>

<details><summary><b>8.</b> What does IoU measure, and what is it used for in object detection?</summary>

IoU (intersection over union) measures the overlap of two bounding boxes:
$|A\cap B|/|A\cup B|\in[0,1]$. It decides whether a predicted box counts as a "hit" of the
ground truth (e.g. IoU $\ge 0.5$) and enters the mAP computation.
</details>

<details><summary><b>9.</b> How does a vision transformer "tokenize" an image, and what is the price compared to a CNN?</summary>

It splits the image into fixed patches (e.g. 16×16), embeds each patch linearly and treats
the patch sequence as a token sequence for a transformer encoder. Price: ViTs have fewer
built-in image priors (locality, translation equivariance) and therefore need more
data/pretraining to work well.
</details>

<details><summary><b>10.</b> You have 500 labelled images of a niche domain and a laptop. Which way do you choose and why?</summary>

**Feature extraction (B)** with a small pretrained backbone (e.g. ResNet18/MobileNet): 500
images are too few for from-scratch training, but enough to train a small classifier on
frozen pretrained features — fast, CPU-capable, robust against overfitting. With a bit more
data, possibly light fine-tuning of the top layers.
</details>

---

## 8 · Literature & sources

*Legend: (free) = freely available, (beginner) = beginner-friendly, (in-depth) = advanced.*

**Textbooks / courses (free)**
- **Stanford CS231n** *Convolutional Neural Networks for Visual Recognition* — notes +
  videos. The reference course for CV. (beginner to in-depth)
- Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., free online) — broad
  classical + modern CV. (in-depth, free)
- Goodfellow, Bengio, Courville, *Deep Learning* — ch. 9 (convolutional networks). (free)
- **PyTorch tutorials**: *Transfer Learning for Computer Vision* & *torchvision models* —
  exactly the API of this module. (beginner, free)

**Key papers (free on arXiv)**
- Krizhevsky et al. (2012): *ImageNet Classification with Deep CNNs* (**AlexNet**).
- Simonyan & Zisserman (2014): *Very Deep CNNs* (**VGG**).
- He et al. (2015): *Deep Residual Learning* (**ResNet**).
- Howard et al. (2017/2019): *MobileNets* / *MobileNetV3*.
- Yosinski et al. (2014): *How transferable are features in deep neural networks?* — the
  empirical foundation of transfer learning.
- Ren et al. (2015): *Faster R-CNN*; Redmon et al. (2016): *YOLO*; Ronneberger et al. (2015):
  *U-Net*; He et al. (2017): *Mask R-CNN*.
- Dosovitskiy et al. (2021): *An Image is Worth 16×16 Words* (**ViT**).

**Interactive / blogs (free)**
- *CNN Explainer* (poloclub.github.io/cnn-explainer) — an interactive CNN visualization. (beginner)
- Distill.pub — *Feature Visualization*, *Building Blocks of Interpretability*.
- torchvision docs: models, weights, `transforms` — the practical reference.

---

## The three projects

All projects are **CPU-friendly** (no GPU/no long training) and use real images:

- **01 – basic** (`projects/01-basic/`): **Convolution & filters by hand + learned
  filters.** Guided notebook: implement 2D convolution yourself, apply classical filters
  (Gaussian, Sobel, gradient magnitude) to a real image — and then visualize the first
  filters learned by a **pretrained ResNet** (strikingly similar to the hand-designed
  ones). Plenty of instruction, no training.
- **02 – medium** (`projects/02-medium/`): **Transfer learning as a feature extractor.**
  Python project with a test suite: use a frozen pretrained backbone as a feature extractor
  on **EuroSAT** (satellite images), train a small classifier and put it up against a raw
  pixel baseline (~0.94 vs. ~0.41). Little instruction.
- **03 – final** (`projects/03-final/`): **Image classifier for a new domain — three ways
  compared.** No code given: from-scratch CNN vs. feature extraction vs. (light)
  fine-tuning on EuroSAT, with clean evaluation and error analysis. The master's-level
  capstone — consolidates "without pretrained" **and** "with pretrained".

Details, setup and reference solutions are in the `README.md` of each project folder.

---
# Modul 11 — Computer Vision (deutsche Fassung)

> **Worum geht es?** Wie bringt man einem Computer bei, **Bilder zu verstehen** —
> Objekte zu erkennen, Szenen zu klassifizieren, Regionen zu segmentieren? Wir beginnen
> bei der **Faltung** (der Grundoperation der Bildverarbeitung), bauen zu **Convolutional
> Neural Networks (CNNs)** auf und widmen dann einen großen Teil den **vortrainierten
> Modellen** und dem **Transfer Learning** — dem Ansatz, mit dem man heute *ohne* großes
> eigenes Training und *ohne* GPU-Cluster starke Bildmodelle baut. Genau das nutzen wir in
> den Projekten. Damit du aber *verstehst*, was da passiert, erklärt dieses Skript
> ausführlich **beides**: wie man ein Netz **von Grund auf** trainieren würde *und* wie und
> warum vortrainierte Modelle funktionieren.

**Hilfreiche Vorkenntnisse:** lineare Algebra (Matrizen/Tensoren), Grundlagen neuronaler
Netze und Gradientenabstieg, PyTorch-Basics.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 04/05 (Machine Learning 1/2)** — Klassifikation, Training/Validierung, und in
  **05-P02** hast du bereits ein **CNN von Grund auf** auf Fashion-MNIST trainiert. Dieses
  Modul vertieft CNNs und ergänzt den heute dominierenden Weg: **Transfer Learning**.
- **Modul 09/10** — Attention/Transformer (für den Abschnitt **Vision Transformer**).

---

## Lernziele

Nach diesem Modul kannst du …

- die **Faltung** (Convolution) als Operation erklären und von Hand rechnen (Kernel,
  Stride, Padding, Rezeptives Feld) — und klassische Filter (Gauß, Sobel) anwenden;
- die Bausteine eines **CNN** benennen (Conv, Pooling, Aktivierung, BatchNorm, FC) und
  erklären, *warum* CNNs für Bilder gut funktionieren (lokale Konnektivität, Gewichts-
  Teilung, Hierarchie);
- ein CNN **von Grund auf trainieren** — die komplette Pipeline (Daten, Augmentation,
  Verlust, Optimierung) beschreiben, inklusive *warum das viele Daten und Rechenzeit
  braucht*;
- **erklären, was ein vortrainiertes Modell ist**, warum die gelernten Features
  übertragbar sind, und die **drei Nutzungsarten** (direkte Inferenz, Feature-Extraktion,
  Fine-Tuning) **präzise und praktisch** anwenden — inklusive der genauen torchvision-API;
- klassische **Architekturen** einordnen (LeNet → AlexNet → VGG → ResNet) und den **Vision
  Transformer** verstehen;
- die zentralen **CV-Aufgaben** unterscheiden (Klassifikation, Objekterkennung,
  Segmentierung) und ihre Metriken (Accuracy, IoU, mAP) benennen.

---

## 1 · Grundlagen — Bild, Faltung, Filter

### 1.1 Was ist ein Bild für den Computer?

Ein digitales Bild ist ein **Raster von Pixeln**. Ein Graustufenbild der Größe $H\times W$
ist eine Matrix von Helligkeitswerten (meist $0$–$255$ oder normiert $[0,1]$). Ein
Farbbild hat **drei Kanäle** (Rot, Grün, Blau) und ist ein **Tensor** der Form
$3\times H\times W$ (in PyTorch: `(C, H, W)`, im Batch `(N, C, H, W)`). Alles, was folgt,
operiert auf diesen Tensoren.

### 1.2 Die Faltung (Convolution)

Die **Faltung** schiebt einen kleinen **Kernel** (Filter) $K$ der Größe $k\times k$ über
das Bild $I$ und berechnet an jeder Position ein gewichtetes Lokal-Mittel:
$$(I * K)(i,j)=\sum_{u=-a}^{a}\sum_{v=-a}^{a} I(i+u,\,j+v)\,K(u,v),\qquad a=\tfrac{k-1}{2}.$$
(In der Praxis/im Deep Learning ist das technisch eine **Kreuzkorrelation** — der Kernel
wird nicht gespiegelt —, aber alle nennen es „Convolution".)

**Wichtige Begriffe:**
- **Padding** $p$: Rand mit Nullen auffüllen, damit die Ausgabe nicht schrumpft. Ohne
  Padding verliert man an jedem Rand $a$ Pixel.
- **Stride** $s$: Schrittweite. $s=2$ überspringt jede zweite Position → halbe Auflösung.
- **Ausgabegröße:** $\displaystyle H_{\text{out}}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1$
  (analog für die Breite).
- **Rezeptives Feld:** der Bildbereich, der eine Ausgabe-Einheit beeinflusst. Es *wächst*
  mit der Netztiefe — tiefe Schichten „sehen" größere Bildregionen.

**Kleines durchgerechnetes Beispiel** — vertikaler **Sobel-Kernel** (Kantendetektor) auf
einem 3×3-Ausschnitt:
$$K_x=\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix},\quad
I=\begin{pmatrix}10&10&90\\10&10&90\\10&10&90\end{pmatrix}
\;\Rightarrow\; (I*K_x)_{\text{Mitte}} = (-1\!\cdot\!10 + 1\!\cdot\!90)+(-2\!\cdot\!10+2\!\cdot\!90)+(-1\!\cdot\!10+1\!\cdot\!90)=320.$$
Der große positive Wert zeigt eine **vertikale Kante** (links dunkel, rechts hell) an. In
einer homogenen Region wäre die Antwort $0$. Genau so „findet" ein Filter Struktur.

### 1.3 Klassische Filter (hand-entworfen)

- **Gauß-Weichzeichner** (Glättung/Rauschunterdrückung): Kernel mit Gauß-Gewichten,
  Summe $1$.
- **Sobel $K_x, K_y$**: Ableitungen in x/y; **Gradientenbetrag**
  $G=\sqrt{G_x^2+G_y^2}$ hebt **Kanten** hervor (Grundlage des Canny-Detektors).
- **Schärfen**: $\begin{smallmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{smallmatrix}$ betont
  Kontraste.

Historisch entwarf man solche Filter **von Hand** und baute Feature-Deskriptoren wie
**SIFT** und **HOG** darauf auf. Die zentrale Idee von CNNs ist: **Lerne die Filter aus
Daten**, statt sie zu entwerfen. (Projekt 01 zeigt genau diese Brücke: hand-entworfene
Filter vs. die von einem Netz *gelernten* ersten Filter — sie sehen sich verblüffend
ähnlich.)

---

## 2 · Convolutional Neural Networks (CNNs)

### 2.1 Warum nicht einfach ein MLP?

Ein voll verbundenes Netz (Modul 05) auf einem $224\times224\times3$-Bild hätte in der
ersten Schicht $\sim150\,000$ Eingänge pro Neuron — **Milliarden** Gewichte, keine
Nutzung der 2D-Struktur, keine Translationsinvarianz. CNNs lösen das durch drei Prinzipien:
1. **Lokale Konnektivität:** ein Neuron sieht nur ein kleines Fenster (den Kernel).
2. **Gewichts-Teilung (weight sharing):** *derselbe* Kernel gleitet über das ganze Bild →
   wenige Parameter, und ein Merkmal wird **überall** erkannt (Translationsäquivarianz).
3. **Hierarchie:** viele Schichten stapeln → frühe lernen Kanten/Farben, mittlere
   Texturen/Teile, späte ganze Objekte.

### 2.2 Bausteine

- **Conv-Schicht:** $C_{\text{out}}$ Filter der Form $C_{\text{in}}\times k\times k$;
  erzeugt $C_{\text{out}}$ **Feature-Maps**. Parameter: $C_{\text{out}}(C_{\text{in}}k^2+1)$.
- **Aktivierung:** meist **ReLU** ($\max(0,x)$) — nichtlinear, sonst wäre der Stapel linear.
- **Pooling:** **Max-Pooling** $2\times2$ halbiert die Auflösung, macht robuster gegen
  kleine Verschiebungen und vergrößert das rezeptive Feld. (Alternativ Stride-Conv.)
- **Batch-Normalisierung:** normiert Aktivierungen pro Mini-Batch (Mittel 0, Varianz 1,
  plus lernbare Skala/Verschiebung) → schnelleres, stabileres Training.
- **Fully-Connected (FC) Kopf:** am Ende werden die Feature-Maps zu einem Vektor
  „geplättet" (oder global gemittelt, *Global Average Pooling*) und auf die Klassen-Logits
  abgebildet. **Softmax + Cross-Entropy** wie bei jeder Klassifikation.

Ein typisches CNN ist also: `[Conv → BN → ReLU → Pool] × N → GlobalAvgPool → FC → Softmax`.

### 2.3 Ein CNN **von Grund auf** trainieren (der „ohne pretrained"-Weg)

Genau das, was man macht, wenn man **kein** vortrainiertes Modell nutzt — die volle
Pipeline:

1. **Daten:** viele **gelabelte** Bilder. Aufteilen in Train/Val/Test. Vorverarbeiten
   (auf feste Größe bringen, zu Tensor, **normalisieren** auf Mittel 0 / Varianz 1 pro
   Kanal).
2. **Data Augmentation:** künstliche Vielfalt erzeugen — zufälliges Spiegeln, Zuschneiden,
   Rotieren, Farb-Jitter. Das wirkt wie „mehr Daten" und **reduziert Overfitting** (wichtig,
   weil CNNs viele Parameter haben).
3. **Modell:** die Architektur aus 2.2 definieren (`nn.Conv2d`, `nn.BatchNorm2d`,
   `nn.ReLU`, `nn.MaxPool2d`, `nn.Linear`).
4. **Verlust & Optimierung:** `CrossEntropyLoss`; Optimierer **SGD mit Momentum** oder
   **Adam**; eine **Lernraten-Strategie** (z. B. schrittweises Absenken). Über viele
   **Epochen** trainieren, dabei die **Validierungs-Accuracy** überwachen.
5. **Regularisierung:** Weight Decay, Dropout, Early Stopping gegen Overfitting.

> **Der Haken — und warum wir es meist *nicht* so machen:** Damit die frühen Schichten
> gute, allgemeine Filter lernen, braucht ein CNN **sehr viele** gelabelte Bilder
> (ImageNet: 1,2 Mio. Bilder, 1000 Klassen) und **viel Rechenzeit** (Stunden bis Tage auf
> GPUs). Für die meisten realen Aufgaben hat man weder die Datenmenge noch die Hardware.
> **Genau hier setzt Transfer Learning an** (Abschnitt 3). — In **Modul 05-P02** hast du ein
> kleines CNN from scratch trainiert; das ging nur, weil Fashion-MNIST winzig und einfach
> ist. Für echte, hochauflösende Bilder ist der From-Scratch-Weg auf einem Laptop
> unpraktikabel.

### 2.4 Klassische Architekturen (kurze Ahnenreihe)

- **LeNet-5** (LeCun, 1998): das erste erfolgreiche CNN (Ziffernerkennung).
- **AlexNet** (2012): gewann ImageNet, löste die Deep-Learning-Welle aus — ReLU, Dropout,
  GPU-Training.
- **VGG** (2014): sehr einheitlich, nur $3\times3$-Convs, sehr tief → zeigte, dass **Tiefe**
  zählt (aber teuer).
- **ResNet** (2015): führte **Residual-Verbindungen** ein — $y=\mathcal{F}(x)+x$. Der
  „Shortcut" lässt Gradienten ungehindert durch sehr tiefe Netze fließen und löst das
  **Degradations-/Vanishing-Gradient-Problem**. Damit wurden Netze mit 50–150+ Schichten
  trainierbar. ResNet ist bis heute das Arbeitspferd (und unser Standard-Backbone).
- **Effiziente Netze** (MobileNet, EfficientNet): auf wenige Parameter/FLOPs optimiert →
  laufen sogar auf CPU/Handy. Deshalb nutzen wir in den Projekten **MobileNetV3** (klein,
  schnell) und **ResNet18** (klein, klassisch).

---

## 3 · Vortrainierte Modelle & Transfer Learning  ⭐

Der wichtigste praktische Teil dieses Moduls.

### 3.1 Was **ist** ein vortrainiertes Modell?

Ein **vortrainiertes (pretrained) Modell** ist ein neuronales Netz, dessen **Gewichte
bereits gelernt wurden** — typischerweise durch Training auf einem **riesigen**, allgemeinen
Datensatz (fast immer **ImageNet**: 1,2 Mio. Bilder, 1000 Objektklassen). Jemand (z. B. das
PyTorch-Team) hat die teure Trainingsarbeit **einmal** gemacht; das Ergebnis sind die
gelernten **Parameter** (eine Datei mit Millionen Zahlen). Wenn du das Modell mit diesen
Gewichten lädst, bekommst du ein Netz, das bereits **sehen kann** — es hat brauchbare
visuelle Merkmale gelernt.

**Warum ist das nützlich (der Kern des Transfer Learning)?** Die **frühen und mittleren
Schichten** eines auf ImageNet trainierten CNN lernen **allgemeine** Merkmale — Kanten,
Farben, Texturen, Formen, Objektteile —, die **nicht** ImageNet-spezifisch sind, sondern
für **fast jede** Bildaufgabe nützlich. Nur die **letzte** Schicht (der Klassifikations-
kopf) ist auf die 1000 ImageNet-Klassen spezialisiert. Idee: **behalte die gelernten
Merkmale, ersetze/justiere nur den aufgabenspezifischen Teil.** So überträgt man Wissen von
der großen Quell-Aufgabe auf die eigene, oft datenarme Ziel-Aufgabe — mit einem Bruchteil
der Daten und Rechenzeit.

> Das ist dasselbe Prinzip wie beim **Vortraining in NLP** (Modul 09: BERT/GPT; Modul 10:
> XLM-R): erst allgemein auf riesigen Daten vortrainieren, dann auf die konkrete Aufgabe
> spezialisieren.

### 3.2 Die drei Nutzungsarten — **genau, wie man es macht**

Angenommen, wir haben ein pretrained Backbone (z. B. `resnet18`).

**(A) Direkte Inferenz** — das Modell *unverändert* für seine ursprüngliche Aufgabe nutzen
(ImageNet-Klassifikation). Kein Training. Anwendung: „Was ist auf diesem Bild?" unter den
1000 ImageNet-Klassen.

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1     # ein konkretes Gewichts-Set
model = resnet18(weights=weights)            # lädt die Architektur MIT gelernten Gewichten
model.eval()                                 # Inferenzmodus (BatchNorm/Dropout einfrieren)

preprocess = weights.transforms()            # GENAU die Vorverarbeitung, mit der trainiert wurde
#   -> resize/center-crop auf 224, zu Tensor, Normalisierung mit ImageNet-Mittel/Std
x = preprocess(pil_image).unsqueeze(0)       # (1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)                        # (1, 1000)
pred = logits.argmax(1).item()
label = weights.meta["categories"][pred]     # Klassenname
```

> **Kritisch:** Ein pretrained Modell **muss** mit **genau der Vorverarbeitung** gefüttert
> werden, mit der es trainiert wurde — insbesondere die **Normalisierung** mit dem
> ImageNet-Mittel `[0.485, 0.456, 0.406]` und Std `[0.229, 0.224, 0.225]`. `weights.
> transforms()` liefert diese Pipeline fertig. Falsche Normalisierung → schlechte Ergebnisse.

**(B) Feature-Extraktion** (das Backbone „einfrieren") — das Netz als **fixen
Merkmalsextraktor** nutzen. Man entfernt den ImageNet-Kopf, schickt eigene Bilder durch das
**eingefrorene** Netz und erhält pro Bild einen **Feature-Vektor** (Embedding). Darauf
trainiert man einen **neuen, kleinen Klassifikator** (z. B. logistische Regression oder eine
FC-Schicht) für die *eigenen* Klassen. Sehr **billig** (nur Vorwärtsdurchläufe, kein
Backprop durchs Backbone) — genau das nutzen wir in Projekt 02.

```python
model = resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False                  # einfrieren: keine Gradienten
model.fc = torch.nn.Identity()               # Klassifikationskopf entfernen -> gibt 512-dim Features
model.eval()
with torch.no_grad():
    features = model(preprocess_batch)       # (N, 512) Embeddings
# darauf einen eigenen Klassifikator trainieren (LogReg / kleine nn.Linear)
```

**(C) Fine-Tuning** — das Backbone (ganz oder teilweise) **weiter-trainieren**. Man ersetzt
den Kopf durch einen neuen für die eigenen Klassen und trainiert das Netz mit einer
**kleinen Lernrate** weiter, sodass die vortrainierten Gewichte nur *sanft* an die neue
Aufgabe angepasst werden. Oft friert man die frühen Schichten ein (die sind schon gut) und
justiert nur die späteren. Bringt meist die **höchste Genauigkeit**, kostet aber mehr
Rechenzeit als (B).

```python
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_KLASSEN)   # neuer Kopf
# optional: frühe Schichten einfrieren; nur model.layer4 + model.fc trainieren
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)       # KLEINE Lernrate!
# normale Trainingsschleife (Teacher-Forcing-frei; Standard-Klassifikation)
```

**Faustregel — welchen Modus?**

| Situation | Empfehlung |
|---|---|
| Aufgabe = ImageNet-Klassen, kein eigenes Label | **(A)** direkte Inferenz |
| Wenige eigene Daten, Ziel ähnelt ImageNet | **(B)** Feature-Extraktion (schnell, robust) |
| Mittel/viele eigene Daten, andere Domäne, max. Genauigkeit | **(C)** Fine-Tuning |
| Sehr viele Daten + Rechenpower + exotische Domäne | ggf. **from scratch** (Abschnitt 2.3) |

### 3.3 From scratch vs. pretrained — der Vergleich

| | From scratch | Pretrained (Transfer) |
|---|---|---|
| Datenbedarf | sehr hoch (10⁴–10⁶+) | gering (10²–10³ genügen oft) |
| Rechenzeit | Stunden–Tage (GPU) | Minuten (oft CPU) |
| Startpunkt | zufällige Gewichte | gelernte, allgemeine Merkmale |
| Typische Genauigkeit bei wenig Daten | niedrig (overfittet) | hoch |

Projekt 02 zeigt das quantitativ: auf **Satellitenbildern** (EuroSAT — eine ganz andere
Domäne als ImageNet!) hebt ein eingefrorenes ImageNet-Backbone die Genauigkeit von
**~0.41** (Klassifikator auf rohen Pixeln) auf **~0.94** (Klassifikator auf pretrained
Features) — *ohne* das Backbone zu trainieren. Das ist die Kraft übertragbarer Merkmale.

---

## 4 · Über Klassifikation hinaus — die CV-Aufgaben

- **Bildklassifikation:** ein Label pro Bild („Katze"). Metrik: **Accuracy** (Top-1/Top-5).
- **Objekterkennung (Detection):** *mehrere* Objekte mit **Bounding Boxes** + Klasse.
  Familien: **R-CNN → Fast/Faster R-CNN** (Region-Vorschläge + Klassifikation) und
  **einstufige** Detektoren **YOLO/SSD** (schnell, ein Durchlauf). Kernmetrik: **IoU**
  (*Intersection over Union* zweier Boxen) und **mAP** (mean Average Precision über
  Klassen/IoU-Schwellen).
- **Semantische Segmentierung:** *jedes Pixel* bekommt eine Klasse (Straße/Auto/Himmel),
  aber Instanzen werden nicht getrennt. Architekturen: **FCN**, **U-Net**
  (Encoder-Decoder mit Skip-Connections), **DeepLab**.
- **Instanz-Segmentierung:** Pixelmasken **pro Objektinstanz** — **Mask R-CNN**.

Alle diese Modelle gibt es in torchvision **vortrainiert** (`torchvision.models.detection`,
`...segmentation`) und werden analog zu Abschnitt 3 genutzt.

---

## 5 · Vision Transformer (ViT) — Attention für Bilder

Der **Vision Transformer** (Dosovitskiy et al. 2021) überträgt den Transformer aus
Modul 09/10 auf Bilder — **ohne** Faltung:

1. Zerlege das Bild in feste **Patches** (z. B. $16\times16$). Jeder Patch wird linear zu
   einem Vektor eingebettet → eine **Sequenz von „Bild-Tokens"** (Patches = Tokens, exakt
   die Tokenisierungs-Analogie aus NLP).
2. Addiere **Positional Embeddings** (Patches haben eine Position) und einen lernbaren
   **[CLS]-Token**.
3. Schicke die Sequenz durch einen **Standard-Transformer-Encoder** (Self-Attention wie in
   Modul 09). Der [CLS]-Token-Output geht in einen Klassifikationskopf.

**Trade-off:** ViTs haben weniger eingebaute Bild-Vorannahmen (*inductive bias*) als CNNs
(keine Lokalität/Translationsäquivarianz „geschenkt") und brauchen daher **noch mehr
Daten** oder starkes Vortraining — weshalb man sie fast immer **pretrained** verwendet.
Moderne Hybride (ConvNeXt, Swin) verbinden beide Welten. Auch ViTs liegen in torchvision
vortrainiert bereit und werden wie in Abschnitt 3 genutzt.

---

## 6 · Zusammenfassung / Cheat-Sheet

| Begriff | Kern in einem Satz |
|---|---|
| **Bild-Tensor** | `(C,H,W)` bzw. `(N,C,H,W)`; Werte pro Kanal normiert. |
| **Faltung** | Kernel gleitet übers Bild, gewichtete Lokal-Summe; Padding/Stride steuern Größe. |
| **Ausgabegröße** | $\lfloor (H+2p-k)/s\rfloor+1$. |
| **Rezeptives Feld** | Bildbereich, der eine Ausgabe beeinflusst; wächst mit Tiefe. |
| **Sobel / Gauß** | hand-entworfene Kanten- / Glättungsfilter; CNNs *lernen* solche Filter. |
| **CNN-Prinzipien** | lokale Konnektivität, Gewichts-Teilung, Feature-Hierarchie. |
| **Pooling** | Auflösung reduzieren, Robustheit + rezeptives Feld erhöhen. |
| **BatchNorm** | Aktivierungen pro Batch normieren → stabileres Training. |
| **ResNet / Residual** | $y=\mathcal F(x)+x$; Shortcut ermöglicht sehr tiefe Netze. |
| **Pretrained Modell** | Netz mit auf Riesendaten (ImageNet) gelernten, allgemeinen Gewichten. |
| **Transfer (A/B/C)** | Inferenz / Feature-Extraktion (Backbone einfrieren) / Fine-Tuning (klein-LR weiter). |
| **`weights.transforms()`** | liefert die exakte, passende Vorverarbeitung (inkl. ImageNet-Normalisierung). |
| **Detection** | Boxes+Klassen; R-CNN vs. YOLO/SSD; Metriken IoU, mAP. |
| **Segmentierung** | pro-Pixel-Klasse (FCN/U-Net) bzw. pro-Instanz (Mask R-CNN). |
| **ViT** | Bild = Patch-Sequenz → Transformer-Encoder; braucht viel Vortraining. |

**Formeln zum Merken:** Faltung $(I*K)(i,j)=\sum_{u,v}I(i+u,j+v)K(u,v)$; Ausgabegröße
$\lfloor(H+2p-k)/s\rfloor+1$; Residual $y=\mathcal F(x)+x$;
IoU $=\frac{|A\cap B|}{|A\cup B|}$.

---

## 7 · Selbsttest

<details><summary><b>1.</b> Ein 5×5-Kernel, Stride 2, Padding 2 auf einem 64×64-Bild — wie groß ist die Ausgabe?</summary>

$\lfloor(64+2\cdot2-5)/2\rfloor+1=\lfloor63/2\rfloor+1=31+1=32$. Also $32\times32$.
</details>

<details><summary><b>2.</b> Nenne die drei Prinzipien, mit denen CNNs die Parameterexplosion eines MLP vermeiden.</summary>

Lokale Konnektivität (Neuron sieht nur ein kleines Fenster), Gewichts-Teilung (ein Kernel
gleitet über das ganze Bild → wenige Parameter + Translationsäquivarianz) und Feature-
Hierarchie (gestapelte Schichten von Kanten zu Objekten).
</details>

<details><summary><b>3.</b> Was ist ein vortrainiertes Modell, und *warum* sind seine Merkmale übertragbar?</summary>

Ein Netz, dessen Gewichte bereits auf einem großen, allgemeinen Datensatz (ImageNet)
gelernt wurden. Seine frühen/mittleren Schichten kodieren **allgemeine** visuelle Merkmale
(Kanten, Texturen, Teile), die nicht datensatzspezifisch sind, sondern für fast jede
Bildaufgabe nützlich — nur der letzte Kopf ist auf die Quell-Klassen spezialisiert.
Deshalb kann man die Merkmale behalten und nur den Kopf anpassen.
</details>

<details><summary><b>4.</b> Beschreibe die drei Nutzungsarten eines pretrained Modells und wann man welche wählt.</summary>

(A) **Direkte Inferenz** — unverändert für die Original-Aufgabe (ImageNet-Klassen), kein
Training. (B) **Feature-Extraktion** — Backbone einfrieren, Kopf entfernen, Embeddings
erzeugen, neuen kleinen Klassifikator trainieren; bei wenig Daten/ähnlicher Domäne, sehr
billig. (C) **Fine-Tuning** — neuen Kopf setzen und das Netz mit kleiner Lernrate
weitertrainieren; bei mehr Daten/anderer Domäne für maximale Genauigkeit.
</details>

<details><summary><b>5.</b> Warum ist die *Normalisierung* bei der Nutzung eines pretrained Modells kritisch, und wie bekommt man die richtige?</summary>

Das Netz hat gelernt, Eingaben in einer bestimmten Statistik zu erwarten (ImageNet-Mittel/
Std pro Kanal). Füttert man anders skalierte/normierte Bilder, passen die Aktivierungen
nicht zu den gelernten Gewichten → schlechte Ergebnisse. `weights.transforms()` liefert die
exakt passende Vorverarbeitung (Resize/Crop + genau diese Normalisierung).
</details>

<details><summary><b>6.</b> Warum kann man ein CNN nicht einfach beliebig tief machen, und wie lösen ResNets das?</summary>

Sehr tiefe Netze leiden unter verschwindenden/degradierenden Gradienten — das Training wird
*schlechter*, nicht nur langsamer. **Residual-Verbindungen** $y=\mathcal F(x)+x$ geben den
Gradienten einen ungehinderten „Shortcut"-Pfad, sodass sehr tiefe Netze (50–150+ Schichten)
trainierbar werden.
</details>

<details><summary><b>7.</b> Warum braucht Training *from scratch* so viel mehr Daten als Transfer Learning?</summary>

From scratch startet mit zufälligen Gewichten — das Netz muss *alle* Merkmale (von Kanten
bis Objekten) selbst aus den Labels lernen, was viele Beispiele erfordert, sonst overfittet
es. Transfer Learning startet mit bereits gelernten, allgemeinen Merkmalen und muss nur noch
die aufgabenspezifische Abbildung lernen — das geht mit wenigen Beispielen.
</details>

<details><summary><b>8.</b> Was misst IoU, und wozu dient es in der Objekterkennung?</summary>

IoU (Intersection over Union) misst die Überlappung zweier Bounding Boxes:
$|A\cap B|/|A\cup B|\in[0,1]$. Es entscheidet, ob eine vorhergesagte Box als „Treffer" der
Ground-Truth zählt (z. B. IoU $\ge 0.5$) und geht in die mAP-Berechnung ein.
</details>

<details><summary><b>9.</b> Wie „tokenisiert" ein Vision Transformer ein Bild, und was ist der Preis gegenüber einem CNN?</summary>

Er zerlegt das Bild in feste Patches (z. B. 16×16), bettet jeden Patch linear ein und
behandelt die Patch-Folge als Token-Sequenz für einen Transformer-Encoder. Preis: ViTs
haben weniger eingebaute Bild-Vorannahmen (Lokalität, Translationsäquivarianz) und brauchen
daher mehr Daten/Vortraining, um gut zu funktionieren.
</details>

<details><summary><b>10.</b> Du hast 500 gelabelte Bilder einer Nischen-Domäne und einen Laptop. Welchen Weg wählst du und warum?</summary>

**Feature-Extraktion (B)** mit einem kleinen pretrained Backbone (z. B. ResNet18/MobileNet):
500 Bilder sind zu wenig für From-Scratch-Training, aber genug, um auf eingefrorenen
pretrained Features einen kleinen Klassifikator zu trainieren — schnell, CPU-tauglich,
robust gegen Overfitting. Bei etwas mehr Daten ggf. leichtes Fine-Tuning der obersten
Schichten.
</details>

---

## 8 · Literatur & Quellen

**Lehrbücher / Kurse (kostenlos 💰)**
- **Stanford CS231n** *Convolutional Neural Networks for Visual Recognition* — Notes +
  Videos. Der Referenzkurs für CV. 🟢 *einsteigerfreundlich bis vertiefend*
- Szeliski, *Computer Vision: Algorithms and Applications* (2. Aufl., frei online) —
  breites klassisches + modernes CV. *vertiefend* 🟢
- Goodfellow, Bengio, Courville, *Deep Learning* — Kap. 9 (Convolutional Networks). 🟢
- **PyTorch-Tutorials**: *Transfer Learning for Computer Vision* & *torchvision models* —
  genau die API dieses Moduls. 🟢 *einsteigerfreundlich*

**Schlüssel-Papers (frei auf arXiv 💰)**
- Krizhevsky et al. (2012): *ImageNet Classification with Deep CNNs* (**AlexNet**). 🟢
- Simonyan & Zisserman (2014): *Very Deep CNNs* (**VGG**). 🟢
- He et al. (2015): *Deep Residual Learning* (**ResNet**). 🟢
- Howard et al. (2017/2019): *MobileNets* / *MobileNetV3*. 🟢
- Yosinski et al. (2014): *How transferable are features in deep neural networks?* — die
  empirische Grundlage des Transfer Learning. 🟢
- Ren et al. (2015): *Faster R-CNN*; Redmon et al. (2016): *YOLO*; Ronneberger et al. (2015):
  *U-Net*; He et al. (2017): *Mask R-CNN*. 🟢
- Dosovitskiy et al. (2021): *An Image is Worth 16×16 Words* (**ViT**). 🟢

**Interaktiv / Blogs (kostenlos 💰)**
- *CNN Explainer* (poloclub.github.io/cnn-explainer) — interaktive CNN-Visualisierung. 🟢 *einsteigerfreundlich*
- Distill.pub — *Feature Visualization*, *Building Blocks of Interpretability*. 🟢
- torchvision-Doku: Modelle, Gewichte, `transforms` — die praktische Referenz. 🟢

---

## Die drei Projekte

Alle Projekte sind **CPU-freundlich** (kein GPU/keine langen Trainings) und nutzen echte
Bilder:

- **01 – basic** (`projects/01-basic/`): **Faltung & Filter von Hand + gelernte Filter.**
  Geführtes Notebook: 2D-Faltung selbst implementieren, klassische Filter (Gauß, Sobel,
  Gradientenbetrag) auf ein echtes Bild anwenden — und dann die von einem **pretrained
  ResNet** gelernten ersten Filter visualisieren (verblüffend ähnlich zu den
  hand-entworfenen). Viel Anleitung, kein Training.
- **02 – medium** (`projects/02-medium/`): **Transfer Learning als Feature-Extraktor.**
  Python-Projekt mit Testsuite: ein eingefrorenes pretrained Backbone als Merkmalsextraktor
  auf **EuroSAT** (Satellitenbilder) nutzen, einen kleinen Klassifikator trainieren und
  gegen eine Rohpixel-Baseline stellen (~0.94 vs. ~0.41). Wenig Anleitung.
- **03 – final** (`projects/03-final/`): **Bildklassifikator für eine neue Domäne — drei
  Wege im Vergleich.** Keine Code-Vorgabe: from-scratch-CNN vs. Feature-Extraktion vs.
  (leichtes) Fine-Tuning auf EuroSAT, mit sauberer Evaluation und Fehleranalyse. Der
  Master-Level-Abschluss — konsolidiert „ohne pretrained" **und** „mit pretrained".

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
