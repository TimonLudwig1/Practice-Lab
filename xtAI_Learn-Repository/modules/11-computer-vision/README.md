# Modul 11 — Computer Vision

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
