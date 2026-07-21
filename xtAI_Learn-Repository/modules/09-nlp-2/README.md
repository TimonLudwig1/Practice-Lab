# Modul 09 — Natural Language Processing 2

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
