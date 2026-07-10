# Modul 10 — Multilingual NLP

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

- **01 – basic** (`projekte/01-basic/`): **Mehrsprachige Subword-Tokenisierung mit
  SentencePiece.** Geführtes Notebook: ein gemeinsames BPE-/Unigram-Vokabular auf DE+EN
  trainieren, Tokenisierung analysieren, **fertility** und Vokabular-Teilung zwischen den
  Sprachen messen, mit Wort-/Zeichen-Tokenisierung vergleichen. Viel Anleitung.
- **02 – medium** (`projekte/02-medium/`): **Cross-linguale Embeddings alignen
  (Procrustes + CSLS).** Python-Projekt mit Testsuite: monolinguale Embeddings pro Sprache,
  orthogonales Alignment über die SVD selbst herleiten, per **Precision@1** und **CSLS**
  ein Wörterbuch „übersetzen". Wenig Anleitung.
- **03 – final** (`projekte/03-final/`): **Ein NMT-Transformer von Grund auf (DE→EN).**
  Keine Code-Vorgabe: vollständiger Encoder-Decoder mit **Cross-Attention**, Teacher
  Forcing, Greedy/Beam-Decoding und **BLEU**-Evaluation auf Tatoeba. Der Master-Level-
  Abschluss, der Modul 09 und 10 zusammenführt.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
