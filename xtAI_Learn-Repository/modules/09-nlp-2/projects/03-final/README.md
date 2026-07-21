# Projekt 03 (final) — Ein Zeichen-Level-GPT von Grund auf

**Modul 09 — NLP 2** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieses Format & dieser Praxisbezug?

Das Abschlussprojekt konsolidiert das ganze Modul: Du baust einen **Decoder-only-
Transformer (GPT)** und trainierst ihn **autoregressiv** — dieselbe Modellklasse, die
hinter GPT-2/3/4 und praktisch allen modernen LLMs steht, nur klein. Auf **Zeichen-Ebene**
(statt Subword-BPE), damit die ganze Pipeline ohne externe Tokenizer und ohne GPU-Cluster
in **wenigen Minuten** läuft und du *jeden* Baustein selbst schreibst. Das ist der
direkte, ehrliche „build GPT from scratch"-Praxisbezug: am Ende generiert *dein* Modell
neuen Text im Stil des Trainingskorpus.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Du entwirfst Dateien,
Klassen und Trainingsschleife selbst und wendest an, was du in Projekt 01 (Embeddings,
Training) und 02 (Self-Attention, Multi-Head, Encoder-Block, Masken) gelernt hast. Eine
vollständige, getestete Musterlösung liegt in [`solution/`](solution/) — **erst selbst
bauen.**

## Ziel

Trainiere ein Sprachmodell, das den **nächsten Buchstaben** vorhersagt, und nutze es zur
**Textgenerierung**. Am Ende soll `generate(...)` aus einem Startkontext neuen,
stilistisch passenden Text produzieren.

Datensatz: **Tiny Shakespeare** (~1,1 MB, ein einziger Textstrom aus Shakespeare-Stücken).
Klein genug fürs CPU/MPS-Training, groß genug für erkennbare Struktur (Sprechernamen in
Großbuchstaben, Zeilenumbrüche, Dialog). Automatischer Download, siehe `datasets/`.

## Vorwissen

- **Skript** Teil 4–5 (Self-Attention, Multi-Head, Positional Encoding, Decoder/kausale
  Maske, autoregressives Sprachmodell, GPT vs. BERT).
- **Projekt 02** — deine Attention-Bausteine sind fast direkt wiederverwendbar; neu ist
  nur die **kausale Maske**.
- PyTorch (`nn.Module`, `nn.Embedding`, `AdamW`, `F.cross_entropy`, `torch.multinomial`).

## Fachliche Spezifikation

Baue einen **GPT** mit dieser Architektur (Radford et al. 2018; Pre-LayerNorm wie GPT-2):

1. **Tokenisierung (Zeichen-Level):** bijektive Abbildung `stoi`/`itos` über alle im
   Korpus vorkommenden Zeichen (Vokabular ≈ 65). `encode: str → [int]`, `decode` invers.
2. **Batches:** Zufällige Fenster der Länge `block_size` aus dem Text; Eingabe `x = data[i:i+T]`,
   Ziel `y = data[i+1:i+1+T]` (um eins verschoben = „nächstes Zeichen").
3. **Einbettung:** Token-Embedding $\mathbf{E}\in\mathbb{R}^{V\times d}$ **plus** eine
   **gelernte** Positional-Embedding $\mathbf{P}\in\mathbb{R}^{T_{\max}\times d}$
   (Alternative sinusoidal aus Projekt 02 ist ebenfalls erlaubt — begründe die Wahl).
4. **Kausale Multi-Head-Self-Attention:** wie in Projekt 02, aber mit einer **unteren
   Dreiecks-Maske**, sodass Position $t$ nur auf $\le t$ schaut:
   $$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}+M\Big)V,\qquad
   M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j> i\end{cases}$$
   Das macht das Modell **autoregressiv**: kein Token sieht seine Zukunft.
5. **Transformer-Block (Pre-LN):**
   $x \leftarrow x + \text{Attn}(\text{LN}(x))$, danach
   $x \leftarrow x + \text{FFN}(\text{LN}(x))$, mit FFN = Linear→GELU→Linear (innere
   Dimension $4d$). Stapele $N$ Blöcke.
6. **Kopf:** finale LayerNorm, dann lineare Projektion auf $V$ Logits. **Weight tying**
   (Kopf-Gewicht = Token-Embedding) ist empfohlen.
7. **Verlust:** Cross-Entropy zwischen Logits $(B,T,V)$ und Zielen $(B,T)$ über *alle*
   Positionen gleichzeitig. Optimizer: **AdamW**.
8. **Generierung:** autoregressiv — Kontext auf `block_size` kürzen, Logits des letzten
   Zeitschritts nehmen, durch **Temperatur** teilen, optional **top-k** filtern,
   `softmax` → `torch.multinomial` → anhängen, wiederholen.

## Milestones (empfohlene Reihenfolge)

1. **Daten & Tokenizer**: Download/Cache, `encode`/`decode`, Train/Val-Split (z. B. 90/10),
   `get_batch`.
2. **Modell**: kausale Attention → Block → GPT. Prüfe die **Formen** (ein Forward mit
   Dummy-Batch, Logits $(B,T,V)$) und dass der **Anfangs-Loss** ≈ $\ln V \approx 4.17$
   ist (uniforme Vorhersage — ein guter Sanity-Check!).
3. **Training**: Schleife mit periodischer Train/Val-Loss-Schätzung.
4. **Generierung**: Sampling mit Temperatur & top-k; beobachte, wie der Text mit
   sinkendem Loss lesbarer wird.
5. **Analyse** (schriftlich, s. u.).

## Was am Ende funktionieren soll

- Training läuft stabil, **Val-Loss fällt** von ≈ 4.2 auf **≈ 1.8** (Referenz-Setup:
  4 Blöcke, 4 Köpfe, $d=128$, `block_size=128`, ~3000 Iterationen; **~2 min auf MPS**,
  einige Minuten auf CPU).
- `generate` produziert **erkennbar Shakespeare-artigen** Text: Sprechernamen in
  Großbuchstaben, Zeilen-/Dialogstruktur, überwiegend echte englische Wörter (bei diesem
  Loss noch mit Fantasiewörtern und ohne globalen Sinn — genau der erwartete Stand).
- Ein kurzer **Analyseteil** (`ANALYSE.md` oder Docstring/Notebook), der belegt:

  1. **Sanity-Check:** Warum ist der Start-Loss ≈ $\ln V$? (Zeige die Rechnung.)
  2. **Kausalität:** Weise nach, dass deine Maske wirkt — z. B. dass eine Änderung an
     Position $t{+}k$ die Logits an Position $t$ **nicht** verändert.
  3. **Temperatur:** Vergleiche Proben bei $T=0.5$, $0.8$, $1.2$. Was passiert an den
     Extremen (repetitiv vs. inkohärent) und warum?
  4. **BPE-Ausblick:** Warum arbeiten echte LLMs auf **Subword-Tokens** (BPE/WordPiece,
     Skript Teil 6) statt auf Zeichen? Nenne mindestens zwei konkrete Vorteile.

## Bewertungsmaßstab (Master-Niveau)

- **Korrektheit der Kausalmaske** (kein Informationsleck aus der Zukunft) — der zentrale
  Punkt eines Decoders.
- Vollständige, saubere Architektur (Pre-LN-Blöcke, Residuals, Positional-Embedding,
  Weight-Tying) und stabile Optimierung.
- Funktionierende Generierung mit Temperatur & top-k.
- Nachvollziehbare, quantitativ belegte Analyse.

## Setup

```bash
source ../../../../.venv/bin/activate
# eigene Umsetzung bauen, dann z. B.:  python gpt.py
```

Benötigt nur `torch` (in der Repo-`requirements.txt`). Der Korpus wird beim ersten Lauf
nach `datasets/` geladen (siehe `datasets/README.md`) und nicht eingecheckt.

## Musterlösung

[`solution/gpt.py`](solution/gpt.py) — ein vollständiges, getestetes ~0,8-Mio.-Parameter-GPT
(kausale Attention, Pre-LN-Blöcke, Weight-Tying, Temperatur/top-k-Sampling). Referenzlauf:
Val-Loss 1.77 nach 3000 Iterationen, ~1:50 min auf Apple-MPS. **Erst selbst bauen.**
