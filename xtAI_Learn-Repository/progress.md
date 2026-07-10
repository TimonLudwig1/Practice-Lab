# Fortschritt — xtAI Lern-Repository

## NEXT ACTION
Als Nächstes zu bauen: **Modul 11 „Computer Vision"** — noch nichts gebaut (Skript + 3 Projekte). Baut auf ML 1+2 (Modul 04/05, inkl. CNN Fashion-MNIST in 05-P02) auf. Themen-Ideen: Bildbildung/Filter (Faltung, Kanten/Gauss/Sobel), klassische Features (SIFT/HOG konzeptionell), CNNs (Recap 05 + tiefer: Architekturen LeNet/AlexNet/ResNet, Transfer Learning), Aufgaben (Klassifikation, Objekterkennung, Segmentierung), evtl. Vision Transformer (ViT — Bezug Modul 09/10 Attention). **WICHTIG Hardware-Constraint (User-MacBook wird heiß): KEINE rechenintensiven Trainings! Kleine Modelle/Datensätze, CPU-freundlich, Sekunden-Minuten. Ggf. vortrainierte Feature-Extraktion statt from-scratch-Training, oder klassische CV (Filter/Features) die kein GPU braucht.** torchvision ist installiert (Datensätze/Transforms/pretrained). Nach 11 → Modul 12 Image Processing.
Letzte abgeschlossene Einheit: **Modul 10 „Multilingual NLP" (komplett: Skript + 3 Projekte, getestet).**
Modul 10 Ergebnis: P01-basic SentencePiece-Tokenisierung (Notebook, fertility+Vokab-Bias×2). P02-medium Procrustes+CSLS-Alignment (py, P@1 0.975, 8 Tests, Skript-Fix X^T Y). P03-final MÜ statistisch+neuronal (KEINE Code-Vorgabe, loesung/): IBM Model 1 EM (BLEU~18, Alignments, ~3s CPU) + Encoder-Decoder-Transformer mit Cross-Attention from scratch, verifiziert per 10 Tests + Toy-Umkehr-Task (Exact-Match 0.87, anti-diagonale Cross-Attn, ~20s CPU). **User-Hardware-Constraint entdeckt: NMT-Volltraining zu heiß für MacBook → auf IBM1+Toy umgestellt. Für alle künftigen Module: KEINE teuren Trainings!**
Modul-10-Infra: Tatoeba DE-EN via manythings.org/anki/deu-eng.zip — BROWSER-HEADER NÖTIG (Referer=https://www.manythings.org/anki/, sonst 406). sentencepiece 0.2.1 ok. torch 2.12.1 im .venv (MPS da, aber CPU bevorzugen bei kleinen Modellen). .gitignore deckt daten/ *.txt/*.zip/*.model/*.vocab/*.npz. datasets+nltk+sacrebleu+gensim FEHLEN → BLEU/Tokenizer from scratch.
Letzte abgeschlossene Einheit: **Modul 09 „NLP 2" (komplett: Skript + 3 Projekte, getestet).**
Modul 09 Ergebnis: P01-basic LSTM-Sentiment (UCI echt, BoW 0.86 > LSTM 0.79, datenhungrig-Aha). P02-medium SDPA+MHA+Encoder von Hand (synthet. Negations-Task: BoW 0.50 vs Transformer 1.00, 8/8 Tests grün, TODOs im Root). P03-final Char-GPT from scratch (KEINE Code-Vorgabe, README-Spec + loesung/gpt.py): Decoder-only, kausale MHA, Pre-LN-Blöcke, Weight-Tying, Temperatur/top-k, Tiny-Shakespeare, Val-Loss 4.2→1.77 in ~1:50 min MPS, erkennbar Shakespeare-artiger Output. Env: torch 2.12.1 im Repo-.venv (NICHT System-python3), MPS. nbconvert: `.venv/bin/jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=python3`.
Umgebung: torch 2.12.1 im Repo-.venv (nicht das System-python3 mit 2.9.1!), MPS verfügbar. nbconvert-Ausführung: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 <nb>`. Build-Skripte im scratchpad (nbformat, erzeugt stub+loesung aus gemeinsamen Bausteinen).
01-basic Ergebnis: UCI Sentiment Labelled Sentences (imdb+amazon+yelp, 3000 Sätze, Download-URL s.u.). LSTM (Embedding+pack_padded_sequence+LSTM+Linear, BCEWithLogitsLoss) Test-Acc ~0.79 (Overfitting, Train ~0.99), BoW+LogReg-Baseline ~0.86 GEWINNT → Aha-Moment „neuronal ist datenhungrig". Läuft in ~34s auf MPS. Download-URL: https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip (200 OK, Mozilla-UA). Notebook-Muster wie 08-basic: Root .ipynb mit TODOs (Teil A vorgegeben), loesung/<name>_loesung.ipynb voll gelöst + getestet.
Umgebung Modul 09: torch 2.12.1, MPS (Apple-GPU) verfügbar, CUDA nein. Projekte klein/CPU-freundlich, device-agnostisch (CPU default für Reproduzierbarkeit, MPS optional). Datensatz-Idee 01-basic: kleiner Sentiment-Datensatz (z.B. via sklearn oder Gutenberg-ähnlich); ggf. synthetisch/klein halten, damit CPU-Training in Minuten läuft.
Inhalt-Zuschnitt Modul 09 (NLP 2 = neuronal, baut auf 08 auf): Wort-Embeddings-Wiederholung → neuronale Sprachmodelle (feedforward), RNN/LSTM/GRU (Vanishing Gradient → Rückverweis Modul 05), seq2seq + Attention (Bahdanau/Luong), DER Transformer (Self-Attention, Multi-Head, Positional Encoding, vollständige Herleitung), Vortraining/Transfer (BERT masked-LM vs GPT autoregressiv, Fine-Tuning), Tokenisierung (BPE/WordPiece/SentencePiece — in Modul 08/10 erwähnt), Evaluation (BLEU/ROUGE/Perplexität), Ausblick LLMs/Prompting/RAG. Projekte: basic Embeddings+RNN-Klassifikation (PyTorch), medium seq2seq oder Attention/Mini-Transformer-Baustein, final Transformer-Anwendung (Fine-Tuning oder from-scratch Self-Attention). Achtung PyTorch: in Modul 05 genutzt (MLP/CNN) — vorhanden. GPU? vermutlich CPU-only → kleine Modelle/Datensätze wählen.
Letzte abgeschlossene Einheit: **Modul 08 „NLP 1" (komplett: Skript + 3 Projekte, getestet).**
Konnektivität: Gutenberg, UD-EWT-GitHub-raw (kann 429→Retry eingebaut), sklearn-20ng-Host erreichbar. daten/ pro Projekt, gitignore deckt *.txt/*.conllu/*.csv/*.zip/*.tsv/*.db. sklearn-20ng cached in ~/scikit_learn_data.
Umgebung Modul 08: sklearn 1.9, numpy 2.5, pandas 3.0, matplotlib, scipy vorhanden; nltk/gensim/datasets FEHLEN → NLP-Kerne from scratch (didaktisch besser). Datenpläne: 01-basic n-Gramm-LM auf Gutenberg-Text (urllib-Download+Cache, .gitignore); 02-medium Textklassifikation auf sklearn 20newsgroups (Download); 03-final POS-Tagging HMM+Viterbi auf Universal Dependencies EN-EWT (.conllu via urllib). Vor Projektbau Konnektivität testen, sonst synthetisch.
Inhalt-Zuschnitt Modul 08 (NLP 1, klassisch/statistisch, baut auf 07 auf — HMM/Viterbi dort schon erklärt): Text-Preprocessing (Tokenisierung, Stemming/Lemma, Normalisierung), N-Gramm-Sprachmodelle + Glättung (Laplace/Kneser-Ney) + Perplexität, Textklassifikation (Naive Bayes — Achtung: Spamfilter schon in Modul 01 P03; hier tiefer/anders, z.B. Sentiment; Logistische Regression/MaxEnt), Wortrepräsentationen (TF-IDF, Word2Vec/GloVe, Embeddings), Sequenz-Labeling (POS-Tagging via HMM/Viterbi → Rückverweis Modul 07; MEMM/CRF), Syntax/Parsing (CFG, CKY, Dependency). NLP 2 (Modul 09) macht dann neuronal/Transformer/seq2seq.
Letzte abgeschlossene Einheit: **Modul 07 „Theorie der KI 2" (komplett: Skript + 3 Projekte, getestet).**
Notebook-Baumuster Module 06/07: scratchpad build-Skripte (Aufgabe+Lösung aus Bausteinen, nbconvert-getestet). py-Projekte: Lösung in loesung/, Skelette mit TODO/NotImplementedError im Root, Testsuite grün.
Inhalt-Zuschnitt Modul 07 (baut auf 06 auf): Planung (STRIPS/PDDL, Zustandsraum- vs. Plan-Suche, GraphPlan), Handeln unter Unsicherheit (Wahrscheinlichkeit, Bayes-Netze + Inferenz, ggf. HMM/Entscheidungstheorie/MEU, Value/Policy Iteration als Brücke zu RL-Modulen), nichtmonotones Schließen, evtl. Beschreibungslogiken/Ontologien. Modul 06 hat schon auf „Description Logics/Datalog als entscheidbare FOL-Fragmente" vorausverwiesen — dort anknüpfen.
Letzte abgeschlossene Einheit: **Modul 06 „Theorie der KI 1" (komplett: Skript + 3 Projekte, getestet).**
Kontext für später: Für Modul 10 (Multilingual NLP) sind `sentencepiece`+`sacremoses` bereits in requirements.txt; Datenquelle verifiziert: Tatoeba EN-DE-Paare via manythings.org/anki (User-Agent-Header nötig, sonst 406).
Letzte abgeschlossene Einheit: **Modul 05 „Machine Learning 2" (komplett: Skript + 3 Projekte, getestet).**

**Refactor (Code-Vorgabe-Regel, neu in CLAUDE.md):** final = KEIN vorgeschriebener Code, medium = nur wenig/vereinzelt Inspiration. Umgesetzt für die **vor der Regel gebauten Module 01–04**: Alle 4 Final-Notebooks haben jetzt leere Code-Zellen (nur Schritt-für-Schritt-Anleitung + Erwartungswerte/Selbstchecks in Markdown; Lösungen unverändert in `loesung/`); READMEs 01–03 entsprechend angepasst. Mediums 01–04 blieben unverändert (Kern war schon TODO, nur Viz/Checks vorgegeben). **Modul 05+ wurde bereits nach der neuen Regel gebaut — nicht nachbearbeiten.** Regel für alle künftigen Module beachten.

## Fortschrittstabelle

| Nr | Modul | Status | Formate | Getestet | Notizen |
|----|-------|--------|---------|----------|---------|
| 01 | Introduction in AI | fertig-getestet | Notebook (basic, final), py (medium) | ✓ alle 3 | A*-Labyrinth · Tic-Tac-Toe Minimax · Naive-Bayes-Spamfilter (echte UCI-SMS-Daten, Acc. 98,6 %) |
| 02 | Data Science 1 | fertig-getestet | Notebook (alle 3) | ✓ alle 3 | pandas-Basics (Penguins) · Datenbereinigung (synthetisch, Abnahmetest) · EDA (Bike Sharing UCI, echte Daten) |
| 03 | Data Science 2 | fertig-getestet | Notebook (alle 3) | ✓ alle 3 | SQL/SQLite (synth. Shop-DB) · Bootstrap/Permutation (synth. A/B-Test) · Regression+Prognose (Bike Sharing UCI, echt) |
| 04 | Machine Learning 1 | fertig-getestet | Notebook (alle 3) | ✓ alle 3 | kNN von Hand (Penguins) · Modellrennen Pipelines/CV/GridSearch (Breast Cancer, echt) · Einkommensvorhersage inkl. Kosten-Schwelle + Fairness-Check (UCI Adult, echt) |
| 05 | Machine Learning 2 | fertig-getestet | Notebook (alle 3) | ✓ alle 3 | Master-Niveau. P01: MLP von Hand + Gradient Check (6e-11, Acc 0,95) · P02: CNN Fashion-MNIST, Optimierer-/Regularisierungs-Ablation (echt) · P03: Kundensegmentierung k-Means/GMM-EM/DBSCAN/Ward + PCA/t-SNE + BIC/ARI (UCI Wholesale, echt; GMM-full bester ARI 0,62, DBSCAN scheitert instruktiv) |
| 06 | Theorie der KI 1 | fertig-getestet | Notebook (basic), py (medium+final) | ✓ alle 3 | P01: BFS/UCS/IDDFS/A* (Romania 418km optimal, Dominanz 5257≫457≫147). P02: CSP Backtracking+MRV+AC-3 & DPLL-SAT, Sudoku beide Wege identisch. P03-final (keine Code-Vorgabe): FOL-Resolutions-Theorembeweiser — KNF+Skolemisierung, Unifikation+Occurs-Check, Resolution+Faktorisierung, Set-of-Support; West-Kriminalfall bewiesen (~600 Schritte) |
| 07 | Theorie der KI 2 | fertig-getestet | Notebook (basic), py (medium+final) | ✓ alle 3 | P01: STRIPS-Planer (Sussman, BFS vs A*(h_add) 18 vs 11). P02: Bayes-Netz (Enumeration=VE exakt, Likelihood Weighting; Alarm 0.2842, explaining away 0.374→0.003). P03-final (keine Code-Vorgabe): MDP-Agent Value+Policy Iteration auf 4x3-Gridworld, AIMA-Utilities reproduziert, VI 34/PI 5 Iter. gleiche Policy, γ-/Reward-Studien |
| 08 | NLP 1 | fertig-getestet | Notebook (basic), py (medium+final) | ✓ alle 3 | P01: N-Gramm-LM Sherlock (Add-1-Trigramm>Unigramm-Aha, Interpolation ~190). P02: NB von Hand vs TF-IDF+LogReg (20ng, beide ~0.89). P03-final (keine Code-Vorgabe): HMM-POS-Tagger+Viterbi auf UD-EWT, Acc 0.912 (unbekannte Wörter 0.65 = Flaschenhals), Suffix-Signaturen, Testsuite grün |
| 09 | NLP 2 | fertig-getestet | Notebook (basic), py (medium+final) | ✓ alle 3 | P01: LSTM-Sentiment (UCI echt) — BoW 0.86 schlägt LSTM 0.79 (neuronal datenhungrig). P02: SDPA+MHA+sinus.PE+Encoder von Hand, synthet. Negations-Task → BoW 0.50 vs Transformer 1.00, 8/8 Tests grün. P03-final (keine Code-Vorgabe): Char-GPT from scratch (Decoder-only, kausale MHA, Pre-LN, Weight-Tying, Temp/top-k) auf Tiny-Shakespeare, Val-Loss 1.77, erkennbar Shakespeare-artig |
| 10 | Multilingual NLP | fertig-getestet | Notebook (basic), py (medium+final) | ✓ alle 3 | Skript (+Fix Procrustes X^T Y). P01: SentencePiece DE+EN, fertility+Vokab-Bias×2. P02: Procrustes+CSLS von Hand (PPMI+SVD, ~2800 Anker), P@1 0.975, 8/8 Tests. P03-final (keine Code-Vorgabe): IBM Model 1 EM (BLEU~18, Alignments) + Enc-Dec-Transformer Cross-Attention from scratch, verifiziert per 10 Tests + Toy-Umkehr (Exact 0.87, anti-diag. Cross-Attn) — alles CPU/billig (kein NMT-Volltraining, MacBook-Hitze) |
| 11 | Computer Vision | projekte-in-arbeit | Notebook (basic), py (medium+final gepl.) | P01 ✓ | Skript fertig. P01 ✓: Faltung+Sobel von Hand + gelernte ResNet-conv1-Filter (Grace-Hopper-Bild, ~7s). P02 (Transfer/Feature-Extraktion EuroSAT) + P03-final offen |
| 12 | Image Processing & Comp. Photography | offen | — | — | — |
| 13 | RL & Computational Decision-Making | offen | — | — | — |
| 14 | Deep RL for Optimal Control | offen | — | — | — |
| 15 | ML for Networks 1 | offen | — | — | — |
| 16 | ML for Networks 2 | offen | — | — | — |
| 17 | Core XR | offen | — | — | — |
| 18 | Multimodal Interfaces | offen | — | — | — |
| 19 | 3D User Interfaces | offen | — | — | — |
| 20 | 3D Point Cloud Processing | offen | — | — | — |
| 21 | Robotics 1 | offen | — | — | — |
| 22 | Robotics 2 | offen | — | — | — |
| 23 | Advanced Automation | offen | — | — | — |
| 24 | Self-aware Computing | offen | — | — | — |
| 25 | Interaktive Computergraphik | offen | — | — | — |
| 26 | Music Information Retrieval | offen | — | — | — |
| 27 | Remote Sensing | offen | — | — | — |
| 28 | ML in der Bioinformatik | offen | — | — | — |
| 29 | Datenbanken 2 | offen | — | — | — |
| 30 | Simulationstechnik | offen | — | — | — |
| 31 | Sicherheit von Softwaresystemen | offen | — | — | — |
| 32 | Deduktive Datenbanken | offen | — | — | — |
| 33 | Logische Programmierung | offen | — | — | — |
| 34 | Systems Benchmarking | offen | — | — | — |
| 35 | Fortgeschrittenes Programmieren | offen | — | — | — |
| 36 | Selected Topics (Leitfaden) | offen | — | — | nur Leitfaden |
| 37–41 | Struktur-Module | offen | — | — | nur Kurz-READMEs |

## Konventionen
- Status: `offen` · `skript-in-arbeit` · `skript-fertig` · `projekte-in-arbeit` · `fertig-getestet`
- Nach jedem Teilschritt: erst diese Datei aktualisieren, dann Git-Commit.
- Neue Session: NUR diese Datei lesen und direkt bei NEXT ACTION weitermachen.