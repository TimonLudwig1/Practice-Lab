# Projekt 02 (medium) — Cross-linguale Embeddings alignen (Procrustes + CSLS)

**Modul 10 — Multilingual NLP** · Format: **Python-Projekt** (mehrere Module + Testsuite)

## Warum dieses Format?

Das Kernstück ist lineare Algebra — die **geschlossene orthogonale Procrustes-Lösung** über
die SVD — plus ein Retrieval-Maß gegen **Hubness**. Als **Codebasis** trennst du die
Infrastruktur (Embeddings bauen, Anker schürfen) vom eigentlichen Lernziel (Alignment +
Retrieval), und die **Testsuite** verifiziert deine Mathematik auf kontrollierten Daten,
bei denen die richtige Antwort bekannt ist.

## Ziel

Du richtest zwei **getrennt** trainierte Wort-Embedding-Räume (Deutsch & Englisch) so
aus, dass man ein Wörterbuch „übersetzen" kann, indem man das Quellwort in den Zielraum
abbildet und den nächsten Nachbarn nimmt (Skript-Abschnitt 3). Konkret lernst du:

- die **orthogonale Procrustes-Lösung** $W=UV^\top$ aus $\mathrm{SVD}(X^\top Y)$ — inklusive
  der **richtigen Richtung** (Quelle$^\top$·Ziel);
- warum die **Orthogonalitäts-Beschränkung** die Geometrie (und damit Kosinus-Nachbarschaften)
  erhält;
- **Precision@1** als Retrieval-Metrik und das **Hubness**-Problem;
- **CSLS** als Korrektur gegen Hubs — und die ehrliche Beobachtung, *wann* es hilft.

## Vorwissen

- **Skript** Abschnitt 3 (Alignment, Procrustes, SVD-Lösung, Hubness, CSLS).
- Lineare Algebra: SVD, Orthogonalität, Frobenius-Norm.
- NumPy (`np.linalg.svd`, Broadcasting, `argmax`, `argpartition`).

## Projektstruktur

```
02-medium/
  align.py         # Procrustes + NN + CSLS + P@1         <- DU (Aufgabe 1 + 2)
  embeddings.py    # PPMI+SVD-Embeddings, Anker schürfen    (vorgegeben)
  data.py          # Tatoeba laden + Test-Lexikon           (vorgegeben)
  run.py           # volle Pipeline + NN-vs-CSLS-Vergleich  (vorgegeben)
  test_align.py    # Testsuite (8 Tests, synthetisch)       (vorgegeben)
  solution/         # vollständige, getestete Musterlösung
```

## Aufgabenstellung

Wenig Vorgabe — die beiden mathematischen Kerne sind deine (`# TODO` in `align.py`):

1. **`orthogonal_procrustes(X, Y)`**: die geschlossene Lösung $W=UV^\top$ mit
   $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$. Achte auf die Richtung: $X^\top Y$ (Quelle$^\top$
   Ziel), damit $XW\approx Y$ gilt. `nearest_neighbor` und `precision_at_1` sind gegeben.
2. **`csls(query_vecs, tgt_emb, src_pool, k)`**: das CSLS-Retrieval
   $2\cos - r_T - r_S$ mit den beiden Nachbarschafts-Dichten (Helfer `_topk_mean` gegeben).

**Vorgehen:**

```bash
source ../../../../.venv/bin/activate
python test_align.py     # rot -> Aufgaben füllen -> alle 8 Tests grün
python run.py            # echte DE-EN-Daten: Embeddings, Anker, Alignment, P@1
```

Die Infrastruktur (`embeddings.py`) baut PPMI+SVD-Embeddings pro Sprache (auf Tatoeba
ergeben sich brauchbare Vektoren: *king~queen*, *water~boils*) und **schürft automatisch
~2800 Anker-Paare** aus den parallelen Sätzen (Mutual-Nearest-Neighbour über Kookkurrenz).
So sind Anker (Training) und Test-Lexikon sauber getrennt.

## Was am Ende funktionieren soll

- `python test_align.py` → **alle 8 Tests grün**: Procrustes rekonstruiert eine bekannte
  Rotation exakt, ist orthogonal, hat die richtige Richtung, denoised; NN & CSLS sind auf
  sauberen Räumen perfekt; **CSLS korrigiert einen konstruierten Hub**, wo NN scheitert.
- `python run.py` → **Precision@1 ≈ 0.95–0.98** auf dem 40-Wörter-Test-Lexikon, mit
  Beispielübersetzungen (`water→wasser`, `king→könig`, …).

> **Ehrliche Beobachtung:** Auf gut getrennten Inhaltswörtern ist NN bereits nahezu
> perfekt; CSLS liegt hier gleichauf oder minimal darunter. Sein Nutzen zeigt sich vor
> allem bei **Hubness** in schwierigerem, verrauschterem Retrieval — genau das isoliert der
> synthetische Hub-Test. „Mehr Methode" ist nicht immer „mehr Genauigkeit", wenn das
> Problem schon leicht ist.

## Musterlösung

Vollständig in [`solution/`](solution/) (alle Tests grün, P@1 ≈ 0.95–0.98). Erst selbst
versuchen — die Root-`align.py` wirft `NotImplementedError`, bis du die TODOs füllst.
