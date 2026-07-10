# Projekt 02 (medium) — Bayes-Netz: exakte & approximative Inferenz

**Modul 07 — Theorie der KI 2** · Format: **Python-Projekt** (mehrere Module + Tests)

## Warum dieses Format?

Inferenz in Bayes-Netzen ist eine *Algorithmen-Familie* mit klarer Struktur —
eine Netz-Datenstruktur, mehrere austauschbare Inferenzmaschinen, eine Testsuite,
die ihre Übereinstimmung prüft. Das ist eine kleine **Codebasis**, kein
exploratives Notebook. Genau so nutzt man Bibliotheken wie `pgmpy` in der Praxis —
und indem du die Verfahren selbst schreibst, verstehst du, was dort *drin* passiert.

## Ziel

Du implementierst die drei zentralen Inferenzverfahren aus Teil 2 des Skripts und
prüfst, dass sie **dieselbe Verteilung** liefern (die exakten exakt, das
approximative nahe daran):

1. **Inferenz durch Aufzählung** (`enumeration_ask`) — die direkte Summe über die
   Verbund-Faktorisierung.
2. **Variable Elimination** (`elimination_ask`) — Faktoren mit **punktweisem
   Produkt** und **summing out**.
3. **Likelihood Weighting** (`likelihood_weighting`) — gewichtetes Sampling.

Validiert wird an **Pearls Alarm-Netz** (mit dem bekannten Resultat
$P(\text{Burglary}\mid j,m)=0{,}284$) und einem kleinen **medizinischen
Diagnosenetz**. Nebenbei machst du **„explaining away"** sichtbar.

## Vorwissen

- Skript Teil 2 (Bayes-Netze, Faktorisierung, d-Separation, Variable Elimination,
  Sampling/Likelihood Weighting).
- Python: Rekursion, Generatoren (`yield`), `dict`, `random`.

## Projektstruktur

```
02-medium/
  bayesnet.py         # Netz-Datenstruktur + Alarm-/Diagnosenetz   (vorgegeben)
  inference.py        # die drei Inferenzverfahren     <- DU: 3 Kerne
  demo.py             # Vergleichs-Demonstration        (vorgegeben)
  test_inference.py   # Testsuite                        (vorgegeben)
  loesung/            # vollständige Musterlösung
```

Vorgegeben ist die **Netz-Datenstruktur** (`BayesNode.p`, `.sample`), die
**Faktor-Plumbing** (`make_factor`, `bool_events`, die `elimination_ask`-
Orchestrierung) und die beiden Beispielnetze. **Du implementierst die drei
konzeptionellen Kerne**, markiert mit `TODO`:

- `enumerate_all` (die Aufzählungs-Rekursion),
- `Factor.pointwise_product` und `Factor.sum_out` (die zwei Faktor-Operationen),
- `weighted_sample` und `likelihood_weighting` (das gewichtete Sampling).

## Aufgabenstellung (Schritt für Schritt)

Teste nach jedem Kern mit `python test_inference.py`:

1. **Aufzählung** — `enumerate_all`: rekursiv über die topologisch geordneten
   Variablen; Evidenzvariablen einsetzen, versteckte aussummieren.
   *Test:* `test_enumeration_alarm` (Ziel: $0{,}2842$).
2. **Variable Elimination** — `Factor.pointwise_product` (Produkt über die
   Vereinigung der Variablen) und `Factor.sum_out` (eine Variable heraussummieren).
   Die Orchestrierung ruft deine Methoden auf.
   *Test:* `test_elimination_matches_enumeration` (muss *exakt* mit Aufzählung übereinstimmen).
3. **Likelihood Weighting** — `weighted_sample` (Evidenz gewichten statt samplen)
   und `likelihood_weighting` (N Stichproben aggregieren).
   *Test:* `test_likelihood_weighting`.

Führe abschließend `python demo.py` aus: Vergleich aller drei Verfahren plus die
Explaining-away-Demonstration.

## Was am Ende funktionieren soll

```bash
source ../../../../.venv/bin/activate    # nur Standardbibliothek nötig
python test_inference.py     # -> "Alle Tests bestanden."
python demo.py               # -> drei Verfahren konsistent, explaining away sichtbar
```

Referenz: $P(\text{Burglary}\mid j,m)=0{,}2842$ (Aufzählung = VE exakt gleich);
Likelihood Weighting nahe daran. **Explaining away:**
$P(B\mid A)=0{,}374 \to P(B\mid A,E)=0{,}003$ — die Erdbebenmeldung „erklärt den
Alarm weg".

> **Wichtige Beobachtung (steht so in der Musterlösung):** Bei Evidenz *an den
> Blättern* (JohnCalls/MaryCalls) hat Likelihood Weighting **hohe Varianz** und
> konvergiert langsam — ein realer Nachteil des Verfahrens, kein Bug. Deshalb ist
> die Toleranz im Test großzügig. Bei Evidenz *an den Wurzeln* (Diagnosenetz mit
> `Smoker`) ist LW deutlich genauer.

## Reflexion (schriftlich, kurz)

1. Warum stimmen Aufzählung und Variable Elimination **exakt** überein, während
   Likelihood Weighting nur näherungsweise trifft?
2. Wovon hängt die Effizienz von Variable Elimination ab? (Skript: Baumweite /
   Eliminationsreihenfolge.) Probiere im Code eine andere Reihenfolge.
3. Warum verwirft Likelihood Weighting keine Stichproben (anders als Rejection
   Sampling), und warum hilft das bei seltener Evidenz nur teilweise?
4. Erkläre „explaining away" mit d-Separation: Welcher Pfad öffnet sich, sobald
   `Alarm` beobachtet ist?

## Musterlösung

Vollständig in **`loesung/`** — alle Tests bestehen, `demo.py` läuft durch. Erst
nach eigenem Versuch ansehen.
