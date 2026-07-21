# Project 02 (medium) — A Bayesian network: exact and approximate inference

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 07 — Theory of AI 2** · Format: **Python project** (several modules + tests)

## Why this format?

Inference in Bayesian networks is a *family of algorithms* with a clear
structure — one network data structure, several interchangeable inference
engines, and a test suite that checks that they agree. This is a small **code
base**, not an exploratory notebook. That is exactly how one uses libraries such
as `pgmpy` in practice — and by writing the procedures yourself you understand
what happens *inside* them.

## Goal

You implement the three central inference procedures from part 2 of the script
and check that they deliver **the same distribution** (the exact ones exactly,
the approximate one close to it):

1. **Inference by enumeration** (`enumeration_ask`) — the direct sum over the
   joint factorization.
2. **Variable elimination** (`elimination_ask`) — factors with the **pointwise
   product** and **summing out**.
3. **Likelihood weighting** (`likelihood_weighting`) — weighted sampling.

The validation happens on **Pearl's alarm network** (with the well-known result
$P(\text{Burglary}\mid j,m)=0.284$) and a small **medical diagnosis network**.
Along the way you make **"explaining away"** visible.

## Prior knowledge

- Part 2 of the script (Bayesian networks, factorization, d-separation, variable
  elimination, sampling/likelihood weighting).
- Python: recursion, generators (`yield`), `dict`, `random`.

## Project structure

```
02-medium/
  bayesnet.py         # the network data structure + the alarm/diagnosis nets  (given)
  inference.py        # the three inference procedures     <- YOU: 3 cores
  demo.py             # the comparison demonstration        (given)
  test_inference.py   # the test suite                      (given)
  solution/           # the complete reference solution
```

What is given is the **network data structure** (`BayesNode.p`, `.sample`), the
**factor plumbing** (`make_factor`, `bool_events`, the `elimination_ask`
orchestration) and the two example networks. **You implement the three
conceptual cores**, marked with `TODO`:

- `enumerate_all` (the enumeration recursion),
- `Factor.pointwise_product` and `Factor.sum_out` (the two factor operations),
- `weighted_sample` and `likelihood_weighting` (the weighted sampling).

## Tasks (step by step)

Test after every core with `python test_inference.py`:

1. **Enumeration** — `enumerate_all`: recursively over the topologically ordered
   variables; substitute the evidence variables, sum out the hidden ones.
   *Test:* `test_enumeration_alarm` (the target: $0.2842$).
2. **Variable elimination** — `Factor.pointwise_product` (the product over the
   union of the variables) and `Factor.sum_out` (summing one variable out). The
   orchestration calls your methods.
   *Test:* `test_elimination_matches_enumeration` (it must agree *exactly* with
   enumeration).
3. **Likelihood weighting** — `weighted_sample` (weight the evidence instead of
   sampling it) and `likelihood_weighting` (aggregate N samples).
   *Test:* `test_likelihood_weighting`.

Finally run `python demo.py`: a comparison of all three procedures plus the
explaining-away demonstration.

## What should work in the end

```bash
source ../../../../.venv/bin/activate    # only the standard library is needed
python test_inference.py     # -> "All tests passed."
python demo.py               # -> the three procedures are consistent, explaining away is visible
```

Reference: $P(\text{Burglary}\mid j,m)=0.2842$ (enumeration = VE, exactly equal);
likelihood weighting close to it. **Explaining away:**
$P(B\mid A)=0.374 \to P(B\mid A,E)=0.003$ — the earthquake report "explains the
alarm away".

> **An important observation (it is stated this way in the reference solution):**
> with evidence *at the leaves* (JohnCalls/MaryCalls), likelihood weighting has
> **high variance** and converges slowly — a real disadvantage of the procedure,
> not a bug. That is why the tolerance in the test is generous. With evidence *at
> the roots* (the diagnosis network with `Smoker`), LW is considerably more
> accurate.

## Reflection (in writing, briefly)

1. Why do enumeration and variable elimination agree **exactly**, while
   likelihood weighting only hits it approximately?
2. What does the efficiency of variable elimination depend on? (Script:
   treewidth / the elimination order.) Try a different order in the code.
3. Why does likelihood weighting discard no samples (unlike rejection sampling),
   and why does that help only partially when the evidence is rare?
4. Explain "explaining away" in terms of d-separation: which path opens as soon
   as `Alarm` is observed?

## Reference solution

Complete in **`solution/`** — all tests pass, `demo.py` runs through. Look only
after your own attempt.

---
---

# Projekt 02 (medium) — Bayes-Netz: exakte & approximative Inferenz (deutsche Fassung)

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
  solution/            # vollständige Musterlösung
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

Vollständig in **`solution/`** — alle Tests bestehen, `demo.py` läuft durch. Erst
nach eigenem Versuch ansehen.
