# Project 03 (final) — A resolution theorem prover for first-order logic

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 06 — Theory of AI 1** · Format: **a Python project, built from scratch by you**

> **This is the final project of the module.** There is **no given code** — you
> design and implement the complete prover yourself. The project consolidates
> part 3 (propositional logic, CNF, resolution) **and** part 4 (first-order
> logic, unification, Skolemization, Herbrand) into a single, running system.
> Level: a genuine master's examination piece.

## Why this format and this topic?

An automatic theorem prover is the heart of symbolic AI and the direct
implementation of the central theorem of the module: **$\mathrm{KB}\models\alpha
\iff \mathrm{KB}\land\lnot\alpha$ is unsatisfiable**. If you build it yourself,
you will not merely have understood every step of the inference pipeline — CNF
conversion, unification, resolution — but *mastered* it. A real code base (not
a notebook) is right here, because the system consists of clearly separated
components with data structures and a control loop.

## Goal

Build a program that decides correctly, for a **knowledge base (KB)** of
first-order formulas and a **query $\alpha$**, whether $\mathrm{KB} \models
\alpha$ holds — by **resolution refutation**. In the end your prover must solve
the four test scenarios below correctly and, in the positive case, print a
comprehensible **derivation of the empty clause**.

## Prior knowledge

The whole script, in particular:
- **Parts 3.4–3.6:** CNF, the resolution rule, the empty clause, refutation
  completeness, DPLL (for comparison).
- **Parts 4.3–4.4:** unification and the MGU, the occurs check, Skolemization,
  FOL resolution, the Herbrand theorem.

Python: recursive data structures, classes/`dataclass`, sets, recursion.

## What you should build — the components

First design a **representation** for terms (variables, constants, function
terms), atoms, formulas (connectives + quantifiers) and clauses (sets of
literals). Then implement the pipeline:

1. **CNF conversion** of an arbitrary closed FOL formula, in exactly the steps
   from the script:
   - eliminate equivalences ($\Leftrightarrow$) and implications ($\Rightarrow$);
   - **negation normal form (NNF)**: move negations inwards (De Morgan **and**
     the quantifier duality $\lnot\forall \equiv \exists\lnot$, $\lnot\exists \equiv \forall\lnot$);
   - **standardize the variables apart** (rename every bound variable uniquely);
   - **Skolemization**: eliminate $\exists$ — a Skolem *constant* outside every
     $\forall$, a Skolem *function* of the surrounding $\forall$ variables inside;
   - **drop the universal quantifiers** (the remaining variables are implicitly
     universally quantified);
   - **distribute** ($\lor$ over $\land$) and **extract the clauses**.
2. **Unification** of two terms/atoms returning the substitution (the MGU) —
   **including the occurs check** (bind $x \mapsto t$ only if $x$ does not
   occur in $t$).
3. **Resolution** of two clauses: **rename the variables freshly** beforehand
   (standardizing apart against name collisions), find complementary,
   **unifiable** pairs of literals, form the resolvent and apply the
   substitution. Add **factoring** (unifying two literals of the same name and
   polarity within one clause) for completeness.
4. **The main refutation loop**: form the KB clauses + the clauses of
   $\lnot\alpha$, then resolve until the **empty clause** arises (→ proved) or
   nothing new arises any more (→ not provable). **A recommendation:** use the
   **set-of-support strategy** (only clauses derived from $\lnot\alpha$ serve as
   the "given clause") — it is refutation complete as long as the KB is
   satisfiable, and it avoids the explosion of blind saturation. Carry
   **parent pointers** so that you can print the derivation.

## The task — your prover must solve these four scenarios

**(A) FOL, the "Colonel West" criminal case** (the classical serious example):

> It is a crime for an American to sell weapons to hostile nations. The country
> *Nono* owns missiles, and all of Nono's missiles were sold by Colonel *West*,
> who is an American. Missiles are weapons. An enemy of America counts as
> *hostile*. Nono is an enemy of America.
>
> **Query:** is West a criminal? — Your prover should prove **yes**.

Formalize the eight sentences as FOL formulas (predicates such as `American/1`,
`Weapon/1`, `Sells/3`, `Hostile/1`, `Missile/1`, `Owns/2`, `Enemy/2`,
`Criminal/1`; constants `West, Nono, America, M1`) and prove `Criminal(West)`.

**(B) FOL with a Skolem function** — "everyone has an ancestor":

> $\forall x\,\exists y\ \mathrm{Parent}(y,x)$ (everyone has a parent);
> $\forall x\,\forall y\ (\mathrm{Parent}(y,x) \Rightarrow \mathrm{Ancestor}(y,x))$.
> **Query:** $\forall x\,\exists y\ \mathrm{Ancestor}(y,x)$.

Here your Skolemization *must* generate a Skolem **function** (the parent
depends on $x$) and your unification must unify into function terms. Check that
the proof succeeds.

**(C) Pure propositional logic** as a special case — modus tollens:
$\{P \Rightarrow Q,\ \lnot Q\} \models \lnot P$. (Variable-free; it tests that
your FOL prover handles propositional logic correctly as a special case.)

**(D) The counter-check (soundness!):** something that does **not** follow must
**not** be proved — for instance `Raining` does not follow from
$\{\mathrm{Sunny},\ \mathrm{Sunny}\Rightarrow\mathrm{Warm}\}$. Your prover has
to terminate here and return "not provable" (the KB is finite enough for the
saturation to stop).

**In addition (a mandatory demo):** show explicitly that `unify(x, f(x))` is
rejected by the **occurs check** (returning "no unifier").

## Acceptance criteria

Your project is finished when:

- [ ] the CNF conversion performs implications/NNF/standardizing apart/
      Skolemization/distribution correctly (checkable on the intermediate output);
- [ ] unification returns the **MGU** and performs the **occurs check**;
- [ ] scenario **(A)** proves `Criminal(West)` and prints the **derivation of the
      empty clause** (the chain of resolution steps with the parent clauses);
- [ ] scenario **(B)** succeeds with a **Skolem function**;
- [ ] scenario **(C)** (propositional logic) succeeds;
- [ ] scenario **(D)** is correctly **not** proved (soundness) and terminates;
- [ ] the occurs check demo shows the expected result.

## Self-check questions (answer them in writing)

1. **Why refutation instead of direct derivation?** Explain on example (A) why
   you add $\lnot\mathrm{Criminal}(\mathrm{West})$ instead of reasoning
   "forwards". (Script: resolution is only *refutation* complete.)
2. **Why does Skolemization preserve only satisfiability, not equivalence** —
   and why is that enough for the refutation procedure?
3. **What is the renaming of variables before every resolution for?** Construct
   an example in which omitting it leads to a wrong (too specific) MGU.
4. **Why does your prover terminate on (D), but not in general?** Connect that
   with the **semi-decidability** of FOL validity (script 4.5): what happens
   with a non-entailed query over a KB with function symbols and an infinite
   Herbrand universe?
5. **Where would DPLL (part 3.6) be usable here, and where not?** (Keyword: the
   propositional special case vs. variables/unification.)

## Extensions (optional, for going deeper)

- **The answer literal:** instead of only "yes/no", also *who* the culprit is,
  by coupling the query $\exists x\,\mathrm{Criminal}(x)$ with an `Answer(x)`
  literal and reading off the binding.
- **Subsumption and tautology elimination**, to keep the clause set small.
- **The "curiosity killed the cat" KB** (AIMA) — it needs Skolemization *and*
  factoring; a real stress test of your completeness.

## Reference solution

**`solution/`** holds a complete, tested reference implementation:
- `logic.py` — terms/formulas, the complete CNF pipeline, unification with the
  occurs check, resolution + factoring, the set-of-support refutation loop with
  proof output.
- `scenarios.py` — the four scenarios (A)–(D) + the occurs check demo;
  `python scenarios.py` prints the full refutation chain for (A).
- `test_prover.py` — the automatic acceptance test; `python test_prover.py` →
  "All tests passed."

Reference values: (A) is proved in about 600 resolution steps, (B) in 2, (C) in
2, and (D) terminates correctly without a proof. **Look only after your own
attempt** — the learning is in building it yourself.

```bash
source ../../../../.venv/bin/activate    # only the standard library is needed
cd solution && python scenarios.py       # the demo with proof output
python test_prover.py                    # the acceptance test
```

---
---

# Projekt 03 (final) — Ein Resolutions-Theorembeweiser für die Prädikatenlogik (deutsche Fassung)

**Modul 06 — Theorie der KI 1** · Format: **Python-Projekt, von Grund auf selbst gebaut**

> **Dies ist das Abschlussprojekt des Moduls.** Es gibt **keinen vorgegebenen
> Code** — du entwirfst und implementierst den kompletten Beweiser selbst. Das
> Projekt konsolidiert Teil 3 (Aussagenlogik, KNF, Resolution) **und** Teil 4
> (Prädikatenlogik, Unifikation, Skolemisierung, Herbrand) zu einem einzigen,
> lauffähigen System. Niveau: echte Master-Prüfungsleistung.

## Warum dieses Format & dieses Thema?

Ein automatischer Theorembeweiser ist das Herzstück der symbolischen KI und die
direkte Umsetzung des zentralen Satzes des Moduls: **$\mathrm{KB}\models\alpha
\iff \mathrm{KB}\land\lnot\alpha$ unerfüllbar**. Wenn du ihn selbst baust, hast
du jeden Schritt der Inferenzpipeline — KNF-Umwandlung, Unifikation, Resolution
— nicht nur verstanden, sondern *beherrscht*. Eine echte Codebasis (kein
Notebook) ist hier richtig, weil das System aus klar getrennten Komponenten mit
Datenstrukturen und einer Kontrollschleife besteht.

## Ziel

Baue ein Programm, das für eine **Wissensbasis (KB)** aus prädikatenlogischen
Formeln und eine **Anfrage $\alpha$** korrekt entscheidet, ob $\mathrm{KB}
\models \alpha$ gilt — durch **Resolutions-Widerlegung**. Am Ende muss dein
Beweiser die vier Testszenarien unten korrekt lösen und für den positiven Fall
eine nachvollziehbare **Ableitung der leeren Klausel** ausgeben.

## Vorwissen

Das gesamte Skript, insbesondere:
- **Teil 3.4–3.6:** KNF, Resolutionsregel, leere Klausel, Widerlegungsvollständigkeit, DPLL (zum Vergleich).
- **Teil 4.3–4.4:** Unifikation & MGU, Occurs-Check, Skolemisierung, FOL-Resolution, Herbrand-Theorem.

Python: rekursive Datenstrukturen, Klassen/`dataclass`, Mengen, Rekursion.

## Was du bauen sollst — die Komponenten

Entwirf zuerst eine **Repräsentation** für Terme (Variablen, Konstanten,
Funktionsterme), Atome, Formeln (Junktoren + Quantoren) und Klauseln (Mengen von
Literalen). Dann implementiere die Pipeline:

1. **KNF-Umwandlung** einer beliebigen geschlossenen FOL-Formel, in genau den
   Schritten aus dem Skript:
   - Äquivalenzen ($\Leftrightarrow$) und Implikationen ($\Rightarrow$) eliminieren;
   - **Negationsnormalform (NNF)**: Negationen nach innen ziehen (De Morgan **und**
     Quantoren-Dualität $\lnot\forall \equiv \exists\lnot$, $\lnot\exists \equiv \forall\lnot$);
   - **Variablen standardisieren** (jede gebundene Variable eindeutig umbenennen);
   - **Skolemisierung**: $\exists$ eliminieren — Skolem-*Konstante* außerhalb jedes
     $\forall$, Skolem-*Funktion* der umgebenden $\forall$-Variablen innerhalb;
   - **All-Quantoren weglassen** (verbleibende Variablen sind implizit allquantifiziert);
   - **distribuieren** ($\lor$ über $\land$) und **Klauseln extrahieren**.
2. **Unifikation** zweier Terme/Atome mit Rückgabe der Substitution (MGU) —
   **inklusive Occurs-Check** (Bindung $x \mapsto t$ nur, wenn $x$ nicht in $t$ vorkommt).
3. **Resolution** zweier Klauseln: Variablen vorher **frisch umbenennen**
   (Standardisieren gegen Namenskollision), komplementäre, **unifizierbare**
   Literalpaare finden, Resolvente bilden und die Substitution anwenden. Ergänze
   **Faktorisierung** (zwei gleichnamige Literale gleicher Polarität in einer
   Klausel unifizieren) für die Vollständigkeit.
4. **Refutations-Hauptschleife**: KB-Klauseln + Klauseln von $\lnot\alpha$
   bilden, dann resolvieren, bis die **leere Klausel** entsteht (→ bewiesen) oder
   nichts Neues mehr entsteht (→ nicht beweisbar). **Empfehlung:** Nutze die
   **Set-of-Support-Strategie** (nur aus $\lnot\alpha$ abgeleitete Klauseln als
   „given clause") — sie ist refutationsvollständig, solange die KB erfüllbar ist,
   und vermeidet die Explosion der blinden Saturierung. Führe **Elternzeiger** mit,
   um die Ableitung ausgeben zu können.

## Aufgabe — diese vier Szenarien muss dein Beweiser lösen

**(A) FOL, der „Colonel-West"-Kriminalfall** (der klassische Ernstfall):

> Es ist ein Verbrechen für einen Amerikaner, Waffen an feindliche Nationen zu
> verkaufen. Das Land *Nono* besitzt Raketen, und alle Raketen von Nono wurden
> von Colonel *West* verkauft, der Amerikaner ist. Raketen sind Waffen. Ein Feind
> Amerikas gilt als feindlich (*hostile*). Nono ist ein Feind Amerikas.
>
> **Anfrage:** Ist West ein Krimineller? — Dein Beweiser soll **ja** beweisen.

Formalisiere die acht Sätze als FOL-Formeln (Prädikate wie `American/1`,
`Weapon/1`, `Sells/3`, `Hostile/1`, `Missile/1`, `Owns/2`, `Enemy/2`,
`Criminal/1`; Konstanten `West, Nono, America, M1`) und beweise `Criminal(West)`.

**(B) FOL mit Skolem-Funktion** — „Jeder hat einen Vorfahren":

> $\forall x\,\exists y\ \mathrm{Parent}(y,x)$ (jeder hat einen Elter);
> $\forall x\,\forall y\ (\mathrm{Parent}(y,x) \Rightarrow \mathrm{Ancestor}(y,x))$.
> **Anfrage:** $\forall x\,\exists y\ \mathrm{Ancestor}(y,x)$.

Hier *muss* deine Skolemisierung eine Skolem-**Funktion** erzeugen (der Elter
hängt von $x$ ab) und deine Unifikation muss in Funktionsterme hinein unifizieren.
Prüfe, dass der Beweis gelingt.

**(C) Reine Aussagenlogik** als Sonderfall — Modus Tollens:
$\{P \Rightarrow Q,\ \lnot Q\} \models \lnot P$. (Variablenfrei; testet, dass dein
FOL-Beweiser die Aussagenlogik als Spezialfall korrekt behandelt.)

**(D) Gegenprobe (Korrektheit!):** Etwas, das **nicht** folgt, darf **nicht**
bewiesen werden — z. B. folgt `Raining` nicht aus $\{\mathrm{Sunny},\
\mathrm{Sunny}\Rightarrow\mathrm{Warm}\}$. Dein Beweiser muss hier terminieren und
„nicht beweisbar" liefern (die KB ist endlich genug, dass die Saturierung stoppt).

**Zusätzlich (Pflicht-Demo):** Zeige explizit, dass `unify(x, f(x))` durch den
**Occurs-Check** abgelehnt wird (Rückgabe „kein Unifikator").

## Akzeptanzkriterien (Abnahmetest)

Dein Projekt ist fertig, wenn:

- [ ] die KNF-Umwandlung Implikationen/NNF/Standardisierung/Skolemisierung/
      Distribution korrekt durchführt (prüfbar an Zwischenausgaben);
- [ ] Unifikation den **MGU** liefert und den **Occurs-Check** durchführt;
- [ ] Szenario **(A)** `Criminal(West)` beweist und die **Ableitung der leeren
      Klausel** ausgibt (Kette der Resolutionsschritte mit Elternklauseln);
- [ ] Szenario **(B)** mit **Skolem-Funktion** gelingt;
- [ ] Szenario **(C)** (Aussagenlogik) gelingt;
- [ ] Szenario **(D)** korrekt **nicht** bewiesen wird (Korrektheit) und terminiert;
- [ ] die Occurs-Check-Demo das erwartete Ergebnis zeigt.

## Selbstcheck-Fragen (schriftlich beantworten)

1. **Warum Refutation statt direkter Ableitung?** Erkläre am Beispiel (A), warum
   du $\lnot\mathrm{Criminal}(\mathrm{West})$ hinzufügst, statt „vorwärts" zu schließen.
   (Skript: Resolution ist nur *widerlegungs*vollständig.)
2. **Warum erhält Skolemisierung nur die Erfüllbarkeit, nicht die Äquivalenz** —
   und warum genügt das dem Refutationsverfahren?
3. **Wozu das Umbenennen der Variablen vor jeder Resolution?** Konstruiere ein
   Beispiel, in dem Weglassen zu einem falschen (zu spezifischen) MGU führt.
4. **Warum terminiert dein Beweiser bei (D), aber im Allgemeinen nicht?** Verbinde
   das mit der **Semi-Entscheidbarkeit** der FOL-Gültigkeit (Skript 4.5): Was
   passiert bei einer nicht-folgernden Anfrage über einer KB mit Funktionssymbolen
   und unendlichem Herbrand-Universum?
5. **Wo wäre DPLL (Teil 3.6) hier einsetzbar, wo nicht?** (Stichwort: aussagenlogischer
   Spezialfall vs. Variablen/Unifikation.)

## Erweiterungen (optional, für Vertiefung)

- **Antwortliteral (answer literal):** Statt nur „ja/nein" auch *wer* der Täter ist,
  indem du die Anfrage $\exists x\,\mathrm{Criminal}(x)$ mit einem `Answer(x)`-Literal
  koppelst und die Bindung ausliest.
- **Subsumtion & Tautologie-Elimination**, um die Klauselmenge klein zu halten.
- **Die „Curiosity killed the cat"-KB** (AIMA) — braucht Skolemisierung *und*
  Faktorisierung; ein echter Härtetest deiner Vollständigkeit.

## Musterlösung

In **`solution/`** liegt eine vollständige, getestete Referenzimplementierung:
- `logic.py` — Terme/Formeln, komplette KNF-Pipeline, Unifikation mit Occurs-Check,
  Resolution + Faktorisierung, Set-of-Support-Refutationsschleife mit Beweisausgabe.
- `scenarios.py` — die vier Szenarien (A)–(D) + Occurs-Check-Demo; `python scenarios.py`
  gibt für (A) die volle Widerlegungskette aus.
- `test_prover.py` — automatische Abnahme; `python test_prover.py` → „Alle Tests bestanden."

Referenzwerte: (A) wird in ~600 Resolutionsschritten bewiesen, (B) in 2, (C) in 2,
(D) terminiert korrekt ohne Beweis. **Erst nach eigenem Versuch ansehen** — der
Lerngewinn steckt im selbst Bauen.

```bash
source ../../../../.venv/bin/activate    # nur Standardbibliothek nötig
cd solution && python scenarios.py       # Demo mit Beweis-Ausgabe
python test_prover.py                    # Abnahmetest
```
