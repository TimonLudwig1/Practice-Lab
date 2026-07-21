# Projekt 03 (final) — Ein Resolutions-Theorembeweiser für die Prädikatenlogik

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
cd loesung && python scenarios.py        # Demo mit Beweis-Ausgabe
python test_prover.py                    # Abnahmetest
```
