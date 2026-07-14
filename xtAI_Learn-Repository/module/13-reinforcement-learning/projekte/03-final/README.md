# Projekt 03 (final) — Bestandsmanagement mit Reinforcement Learning

**Format: Python-Projekt, _ohne Code-Vorgabe_.** Dieses Abschlussprojekt bekommt **kein
Gerüst** — du baust alles selbst: Umgebung, Referenzlösung, Lerner, Auswertung. Das ist die
Master-Prüfungsleistung des Moduls: ein realistisches Entscheidungsproblem als MDP formulieren,
es modellfrei lösen **und** die Lösung gegen ein exaktes Optimum validieren.

**Warum ein `.py`-Projekt (kein Notebook)?** Weil hier mehrere Komponenten (Umgebung, DP-Referenz,
vier Lern-Algorithmen, Auswertung) sauber getrennt und *testbar* zusammenspielen — genau die
Architektur echter RL-Software. Ein Notebook würde das verwischen.

---

## Das Szenario (Praxisbezug)

Ein Händler steuert das **Lager** eines Produkts — ein klassisches Operations-Research-Problem
mit direktem Praxisbezug (Supply-Chain, Retail, Ersatzteillogistik). Jede Periode:

1. Bestand $i$ beobachten (Zustand).
2. Menge $a$ **bestellen** (Aktion) — trifft sofort ein, Bestand $x=i+a$ (gedeckelt durch
   Lagerkapazität $M$).
3. **Stochastische Nachfrage** $D\sim\text{Poisson}(\lambda)$ tritt ein.
4. Verkauf $=\min(x,D)$; nicht befriedigte Nachfrage geht **verloren** (*lost sales*);
   neuer Bestand $i'=\max(x-D,0)$.

**Kosten je Periode** (negative Belohnung):
$$r = -\big(\underbrace{K\,\mathbb 1[a>0]}_{\text{Bestellfixkosten}} + \underbrace{c\,a}_{\text{Stückkosten}} + \underbrace{h\,\max(x-D,0)}_{\text{Lagerkosten}} + \underbrace{p\,\max(D-x,0)}_{\text{Fehlmengenstrafe}}\big).$$

Die **Spannung**: viel bestellen → hohe Lagerkosten; wenig bestellen → teure Fehlmengen; oft
bestellen → viele Fixkosten. Das Optimum balanciert alles.

> **Theorie-Anker (den du bestätigen sollst):** Bei **Fixkosten** $K>0$ ist die optimale Politik
> nachweislich eine **(s, S)-Politik**: fällt der Bestand auf oder unter einen **Bestellpunkt
> $s$**, fülle auf ein **Zielniveau $S$** auf; sonst bestelle nichts. Ohne Fixkosten wird daraus
> eine **Base-Stock-Politik** ($s=S-1$). Dein RL-Agent soll diese Struktur *selbst entdecken*,
> ohne sie einprogrammiert zu bekommen.

## Lernziele dieses Projekts

- Ein reales Entscheidungsproblem **präzise als MDP** modellieren (Zustände, Aktionen mit
  Zulässigkeits-Constraints, stochastische Dynamik, Kostenfunktion).
- Erkennen, dass hier das **Modell bekannt** ist → **Value Iteration** (Modul 07!) liefert die
  *exakte* Optimalpolitik als **Referenz-Messlatte**.
- **Vier modellfreie** Verfahren implementieren, die das Modell **nicht** benutzen dürfen —
  MC-Control, SARSA, Q-Learning, Expected SARSA — und ihre gelernten Politiken **quantitativ**
  gegen das DP-Optimum bewerten.
- Eine **Hyperparameter-Studie** durchführen und interpretieren.

## Aufgabenstellung (Schritt für Schritt)

Baue die folgenden Bausteine (Dateinamen als Vorschlag):

1. **`inventory_env.py` — die Umgebung.** Zustände $0..M$, Aktionen $0..M$ mit **Maske**
   (in Zustand $i$ nur $a\le M-i$ zulässig). Eine Gym-artige API `reset()`/`step(a) → (i',r,False)`
   (fortlaufende Aufgabe, kein Terminalzustand). Nachfrage per Poisson-PMF (Tail auf $D_{\max}$
   bündeln, damit sie sich zu 1 summiert). **Zusätzlich** — nur für die Referenz — ein
   *explizites Modell*: `expected_reward(i,a)` und `transition_probs(i,a)`.
2. **`dp_reference.py` — Value Iteration** auf dem bekannten Modell → optimale Politik, $V^*$,
   $Q^*$. Plus eine **exakte Policy-Bewertung** `policy_value(π)` (löse
   $(I-\gamma P_\pi)V=R_\pi$), um beliebige Politiken zu benoten.
3. **`agents.py` — die vier Lerner** (alle ε-greedy, mit Aktionsmasken):
   MC-Control (every-visit, Episoden fester Länge), **SARSA**, **Q-Learning**, **Expected SARSA**.
   *Keiner* darf `expected_reward`/`transition_probs` benutzen — nur $(s,a,r,s')$-Erfahrung.
4. **`run.py` — das Experiment.** Trainiere alle, extrahiere die greedy-Politik, bewerte sie
   exakt und vergleiche: **Optimalitätslücke** in % gegenüber $V^*$ und Aktions-Übereinstimmung.
   Plotte Lernkurven und die **Ziel-Lagerstand-Kurven** ($x=i+a$) von gelernter vs. optimaler
   Politik. Führe die **α-Studie** für Q-Learning durch.
5. **`test_*.py` — Tests** (PMF normiert, Kostenformel, Masken, Transitions summieren zu 1,
   VI zeigt (s,S)-Struktur, `policy_value` reproduziert $V^*$, Q-Learning kommt nah ans Optimum).

**Parameter-Vorschlag:** $M=20$, $\lambda=8$, $c=2$, $K=10$, $h=1$, $p=6$, $\gamma=0.95$.
(Diese Werte erzeugen eine gut sichtbare (s,S)-Struktur; experimentiere ruhig.)

## Was am Ende funktionieren / herauskommen soll

- **Value Iteration** liefert eine saubere **(s,S)-Politik** — z. B. mit den Vorschlagswerten:
  Bestellpunkt $s\approx4$, Zielniveau $S\approx16$ (unter 5 Stück → auffüllen auf 16, sonst
  nichts bestellen).
- Die **modellfrei** gelernten Politiken liegen **nahe am Optimum**: Optimalitätslücke typisch
  **1–7 %** (bei $\varepsilon=0.1$, $\alpha=0.1$, ~300 000 Schritten). Größenordnung z. B.:

  | Verfahren | Opt.-Lücke |
  |---|---|
  | MC-Control | ~1–2 % |
  | Q-Learning | ~3 % |
  | Expected SARSA | ~4 % |
  | SARSA | ~7 % |

- **Interpretation, die du liefern sollst:**
  - Warum ist **SARSA** (on-policy) tendenziell etwas weiter vom Optimum als **Q-Learning**
    (off-policy)? → gleiche Logik wie beim Cliff Walking in Projekt 02: SARSA preist das
    ε-Explorationsverhalten in seine Werte ein und lernt eine leicht *konservativere* Politik.
  - Warum ist die **Aktions-Übereinstimmung** oft nur ~50–70 %, obwohl der **Wert** fast optimal
    ist? → Weil die Wertlandschaft um das Optimum **flach** ist (mehrere Bestellmengen sind fast
    gleich gut) und selten besuchte Zustände (hoher Bestand) verrauschte Q-Werte haben, aber
    kaum zum Ertrag beitragen. **Wichtige RL-Erkenntnis: near-optimale *Performance* ≠
    exakt-optimale *Politik*.**
  - **α-Studie:** zu kleines $\alpha$ lernt im festen Budget nicht fertig, zu großes $\alpha$
    ist zu verrauscht → es gibt ein **Sweet Spot** (bei den Vorschlagswerten um $\alpha\approx0.05$).

## Referenzlösung

Eine vollständige, getestete Musterlösung liegt in [`loesung/`](loesung/) (Umgebung, DP-Referenz,
vier Lerner, `run.py`, 10 Tests). **Sieh erst hinein, wenn du es selbst versucht hast.**

Ausführen:
```bash
/.../xtAI_Learn-Repository/.venv/bin/python loesung/test_inventory.py   # 10 Tests, ~1-2 s
/.../xtAI_Learn-Repository/.venv/bin/python loesung/run.py              # Experiment + Plot, ~30 s
```
Reines `numpy` (+ `matplotlib` optional). Alles **tabellarisch, CPU, Sekunden** — kein Training
schwerer Modelle.

## Erweiterungen (für die besonders Motivierten)

- **Backorders** statt lost sales (unbefriedigte Nachfrage wird nachgeliefert → negativer
  Bestand als Zustand): wie ändert sich die optimale Politik?
- **Lieferzeit (lead time)** $L>0$: Bestellungen treffen erst nach $L$ Perioden ein → der
  Zustand muss die *Pipeline* offener Bestellungen enthalten (größerer Zustandsraum).
- **Nichtstationäre Nachfrage** (saisonaler $\lambda_t$): konstantes $\alpha$ statt sample-average
  (vgl. Bandit-Projekt) — schlägt es jetzt die DP-Politik für *festes* $\lambda$?
- **Base-Stock ohne Fixkosten** ($K=0$): bestätige, dass $s=S-1$ wird (jede Periode auffüllen).
- Vergleiche gegen die **Newsvendor-Näherung**: $S \approx$ das $\frac{p}{p+h}$-Quantil der
  Nachfrageverteilung — wie nah liegt die exakte (s,S)-Lösung daran?
