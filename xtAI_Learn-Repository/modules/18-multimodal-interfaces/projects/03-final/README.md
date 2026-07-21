# P03 (final) — „Put-that-there": ein multimodaler Referenz-Interpreter

**Modul 18 — Multimodal Interfaces** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Dies ist das Abschlussprojekt. Es gibt **keinen vorgegebenen Code** — du baust das System selbst aus dem, was du in Modul 18 gelernt hast. Die Referenzlösung liegt in [`solution/`](solution/); sieh sie dir **erst nach eigenem Versuch** an. Diese README ist die vollständige Spezifikation.

## Worum es geht

Du baust den Kern von Richard Bolts legendärem System von 1980 (Skript, Abschnitt 2 & 12) nach: einen Interpreter, der einen **gesprochenen deiktischen Befehl** („… put **that** …") mit einer **gleichzeitigen Zeigegeste** verschmilzt, um aufzulösen, *welches Objekt* der Szene gemeint ist. Das ist der **synergistische** CASE-Fall (parallel + fusioniert) mit **komplementären** Modalitäten (CARE) — der anspruchsvollste und interessanteste Fall multimodaler Interaktion.

Der Clou: Weder das Zeigen allein (räumlich mehrdeutig — der Zeiger kommt mehreren Objekten nahe) noch der zeitliche noch der semantische Kontext allein genügen. Erst ihre **multiplikative Fusion** (Bayes-Produkt) löst die Referenz zuverlässig auf. Du wirst das **quantitativ nachweisen** — inklusive **Mutual Disambiguation** und der Rolle des zeitlichen Versatzes.

## Lernziel

Du wendest alle drei Fusionsebenen des Moduls an einem realistischen, asynchronen Szenario an:
- die **Bayes-Produktregel** über drei Faktoren (Skript, Abschnitt 12),
- **zeitliche Fusion** mit einem Zeitfenster und dem empirischen Versatz „Geste führt Wort" (Abschnitt 11),
- **Mutual Disambiguation** und deren Messung (Abschnitt 10),
- saubere **Ablations-Evaluation** (Abschnitt 16).

## Vorwissen

Die gesamte Modul-18-[README](../../README.md), besonders **Abschnitte 7 (Bayes-Produkt), 11 (Zeitfenster), 12 (Referenzauflösung), 16 (Evaluation)**. Aus P01/P02 die Fusionsmechanik. Numpy.

---

## Aufgabenstellung (Spezifikation)

### 1. Szene & asynchrone Ereignisströme (Datengenerator)

Baue einen **reproduzierbaren, offengelegten** Generator (fester Seed) — die Datenwahl ist bewusst **synthetisch**, weil nur so die *ground truth* (welches Objekt wirklich gemeint war) und die eingebauten Mehrdeutigkeiten exakt kontrollierbar sind; echte Zeige-/Sprachdaten hätten weder Labels noch die gezielten Fallen. Modelliere:

- **Szene**: $n$ Objekte, jedes mit einer 2D-Bildschirmposition $\mathbf{p}_o \in [0,1]^2$ und einem **Typ** $\tau_o$ aus z. B. `{button, slider, text, image}`.
- **Sprach-Strom**: ein deiktisches Wort „that" zum Zeitpunkt $t_{\text{word}}$, plus eine **verrauschte ASR-Verteilung** $q$ über einen Typ-Nomen-Slot („… that **button**"). Modelliere die ASR-Unsicherheit: mit Wahrscheinlichkeit `asr_correct_prob` liegt die Verteilung auf dem *wahren* Typ, sonst auf einem falschen; manchmal wird gar kein Nomen gesprochen ($q$ uniform).
- **Gesten-Strom**: eine über die Zeit **abgetastete, verrauschte 2D-Zeigeposition** $\mathbf{r}(t)$ (Zeitstempel + Position).

**Baue gezielt zwei Mehrdeutigkeiten ein** (das ist der didaktische Kern):
1. **Decoy (zeitliche Falle)**: Lass den Zeiger seine Bewegung **auf einem anderen Objekt** beginnen und von dort zum Ziel wandern. So kommt der Zeiger dem Decoy räumlich sehr nahe — aber zur **falschen Zeit** (ganz am Anfang, weit vor $t_{\text{word}}$). Nur der **zeitliche** Faktor kann ihn ausschließen.
2. **Zwilling (semantische Falle)**: Platziere mit ~50 % Wahrscheinlichkeit ein zweites Objekt **direkt neben dem Ziel**, aber mit **anderem Typ**. Räumlich und zeitlich fast identisch zum Ziel — nur der **semantische** Faktor (der gehörte Typ) kann ihn trennen.

Wichtig: Modelliere den **Versatz „Geste führt Wort"** — der Zeiger erreicht das Ziel etwas *vor* dem deiktischen Wort ($t_{\text{arrive}} = t_{\text{word}} + \mu$ mit $\mu < 0$, empirisch ~$-0.15$ s). Danach wandert der Zeiger weiter (zur „there"-Position), sodass das Ziel ein *scharfer Wegpunkt* ist, nicht ein dauerndes Verweilen.

### 2. Der Interpreter (die Fusion)

Für jedes Objekt $o$ berechne $\log P(o) = \log P_{\text{point}}(o) + \log P_{\text{temp}}(o) + \log P_{\text{sem}}(o)$, dann Softmax über die Objekte. Aufgelöst wird $\arg\max_o P(o)$. Die drei Faktoren:

$$P_{\text{point}}(o) \propto \exp\!\Big(-\frac{d_{\min}(o)^2}{2\sigma^2}\Big), \qquad
  P_{\text{temp}}(o) \propto \exp\!\Big(-\frac{(\delta_o - \mu)^2}{2\tau^2}\Big), \qquad
  P_{\text{sem}}(o) = q[\tau_o]$$

wobei $d_{\min}(o)$ die minimale Zeiger-Objekt-Distanz über den ganzen Strom ist und $\delta_o = t_{\text{near}}(o) - t_{\text{word}}$ der zeitliche Abstand des Moments größter Nähe zum deiktischen Wort. $P_{\text{point}}$ fragt *„kam der Zeiger je nahe?"*, $P_{\text{temp}}$ *„geschah das zur richtigen Zeit?"*, $P_{\text{sem}}$ *„passt der Typ zum Gehörten?"*.

Baue die Faktoren **einzeln abschaltbar** (Flags), damit du Ablationen fahren kannst.

### 3. Evaluation

Erzeuge einen großen Pool von Kommandos (z. B. 3000 über wechselnde Szenen) und miss die **Auflösungs-Genauigkeit** (aufgelöstes == wahres Objekt) für:

- **Ablationen**: nur Zeigen · nur Semantik · Zeigen+Semantik (ohne Zeit) · Zeigen+Zeit (ohne Semantik) · **volle Fusion**. Plus eine **naive Baseline** (das Objekt, das dem Zeiger *exakt* zum Wortzeitpunkt am nächsten war).
- **Mutual Disambiguation**: Zähle die Kommandos, bei denen **Zeigen allein danebenliegt**, und davon den Anteil, den die **volle Fusion rettet**.
- **μ-Studie**: Variiere den angenommenen Versatz $\hat\mu$ und zeige, dass die Genauigkeit bei $\hat\mu \approx \mu_{\text{wahr}}$ am höchsten ist und bei der naiven Annahme $\hat\mu = 0$ („gleichzeitig") deutlich einbricht.

Erzeuge Plots (nach `results/`, gitignored) und eine Testsuite (`__main__`-Runner, da kein pytest).

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

Deine Zahlen dürfen abweichen (andere Parameter/Seeds), aber die **Rangfolge und die Geschichte** müssen stimmen:

| Konfiguration | Genauigkeit (Referenz) |
|---|---|
| nur Zeigen ($P_{\text{point}}$) | ~0.48 — **mehrdeutig** (Decoy ≈ Ziel räumlich) |
| nur Semantik ($P_{\text{sem}}$) | ~0.36 — allein schwach |
| Zeigen + Semantik (ohne Zeit) | ~0.69 |
| Zeigen + Zeit (ohne Semantik) | ~0.83 — **Zeit eliminiert den Decoy** |
| **volle Fusion** | **~0.89 — Maximum** |
| naiv (nächstes Objekt @ $t_{\text{word}}$) | ~0.48 — versagt (Zeiger schon weitergewandert) |

- **Mutual Disambiguation**: In der Referenz liegt Zeigen-allein bei ~1560/3000 Kommandos falsch — davon rettet die volle Fusion **~86 %**. Das ist der quantitative Beweis, dass der zeitliche + semantische Kontext die räumliche Mehrdeutigkeit auflöst.
- **μ-Studie**: bestes $\hat\mu \approx -0.2$ s (nahe am wahren $-0.15$); $\hat\mu = 0$ verliert ~9 Prozentpunkte.

> **Die große Lehre des Moduls, an einem System gezeigt:** Die drei Modalitäts-Faktoren sind **komplementär, nicht redundant** — jeder löst eine *andere* Mehrdeutigkeit (Zeit → Decoy, Semantik → Zwilling, Raum → grobe Position). Deshalb schlägt die volle Fusion jede Teilkombination, und deshalb ist „Put-that-there" ein *synergistisches* Interface: keine Modalität ist entbehrlich.

## Setup & Ausführen

```bash
cd modules/18-multimodal-interfaces/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_putthatthere.py   # Testsuite
/Users/.../.venv/bin/python run.py                  # Evaluation + Plots
```

Nur `numpy` + `matplotlib` nötig. Laufzeit wenige Sekunden (reine CPU, kein Training).

## Lösung

Vollständige Referenz in [`solution/`](solution/): `putthatthere.py` (Generator + Interpreter), `run.py` (Ablation, Mutual Disambiguation, μ-Studie, Plots), `test_putthatthere.py` (8 Tests). **Erst selbst bauen!**

## Rückblick & Ausblick

Damit schließt Modul 18: Du hast die multimodale Fusion von der **inverse-Varianz-Wahrnehmung** (P01) über **early/late Fusion & Mutual Disambiguation** (P02) bis zum **vollständigen synergistischen Interpreter** (P03) durchgearbeitet. Die Zeige- und Referenzmathematik hier ist direkte Vorarbeit für **Modul 19 „3D User Interfaces"** (Selektion/Manipulation im 3D-Raum).
