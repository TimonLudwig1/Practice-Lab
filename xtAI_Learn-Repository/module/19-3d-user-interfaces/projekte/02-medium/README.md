# P02 (medium) — Zeige-Präzision & Reichweite: angulares Fitts' Law und Go-Go

**Modul 19 — 3D User Interfaces** · Format: **Python-Modul + Testsuite**

## Ziel

Du modellierst zwei Selektionstechniken **mechanistisch** (nicht über angenommene Kurven) und weist die zentralen Aussagen des Skripts quantitativ nach:

1. **Ray-Casting-Präzision fällt mit der Distanz.** Ein Ziel (Radius $r$) in Distanz $L$ subtendiert nur den angularen Radius $\theta_r\approx\arctan(r/L)$ — der **mit $L$ schrumpft**. Bei festem angularem Zeigerauschen $\sigma_\theta$ bricht die Trefferrate ein (angulares Fitts' Law, Skript Kap. 10).
2. **Go-Go tauscht Reichweite gegen Präzision.** Die nichtlineare Abbildung $r_v = r_r + k(r_r-D)^2$ (für $r_r\ge D$) verlängert den Arm dramatisch — aber die **C/D-Verstärkung** $g=\mathrm{d}r_v/\mathrm{d}r_r$ verstärkt jenseits von $D$ auch das Handrauschen ($\sigma_v = g\,\sigma_r$), sodass die Präzision im gestreckten Bereich sinkt. Die reine **Virtual Hand** ($k=0$) kann jenseits der Armlänge gar nichts erreichen.
3. **Angulares Fitts' Law** als Charakterisierung: $ID=\log_2(\theta_D/\theta_W+1)$ steigt mit der Distanz, die Bewegungszeit wächst — per Regression aus simulierten Trials gezeigt.

## Warum dieses Format?

Ein **Python-Modul mit Testsuite** — die Physik der Techniken (Trefferwahrscheinlichkeiten, Go-Go-Funktion) lässt sich so als getestete, über Parameter (Distanz, $k$, $\sigma$) systematisch variierbare Funktionen fassen.

## Warum synthetische Daten?

Die Aussagen sind **Modelle über Rauschen und Geometrie**. Nur mit kontrollierten Parametern lässt sich isolieren, *warum* Ray-Casting mit der Distanz zerfällt (angulare Schrumpfung) und Go-Go im Fernbereich unpräzise wird (Gain-Verstärkung). Alles reproduzierbar (fester Seed in den Simulationen).

## Vorwissen

**P01** dieses Moduls, **Kapitel 6 (Go-Go) & 10 (angulares Fitts)** des [Skripts](../../README.md), Fitts' Law aus Modul 17, Gauß/Rayleigh-Grundlagen.

## Aufgabenstellung

Öffne `selection.py`. Gain, Inverse, Fitts-Fit und die Experiment-/Test-Skripte sind vorgegeben — du implementierst die **drei Kernfunktionen** (`# TODO` / `NotImplementedError`):

1. **`p_hit_raycasting(r, L, sigma_theta)`** — Rayleigh-Trefferwahrscheinlichkeit $1-\exp(-\theta_r^2/2\sigma_\theta^2)$ mit $\theta_r=\arctan(r/L)$.
2. **`go_go(r_r, D, k)`** — die stückweise Go-Go-Funktion (`np.where`).
3. **`p_hit_gogo(r, L, D, k, sigma_r, arm_length)`** — drei Schritte: reale Handdistanz via `go_go_inverse`, Erreichbarkeitscheck gegen die Armlänge, Trefferwahrscheinlichkeit mit gain-verstärktem Rauschen.

Dann:

```bash
cd module/19-3d-user-interfaces/projekte/02-medium
/Users/.../.venv/bin/python test_selection.py   # 7 Tests -> alle PASS
/Users/.../.venv/bin/python run.py               # 3 Experimente + Plots
```

## Was am Ende herauskommt (Erwartungswerte)

**Ray-Casting vs. Distanz** ($r=0.1$ m, $\sigma_\theta=1°$): P(Treffer) fällt von ~1.0 bei $L=1$ m über 0.64 bei 4 m auf **0.06 bei 16 m** — doppelte Distanz halbiert die angulare Größe.

**Go-Go** ($D=0.45$, $k=60$, Arm $0.7$ m): max. Reichweite **4.45 m (×6.4 Arm)**. Trefferrate bei $r=0.08$ m: nah (≤0.45 m) ~1.0, dann fallend — 0.62 bei 1 m, 0.29 bei 2 m, **0.16 bei 3.5 m** (Gain wächst 1→27). Virtual Hand ($k=0$): **0.0 jenseits 0.7 m**.

**Angulares Fitts**: Fit gewinnt $a,b$ zurück ($R^2\approx0.96$); $ID$ steigt mit $L$ von ~1.9 auf ~5.4 bit, $MT$ von 0.52 auf 1.24 s.

> **Die Lehre.** Es gibt **keine universell beste** Selektionstechnik: Ray-Casting reicht beliebig weit, verliert aber angular an Präzision; Go-Go ist im Nahbereich exakt und weit reichend, aber im Streckbereich zittrig; Virtual Hand ist präzise, aber kurz. Die Wahl hängt von **Zieldistanz und -größe** ab — genau das vermisst das Final-Projekt in einer vollen vergleichenden Studie.

## Lösung

Vollständige Referenz in [`loesung/`](loesung/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: ein kompletter Selektionstechnik-Vergleich (Ray-Casting vs. Cone/Bubble) **unter Gedränge und Verdeckung**, mit ISO-9241-9-Throughput und Statistik. Keine Code-Vorgabe.
