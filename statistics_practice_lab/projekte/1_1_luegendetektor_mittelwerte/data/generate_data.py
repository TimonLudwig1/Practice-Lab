"""Datengenerator für Projekt 1.1 — Der Lügendetektor für Mittelwerte.

Wir erzeugen DREI realistische Gehaltsdatensätze (Jahresbruttogehalt in Euro),
deren *wahren* Datengenerierungsprozess (DGP) wir kennen. Genau das ist die
didaktische Superkraft simulierter Daten: Wir wissen, welche Verteilung
dahintersteckt, und können jede Kennzahl daran messen.

Die drei Welten:

1. ``symmetric`` — Normalverteilung. Ein Betrieb mit sehr homogenem Lohngefüge.
   Mittelwert und Median liegen fast aufeinander → hier lässt sich kaum lügen.

2. ``right_skewed`` — Lognormalverteilung + einige "Superreiche" (CEOs, Stars).
   Ein langer rechter Schwanz zieht den Mittelwert nach oben, der Median bleibt
   ruhig → die klassische Bühne für Durchschnittslügen.

3. ``bimodal`` — Mischung zweier Berufsgruppen (z.B. Pflege/Einzelhandel vs.
   IT/Ingenieurwesen). Zwei Gipfel, dazwischen ein Tal, in dem kaum jemand
   verdient → Mittelwert UND Median landen im leeren Tal und lügen gemeinsam.

Aufruf als Skript schreibt die drei Datensätze als CSV nach ``./`` (data/).
Im Notebook importieren wir stattdessen ``generate_all()`` direkt, damit der
DGP sichtbar bleibt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reproduzierbarkeit ist Pflicht (siehe CLAUDE.md): fester Seed, modernes API.
SEED = 42
N = 5_000  # Beschäftigte pro "Erhebung" — groß genug für stabile Kennzahlen.


def generate_symmetric(rng: np.random.Generator, n: int = N) -> np.ndarray:
    """Symmetrische Gehälter: Normalverteilung um 45.000 € (SD 8.000 €).

    Wahrer DGP: X ~ Normal(μ=45000, σ=8000), untergrenzt bei 20.000 €
    (Mindestlohn-Effekt; die Kappung erzeugt nur eine minimale Schiefe).
    Für eine Normalverteilung gilt: Erwartungswert = Median = μ.
    """
    salaries = rng.normal(loc=45_000, scale=8_000, size=n)
    # Niemand verdient unter ~20.000 €/Jahr (Vollzeit-Mindestlohn-Größenordnung).
    return np.clip(salaries, 20_000, None)


def generate_right_skewed(rng: np.random.Generator, n: int = N) -> np.ndarray:
    """Rechtsschiefe Gehälter: Lognormal + einige Superverdiener.

    Wahrer DGP:
      - Basis: X ~ Lognormal, sodass der Median = exp(μ_log) = 38.000 € ist
        und σ_log = 0.55 einen realistischen rechten Schwanz erzeugt.
      - Zusätzlich ersetzen wir 5 zufällige Personen durch "Superreiche"
        (2 bis 15 Mio. €) — CEOs, Stars, Erben. Sie sind selten, aber sie
        allein reichen, um den Mittelwert dramatisch nach oben zu hebeln.

    Für die Lognormal-Basis gilt analytisch:
      Median = exp(μ_log),  Mittelwert = exp(μ_log + σ_log²/2) > Median.
    """
    mu_log = np.log(38_000)  # -> Median der Basisverteilung = 38.000 €
    sigma_log = 0.55
    salaries = rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)

    # 5 Superreiche einstreuen (absichtliche Extremwerte, im Aufgabenteil der
    # eigentliche Grund, warum der Mittelwert lügt).
    n_super = 5
    idx = rng.choice(n, size=n_super, replace=False)
    salaries[idx] = rng.uniform(2_000_000, 15_000_000, size=n_super)
    return salaries


def generate_bimodal(rng: np.random.Generator, n: int = N) -> np.ndarray:
    """Bimodale Gehälter: zwei Berufsgruppen mit klarer Lücke dazwischen.

    Wahrer DGP: Mischung zweier Normalverteilungen
      - Gruppe A (60 %): Pflege/Einzelhandel ~ Normal(31000, 4000)
      - Gruppe B (40 %): IT/Ingenieurwesen  ~ Normal(66000, 9000)
    Die beiden Gipfel liegen weit auseinander; im Tal bei ~48.000 € verdient
    fast niemand — genau dort landen aber Mittelwert und Median.
    """
    p_group_a = 0.60
    is_group_a = rng.random(n) < p_group_a

    salaries = np.empty(n)
    n_a = int(is_group_a.sum())
    salaries[is_group_a] = rng.normal(31_000, 4_000, size=n_a)
    salaries[~is_group_a] = rng.normal(66_000, 9_000, size=n - n_a)
    return np.clip(salaries, 20_000, None)


def generate_all(seed: int = SEED, n: int = N) -> dict[str, np.ndarray]:
    """Erzeuge alle drei Datensätze mit EINEM gemeinsamen Seed.

    Wir ziehen die drei Datensätze nacheinander aus demselben Generator, damit
    das gesamte Projekt aus einem einzigen Seed reproduzierbar ist.
    """
    rng = np.random.default_rng(seed)
    return {
        "symmetric": generate_symmetric(rng, n),
        "right_skewed": generate_right_skewed(rng, n),
        "bimodal": generate_bimodal(rng, n),
    }


def main() -> None:
    """Als Skript: Datensätze erzeugen und als CSV neben dieses Skript legen."""
    here = Path(__file__).parent
    data = generate_all()
    for name, values in data.items():
        df = pd.DataFrame({"salary_eur": values})
        out = here / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"{name:14s} -> {out.name}  (n={len(values)}, "
              f"mean={values.mean():,.0f} €, median={np.median(values):,.0f} €)")


if __name__ == "__main__":
    main()
