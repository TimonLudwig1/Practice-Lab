"""Datengenerator für Projekt 1.2 — Streuung zum Anfassen.

Zwei Kaffeemaschinen füllen 200-ml-Tassen (Fassungsvermögen bis 250 ml).
Beide zielen auf denselben Sollwert — sie unterscheiden sich NUR in der Streuung.
Das ist der didaktische Kern: gleicher Mittelwert, völlig anderes Verhalten.

Wahrer DGP:
  - Maschine A ("Präzise"):  Füllmenge ~ Normal(μ=235 ml, σ=5 ml)
  - Maschine B ("Schludrig"): Füllmenge ~ Normal(μ=235 ml, σ=15 ml)

Beide Male ist der Erwartungswert 235 ml. Bei 250 ml läuft die Tasse über.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 10_000            # Tassen pro Maschine (Projektvorgabe: >= 10.000)
TARGET_ML = 235.0     # gemeinsamer Sollwert / Erwartungswert beider Maschinen
CUP_CAPACITY_ML = 250.0  # ab hier läuft die Tasse über

MACHINE_A_SD = 5.0    # präzise
MACHINE_B_SD = 15.0   # schludrig


def generate(seed: int = SEED, n: int = N) -> dict[str, np.ndarray]:
    """Erzeuge Füllmengen beider Maschinen aus einem gemeinsamen Seed."""
    rng = np.random.default_rng(seed)
    return {
        "machine_a": rng.normal(TARGET_ML, MACHINE_A_SD, size=n),
        "machine_b": rng.normal(TARGET_ML, MACHINE_B_SD, size=n),
    }


def add_outliers(values: np.ndarray, seed: int = SEED, n_out: int = 5) -> np.ndarray:
    """Stretch: 5 grobe Fehlbefüllungen einstreuen (z.B. Maschine 'rülpst' ~400 ml).

    Ausreißer sind hier absichtlich extrem, um zu zeigen, welche Streuungsmaße
    explodieren (Varianz, SD, Spannweite) und welche stabil bleiben (IQR, MAD).
    """
    rng = np.random.default_rng(seed + 1)
    out = values.copy()
    idx = rng.choice(len(out), size=n_out, replace=False)
    out[idx] = rng.uniform(380, 420, size=n_out)
    return out


def main() -> None:
    here = Path(__file__).parent
    data = generate()
    for name, v in data.items():
        pd.DataFrame({"fill_ml": v}).to_csv(here / f"{name}.csv", index=False)
        print(f"{name:10s} -> mean={v.mean():.2f} ml, sd={v.std(ddof=1):.2f} ml, "
              f"überlaufend={np.mean(v > CUP_CAPACITY_ML) * 100:.2f}%")


if __name__ == "__main__":
    main()
