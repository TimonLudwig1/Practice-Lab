"""Synthetische Daten einer XR-Nutzerstudie: 3 DoF vs. 6 DoF.

WARUM SYNTHETISCH? Eine echte Studie braucht Probanden, Wochen Zeit und eine Ethikkommission.
Fuer das Erlernen der AUSWERTUNG ist ein simulierter Datensatz sogar besser: die "Wahrheit"
(die echten Effekte, die Ordnungseffekte) ist bekannt, also kann man pruefen, ob die Statistik
sie wiederfindet - und was passiert, wenn man Fehler macht. Der Generator ist vollstaendig
offengelegt, damit jede Annahme sichtbar ist.

Studiendesign (Skript 5.2): WITHIN-SUBJECT (jede Person testet BEIDE Bedingungen) mit
COUNTERBALANCING (die Haelfte beginnt mit 3 DoF, die andere mit 6 DoF). Das ist in XR die
richtige Wahl, weil die individuellen Unterschiede (Anfaelligkeit, Erfahrung) riesig sind.

Eingebaute "Wahrheit" (die die Auswertung wiederfinden soll):
  * presence (IPQ, 1-7, hoeher=besser): 6 DoF deutlich hoeher (grosser Effekt)
  * sickness (SSQ-Delta, hoeher=schlechter): 3 DoF deutlich schlimmer (grosser Effekt)
  * time (Aufgabenzeit in s, niedriger=besser): 6 DoF schneller (grosser Effekt)
  * comfort (1-7, hoeher=besser): 6 DoF nur LEICHT besser (kleiner Effekt -> lehrreich fuer
    Mehrfachvergleiche: roh signifikant, nach Bonferroni nicht mehr)

Zwei absichtlich eingebaute ORDNUNGSEFFEKTE (der Grund fuer Counterbalancing):
  * Carryover: in der ZWEITEN Sitzung ist die Uebelkeit hoeher (sie klingt nicht ganz ab).
  * Lerneffekt: in der ZWEITEN Sitzung ist man bei der Aufgabe schneller.
Ohne Counterbalancing wuerden diese die Bedingungseffekte verzerren (run.py zeigt das).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# die "wahren" Effekte (6 DoF relativ zu 3 DoF) - hierhin soll die Auswertung zurueckfinden
WAHRHEIT = {
    "presence_effekt": +1.2,     # IPQ-Punkte
    "sickness_effekt": -12.0,    # SSQ: 6 DoF weniger krank (also 3 DoF +12)
    "time_effekt": -8.0,         # Sekunden schneller
    "comfort_effekt": +0.5,      # kleiner Komfort-Vorteil
    "carryover_sickness": +6.0,  # zweite Sitzung kraenker
    "lerneffekt_time": -5.0,     # zweite Sitzung schneller
}


def generate_study(n_participants: int = 24, seed: int = 3) -> pd.DataFrame:
    """Erzeugt einen Long-Format-DataFrame mit 2 Zeilen je Proband (eine je Bedingung).

    Spalten: participant, condition ('3DoF'/'6DoF'), position (1=erste Sitzung, 2=zweite),
             presence, sickness, time, comfort.
    """
    rng = np.random.default_rng(seed)
    n = n_participants

    # individuelle Merkmale (in XR gross - deshalb within-subject)
    anfaelligkeit = rng.normal(0, 1, n)     # sickness-prone
    geschick = rng.normal(0, 1, n)          # schneller bei der Aufgabe
    praesenz_basis = rng.normal(0, 1, n)    # generell empfaenglich fuer Praesenz
    komfort_basis = rng.normal(0, 0.8, n)

    # Counterbalancing: exakt die Haelfte beginnt mit 3 DoF (0), die andere mit 6 DoF (1)
    reihenfolge = np.tile([0, 1], n // 2)
    rng.shuffle(reihenfolge)

    zeilen = []
    for i in range(n):
        for cond in ("3DoF", "6DoF"):
            beginnt_mit_dieser = (reihenfolge[i] == 0 and cond == "3DoF") or \
                                 (reihenfolge[i] == 1 and cond == "6DoF")
            position = 1 if beginnt_mit_dieser else 2

            presence = (4.0 + praesenz_basis[i] * 0.8
                        + (WAHRHEIT["presence_effekt"] if cond == "6DoF" else 0)
                        + rng.normal(0, 0.6))
            sickness = (10 + anfaelligkeit[i] * 8
                        + (0 if cond == "6DoF" else -WAHRHEIT["sickness_effekt"])   # 3 DoF +12
                        + (WAHRHEIT["carryover_sickness"] if position == 2 else 0)
                        + rng.normal(0, 5))
            time = (40 - geschick[i] * 6
                    + (WAHRHEIT["time_effekt"] if cond == "6DoF" else 0)
                    + (WAHRHEIT["lerneffekt_time"] if position == 2 else 0)
                    + rng.normal(0, 4))
            comfort = (4.5 + komfort_basis[i]
                       + (WAHRHEIT["comfort_effekt"] if cond == "6DoF" else 0)
                       + rng.normal(0, 0.65))

            zeilen.append(dict(
                participant=i, condition=cond, position=position,
                presence=round(float(np.clip(presence, 1, 7)), 2),
                sickness=round(float(max(0, sickness)), 1),
                time=round(float(max(5, time)), 1),
                comfort=round(float(np.clip(comfort, 1, 7)), 2),
            ))
    return pd.DataFrame(zeilen)


def generate_naiv_ohne_counterbalancing(n_participants: int = 24, seed: int = 3) -> pd.DataFrame:
    """Dieselbe Studie, aber ALLE machen 3 DoF zuerst, 6 DoF zweite (kein Counterbalancing).

    Nur fuer die Gegenprobe in run.py: hier faellt der Carryover-Effekt komplett auf die
    6-DoF-Bedingung und maskiert deren Vorteil.
    """
    rng = np.random.default_rng(seed)
    n = n_participants
    anfaelligkeit = rng.normal(0, 1, n)
    zeilen = []
    for i in range(n):
        for cond, position in [("3DoF", 1), ("6DoF", 2)]:     # immer 3 DoF zuerst
            sickness = (10 + anfaelligkeit[i] * 8
                        + (0 if cond == "6DoF" else -WAHRHEIT["sickness_effekt"])
                        + (WAHRHEIT["carryover_sickness"] if position == 2 else 0)
                        + rng.normal(0, 5))
            zeilen.append(dict(participant=i, condition=cond,
                               sickness=round(float(max(0, sickness)), 1)))
    return pd.DataFrame(zeilen)


if __name__ == "__main__":
    df = generate_study()
    print(df.head(6).to_string(index=False))
    print(f"\n{df.participant.nunique()} Probanden, {len(df)} Zeilen")
    print("Counterbalancing:", df.groupby("condition")["position"].apply(
        lambda s: (s == 1).sum()).to_dict(), "beginnen je mit dieser Bedingung")
