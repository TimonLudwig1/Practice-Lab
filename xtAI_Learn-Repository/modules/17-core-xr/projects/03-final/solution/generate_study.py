"""Synthetic data of an XR user study: 3 DoF vs. 6 DoF.

WHY SYNTHETIC? A real study needs participants, weeks of time and an ethics committee. For
learning the ANALYSIS a simulated dataset is even better: the "truth" (the real effects, the
order effects) is known, so you can check whether the statistics recover it - and what happens if
you make mistakes. The generator is fully disclosed so that every assumption is visible.

Study design (script 5.2): WITHIN-SUBJECT (every person tests BOTH conditions) with
COUNTERBALANCING (half start with 3 DoF, the other half with 6 DoF). In XR that is the right
choice, because the individual differences (susceptibility, experience) are enormous.

The built-in "truth" (which the analysis is supposed to recover):
  * presence (IPQ, 1-7, higher=better): 6 DoF clearly higher (a large effect)
  * sickness (the SSQ delta, higher=worse): 3 DoF clearly worse (a large effect)
  * time (task time in s, lower=better): 6 DoF faster (a large effect)
  * comfort (1-7, higher=better): 6 DoF only SLIGHTLY better (a small effect -> instructive for
    multiple comparisons: significant raw, no longer after Bonferroni)

Two deliberately built-in ORDER EFFECTS (the reason for counterbalancing):
  * carryover: in the SECOND session the nausea is higher (it does not fully subside).
  * learning: in the SECOND session you are faster at the task.
Without counterbalancing these would bias the condition effects (run.py shows that).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# the "true" effects (6 DoF relative to 3 DoF) - this is what the analysis should find its way
# back to
TRUTH = {
    "presence_effect": +1.2,       # IPQ points
    "sickness_effect": -12.0,      # SSQ: 6 DoF less sick (so 3 DoF +12)
    "time_effect": -8.0,           # seconds faster
    "comfort_effect": +0.5,        # a small comfort advantage
    "carryover_sickness": +6.0,    # the second session is sicker
    "learning_effect_time": -5.0,  # the second session is faster
}


def generate_study(n_participants: int = 24, seed: int = 3) -> pd.DataFrame:
    """Produces a long-format DataFrame with 2 rows per participant (one per condition).

    Columns: participant, condition ('3DoF'/'6DoF'), position (1=first session, 2=second),
             presence, sickness, time, comfort.
    """
    rng = np.random.default_rng(seed)
    n = n_participants

    # individual traits (large in XR - which is why within-subject)
    susceptibility = rng.normal(0, 1, n)     # sickness-prone
    skill = rng.normal(0, 1, n)              # faster at the task
    presence_baseline = rng.normal(0, 1, n)  # generally receptive to presence
    comfort_baseline = rng.normal(0, 0.8, n)

    # counterbalancing: exactly half start with 3 DoF (0), the other half with 6 DoF (1)
    order = np.tile([0, 1], n // 2)
    rng.shuffle(order)

    rows = []
    for i in range(n):
        for cond in ("3DoF", "6DoF"):
            starts_with_this = (order[i] == 0 and cond == "3DoF") or \
                               (order[i] == 1 and cond == "6DoF")
            position = 1 if starts_with_this else 2

            presence = (4.0 + presence_baseline[i] * 0.8
                        + (TRUTH["presence_effect"] if cond == "6DoF" else 0)
                        + rng.normal(0, 0.6))
            sickness = (10 + susceptibility[i] * 8
                        + (0 if cond == "6DoF" else -TRUTH["sickness_effect"])   # 3 DoF +12
                        + (TRUTH["carryover_sickness"] if position == 2 else 0)
                        + rng.normal(0, 5))
            time = (40 - skill[i] * 6
                    + (TRUTH["time_effect"] if cond == "6DoF" else 0)
                    + (TRUTH["learning_effect_time"] if position == 2 else 0)
                    + rng.normal(0, 4))
            comfort = (4.5 + comfort_baseline[i]
                       + (TRUTH["comfort_effect"] if cond == "6DoF" else 0)
                       + rng.normal(0, 0.65))

            rows.append(dict(
                participant=i, condition=cond, position=position,
                presence=round(float(np.clip(presence, 1, 7)), 2),
                sickness=round(float(max(0, sickness)), 1),
                time=round(float(max(5, time)), 1),
                comfort=round(float(np.clip(comfort, 1, 7)), 2),
            ))
    return pd.DataFrame(rows)


def generate_naive_without_counterbalancing(n_participants: int = 24,
                                            seed: int = 3) -> pd.DataFrame:
    """The same study, but EVERYBODY does 3 DoF first and 6 DoF second (no counterbalancing).

    Only for the counter-check in run.py: here the carryover falls entirely on the 6 DoF
    condition and masks its advantage.
    """
    rng = np.random.default_rng(seed)
    n = n_participants
    susceptibility = rng.normal(0, 1, n)
    rows = []
    for i in range(n):
        for cond, position in [("3DoF", 1), ("6DoF", 2)]:     # always 3 DoF first
            sickness = (10 + susceptibility[i] * 8
                        + (0 if cond == "6DoF" else -TRUTH["sickness_effect"])
                        + (TRUTH["carryover_sickness"] if position == 2 else 0)
                        + rng.normal(0, 5))
            rows.append(dict(participant=i, condition=cond,
                             sickness=round(float(max(0, sickness)), 1)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_study()
    print(df.head(6).to_string(index=False))
    print(f"\n{df.participant.nunique()} participants, {len(df)} rows")
    print("Counterbalancing:", df.groupby("condition")["position"].apply(
        lambda s: (s == 1).sum()).to_dict(), "start with this condition respectively")
