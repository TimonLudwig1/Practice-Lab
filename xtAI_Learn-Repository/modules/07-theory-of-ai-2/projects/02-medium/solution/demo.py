"""A demonstration and comparison of the three inference procedures."""
from bayesnet import alarm_net, diagnosis_net
from inference import enumeration_ask, elimination_ask, likelihood_weighting


def fmt(d):
    return f"P(True)={d[True]:.4f}"


def compare(name, X, e, bn, N=200000):
    enum = enumeration_ask(X, e, bn)
    velim = elimination_ask(X, e, bn)
    lw = likelihood_weighting(X, e, bn, N=N, seed=1)
    print(f"\n{name}:  P({X} | {e})")
    print(f"  Enumeration          : {fmt(enum)}")
    print(f"  Variable elimination : {fmt(velim)}")
    print(f"  Likelihood weighting : {fmt(lw)}   (N={N})")
    assert abs(enum[True] - velim[True]) < 1e-9, "the exact procedures must agree!"
    # LW has high variance with evidence at the LEAVES (like John/Mary) -> a generous bound
    assert abs(enum[True] - lw[True]) < 0.03, "sampling should be roughly close to the exact value"
    return enum[True]


if __name__ == "__main__":
    alarm = alarm_net()
    diag = diagnosis_net()

    p = compare("Alarm network", "Burglary",
                {"JohnCalls": True, "MaryCalls": True}, alarm)
    print(f"  -> expected about 0.2842 (the known AIMA result): {'OK' if abs(p-0.2842)<0.001 else 'DEVIATION'}")

    compare("Alarm network (John only)", "Burglary", {"JohnCalls": True}, alarm)

    # explaining away: an earthquake report lowers P(Burglary | Alarm)
    p1 = enumeration_ask("Burglary", {"Alarm": True}, alarm)[True]
    p2 = enumeration_ask("Burglary", {"Alarm": True, "Earthquake": True}, alarm)[True]
    print(f"\nExplaining away:  P(Burglary|Alarm)={p1:.3f}  ->  "
          f"P(Burglary|Alarm,Earthquake)={p2:.3f}  (it falls: {'OK' if p2 < p1 else 'AN ERROR'})")

    compare("Diagnosis network", "Cancer",
            {"XRay": True, "Dyspnoea": True, "Smoker": True}, diag)

    print("\nAll procedures consistent.")
