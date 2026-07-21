"""Demonstration & Vergleich der drei Inferenzverfahren."""
from bayesnet import alarm_net, diagnosis_net
from inference import enumeration_ask, elimination_ask, likelihood_weighting


def fmt(d):
    return f"P(True)={d[True]:.4f}"


def compare(name, X, e, bn, N=200000):
    enum = enumeration_ask(X, e, bn)
    velim = elimination_ask(X, e, bn)
    lw = likelihood_weighting(X, e, bn, N=N, seed=1)
    print(f"\n{name}:  P({X} | {e})")
    print(f"  Aufzaehlung          : {fmt(enum)}")
    print(f"  Variable Elimination : {fmt(velim)}")
    print(f"  Likelihood Weighting : {fmt(lw)}   (N={N})")
    assert abs(enum[True] - velim[True]) < 1e-9, "exakte Verfahren muessen uebereinstimmen!"
    # LW hat bei Evidenz an den BLAETTERN (wie John/Mary) hohe Varianz -> grosszuegige Schranke
    assert abs(enum[True] - lw[True]) < 0.03, "Sampling sollte grob nahe am Exakten liegen"
    return enum[True]


if __name__ == "__main__":
    alarm = alarm_net()
    diag = diagnosis_net()

    p = compare("Alarm-Netz", "Burglary",
                {"JohnCalls": True, "MaryCalls": True}, alarm)
    print(f"  -> Erwartungswert ~0.2842 (bekanntes AIMA-Resultat): {'OK' if abs(p-0.2842)<0.001 else 'ABWEICHUNG'}")

    compare("Alarm-Netz (nur John)", "Burglary", {"JohnCalls": True}, alarm)

    # explaining away: Erdbebenmeldung senkt P(Burglary | Alarm)
    p1 = enumeration_ask("Burglary", {"Alarm": True}, alarm)[True]
    p2 = enumeration_ask("Burglary", {"Alarm": True, "Earthquake": True}, alarm)[True]
    print(f"\nExplaining away:  P(Burglary|Alarm)={p1:.3f}  ->  "
          f"P(Burglary|Alarm,Earthquake)={p2:.3f}  (sinkt: {'OK' if p2 < p1 else 'FEHLER'})")

    compare("Diagnose-Netz", "Cancer",
            {"XRay": True, "Dyspnoea": True, "Smoker": True}, diag)

    print("\nAlle Verfahren konsistent.")
