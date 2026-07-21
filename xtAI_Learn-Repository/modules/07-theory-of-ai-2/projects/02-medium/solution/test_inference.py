"""The test suite. Run it while you fill in inference.py:

    python test_inference.py

All tests have to end with 'OK'.
"""
from bayesnet import alarm_net, diagnosis_net
from inference import enumeration_ask, elimination_ask, likelihood_weighting


def test_enumeration_alarm():
    bn = alarm_net()
    d = enumeration_ask("Burglary", {"JohnCalls": True, "MaryCalls": True}, bn)
    assert abs(d[True] - 0.2842) < 0.001, d          # the known AIMA result
    print("  Enumeration (alarm) ........ OK  P(Burglary|j,m)=%.4f" % d[True])


def test_elimination_matches_enumeration():
    for bn, X, e in [
        (alarm_net(), "Burglary", {"JohnCalls": True, "MaryCalls": True}),
        (alarm_net(), "Alarm", {"Burglary": True}),
        (diagnosis_net(), "Cancer", {"XRay": True, "Dyspnoea": True, "Smoker": True}),
    ]:
        a = enumeration_ask(X, e, bn)[True]
        b = elimination_ask(X, e, bn)[True]
        assert abs(a - b) < 1e-9, (X, e, a, b)
    print("  Variable elimination == enumeration ... OK")


def test_explaining_away():
    bn = alarm_net()
    p1 = enumeration_ask("Burglary", {"Alarm": True}, bn)[True]
    p2 = enumeration_ask("Burglary", {"Alarm": True, "Earthquake": True}, bn)[True]
    assert p2 < p1, (p1, p2)          # the earthquake "explains" the alarm away
    print("  Explaining away ............ OK  %.3f -> %.3f" % (p1, p2))


def test_likelihood_weighting():
    bn = diagnosis_net()
    X, e = "Cancer", {"XRay": True, "Dyspnoea": True, "Smoker": True}
    exact = enumeration_ask(X, e, bn)[True]
    approx = likelihood_weighting(X, e, bn, N=100000, seed=1)[True]
    assert abs(exact - approx) < 0.02, (exact, approx)
    print("  Likelihood weighting ....... OK  exact=%.4f approx=%.4f" % (exact, approx))


if __name__ == "__main__":
    print("Tests:")
    test_enumeration_alarm()
    test_elimination_matches_enumeration()
    test_explaining_away()
    test_likelihood_weighting()
    print("\nAll tests passed.")
