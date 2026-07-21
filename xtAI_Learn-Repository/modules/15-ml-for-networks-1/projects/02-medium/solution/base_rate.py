"""Der Base-Rate-Fallacy, quantitativ (Axelsson 2000) — Skript Abschnitt 3.1.

Die Kernfrage eines Analysten ist NICHT "wie oft erkennt der Detektor einen Angriff?" (TPR),
sondern: **"Es hat Alarm geschlagen — ist da wirklich was?"** Das ist der positive
Vorhersagewert (PPV / Precision), und er haengt entscheidend von der Basisrate ab:

    P(I|A) = P(A|I)*pi / ( P(A|I)*pi + P(A|~I)*(1-pi) )
           =   TPR*pi  / (   TPR*pi   +    FPR*(1-pi)  )

mit  I = Angriff ("Intrusion"),  A = Alarm,  pi = P(I) = Basisrate.
"""
from __future__ import annotations
import numpy as np


def ppv_bei_basisrate(tpr, fpr, pi):
    """Positiver Vorhersagewert P(Angriff | Alarm) nach Bayes.

    Akzeptiert Skalare oder Arrays (fuer pi). Rueckgabe: gleiche Form wie die Eingabe.
    """
    tpr = np.asarray(tpr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    pi = np.asarray(pi, dtype=float)
    zaehler = tpr * pi
    nenner = tpr * pi + fpr * (1.0 - pi)
    # Wo gar kein Alarm moeglich ist (Nenner 0), ist der PPV undefiniert -> 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(nenner > 0, zaehler / nenner, 0.0)
    return out if out.ndim else float(out)


def benoetigte_fpr(tpr, pi, ziel_ppv):
    """Welche FPR ist noetig, um einen Ziel-PPV zu erreichen? (Umstellung der Bayes-Formel)

        ppv = TPR*pi / (TPR*pi + FPR*(1-pi))
    =>  FPR = TPR*pi*(1-ppv) / (ppv*(1-pi))
    """
    tpr = float(tpr); pi = float(pi); ziel_ppv = float(ziel_ppv)
    if not (0.0 < ziel_ppv < 1.0):
        raise ValueError("ziel_ppv muss in (0,1) liegen")
    return tpr * pi * (1.0 - ziel_ppv) / (ziel_ppv * (1.0 - pi))


def alarme_pro_tag(n_flows_pro_tag, tpr, fpr, pi):
    """Absolute Alarmzahlen pro Tag. Rueckgabe: (echte_alarme, fehlalarme).

    Der wichtigste Reality-Check ueberhaupt: Prozente verschleiern, absolute Zahlen nicht.
    """
    n = float(n_flows_pro_tag)
    echte = tpr * pi * n
    falsch = fpr * (1.0 - pi) * n
    return echte, falsch


def erwartete_kosten(n_flows_pro_tag, tpr, fpr, pi,
                     kosten_fehlalarm, kosten_verpasst):
    """Erwartete Tageskosten eines Betriebspunkts.

    Fehlalarm  -> Analystenzeit (kosten_fehlalarm je Stueck)
    Verpasster Angriff -> Schaden (kosten_verpasst je Stueck)
    """
    n = float(n_flows_pro_tag)
    fehlalarme = fpr * (1.0 - pi) * n
    verpasste = (1.0 - tpr) * pi * n
    return fehlalarme * kosten_fehlalarm + verpasste * kosten_verpasst


def bester_betriebspunkt(fpr_kurve, tpr_kurve, schwellen, n_flows_pro_tag, pi,
                         kosten_fehlalarm, kosten_verpasst):
    """Waehlt aus einer ROC-Kurve den kostenminimalen Betriebspunkt.

    Rueckgabe: (index, schwelle, kosten) des Minimums.
    """
    kosten = np.array([
        erwartete_kosten(n_flows_pro_tag, t, f, pi, kosten_fehlalarm, kosten_verpasst)
        for f, t in zip(fpr_kurve, tpr_kurve)
    ])
    i = int(np.argmin(kosten))
    return i, float(schwellen[i]), float(kosten[i])
