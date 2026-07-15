"""Der Base-Rate-Fallacy, quantitativ (Axelsson 2000) — Skript Abschnitt 3.1.

>>> DEINE AUFGABE <<<  Fuelle die fuenf mit TODO markierten Funktionen. Sie sind alle kurz —
die Schwierigkeit liegt nicht im Code, sondern darin, die Formeln WIRKLICH zu verstehen.

Die Kernfrage eines Analysten ist NICHT "wie oft erkennt der Detektor einen Angriff?" (TPR),
sondern: **"Es hat Alarm geschlagen — ist da wirklich was?"** Das ist der positive
Vorhersagewert (PPV / Precision):

    P(I|A) = P(A|I)*pi / ( P(A|I)*pi + P(A|~I)*(1-pi) )
           =   TPR*pi  / (   TPR*pi   +    FPR*(1-pi)  )

mit  I = Angriff ("Intrusion"),  A = Alarm,  pi = P(I) = Basisrate.

Kontrollrechnung (die dein Code reproduzieren muss):
    TPR=0.99, FPR=0.001, pi=1e-4  ->  PPV ~ 9 %   (also ~91 % Fehlalarme!)

Musterloesung: loesung/base_rate.py — erst selbst versuchen!
"""
from __future__ import annotations
import numpy as np


def ppv_bei_basisrate(tpr, fpr, pi):
    """Positiver Vorhersagewert P(Angriff | Alarm) nach Bayes.

    Muss Skalare UND Arrays fuer `pi` verarbeiten (fuer die Kurve in run.py).
    Randfall: ist gar kein Alarm moeglich (Nenner 0), gib 0.0 zurueck.

    Tipp: np.asarray(..., dtype=float) am Anfang, dann np.where(nenner > 0, ...).
    """
    # TODO
    raise NotImplementedError


def benoetigte_fpr(tpr, pi, ziel_ppv):
    """Welche FPR ist noetig, um einen Ziel-PPV zu erreichen?

    Stelle die Bayes-Formel nach FPR um (Papier und Bleistift!). Wirf einen ValueError,
    wenn ziel_ppv nicht echt zwischen 0 und 1 liegt.

    Selbstcheck: ppv_bei_basisrate(tpr, benoetigte_fpr(tpr, pi, z), pi) == z
    """
    # TODO
    raise NotImplementedError


def alarme_pro_tag(n_flows_pro_tag, tpr, fpr, pi):
    """Absolute Alarmzahlen pro Tag. Rueckgabe: (echte_alarme, fehlalarme).

    Der wichtigste Reality-Check ueberhaupt: Prozente verschleiern, absolute Zahlen nicht.
    Von n Flows sind pi*n Angriffe (davon TPR erkannt) und (1-pi)*n harmlos
    (davon FPR faelschlich alarmiert).
    """
    # TODO
    raise NotImplementedError


def erwartete_kosten(n_flows_pro_tag, tpr, fpr, pi,
                     kosten_fehlalarm, kosten_verpasst):
    """Erwartete Tageskosten eines Betriebspunkts.

    Zwei Fehlerarten kosten Geld:
      Fehlalarm          -> Analystenzeit        (Anzahl: FPR*(1-pi)*n)
      verpasster Angriff -> Schaden              (Anzahl: (1-TPR)*pi*n)
    """
    # TODO
    raise NotImplementedError


def bester_betriebspunkt(fpr_kurve, tpr_kurve, schwellen, n_flows_pro_tag, pi,
                         kosten_fehlalarm, kosten_verpasst):
    """Waehlt aus einer ROC-Kurve den KOSTENMINIMALEN Betriebspunkt.

    Fuer jeden Punkt (fpr, tpr) der Kurve die erwarteten Kosten ausrechnen und das
    Minimum nehmen. Rueckgabe: (index, schwelle, kosten) als (int, float, float).

    Das ist die Bruecke zu Modul 04 (Kosten-Schwelle beim Adult-Projekt): die richtige
    Schwelle folgt aus den KOSTEN, nicht aus 0.5.
    """
    # TODO
    raise NotImplementedError
