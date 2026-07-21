"""Die Motion-to-Photon-Kette und ihre zwei Gegenmittel: Prediction und Timewarp.

>>> DEINE AUFGABE <<<  Fuelle die vier mit TODO markierten Funktionen. `LatencyBudget` und
`ms_to_steps` sind vorgegeben.

Zentrale Groesse (Skript 3): die Zeit von "Kopf bewegt sich" bis "passendes Photon trifft die
Netzhaut". Zu gross (> ~20 ms) => das Bild hinkt der Bewegung hinterher => Cybersickness.

Was hier passiert: Zu jedem Zeitpunkt t ist die WAHRE Kopf-Orientierung `truth[t]`. Das Headset
zeigt aber die Orientierung, die vor `latency` Millisekunden gemessen wurde. Der ANGEZEIGTE
Fehler = |angezeigte Orientierung - wahre Orientierung JETZT|. Genau diesen Winkel spuert der
Nutzer als "die Welt haengt nach".

Alles arbeitet auf Arrays: `truth[i]` ist die wahre yaw-Orientierung zum Sample i. Eine Latenz
von L Millisekunden entspricht L*fs/1000 Samples Verschiebung.

Musterloesung: solution/latency.py — erst selbst versuchen!
"""
from __future__ import annotations
import numpy as np


def ms_to_steps(ms: float, fs: int) -> int:
    """Millisekunden in Sample-Schritte umrechnen (bei Abtastrate fs). Vorgegeben."""
    return int(round(ms / 1000.0 * fs))


class LatencyBudget:
    """Die Motion-to-Photon-Kette als Summe ihrer Glieder (Skript 3.1). Vorgegeben."""
    def __init__(self, **stages_ms):
        self.stages = stages_ms

    @property
    def total_ms(self) -> float:
        return float(sum(self.stages.values()))

    def report(self) -> str:
        zeilen = [f"  {name:12s} {ms:5.1f} ms" for name, ms in self.stages.items()]
        zeilen.append(f"  {'= gesamt':12s} {self.total_ms:5.1f} ms")
        bewertung = "OK (< 20 ms)" if self.total_ms < 20 else "ZU HOCH (>= 20 ms)"
        zeilen.append(f"  Motion-to-Photon: {bewertung}")
        return "\n".join(zeilen)


def displayed_pose(truth: np.ndarray, latency_ms: float, fs: int) -> np.ndarray:
    """Was das Headset OHNE Gegenmittel anzeigt: die Pose von vor `latency_ms`.

    Bauplan:
      - lat = ms_to_steps(latency_ms, fs)
      - jeder Ausgabewert i soll truth[i - lat] sein
      - am Anfang (i - lat < 0) die aelteste Pose halten -> np.clip(indices, 0, len-1)
    Rueckgabe: Array gleicher Laenge wie truth.
    """
    # TODO
    raise NotImplementedError


def predicted_pose(truth: np.ndarray, velocity: np.ndarray, latency_ms: float,
                   horizon_ms: float, fs: int) -> np.ndarray:
    """Prediction (Skript 3.2): statt der alten Pose zeige die um `horizon_ms` EXTRAPOLIERTE.

        angezeigt[i] = truth[i-lat] + velocity[i-lat] * (horizon_ms / 1000)

    velocity ist in Grad/Sekunde, horizon_ms in Millisekunden -> auf Sekunden umrechnen.
    Ideal ist horizon = latency (man rendert genau fuer den Anzeige-Zeitpunkt). Zu grosser
    horizon => Overshoot, v.a. an Richtungswechseln.
    """
    # TODO
    raise NotImplementedError


def timewarped_pose(truth: np.ndarray, render_latency_ms: float,
                    warp_latency_ms: float, fs: int) -> np.ndarray:
    """Orientational Timewarp (Skript 3.2): das fertig gerenderte Bild wird kurz vor der Anzeige
    anhand der NEUESTEN Pose nachgeschoben.

    Fuer eine reine DREHUNG laesst sich das fertige Bild exakt nachkorrigieren (man verschiebt
    nur den Bildausschnitt). Die effektive Latenz sinkt daher von der Render-Latenz auf die viel
    kleinere Warp-Latenz. Konkret heisst das hier: die angezeigte Pose ist einfach
    displayed_pose(truth, warp_latency_ms, fs) - der Bildinhalt (render_latency) ist bei reiner
    Rotation egal und taucht bewusst NICHT in der Formel auf.
    (Bei TRANSLATION waere das anders: dann fehlt verdeckte Information -> Disokklusion.)
    """
    # TODO
    raise NotImplementedError


def angular_error(displayed: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Der vom Nutzer gespuerte Fehler je Zeitpunkt: |angezeigt - wahr| (in Grad)."""
    # TODO
    raise NotImplementedError
