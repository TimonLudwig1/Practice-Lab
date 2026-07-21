"""Die Motion-to-Photon-Kette und ihre zwei Gegenmittel: Prediction und Timewarp.

Zentrale Groesse (Skript 3): die Zeit von "Kopf bewegt sich" bis "passendes Photon trifft die
Netzhaut". Zu gross (> ~20 ms) => das Bild hinkt der Bewegung hinterher => Cybersickness.

Was hier passiert: Zu jedem Zeitpunkt t ist die WAHRE Kopf-Orientierung `truth[t]`. Das Headset
zeigt aber die Orientierung, die vor `latency` Millisekunden gemessen wurde. Der ANGEZEIGTE
Fehler = |angezeigte Orientierung - wahre Orientierung JETZT|. Genau diesen Winkel spuert der
Nutzer als "die Welt haengt nach".
"""
from __future__ import annotations
import numpy as np


def ms_to_steps(ms: float, fs: int) -> int:
    """Millisekunden in Sample-Schritte umrechnen (bei Abtastrate fs)."""
    return int(round(ms / 1000.0 * fs))


class LatencyBudget:
    """Die Motion-to-Photon-Kette als Summe ihrer Glieder (Skript 3.1).

    Prozente verschleiern, Millisekunden nicht: hier sieht man, dass bei 90 Hz allein EIN Frame
    (11.1 ms) fast das gesamte 20-ms-Budget frisst.
    """
    def __init__(self, **stages_ms):
        # z.B. sensor=1, fusion=1, app=4, render=11, scanout=3, display=2
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

    Am Anfang (bevor genug Historie da ist) wird die aelteste Pose gehalten.
    """
    lat = ms_to_steps(latency_ms, fs)
    idx = np.clip(np.arange(len(truth)) - lat, 0, len(truth) - 1)
    return truth[idx]


def predicted_pose(truth: np.ndarray, velocity: np.ndarray, latency_ms: float,
                   horizon_ms: float, fs: int) -> np.ndarray:
    """Prediction (Skript 3.2): statt der alten Pose zeige die um `horizon_ms` EXTRAPOLIERTE.

    pose_angezeigt = alte_pose + alte_geschwindigkeit * horizon
    Ideal ist horizon = latency (man rendert genau fuer den Anzeige-Zeitpunkt). Zu grosser
    horizon => Overshoot, v.a. an Richtungswechseln.
    """
    lat = ms_to_steps(latency_ms, fs)
    idx = np.clip(np.arange(len(truth)) - lat, 0, len(truth) - 1)
    return truth[idx] + velocity[idx] * (horizon_ms / 1000.0)


def timewarped_pose(truth: np.ndarray, render_latency_ms: float,
                    warp_latency_ms: float, fs: int) -> np.ndarray:
    """Orientational Timewarp (Skript 3.2): das fertig gerenderte Bild wird kurz vor der Anzeige
    anhand der NEUESTEN Pose nachgeschoben.

    Fuer eine reine DREHUNG laesst sich das fertige Bild exakt nachkorrigieren (man verschiebt
    nur den Bildausschnitt). Die effektive Latenz sinkt daher von der Render-Latenz auf die viel
    kleinere Warp-Latenz (Sensor -> Scanout, ~1-3 ms). Der Bildinhalt (`render_latency`) ist bei
    reiner Rotation egal — deshalb taucht er hier bewusst nicht in der Formel auf.
    (Bei TRANSLATION waere das anders: dann fehlt verdeckte Information -> Disokklusion.)
    """
    return displayed_pose(truth, warp_latency_ms, fs)


def angular_error(displayed: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Der vom Nutzer gespuerte Fehler je Zeitpunkt: |angezeigt - wahr| (in Grad)."""
    return np.abs(displayed - truth)
