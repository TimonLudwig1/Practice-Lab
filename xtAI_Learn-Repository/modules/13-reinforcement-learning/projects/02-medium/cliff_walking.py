"""Cliff-Walking-Umgebung (Sutton & Barto, Beispiel 6.6) — from scratch, ohne gym.

Ein 4x12-Gitter. Start unten links, Ziel unten rechts. Die untere Reihe *zwischen*
Start und Ziel ist eine "Klippe": betritt man sie, gibt es -100 und man wird zum Start
zurueckgesetzt (die Episode endet dabei NICHT). Jeder andere Schritt kostet -1. Die
Aufgabe ist episodisch und undiskontiert (gamma=1); optimal ist die kuerzeste Route
(13 Schritte, Ertrag -13), die direkt an der Klippenkante entlangfuehrt.

Zustaende sind als ganze Zahlen r*cols + c kodiert (0..47). Aktionen: 0=hoch, 1=rechts,
2=runter, 3=links.

Diese Datei ist INFRASTRUKTUR und vollstaendig vorgegeben — der Lernteil steckt in den
Agenten (td_control.py), nicht in der Umgebung.
"""
from __future__ import annotations
import numpy as np

# Aktionen als (dr, dc)
ACTIONS = {
    0: (-1, 0),   # hoch
    1: (0, +1),   # rechts
    2: (+1, 0),   # runter
    3: (0, -1),   # links
}
ACTION_NAMES = {0: "^", 1: ">", 2: "v", 3: "<"}


class CliffWalking:
    def __init__(self, rows: int = 4, cols: int = 12):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        # Klippe: untere Reihe, Spalten 1 .. cols-2
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self._state = self.start

    # ---- Kodierung ----
    def encode(self, rc: tuple[int, int]) -> int:
        r, c = rc
        return r * self.cols + c

    def decode(self, s: int) -> tuple[int, int]:
        return divmod(s, self.cols)

    # ---- Gym-artige API ----
    def reset(self) -> int:
        self._state = self.start
        return self.encode(self._state)

    def step(self, action: int):
        """Fuehre Aktion aus. Rueckgabe: (next_state, reward, done)."""
        r, c = self._state
        dr, dc = ACTIONS[action]
        # an den Raendern abprallen (bleibt in der Zelle)
        nr = min(max(r + dr, 0), self.rows - 1)
        nc = min(max(c + dc, 0), self.cols - 1)
        pos = (nr, nc)

        if pos in self.cliff:
            reward = -100.0
            self._state = self.start          # Absturz -> zurueck zum Start
            return self.encode(self._state), reward, False
        if pos == self.goal:
            self._state = pos
            return self.encode(pos), -1.0, True
        self._state = pos
        return self.encode(pos), -1.0, False

    # ---- Hilfen fuer Auswertung/Visualisierung ----
    def render_policy(self, greedy_actions: np.ndarray) -> str:
        """ASCII-Karte der greedy-Policy (S=Start, G=Ziel, C=Klippe)."""
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.start:
                    row.append("S")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.cliff:
                    row.append("C")
                else:
                    row.append(ACTION_NAMES[int(greedy_actions[self.encode((r, c))])])
            lines.append(" ".join(row))
        return "\n".join(lines)
