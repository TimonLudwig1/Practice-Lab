"""Cliff-walking environment (Sutton & Barto, example 6.6) — from scratch, without gym.

A 4x12 grid. Start at the bottom left, goal at the bottom right. The bottom row *between*
start and goal is a "cliff": if you step onto it, you get -100 and are reset to the start
(the episode does NOT end there). Every other step costs -1. The task is episodic and
undiscounted (gamma=1); optimal is the shortest route (13 steps, return -13) that runs right
along the cliff edge.

States are encoded as integers r*cols + c (0..47). Actions: 0=up, 1=right, 2=down, 3=left.

This file is INFRASTRUCTURE and fully given — the learning part is in the agents
(td_control.py), not in the environment.
"""
from __future__ import annotations
import numpy as np

# actions as (dr, dc)
ACTIONS = {
    0: (-1, 0),   # up
    1: (0, +1),   # right
    2: (+1, 0),   # down
    3: (0, -1),   # left
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
        # cliff: bottom row, columns 1 .. cols-2
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self._state = self.start

    # ---- encoding ----
    def encode(self, rc: tuple[int, int]) -> int:
        r, c = rc
        return r * self.cols + c

    def decode(self, s: int) -> tuple[int, int]:
        return divmod(s, self.cols)

    # ---- gym-like API ----
    def reset(self) -> int:
        self._state = self.start
        return self.encode(self._state)

    def step(self, action: int):
        """Execute an action. Returns: (next_state, reward, done)."""
        r, c = self._state
        dr, dc = ACTIONS[action]
        # bounce off at the borders (stays in the cell)
        nr = min(max(r + dr, 0), self.rows - 1)
        nc = min(max(c + dc, 0), self.cols - 1)
        pos = (nr, nc)

        if pos in self.cliff:
            reward = -100.0
            self._state = self.start          # fall -> back to the start
            return self.encode(self._state), reward, False
        if pos == self.goal:
            self._state = pos
            return self.encode(pos), -1.0, True
        self._state = pos
        return self.encode(pos), -1.0, False

    # ---- helpers for evaluation/visualization ----
    def render_policy(self, greedy_actions: np.ndarray) -> str:
        """ASCII map of the greedy policy (S=start, G=goal, C=cliff)."""
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
