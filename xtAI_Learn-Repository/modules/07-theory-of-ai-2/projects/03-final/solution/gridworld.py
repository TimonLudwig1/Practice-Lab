"""The classical 4x3 gridworld (Russell & Norvig) as an MDP.

Coordinates (x, y) with (0,0) at the bottom left. The layout:

    y=2 |  .    .    .   +1
    y=1 |  .  WALL   .   -1
    y=0 |  .    .    .    .
          x=0  x=1  x=2  x=3

Movement is noisy: with probability 0.8 in the intended direction, with 0.1 each
to the left/right of it (perpendicular). Against a wall or the border -> stay
put. Every non-terminal step costs R = -0.04 (the living reward).
"""

ACTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
ARROW = {"N": "↑", "S": "↓", "E": "→", "W": "←", None: "·"}


class GridworldMDP:
    def __init__(self, living_reward=-0.04, gamma=1.0):
        self.cols, self.rows = 4, 3
        self.walls = {(1, 1)}
        self.terminals = {(3, 2): +1.0, (3, 1): -1.0}
        self.living_reward = living_reward
        self.gamma = gamma
        self.states = [(x, y) for x in range(self.cols) for y in range(self.rows)
                       if (x, y) not in self.walls]

    def reward(self, s):
        return self.terminals.get(s, self.living_reward)

    def is_terminal(self, s):
        return s in self.terminals

    def actions(self, s):
        return [] if self.is_terminal(s) else list(ACTIONS.keys())

    def _move(self, s, action):
        dx, dy = ACTIONS[action]
        nx, ny = s[0] + dx, s[1] + dy
        if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in self.walls:
            return (nx, ny)
        return s                      # against a wall or the border -> stay

    def transitions(self, s, action):
        """A list of (probability, successor state) for the noisy model."""
        if self.is_terminal(s):
            return [(1.0, s)]
        # the perpendicular wrong directions
        if action in ("N", "S"):
            perp = ("E", "W")
        else:
            perp = ("N", "S")
        outcomes = {}
        for prob, a in [(0.8, action), (0.1, perp[0]), (0.1, perp[1])]:
            s2 = self._move(s, a)
            outcomes[s2] = outcomes.get(s2, 0.0) + prob
        return [(prob, s2) for s2, prob in outcomes.items()]


def show_policy(mdp, policy):
    """An ASCII rendering of a policy (arrows)."""
    lines = []
    for y in range(mdp.rows - 1, -1, -1):
        row = []
        for x in range(mdp.cols):
            s = (x, y)
            if s in mdp.walls:
                row.append(" ## ")
            elif s in mdp.terminals:
                row.append(f" {'+1' if mdp.terminals[s] > 0 else '-1'} ")
            else:
                row.append(f"  {ARROW[policy.get(s)]} ")
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def show_values(mdp, V):
    lines = []
    for y in range(mdp.rows - 1, -1, -1):
        row = []
        for x in range(mdp.cols):
            s = (x, y)
            row.append("  ##  " if s in mdp.walls else f"{V[s]:+.3f}")
        lines.append(" ".join(row))
    return "\n".join(lines)
