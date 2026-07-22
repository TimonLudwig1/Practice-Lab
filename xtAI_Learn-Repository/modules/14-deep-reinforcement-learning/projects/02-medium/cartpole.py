"""CartPole environment by hand (identical to project 01) — infrastructure, fully given.

State s = [x, x_dot, theta, theta_dot], actions 0=push left / 1=push right, reward +1 per step,
end at |x|>2.4, |theta|>12 degrees or 500 steps.
"""
from __future__ import annotations
import math
import numpy as np


class CartPole:
    def __init__(self, seed=None):
        self.g = 9.8; self.mc = 1.0; self.mp = 0.1; self.l = 0.5
        self.fmag = 10.0; self.tau = 0.02
        self.mt = self.mc + self.mp; self.pml = self.mp * self.l
        self.x_thr = 2.4; self.th_thr = 12 * math.pi / 180; self.max_steps = 500
        self.n_states = 4; self.n_actions = 2
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.s.copy()

    def step(self, a):
        x, xd, th, thd = self.s
        f = self.fmag if a == 1 else -self.fmag
        ct = math.cos(th); st = math.sin(th)
        temp = (f + self.pml * thd * thd * st) / self.mt
        thacc = (self.g * st - ct * temp) / (self.l * (4/3 - self.mp * ct * ct / self.mt))
        xacc = temp - self.pml * thacc * ct / self.mt
        x += self.tau * xd; xd += self.tau * xacc
        th += self.tau * thd; thd += self.tau * thacc
        self.s = np.array([x, xd, th, thd]); self.steps += 1
        done = bool(abs(x) > self.x_thr or abs(th) > self.th_thr or self.steps >= self.max_steps)
        return self.s.copy(), 1.0, done
