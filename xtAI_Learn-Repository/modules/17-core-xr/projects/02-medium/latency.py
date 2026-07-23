"""The motion-to-photon chain and its two countermeasures: prediction and timewarp.

>>> YOUR TASK <<<  Fill in the four functions marked with TODO. `LatencyBudget` and
`ms_to_steps` are given.

The central quantity (script 3): the time from "the head moves" to "the matching photon hits the
retina". Too large (> ~20 ms) => the image lags behind the movement => cybersickness.

What happens here: at every point in time t the TRUE head orientation is `truth[t]`. But the
headset displays the orientation that was measured `latency` milliseconds ago. The DISPLAYED
error = |displayed orientation - true orientation NOW|. That is exactly the angle the user feels
as "the world is lagging".

Everything works on arrays: `truth[i]` is the true yaw orientation at sample i. A latency of
L milliseconds corresponds to a shift of L*fs/1000 samples.

Reference solution: solution/latency.py — try it yourself first!
"""
from __future__ import annotations
import numpy as np


def ms_to_steps(ms: float, fs: int) -> int:
    """Convert milliseconds into sample steps (at the sampling rate fs)."""
    return int(round(ms / 1000.0 * fs))


class LatencyBudget:
    """The motion-to-photon chain as the sum of its links (script 3.1).

    Percentages obscure, milliseconds do not: here you see that at 90 Hz a single frame alone
    (11.1 ms) eats almost the entire 20 ms budget.
    """
    def __init__(self, **stages_ms):
        # e.g. sensor=1, fusion=1, app=4, render=11, scanout=3, display=2
        self.stages = stages_ms

    @property
    def total_ms(self) -> float:
        return float(sum(self.stages.values()))

    def report(self) -> str:
        lines = [f"  {name:12s} {ms:5.1f} ms" for name, ms in self.stages.items()]
        lines.append(f"  {'= total':12s} {self.total_ms:5.1f} ms")
        verdict = "OK (< 20 ms)" if self.total_ms < 20 else "TOO HIGH (>= 20 ms)"
        lines.append(f"  Motion-to-photon: {verdict}")
        return "\n".join(lines)


def displayed_pose(truth: np.ndarray, latency_ms: float, fs: int) -> np.ndarray:
    """What the headset displays WITHOUT countermeasures: the pose from `latency_ms` ago.

    Blueprint:
      - lat = ms_to_steps(latency_ms, fs)
      - every output value i should be truth[i - lat]
      - at the beginning (i - lat < 0) hold the oldest pose -> np.clip(indices, 0, len-1)
    Returns: an array of the same length as truth.
    """
    # TODO
    raise NotImplementedError


def predicted_pose(truth: np.ndarray, velocity: np.ndarray, latency_ms: float,
                   horizon_ms: float, fs: int) -> np.ndarray:
    """Prediction (script 3.2): instead of the old pose, display the one EXTRAPOLATED by
    `horizon_ms`.

        displayed[i] = truth[i-lat] + velocity[i-lat] * (horizon_ms / 1000)

    velocity is in degrees/second, horizon_ms in milliseconds -> convert to seconds.
    The ideal is horizon = latency (you render exactly for the moment of display). Too large a
    horizon => overshoot, above all at changes of direction.
    """
    # TODO
    raise NotImplementedError


def timewarped_pose(truth: np.ndarray, render_latency_ms: float,
                    warp_latency_ms: float, fs: int) -> np.ndarray:
    """Orientational timewarp (script 3.2): the finished rendered image is shifted shortly before
    display using the LATEST pose.

    For a pure ROTATION the finished image can be corrected exactly afterwards (you only shift the
    image section). The effective latency therefore drops from the render latency to the much
    smaller warp latency. Concretely that means here: the displayed pose is simply
    displayed_pose(truth, warp_latency_ms, fs) - the image content (render_latency) is irrelevant
    for a pure rotation and deliberately does NOT appear in the formula.
    (With TRANSLATION it would be different: then occluded information is missing -> disocclusion.)
    """
    # TODO
    raise NotImplementedError


def angular_error(displayed: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The error the user feels at each point in time: |displayed - true| (in degrees)."""
    # TODO
    raise NotImplementedError
