"""The experiment: motion-to-photon latency and its countermeasures.

  1. Break down the latency budget (why 90 Hz is tight).
  2. How strongly does the world lag? (the error in degrees over the latency)
  3. Prediction: the sweet spot (horizon = latency) and the overshoot.
  4. Timewarp: why it makes the latency "invisible".

Call:  python run.py        (~2 s)
"""
from __future__ import annotations
import numpy as np

from head_motion import generate_head_yaw
from latency import (LatencyBudget, displayed_pose, predicted_pose, timewarped_pose,
                     angular_error)

FS = 1000


def main():
    t, truth, velocity = generate_head_yaw(duration_s=20.0, fs=FS, seed=0)
    print("=== Head motion (simulated) ===")
    print(f"Duration {t[-1]:.0f}s @ {FS} Hz | max angular velocity "
          f"{np.abs(velocity).max():.0f} deg/s (up to ~500 in reality)")

    # ---------- 1) the latency budget ----------
    print("\n=== 1) The motion-to-photon budget (script 3.1) ===")
    budget = LatencyBudget(sensor=1.5, fusion=1.0, app=4.0, render=11.1, scanout=3.0, display=2.0)
    print(budget.report())
    print("-> A single frame at 90 Hz = 11.1 ms alone eats half the 20 ms budget.")

    # ---------- 2) pure latency ----------
    print("\n=== 2) How strongly does the world lag? (without countermeasures) ===")
    print(f"{'latency':>8s} {'mean error':>13s} {'max error':>11s}")
    for L in (5, 11, 20, 50):
        e = angular_error(displayed_pose(truth, L, FS), truth)
        print(f"{L:6d}ms {e.mean():11.2f}deg {e.max():10.2f}deg")
    print("-> At 50 ms the image deviates from the truth by 5 degrees on average - clearly felt.")

    # ---------- 3) prediction ----------
    print("\n=== 3) Prediction: extrapolating instead of lagging behind ===")
    L = 20
    e_without = angular_error(displayed_pose(truth, L, FS), truth).mean()
    print(f"At {L} ms of latency, the horizon varied (the ideal = the latency):")
    print(f"{'horizon':>9s} {'mean error':>13s} {'max error':>11s} {'vs without':>11s}")
    for H in (0, 10, 20, 30, 40):
        e = angular_error(predicted_pose(truth, velocity, L, H, FS), truth)
        print(f"{H:7d}ms {e.mean():11.3f}deg {e.max():10.2f}deg "
              f"{100*(1-e.mean()/e_without):+9.0f}%")
    print("-> A U shape: horizon = latency is optimal (~-86%). TOO much prediction (40 ms) is as")
    print("   bad as none at all - you extrapolate past the movement.")

    # the overshoot at changes of direction
    accel = np.abs(np.gradient(velocity, 1 / FS))
    turns = accel > np.percentile(accel, 90)
    e_pred = angular_error(predicted_pose(truth, velocity, L, L, FS), truth)
    print(f"\nOvershoot: the prediction error at changes of direction is "
          f"{e_pred[turns].mean():.3f} deg vs. {e_pred[~turns].mean():.3f} deg during smooth "
          f"movement")
    print("-> That is exactly where prediction is off: it keeps extrapolating in the OLD direction.")

    # ---------- 4) timewarp ----------
    print("\n=== 4) Timewarp: shifting the finished image ===")
    e_render = angular_error(displayed_pose(truth, 20, FS), truth).mean()
    e_warp = angular_error(timewarped_pose(truth, render_latency_ms=20, warp_latency_ms=2, fs=FS),
                           truth).mean()
    print(f"  render latency 20 ms, without timewarp: {e_render:.2f} deg")
    print(f"  with timewarp (the warp fetches the pose at 2 ms): {e_warp:.2f} deg  "
          f"({100*(1-e_warp/e_render):.0f}% less)")
    print("-> Timewarp does not make the latency SMALLER, but makes it invisible for the")
    print("   ORIENTATION - and that is exactly what the vestibular system reacts to most")
    print("   sensitively. (With TRANSLATION this would not work: then occluded information")
    print("   is missing -> disocclusion.)")

    # ---------- plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))

        s = slice(2000, 4000)   # a 2 s excerpt
        ax1.plot(t[s], truth[s], "k", lw=2, label="true (now)")
        ax1.plot(t[s], displayed_pose(truth, 50, FS)[s], lw=1.3, label="50 ms latency (lags)")
        ax1.plot(t[s], predicted_pose(truth, velocity, 50, 50, FS)[s], lw=1.3,
                 label="+ prediction")
        ax1.set(xlabel="time [s]", ylabel="yaw [deg]", title="Latency makes the world lag")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        hs = np.arange(0, 45, 2)
        errs = [angular_error(predicted_pose(truth, velocity, 20, h, FS), truth).mean() for h in hs]
        ax2.plot(hs, errs, "o-", lw=2)
        ax2.axvline(20, ls="--", color="crimson", label="horizon = latency")
        ax2.set(xlabel="prediction horizon [ms]", ylabel="mean error [deg]",
                title="Prediction: a U shape around the sweet spot")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        Ls = np.arange(0, 55, 5)
        ax3.plot(Ls, [angular_error(displayed_pose(truth, L, FS), truth).mean() for L in Ls],
                 "o-", lw=2, label="without countermeasures")
        ax3.plot(Ls, [angular_error(predicted_pose(truth, velocity, L, L, FS), truth).mean()
                      for L in Ls], "s-", lw=2, label="with prediction")
        ax3.plot(Ls, [angular_error(timewarped_pose(truth, L, 2, FS), truth).mean()
                      for L in Ls], "^-", lw=2, label="with timewarp")
        ax3.set(xlabel="render latency [ms]", ylabel="mean error [deg]",
                title="The countermeasures compared")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

        plt.tight_layout(); plt.savefig("results/motion_to_photon.png", dpi=110)
        print("\nPlot saved: results/motion_to_photon.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
