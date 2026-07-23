"""Sense-plan-act navigation — evaluation.  (Reference solution P03-final)

    /Users/.../.venv/bin/python run.py    Plots -> results/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navigation import (World, default_world, LANDMARKS, rrt, shortcut, densify,
                       path_length, unicycle_step, measure_ranges, ParticleFilter,
                       pure_pursuit)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

START = np.array([0.5, 0.5])
GOAL = np.array([9.0, 9.0])
MARGIN = 0.3            # robot radius / safety margin during planning
DT = 0.1
NOISE = (0.12, 0.15)    # (sigma_v, sigma_omega) of the motion model
SIGMA_R = 0.25          # measurement noise of the ranges


# ==========================================================================
# PLAN
# ==========================================================================
def plan(world, rng, goal_bias=0.08, step=0.5):
    raw, nodes, _ = rrt(START, GOAL, world, step=step, goal_bias=goal_bias,
                        rng=rng, margin=MARGIN)
    if raw is None:
        return None, None, nodes
    sm = shortcut(raw, world, rng=rng, margin=MARGIN)
    return raw, densify(sm, ds=0.1), nodes


def exp_planning():
    print("=" * 70)
    print("EXPERIMENT 1 — RRT planning: goal bias, step size, smoothing")
    print("=" * 70)
    world = default_world()
    print(f"  {'goal_bias':>9} | {'success':>7} | {'nodes':>7} | {'length raw':>10} | {'smoothed':>10}")
    print("  " + "-" * 54)
    for gb in [0.0, 0.02, 0.05, 0.10, 0.30]:
        succ, nodes_n, lr, ls = 0, [], [], []
        for s in range(20):
            rng = np.random.default_rng(1000 + s)
            raw, sm, nodes = plan(world, rng, goal_bias=gb)
            if raw is not None:
                succ += 1; nodes_n.append(len(nodes))
                lr.append(path_length(raw)); ls.append(path_length(sm))
        print(f"  {gb:9.2f} | {succ/20:7.2f} | {np.mean(nodes_n):7.0f} | "
              f"{np.mean(lr):10.2f} | {np.mean(ls):10.2f}")
    print("  => In this (fairly open) world RRT always finds a solution. The goal bias")
    print("     lowers the search effort noticeably (about 200 -> 120-140 nodes) but barely")
    print("     changes the path length. The big gain comes from SMOOTHING: ~16.3 -> ~13.6 (~17% shorter).")
    print("     This illustrates: RRT is probabilistically complete, but NOT optimal —")
    print("     the raw paths are jagged and too long, so post-processing is mandatory.")

    # image of the planning
    rng = np.random.default_rng(0)
    raw, sm, nodes = plan(world, rng)
    fig, ax = plt.subplots(figsize=(7, 7))
    for o in world.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], color="lightgray"))
        ax.add_patch(plt.Circle(o[:2], o[2] + MARGIN, color="gray", fill=False, ls=":"))
    ax.plot(nodes[:, 0], nodes[:, 1], ".", ms=1.5, color="silver", label=f"RRT tree ({len(nodes)})")
    ax.plot(raw[:, 0], raw[:, 1], "-", color="steelblue", lw=1.2, label=f"raw ({path_length(raw):.1f})")
    ax.plot(sm[:, 0], sm[:, 1], "-", color="crimson", lw=2, label=f"smoothed ({path_length(sm):.1f})")
    ax.plot(*START, "go", ms=10, label="start"); ax.plot(*GOAL, "r*", ms=16, label="goal")
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "kv", ms=8, label="landmarks")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("RRT planning with a safety margin")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "planning.png"), dpi=110); plt.close(fig)
    return sm


# ==========================================================================
# SENSE — localisation
# ==========================================================================
def simulate(path, M=500, seed=0, use_filter=True, max_steps=1200):
    """Drives the path. Control acts on the ESTIMATED pose.
    use_filter=True -> particle filter; False -> pure odometry (dead reckoning).
    Returns a dict with the error histories and the result."""
    rng = np.random.default_rng(seed)
    true = np.array([START[0], START[1], 0.0])
    odo = true.copy()
    pf = ParticleFilter(M, true, rng=rng) if use_filter else None
    world = default_world()
    err_est, err_odo, traj = [], [], [true[:2].copy()]
    collided = False
    for k in range(max_steps):
        est = pf.estimate() if use_filter else odo
        v, om = pure_pursuit(path, est)
        # true motion (noisy) + odometry (knows only the commanded values)
        true = unicycle_step(true, v, om, DT, rng=rng, noise=NOISE)
        odo = unicycle_step(odo, v, om, DT)
        if use_filter:
            pf.predict(v, om, DT, NOISE)
            pf.update(measure_ranges(true, sigma_r=SIGMA_R, rng=rng), sigma_r=SIGMA_R)
            pf.resample_if_needed()
        traj.append(true[:2].copy())
        err_est.append(np.linalg.norm((pf.estimate() if use_filter else odo)[:2] - true[:2]))
        err_odo.append(np.linalg.norm(odo[:2] - true[:2]))
        if world.in_collision(true[:2]):
            collided = True
        if np.linalg.norm(true[:2] - GOAL) < 0.4:
            break
    return dict(steps=k + 1, reached=np.linalg.norm(true[:2] - GOAL) < 0.4,
                final_dist=float(np.linalg.norm(true[:2] - GOAL)), collided=collided,
                err_est=np.array(err_est), err_odo=np.array(err_odo),
                traj=np.array(traj))


def exp_localization(path):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — localisation: particle filter vs. pure odometry")
    print("=" * 70)
    r = simulate(path, M=500, seed=0, use_filter=True)
    print(f"  one run over {r['steps']} steps:")
    print(f"    particle filter error : mean {r['err_est'].mean():.3f} m, final {r['err_est'][-1]:.3f} m")
    print(f"    odometry error        : mean {r['err_odo'].mean():.3f} m, final {r['err_odo'][-1]:.3f} m")
    print("  => The odometry error grows WITHOUT BOUND (a random walk, above all the orientation")
    print("     error); the particle filter stays bounded thanks to the landmark measurements.")

    print("\n  particle count M (5 runs each, mean localisation error):")
    print(f"  {'M':>6} | {'mean error':>13} | {'max error':>11} | {'goal reached':>13}")
    print("  " + "-" * 50)
    Ms, means = [], []
    for M in [10, 50, 200, 1000]:
        e_mean, e_max, reach = [], [], 0
        for s in range(5):
            rr = simulate(path, M=M, seed=100 + s, use_filter=True)
            e_mean.append(rr["err_est"].mean()); e_max.append(rr["err_est"].max())
            reach += rr["reached"]
        Ms.append(M); means.append(np.mean(e_mean))
        print(f"  {M:6d} | {np.mean(e_mean):13.3f} | {np.mean(e_max):11.3f} | {reach:>10d}/5")
    print("  => Too few particles -> the filter cannot cover the state space and drifts;")
    print("     from a few hundred particles the gain saturates (more just costs compute).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(r["err_odo"], label="odometry (dead reckoning)", color="crimson")
    ax1.plot(r["err_est"], label="particle filter", color="steelblue")
    ax1.set_xlabel("time step"); ax1.set_ylabel("localisation error [m]")
    ax1.set_title("odometry drifts, the filter stays bounded"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.semilogx(Ms, means, "o-", color="purple")
    ax2.set_xlabel("particle count M (log)"); ax2.set_ylabel("mean error [m]")
    ax2.set_title("accuracy vs. particle count"); ax2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "localization.png"), dpi=110); plt.close(fig)


# ==========================================================================
# ACT — closed loop
# ==========================================================================
def exp_closed_loop(path):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — closed loop: control on the estimated vs. the odometric pose")
    print("=" * 70)
    print(f"  {'control on':>22} | {'goal reached':>13} | {'final dist':>11} | {'collisions':>11}")
    print("  " + "-" * 64)
    summary = {}
    for label, use_filter in [("particle filter", True), ("odometry only", False)]:
        reach, dists, colls = 0, [], 0
        for s in range(10):
            rr = simulate(path, M=500, seed=200 + s, use_filter=use_filter)
            reach += rr["reached"]; dists.append(rr["final_dist"]); colls += rr["collided"]
        summary[label] = (reach, np.mean(dists), colls)
        print(f"  {label:>22} | {reach:>10d}/10 | {np.mean(dists):11.3f} | {colls:>8d}/10")
    print("  => With the filter the robot reaches the goal reliably and collision-free.")
    print("     On pure odometry it believes it is elsewhere -> it leaves the path,")
    print("     grazes obstacles and misses the goal. Sense-plan-act needs the SENSE.")

    fig, ax = plt.subplots(figsize=(7, 7))
    world = default_world()
    for o in world.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], color="lightgray"))
    ax.plot(path[:, 0], path[:, 1], "--", color="gray", lw=1.5, label="planned path")
    for label, use_filter, col in [("with particle filter", True, "steelblue"),
                                   ("odometry only", False, "crimson")]:
        rr = simulate(path, M=500, seed=200, use_filter=use_filter)
        ax.plot(rr["traj"][:, 0], rr["traj"][:, 1], "-", color=col, lw=2, label=label)
    ax.plot(*START, "go", ms=10); ax.plot(*GOAL, "r*", ms=16)
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "kv", ms=8, label="landmarks")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("driven trajectory: with localisation vs. odometry only")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "navigation.png"), dpi=110); plt.close(fig)
    return summary


if __name__ == "__main__":
    path = exp_planning()
    exp_localization(path)
    exp_closed_loop(path)
    print("\nPlots in results/. Done.")
