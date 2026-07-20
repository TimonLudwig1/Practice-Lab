"""Sense-Plan-Act-Navigation — Auswertung.  (Referenzloesung P03-final)

    /Users/.../.venv/bin/python run.py    Plots -> ergebnisse/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navigation import (World, default_world, LANDMARKS, rrt, shortcut, densify,
                       path_length, unicycle_step, measure_ranges, ParticleFilter,
                       pure_pursuit)

OUT = os.path.join(os.path.dirname(__file__), "ergebnisse")
os.makedirs(OUT, exist_ok=True)

START = np.array([0.5, 0.5])
GOAL = np.array([9.0, 9.0])
MARGIN = 0.3            # Roboterradius / Sicherheitsabstand beim Planen
DT = 0.1
NOISE = (0.12, 0.15)    # (sigma_v, sigma_omega) des Bewegungsmodells
SIGMA_R = 0.25          # Messrauschen der Entfernungen


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
    print("EXPERIMENT 1 — RRT-Planung: Goal-Bias, Schrittweite, Glaettung")
    print("=" * 70)
    world = default_world()
    print(f"  {'goal_bias':>9} | {'Erfolg':>7} | {'Knoten':>7} | {'Laenge roh':>10} | {'geglaettet':>10}")
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
    print("  => In dieser (relativ offenen) Welt findet RRT immer eine Loesung. Der Goal-Bias")
    print("     senkt den Suchaufwand spuerbar (ca. 200 -> 120-140 Knoten), aendert die Pfadlaenge")
    print("     aber kaum. Der grosse Gewinn kommt vom GLAETTEN: ~16.3 -> ~13.6 (rund 17 % kuerzer).")
    print("     Das illustriert: RRT ist probabilistisch vollstaendig, aber NICHT optimal —")
    print("     die rohen Pfade sind zackig und zu lang, deshalb ist Nachbearbeiten Pflicht.")

    # Bild der Planung
    rng = np.random.default_rng(0)
    raw, sm, nodes = plan(world, rng)
    fig, ax = plt.subplots(figsize=(7, 7))
    for o in world.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], color="lightgray"))
        ax.add_patch(plt.Circle(o[:2], o[2] + MARGIN, color="gray", fill=False, ls=":"))
    ax.plot(nodes[:, 0], nodes[:, 1], ".", ms=1.5, color="silver", label=f"RRT-Baum ({len(nodes)})")
    ax.plot(raw[:, 0], raw[:, 1], "-", color="steelblue", lw=1.2, label=f"roh ({path_length(raw):.1f})")
    ax.plot(sm[:, 0], sm[:, 1], "-", color="crimson", lw=2, label=f"geglaettet ({path_length(sm):.1f})")
    ax.plot(*START, "go", ms=10, label="Start"); ax.plot(*GOAL, "r*", ms=16, label="Ziel")
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "kv", ms=8, label="Landmarken")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("RRT-Planung mit Sicherheitsabstand")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "planung.png"), dpi=110); plt.close(fig)
    return sm


# ==========================================================================
# SENSE — Lokalisierung
# ==========================================================================
def simulate(path, M=500, seed=0, use_filter=True, max_steps=1200):
    """Faehrt den Pfad ab. Geregelt wird auf der GESCHAETZTEN Pose.
    use_filter=True -> Partikelfilter; False -> reine Odometrie (dead reckoning).
    Gibt dict mit Fehlerverlaeufen und Ergebnis zurueck."""
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
        # wahre Bewegung (verrauscht) + Odometrie (kennt nur die Sollbefehle)
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
    print("EXPERIMENT 2 — Lokalisierung: Partikelfilter vs. reine Odometrie")
    print("=" * 70)
    r = simulate(path, M=500, seed=0, use_filter=True)
    print(f"  Ein Lauf ueber {r['steps']} Schritte:")
    print(f"    Partikelfilter-Fehler : Mittel {r['err_est'].mean():.3f} m, Ende {r['err_est'][-1]:.3f} m")
    print(f"    Odometrie-Fehler      : Mittel {r['err_odo'].mean():.3f} m, Ende {r['err_odo'][-1]:.3f} m")
    print("  => Der Odometriefehler waechst UNBESCHRAENKT (Random Walk, v.a. der Orientierungs-")
    print("     fehler); der Partikelfilter bleibt durch die Landmarkenmessungen beschraenkt.")

    print("\n  Partikelzahl M (je 5 Laeufe, mittlerer Lokalisierungsfehler):")
    print(f"  {'M':>6} | {'Fehler Mittel':>13} | {'Fehler max':>11} | {'Ziel erreicht':>13}")
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
    print("  => Zu wenige Partikel -> der Filter kann den Zustandsraum nicht abdecken und driftet;")
    print("     ab einigen Hundert Partikeln saettigt der Gewinn (mehr kostet nur Rechenzeit).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(r["err_odo"], label="Odometrie (dead reckoning)", color="crimson")
    ax1.plot(r["err_est"], label="Partikelfilter", color="steelblue")
    ax1.set_xlabel("Zeitschritt"); ax1.set_ylabel("Lokalisierungsfehler [m]")
    ax1.set_title("Odometrie driftet, Filter bleibt beschraenkt"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.semilogx(Ms, means, "o-", color="purple")
    ax2.set_xlabel("Partikelzahl M (log)"); ax2.set_ylabel("mittlerer Fehler [m]")
    ax2.set_title("Genauigkeit vs. Partikelzahl"); ax2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lokalisierung.png"), dpi=110); plt.close(fig)


# ==========================================================================
# ACT — geschlossener Kreis
# ==========================================================================
def exp_closed_loop(path):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — Geschlossener Kreis: Regelung auf geschaetzter vs. odometrischer Pose")
    print("=" * 70)
    print(f"  {'Regelung auf':>22} | {'Ziel erreicht':>13} | {'Endabstand':>11} | {'Kollisionen':>11}")
    print("  " + "-" * 64)
    summary = {}
    for label, use_filter in [("Partikelfilter", True), ("nur Odometrie", False)]:
        reach, dists, colls = 0, [], 0
        for s in range(10):
            rr = simulate(path, M=500, seed=200 + s, use_filter=use_filter)
            reach += rr["reached"]; dists.append(rr["final_dist"]); colls += rr["collided"]
        summary[label] = (reach, np.mean(dists), colls)
        print(f"  {label:>22} | {reach:>10d}/10 | {np.mean(dists):11.3f} | {colls:>8d}/10")
    print("  => Mit Filter erreicht der Roboter das Ziel zuverlaessig und kollisionsfrei.")
    print("     Auf reiner Odometrie glaubt er, woanders zu sein -> er verlaesst den Pfad,")
    print("     streift Hindernisse und verfehlt das Ziel. Sense-Plan-Act braucht das SENSE.")

    fig, ax = plt.subplots(figsize=(7, 7))
    world = default_world()
    for o in world.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], color="lightgray"))
    ax.plot(path[:, 0], path[:, 1], "--", color="gray", lw=1.5, label="geplanter Pfad")
    for label, use_filter, col in [("mit Partikelfilter", True, "steelblue"),
                                   ("nur Odometrie", False, "crimson")]:
        rr = simulate(path, M=500, seed=200, use_filter=use_filter)
        ax.plot(rr["traj"][:, 0], rr["traj"][:, 1], "-", color=col, lw=2, label=label)
    ax.plot(*START, "go", ms=10); ax.plot(*GOAL, "r*", ms=16)
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "kv", ms=8, label="Landmarken")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("Gefahrene Bahn: mit Lokalisierung vs. nur Odometrie")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "navigation.png"), dpi=110); plt.close(fig)
    return summary


if __name__ == "__main__":
    path = exp_planning()
    exp_localization(path)
    exp_closed_loop(path)
    print("\nPlots in ergebnisse/. Fertig.")
