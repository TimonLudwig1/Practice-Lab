"""Experimente zu Zeige-Praezision & Reichweite.  (Referenzloesung P02-medium)

    /Users/.../.venv/bin/python run.py
Plots -> results/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selection import (angular_radius, p_hit_raycasting, go_go, go_go_gain,
                      go_go_inverse, p_hit_gogo, angular_id,
                      simulate_fitts_times, fit_fitts)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def exp_raycasting_vs_distance():
    print("=" * 66)
    print("EXPERIMENT 1 — Ray-Casting-Praezision faellt mit der Distanz")
    print("=" * 66)
    r = 0.10                       # Zielradius 10 cm
    sigma_theta = np.deg2rad(1.0)  # 1 Grad Winkelrauschen (Hand-Tremor)
    Ls = np.array([1, 2, 4, 8, 16, 32.0])
    print(f"  Zielradius r={r} m, Winkelrauschen sigma={np.rad2deg(sigma_theta):.1f} deg")
    print(f"  {'L [m]':>6} | {'theta_W [deg]':>13} | {'P(hit)':>7}")
    print("  " + "-" * 34)
    for L in Ls:
        tW = 2 * np.rad2deg(angular_radius(r, L))
        print(f"  {L:6.0f} | {tW:13.2f} | {p_hit_raycasting(r, L, sigma_theta):7.3f}")
    print("  => Doppelte Distanz ~ halbe angulare Groesse -> Trefferrate bricht ein.")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    Lg = np.linspace(0.5, 32, 200)
    for rr in [0.05, 0.10, 0.20]:
        ax.plot(Lg, [p_hit_raycasting(rr, L, sigma_theta) for L in Lg], label=f"r={rr} m")
    ax.set_xlabel("Zieldistanz L [m]"); ax.set_ylabel("P(Treffer) Ray-Casting")
    ax.set_title("Ray-Casting: Praezision faellt mit Distanz"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "raycasting_distanz.png"), dpi=110); plt.close(fig)


def exp_gogo_reach_and_precision():
    print("\n" + "=" * 66)
    print("EXPERIMENT 2 — Go-Go: Reichweite gewonnen, Praezision im Streckbereich verloren")
    print("=" * 66)
    D, k, arm = 0.45, 60.0, 0.7     # Schwelle, Verstaerkung, Armlaenge
    sigma_r = 0.005                 # 5 mm Hand-Positionsrauschen
    # Reichweite: welche virtuelle Distanz erreicht der voll gestreckte Arm?
    max_reach = float(go_go(arm, D, k))
    print(f"  Armlaenge {arm} m; Virtual-Hand-Reichweite = {arm} m")
    print(f"  Go-Go (D={D}, k={k}): max. Reichweite = {max_reach:.2f} m  "
          f"(x{max_reach/arm:.1f} des Arms)")
    r = 0.08
    print(f"\n  Ziel-Radius r={r} m. Trefferrate Virtual-Hand vs Go-Go:")
    print(f"  {'L [m]':>6} | {'r_r [m]':>7} | {'gain g':>6} | {'VHand':>6} | {'Go-Go':>6}")
    print("  " + "-" * 44)
    for L in [0.3, 0.45, 0.6, 1.0, 2.0, 3.5]:
        r_r = float(go_go_inverse(L, D, k))
        g = float(go_go_gain(r_r, D, k)) if r_r <= arm else np.nan
        p_vh = p_hit_gogo(r, L, D, k=0.0, sigma_r=sigma_r, arm_length=arm)  # k=0 -> reine Virtual Hand
        p_gg = p_hit_gogo(r, L, D, k, sigma_r, arm)
        gtxt = f"{g:6.2f}" if not np.isnan(g) else "  n/a "
        print(f"  {L:6.2f} | {r_r:7.3f} | {gtxt} | {p_vh:6.3f} | {p_gg:6.3f}")
    print("  => Virtual Hand kann nichts jenseits der Armlaenge (0.7 m) treffen (P=0).")
    print("     Go-Go erreicht ferne Ziele, aber die C/D-Verstaerkung g>1 verstaerkt das")
    print("     Rauschen -> Praezision sinkt mit wachsender Streckung.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    rr = np.linspace(0, arm, 200)
    ax1.plot(rr, go_go(rr, D, k), label="Go-Go r_v")
    ax1.plot(rr, rr, "--", color="gray", label="Virtual Hand (1:1)")
    ax1.axvline(D, ls=":", color="red", label=f"Schwelle D={D}")
    ax1.set_xlabel("reale Handdistanz r_r [m]"); ax1.set_ylabel("virtuelle Reichweite r_v [m]")
    ax1.set_title("Go-Go verlaengert nichtlinear"); ax1.legend(); ax1.grid(alpha=0.3)
    Lg = np.linspace(0.2, max_reach * 0.98, 200)
    ax2.plot(Lg, [p_hit_gogo(r, L, D, k, sigma_r, arm) for L in Lg], label="Go-Go")
    ax2.plot(Lg, [p_hit_gogo(r, L, D, 0.0, sigma_r, arm) for L in Lg], "--", label="Virtual Hand")
    ax2.axvline(arm, ls=":", color="gray", label=f"Armlaenge {arm}")
    ax2.set_xlabel("Zieldistanz L [m]"); ax2.set_ylabel("P(Treffer)")
    ax2.set_title(f"Reichweite vs Praezision (r={r} m)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "gogo.png"), dpi=110); plt.close(fig)
    return dict(max_reach=max_reach, arm=arm)


def exp_angular_fitts():
    print("\n" + "=" * 66)
    print("EXPERIMENT 3 — Angulares Fitts' Law: ID waechst mit der Distanz")
    print("=" * 66)
    rng = np.random.default_rng(0)
    a_true, b_true, motor_sigma = 0.15, 0.20, 0.05   # MT = a + b*ID (s)
    # Reciprocal-Tapping-Bedingungen: variiere angulare Amplitude & Breite
    r = 0.10
    Ls = np.array([1, 2, 4, 8, 16.0])
    amp_deg = 30.0                                   # angulare Bewegungsamplitude (fest)
    theta_D = np.full(len(Ls), np.deg2rad(amp_deg))
    theta_W = np.array([2 * angular_radius(r, L) for L in Ls])  # schrumpft mit L
    ids, mts = simulate_fitts_times(theta_D, theta_W, a_true, b_true, motor_sigma, 40, rng)
    a_fit, b_fit, r2 = fit_fitts(ids, mts)
    print(f"  wahre Fitts-Parameter : a={a_true:.3f}s  b={b_true:.3f}s/bit")
    print(f"  gefittete Parameter   : a={a_fit:.3f}s  b={b_fit:.3f}s/bit   R^2={r2:.4f}")
    print(f"  {'L [m]':>6} | {'theta_W [deg]':>13} | {'ID [bit]':>8} | {'MT [s]':>7}")
    print("  " + "-" * 42)
    for L, tW in zip(Ls, theta_W):
        ID = angular_id(np.deg2rad(amp_deg), tW)
        print(f"  {L:6.0f} | {np.rad2deg(tW):13.2f} | {ID:8.2f} | {a_true + b_true*ID:7.3f}")
    print("  => theta_W schrumpft mit L -> ID steigt -> Bewegungszeit waechst (angulares Fitts).")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.scatter(ids, mts, s=6, alpha=0.3, color="steelblue", label="Trials")
    xs = np.linspace(ids.min(), ids.max(), 50)
    ax.plot(xs, a_fit + b_fit * xs, "r-", lw=2, label=f"Fit: MT={a_fit:.2f}+{b_fit:.2f}*ID (R^2={r2:.3f})")
    ax.set_xlabel("Index of Difficulty ID [bit]"); ax.set_ylabel("Bewegungszeit MT [s]")
    ax.set_title("Angulares Fitts' Law"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "angular_fitts.png"), dpi=110); plt.close(fig)
    return dict(a_fit=a_fit, b_fit=b_fit, r2=r2)


if __name__ == "__main__":
    exp_raycasting_vs_distance()
    exp_gogo_reach_and_precision()
    exp_angular_fitts()
    print("\nPlots in results/. Fertig.")
