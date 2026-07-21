"""Tests fuer die Motion-to-Photon-Kette.  Aufruf:  python test_latency.py"""
import numpy as np

from head_motion import generate_head_yaw
from latency import (LatencyBudget, ms_to_steps, displayed_pose, predicted_pose,
                     timewarped_pose, angular_error)

FS = 1000
t, TRUTH, VEL = generate_head_yaw(duration_s=20.0, fs=FS, seed=0)


# ----------------------------- Grundbausteine -----------------------------
def test_ms_to_steps():
    assert ms_to_steps(20, 1000) == 20
    assert ms_to_steps(11.1, 1000) == 11
    assert ms_to_steps(0, 1000) == 0


def test_budget_summiert_und_bewertet():
    b = LatencyBudget(sensor=1.5, fusion=1.0, app=4.0, render=11.1, scanout=3.0, display=2.0)
    assert np.isclose(b.total_ms, 22.6)
    assert "ZU HOCH" in b.report()
    b2 = LatencyBudget(sensor=1, render=8, scanout=2)
    assert b2.total_ms == 11 and "OK" in b2.report()


# ----------------------------- reine Latenz -----------------------------
def test_null_latenz_ist_fehlerfrei():
    d = displayed_pose(TRUTH, 0, FS)
    assert np.array_equal(d, TRUTH)
    assert angular_error(d, TRUTH).max() == 0.0


def test_latenz_verschiebt_um_genau_die_richtige_zeit():
    # bei 20 ms zeigt das Display die Pose von vor 20 Samples (nach der Einschwingzeit)
    d = displayed_pose(TRUTH, 20, FS)
    assert np.allclose(d[100:], TRUTH[80:-20])


def test_mehr_latenz_mehr_fehler():
    fehler = [angular_error(displayed_pose(TRUTH, L, FS), TRUTH).mean() for L in (5, 11, 20, 50)]
    assert fehler == sorted(fehler)          # monoton steigend
    assert fehler[0] > 0


# ----------------------------- Prediction -----------------------------
def test_prediction_horizon_null_ist_reine_latenz():
    p = predicted_pose(TRUTH, VEL, latency_ms=20, horizon_ms=0, fs=FS)
    assert np.array_equal(p, displayed_pose(TRUTH, 20, FS))


def test_prediction_sweet_spot_ist_horizon_gleich_latenz():
    # bei konstanter Winkelgeschwindigkeit ist horizon = latency exakt; hier fast exakt
    e = {H: angular_error(predicted_pose(TRUTH, VEL, 20, H, FS), TRUTH).mean()
         for H in (0, 10, 20, 30, 40)}
    beste = min(e, key=e.get)
    assert beste == 20, e
    assert e[20] < 0.2 * e[0]                 # deutlich besser als ohne Prediction


def test_prediction_ist_u_foermig():
    # zu viel Vorhersage ist wieder schlechter -> U-Form
    e = {H: angular_error(predicted_pose(TRUTH, VEL, 20, H, FS), TRUTH).mean()
         for H in (20, 30, 40)}
    assert e[20] < e[30] < e[40]


def test_prediction_overshoot_an_richtungswechseln():
    # Prediction ist dort am schlechtesten, wo die Bewegung dreht (hohe Beschleunigung)
    beschl = np.abs(np.gradient(VEL, 1 / FS))
    wende = beschl > np.percentile(beschl, 90)
    e = angular_error(predicted_pose(TRUTH, VEL, 20, 20, FS), TRUTH)
    assert e[wende].mean() > 2 * e[~wende].mean(), (e[wende].mean(), e[~wende].mean())


def test_prediction_bei_konstanter_geschwindigkeit_exakt():
    # kuenstlicher Fall: gleichmaessige Drehung -> lineare Extrapolation ist EXAKT
    n = 2000
    truth = np.arange(n) * 0.1          # 0.1 Grad pro Sample = konstante Geschwindigkeit
    vel = np.gradient(truth, 1 / FS)
    p = predicted_pose(truth, vel, latency_ms=20, horizon_ms=20, fs=FS)
    assert angular_error(p, truth)[100:].max() < 1e-6


# ----------------------------- Timewarp -----------------------------
def test_timewarp_reduziert_auf_warp_latenz():
    # Orientational Timewarp: effektive Latenz = Warp-Latenz, unabhaengig von der Render-Latenz
    tw = timewarped_pose(TRUTH, render_latency_ms=50, warp_latency_ms=2, fs=FS)
    assert np.array_equal(tw, displayed_pose(TRUTH, 2, FS))


def test_timewarp_ist_unabhaengig_von_render_latenz():
    a = timewarped_pose(TRUTH, render_latency_ms=20, warp_latency_ms=2, fs=FS)
    b = timewarped_pose(TRUTH, render_latency_ms=50, warp_latency_ms=2, fs=FS)
    assert np.array_equal(a, b)         # bei reiner Drehung zaehlt nur die Warp-Latenz


def test_timewarp_schlaegt_reine_latenz():
    e_ohne = angular_error(displayed_pose(TRUTH, 20, FS), TRUTH).mean()
    e_warp = angular_error(timewarped_pose(TRUTH, 20, 2, FS), TRUTH).mean()
    assert e_warp < 0.2 * e_ohne


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, tf in tests:
        tf(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
