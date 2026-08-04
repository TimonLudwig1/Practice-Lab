"""A complete self-aware system: MAPE-K with a model at run-time, online re-estimation and healing.
(Reference solution P03-final)   Module 24 — Self-aware Computing.

Kounev's three properties, made concrete:
  self-reflective  -- the manager monitors load, capacity, latency and per-replica health
  self-predictive  -- it carries an M/M/c model of itself and INVERTS it to plan capacity
  self-adaptive    -- it scales, and it restarts replicas it believes are sick

Two things make this more than the P02 scaler: the model's parameter (the service rate) is
RE-ESTIMATED ONLINE, so the system survives drift; and failures are detected and healed, which
forces an honest confrontation with the base-rate fallacy. numpy only.
"""
import numpy as np
from math import factorial


# ==========================================================================
# The self-model (M/M/c), as in P01/P02
# ==========================================================================
def erlang_c(c, lam, mu):
    """Probability that an arriving request has to queue. 1.0 if the configuration is unstable."""
    if c < 1 or mu <= 0:
        return 1.0
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    s = sum(a**k / factorial(k) for k in range(c))
    top = a**c / (factorial(c) * (1.0 - rho))
    return top / (s + top)


def latency_p95(lam, mu, c):
    """Predicted 95th-percentile response time of an M/M/c system (inf if unstable)."""
    if c < 1 or mu <= 0 or c * mu <= lam:
        return np.inf
    pq = erlang_c(c, lam, mu)
    wp = 0.0 if pq <= 0.05 else np.log(pq / 0.05) / (c * mu - lam)
    return wp + 1.0 / mu


def min_replicas(lam, mu, slo, c_max=30):
    """The model INVERTED: the smallest replica count predicted to meet the SLO."""
    for c in range(1, c_max + 1):
        if latency_p95(lam, mu, c) < slo:
            return c
    return c_max


# ==========================================================================
# The managed element: a replicated service that drifts and gets sick
# ==========================================================================
class Service:
    """The plant. Its TRUE service rate is hidden from the manager, drifts at a deployment, and
    individual replicas can fall sick (serving at a fraction of the nominal rate)."""

    def __init__(self, n_replicas=8, mu_nominal=10.0, seed=0,
                 p_sick=0.004, sick_factor=0.3, heal_time=3):
        self.mu_nominal = mu_nominal
        self.mu_true = mu_nominal                 # hidden; changes at a deployment
        self.n = n_replicas
        self.sick = np.zeros(n_replicas, bool)
        self.restarting = np.zeros(n_replicas, int)
        self.rng = np.random.default_rng(seed)
        self.p_sick, self.sick_factor, self.heal_time = p_sick, sick_factor, heal_time

    def deploy(self, factor):
        """A code deployment changes the true service rate — the model is now stale."""
        self.mu_true = self.mu_nominal * factor

    def step(self):
        """Advance one interval: sickness arrives, restarts complete."""
        newly = (self.rng.random(self.n) < self.p_sick) & (~self.sick) & (self.restarting == 0)
        self.sick |= newly
        self.restarting = np.maximum(self.restarting - 1, 0)

    def available(self, c=None):
        """Which of the first `c` provisioned replicas are actually serving (not restarting)."""
        mask = self.restarting == 0
        if c is not None:
            active = np.zeros(self.n, bool)
            active[:c] = True
            mask = mask & active
        return mask

    def capacity(self, c):
        """Effective service capacity of the `c` provisioned replicas (restarting ones give none)."""
        rates = np.where(self.sick, self.mu_true * self.sick_factor, self.mu_true)
        return float((rates * self.available(c)).sum())

    def observe_latency(self, lam, c):
        """The TRUE p95 latency users experience, given the capacity actually serving."""
        c_eff = max(int(self.available(c).sum()), 1)
        mu_eff = max(self.capacity(c) / c_eff, 1e-6)
        return latency_p95(lam, mu_eff, c_eff)

    def measure_service_rate(self, c, noise=0.03):
        """A noisy monitor reading of the average per-replica service rate (MAPE 'Monitor')."""
        c_eff = max(int(self.available(c).sum()), 1)
        return (self.capacity(c) / c_eff) * (1 + self.rng.normal(0, noise))

    def health_signal(self, sigma=0.35):
        """A noisy per-replica health signal in ~[0,1]; sick replicas read low, but noise overlaps."""
        true_health = np.where(self.sick, self.sick_factor, 1.0)
        return true_health + self.rng.normal(0, sigma, self.n)

    def restart(self, i):
        """Heal a replica: it becomes healthy but is UNAVAILABLE for heal_time intervals."""
        was_sick = bool(self.sick[i])
        self.sick[i] = False
        self.restarting[i] = self.heal_time
        return was_sick                            # True = the restart was justified


# ==========================================================================
# Workload
# ==========================================================================
def workload(T=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return np.maximum(5.0, 45.0 + 20.0 * np.sin(2 * np.pi * t / T - np.pi / 2) + rng.normal(0, 2.0, T))


# ==========================================================================
# The autonomic manager (MAPE-K)
# ==========================================================================
def run(T=300, seed=0, slo=0.5, adaptive_model=True, healing=True, persistence=3,
        deploy_at=None, deploy_factor=0.7, provisioning_delay=3, margin=1.05,
        detect_threshold=0.55, n_replicas=16, mu_nominal=10.0, sigma=0.35, p_sick=0.004):
    """Run the self-aware system.

    adaptive_model : re-estimate the service rate online (models@run.time) or keep the design-time value
    healing        : restart replicas the detector flags
    persistence    : require k consecutive anomalous intervals before restarting (base-rate defence)
    deploy_at      : interval at which the true service rate changes (model drift)
    """
    lam = workload(T, seed)
    svc = Service(n_replicas=n_replicas, mu_nominal=mu_nominal, seed=seed + 7, p_sick=p_sick)

    mu_hat = mu_nominal                      # THE MODEL AT RUN-TIME (its one parameter)
    c = min_replicas(lam[0], mu_hat, slo)
    pending = []
    streak = np.zeros(n_replicas, int)

    log = dict(latency=[], violated=[], mu_hat=[], mu_true=[], c=[], pred_error=[])
    restarts = tp = fp = 0
    sick_intervals = 0

    for k in range(T):
        if deploy_at is not None and k == deploy_at:
            svc.deploy(deploy_factor)                  # the world changes under the model

        svc.step()
        for (t, v) in pending:                          # provisioning completes
            if t <= k:
                c = v
        pending = [(t, v) for (t, v) in pending if t > k]
        c = int(np.clip(c, 1, n_replicas))

        # ---------- Monitor ----------
        latency = svc.observe_latency(lam[k], c)
        mu_meas = svc.measure_service_rate(c)
        health = svc.health_signal(sigma)
        sick_intervals += int((svc.sick & svc.available(c)).sum())

        # ---------- Analyze: what did the model PREDICT, and how wrong was it? ----------
        predicted = latency_p95(lam[k], mu_hat, c)
        pred_error = abs(min(predicted, 5.0) - min(latency, 5.0))

        log["latency"].append(min(latency, 5.0)); log["violated"].append(latency > slo)
        log["mu_hat"].append(mu_hat); log["mu_true"].append(svc.mu_true)
        log["c"].append(c); log["pred_error"].append(pred_error)

        # ---------- Knowledge: keep the model true (models@run.time) ----------
        if adaptive_model:
            mu_hat = 0.85 * mu_hat + 0.15 * mu_meas     # EWMA re-estimation

        # ---------- Plan + Execute: capacity ----------
        target = int(np.clip(min_replicas(lam[k] * margin, max(mu_hat, 0.1), slo), 1, n_replicas))
        if target != c and not pending:
            pending.append((k + provisioning_delay, target))

        # ---------- Plan + Execute: healing ----------
        anomalous = (health < detect_threshold) & svc.available(c)
        streak = np.where(anomalous, streak + 1, 0)
        if healing:
            for i in range(c):
                if streak[i] >= persistence and svc.restarting[i] == 0:
                    justified = svc.restart(i)
                    tp += int(justified); fp += int(not justified)
                    restarts += 1
                    streak[i] = 0

    return _summary(log, lam, restarts, tp, fp, sick_intervals, T, n_replicas, deploy_at)


def _summary(log, lam, restarts, tp, fp, sick_intervals, T, n_replicas, deploy_at):
    viol = np.array(log["violated"])
    pe = np.array(log["pred_error"])
    out = dict(
        slo_violations=float(viol.mean()),
        mean_prediction_error=float(pe.mean()),
        restarts=restarts, true_positives=tp, false_positives=fp,
        precision=(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        sick_time=sick_intervals / (T * n_replicas),
        mu_hat=np.array(log["mu_hat"]), mu_true=np.array(log["mu_true"]),
        latency=np.array(log["latency"]), c=np.array(log["c"]),
        pred_error=pe, lam=lam,
    )
    if deploy_at is not None:                        # split the metrics around the drift
        out["slo_before"] = float(viol[:deploy_at].mean())
        out["slo_after"] = float(viol[deploy_at:].mean())
        out["pred_error_after"] = float(pe[deploy_at:].mean())
    return out
