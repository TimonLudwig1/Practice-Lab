"""MAPE-K auto-scaling: reactive vs. control-theoretic vs. model-based predictive elasticity.
(Reference solution P02-medium)   Module 24 — Self-aware Computing.

The managed element is a service with `c` identical replicas, modelled as an M/M/c queue. An
autonomic manager runs a MAPE-K loop once per control interval: Monitor (load, utilisation,
latency) -> Analyze/Plan (a scaling policy) -> Execute (a capacity change that only takes effect
after a PROVISIONING DELAY). Everything from scratch (numpy + the Erlang-C formula).
"""
import numpy as np
from math import factorial


# ==========================================================================
# The self-model: M/M/c (given — this is the script's ch. 5, now with c servers)
# ==========================================================================
def erlang_c(c, lam, mu):
    """Probability that an arriving request has to queue (Erlang-C). 1.0 if unstable."""
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    s = sum(a**k / factorial(k) for k in range(c))
    top = a**c / (factorial(c) * (1.0 - rho))
    return top / (s + top)


def response_quantile_mmc(lam, mu, c, p=0.95):
    """p-quantile of the response time for M/M/c.

    The WAITING time is P(W>t) = P_Q exp(-(c mu - lam) t), so its p-quantile is closed form; we
    approximate the response-time quantile as (waiting-time quantile + mean service time). For c=1
    this is within ~1 % of the exact M/M/1 result at the loads that matter.
    """
    if c * mu <= lam:
        return np.inf                      # unstable: unbounded latency
    pq = erlang_c(c, lam, mu)
    wp = 0.0 if pq <= 1.0 - p else np.log(pq / (1.0 - p)) / (c * mu - lam)
    return wp + 1.0 / mu


def min_replicas(lam, mu, slo, c_max=40):
    """Smallest replica count whose predicted p95 latency meets the SLO — the model INVERTED.

    This is the self-predictive step: instead of reacting to a measurement, the system asks its
    model 'how much capacity would this load require?' and answers before acting.
    """
    for c in range(1, c_max + 1):
        if response_quantile_mmc(lam, mu, c) < slo:
            return c
    return c_max


# ==========================================================================
# Workload: a daily pattern plus a flash crowd
# ==========================================================================
def workload(T=200, seed=0, spike=35.0):
    """Arrival rate over T control intervals: daily sinusoid + flash crowd + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = 50.0 + 35.0 * np.sin(2 * np.pi * t / T - np.pi / 2)
    flash = np.where((t > int(0.62 * T)) & (t < int(0.70 * T)), spike, 0.0)
    return np.maximum(5.0, base + flash + rng.normal(0, 2.0, T))


# ==========================================================================
# The three scaling policies
# ==========================================================================
def policy_reactive(state, knob=0.75):
    """Threshold-based with hysteresis: scale out above `knob`, in below `knob - gap`.

    The gap (state["hyst_gap"], default 0.40) is the dead zone that prevents oscillation.
    Purely reactive: it can only respond AFTER utilisation has already risen.
    """
    c, util = state["c"], state["util"]
    gap = state.get("hyst_gap", 0.40)
    if util > knob:
        return c + 1
    if util < knob - gap:
        return c - 1
    return c


def policy_control(state, knob=0.70):
    """Control-theoretic: proportional (Kubernetes-HPA style) plus an integral term.

    desired = ceil(c * (util + 0.5*I) / target_util); the integral removes steady-state offset.
    Given as the worked example — the other two are yours.
    """
    c, util = state["c"], state["util"]
    err = util - knob
    state["integ"] = float(np.clip(0.85 * state.get("integ", 0.0) + err, -1.0, 1.0))
    if util <= 0:
        return c
    return int(np.ceil(c * (util + 0.5 * state["integ"]) / knob))


def policy_predictive(state, knob=1.10):
    """Model-based predictive: forecast the load over the provisioning delay, then invert the
    queueing model to get the capacity that will meet the SLO AT THAT FUTURE LOAD.

    `knob` is a safety margin on the forecast load. This is MPC applied to the system itself
    (module 23): predict a horizon, optimise, act, repeat.
    """
    ewma, prev_ewma = state["ewma"], state["prev_ewma"]
    forecast = ewma + (ewma - prev_ewma) * state["delay"]
    return min_replicas(max(forecast * knob, 1.0), state["mu"], state["slo"])


POLICIES = {"reactive": policy_reactive, "control": policy_control, "predictive": policy_predictive}


# ==========================================================================
# The MAPE-K loop
# ==========================================================================
def simulate(policy, knob=None, T=200, seed=0, mu=10.0, slo=0.5, delay=3, cooldown=2,
             c_min=1, c_max=20, spike=35.0, hyst_gap=0.40, allow_concurrent=False):
    """Run the autonomic manager over T control intervals. Returns the trace and the metrics."""
    lam = workload(T, seed, spike)
    fn = POLICIES[policy]
    c = min_replicas(lam[0], mu, slo)          # start correctly provisioned
    pending = []                                # (effective_at, new_c) — the provisioning delay
    state = dict(c=c, mu=mu, slo=slo, delay=delay, integ=0.0, ewma=lam[0], prev_ewma=lam[0],
                 hyst_gap=hyst_gap)

    trace = dict(c=[], lam=[], latency=[], violated=[])
    adaptations, last_action = 0, -10**9
    for k in range(T):
        # --- Execute (deferred): apply capacity changes whose provisioning has completed
        for (t, v) in pending:
            if t <= k:
                c = v
        pending = [(t, v) for (t, v) in pending if t > k]

        # --- Monitor
        latency = response_quantile_mmc(lam[k], mu, c)
        util = lam[k] / (c * mu)
        trace["c"].append(c); trace["lam"].append(lam[k])
        trace["latency"].append(min(latency, 10.0)); trace["violated"].append(latency > slo)

        # --- Knowledge: keep the smoothed load estimate up to date
        state["prev_ewma"] = state["ewma"]
        state["ewma"] = 0.5 * state["ewma"] + 0.5 * lam[k]
        state.update(c=c, util=util)

        # --- Analyze + Plan
        target = fn(state) if knob is None else fn(state, knob)
        target = int(np.clip(target, c_min, c_max))

        # --- Execute: request the change (cooldown + no concurrent change)
        if target != c and (k - last_action) >= cooldown and (allow_concurrent or not pending):
            pending.append((k + delay, target))
            last_action = k
            adaptations += 1

    return _metrics(trace, lam, mu, slo, adaptations)


def _metrics(trace, lam, mu, slo, adaptations):
    """The elasticity triple (Herbst/Kounev): under-provisioning, over-provisioning, instability."""
    c = np.array(trace["c"])
    needed = np.array([min_replicas(l, mu, slo) for l in lam])
    return dict(
        slo_violations=float(np.mean(trace["violated"])),      # under-provisioning -> SLO
        mean_replicas=float(c.mean()),                          # over-provisioning -> cost
        adaptations=adaptations,                                # instability -> flapping
        under=float(np.mean(np.maximum(needed - c, 0))),
        over=float(np.mean(np.maximum(c - needed, 0))),
        c=c, needed=needed, lam=lam,
        latency=np.array(trace["latency"]),
    )


def oracle(T=200, seed=0, mu=10.0, slo=0.5, spike=35.0):
    """Perfect knowledge, no provisioning delay: the unreachable lower bound on cost."""
    lam = workload(T, seed, spike)
    needed = np.array([min_replicas(l, mu, slo) for l in lam])
    return dict(slo_violations=0.0, mean_replicas=float(needed.mean()), adaptations=None)
