"""Put-that-there — a multimodal reference interpreter.  (The reference solution for P03-final)

Module 18 — Multimodal Interfaces.

Given two ASYNCHRONOUS event streams:
  - Speech: a deictic word ("that") at the time t_word, plus a (noisy) ASR distribution
    over a type-noun slot ("... that BUTTON") -> the semantic cue.
  - Gesture: a sampled, noisy 2D pointing position over time (deixis).

The task: resolve which object of the scene is meant by "that", through the multiplicative
fusion of three factors (the Bayes product, script chapter 12):

    P(obj = o | .)  proportional to  P_sem(o) * P_point(o) * P_temp(o)

  P_point(o) : did the pointer come spatially close to o?   exp(-d_min(o)^2 / 2 sigma^2)
  P_temp(o)  : did that happen at the right TIME (around t_word, the gesture leading slightly)?
               exp(-(delta_o - mu)^2 / 2 tau^2),  delta_o = t_near(o) - t_word
  P_sem(o)   : does the type fit the heard noun?            the ASR distribution q[type_o]

The generator is fully disclosed and reproducible with a seed. It deliberately builds in
two kinds of ambiguity:
  - a "decoy" object that the pointer sweeps over early ON THE WAY to the target
    (spatially close, but at the WRONG time) -> only the TEMPORAL factor resolves that.
  - with 50 % probability a spatial "twin" right next to the target, of a different type
    (spatially + temporally almost identical) -> only the SEMANTIC factor resolves that.
"""
import numpy as np

TYPES = ["button", "slider", "text", "image"]


# ==========================================================================
# The scene & command generator  (disclosed, reproducible)
# ==========================================================================
def make_scene(n_objects=7, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.1, 0.9, size=(n_objects, 2))
    types = rng.integers(0, len(TYPES), n_objects)
    return dict(pos=pos, types=types)


def make_command(scene, seed,
                 mu_true=-0.15,      # the gesture LEADS the word by 150 ms (delta = t_near - t_word < 0)
                 sigma_point=0.035,  # the spatial pointing noise
                 tau_true=0.18,      # the temporal spread
                 t_word=1.0,
                 dt=0.02, t_max=2.0,
                 asr_correct_prob=0.75,  # P(the ASR noun == the true type)
                 noun_prob=0.8,          # P(a type noun is spoken at all)
                 twin_prob=0.5):         # P(a spatial twin of a different type)
    """Produces ONE command on 'scene'. Returns a dict with the ground truth + the streams.

    The pointer starts ON a decoy object and wanders to the target, where it dwells from
    t_arrive on. This way the decoy has a small minimal distance, but at the wrong (early) time.
    """
    rng = np.random.default_rng(seed)
    pos = scene["pos"].copy()
    types = scene["types"].copy()
    n = pos.shape[0]

    target = int(rng.integers(n))
    # decoy != target: the pointer brushes past it on the way
    decoy = int(rng.choice([i for i in range(n) if i != target]))

    # an optional spatial twin: a new object right next to the target, of a different type
    added_twin = False
    if rng.random() < twin_prob:
        direction = rng.normal(size=2); direction /= np.linalg.norm(direction)
        twin_pos = pos[target] + direction * (1.6 * sigma_point + 0.02)
        twin_pos = np.clip(twin_pos, 0.02, 0.98)
        # a type != the target type, so that the semantics can separate it
        twin_type = int(rng.choice([t for t in range(len(TYPES)) if t != types[target]]))
        pos = np.vstack([pos, twin_pos])
        types = np.append(types, twin_type)
        added_twin = True
    n = pos.shape[0]

    # ----- the gesture stream: start(decoy) -> target (a waypoint) -> the "there" target -----
    # The pointer BRUSHES the decoy at the beginning (t~0), reaches the target as a sharp
    # waypoint at t_arrive (= t_word + mu_true, the gesture leading), and then wanders on to
    # a "there" position. This way the target moment is cleanly defined and pointing alone
    # becomes ambiguous (the decoy and the target are both spatially close).
    t = np.arange(0.0, t_max, dt)
    t_arrive = t_word + mu_true               # the arrival at the target (before the word)
    t_there = t_arrive + 0.5                  # afterwards on to the "there" position
    start = pos[decoy] + rng.normal(0, sigma_point, 2)
    there = rng.uniform(0.1, 0.9, 2)          # the free target position of the movement
    ptr = np.empty((t.size, 2))
    for i, ti in enumerate(t):
        if ti <= t_arrive:                    # start -> target
            frac = ti / max(t_arrive, 1e-6)
            ptr[i] = (1 - frac) * start + frac * pos[target]
        elif ti <= t_there:                   # target -> there
            frac = (ti - t_arrive) / (t_there - t_arrive)
            ptr[i] = (1 - frac) * pos[target] + frac * there
        else:                                 # dwell at there
            ptr[i] = there
    ptr = ptr + rng.normal(0, sigma_point, ptr.shape)

    # ----- the speech stream: the ASR distribution over the type noun (the semantic cue) -----
    q = np.full(len(TYPES), 1.0 / len(TYPES))  # uniform = no noun heard
    spoke_noun = rng.random() < noun_prob
    if spoke_noun:
        if rng.random() < asr_correct_prob:
            heard = int(types[target])
        else:
            heard = int(rng.integers(len(TYPES)))
        q = np.full(len(TYPES), 0.1)
        q[heard] += 0.6
        q = q / q.sum()

    return dict(scene_pos=pos, scene_types=types, target=target, decoy=decoy,
                twin=added_twin, t_word=t_word, ptr_t=t, ptr_xy=ptr, q_type=q,
                mu_true=mu_true, sigma_point=sigma_point, tau_true=tau_true)


# ==========================================================================
# The interpreter — the multimodal fusion
# ==========================================================================
def _object_evidence(cmd, o):
    """For object o: the minimal pointer distance d_min and the moment t_near at which it
    occurred. The basis for P_point and P_temp."""
    d = np.linalg.norm(cmd["ptr_xy"] - cmd["scene_pos"][o], axis=1)
    i = int(d.argmin())
    return d[i], cmd["ptr_t"][i]


def resolve(cmd, mu_hat=-0.15, sigma_hat=0.035, tau_hat=0.18,
            use_sem=True, use_point=True, use_temp=True):
    """Resolves the deictic reference. Returns (the posterior over the objects, the argmax).
    Individual factors can be switched off for ablations via the use_* flags."""
    n = cmd["scene_pos"].shape[0]
    logp = np.zeros(n)
    for o in range(n):
        dmin, t_near = _object_evidence(cmd, o)
        lp = 0.0
        if use_point:
            lp += -dmin**2 / (2 * sigma_hat**2)
        if use_temp:
            delta = t_near - cmd["t_word"]
            lp += -(delta - mu_hat)**2 / (2 * tau_hat**2)
        if use_sem:
            lp += np.log(max(cmd["q_type"][cmd["scene_types"][o]], 1e-9))
        logp[o] = lp
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return p, int(p.argmax())


# ==========================================================================
# The naive baseline: the object the pointer was closest to EXACTLY at the word moment
# ==========================================================================
def resolve_naive_at_word(cmd):
    i = int(np.argmin(np.abs(cmd["ptr_t"] - cmd["t_word"])))
    ptr_now = cmd["ptr_xy"][i]
    d = np.linalg.norm(cmd["scene_pos"] - ptr_now, axis=1)
    return int(d.argmin())
