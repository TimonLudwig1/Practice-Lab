"""Put-that-there — ein multimodaler Referenz-Interpreter.  (Referenzloesung P03-final)

Modul 18 — Multimodal Interfaces.

Gegeben zwei ASYNCHRONE Ereignisstroeme:
  - Sprache: ein deiktisches Wort ("that") zum Zeitpunkt t_word, plus eine (verrauschte)
    ASR-Verteilung ueber einen Typ-Nomen-Slot ("... that BUTTON") -> semantischer Hinweis.
  - Geste:  eine ueber die Zeit abgetastete, verrauschte 2D-Zeigeposition (Deixis).

Aufgabe: aufloesen, welches Objekt der Szene mit "that" gemeint ist, durch multiplikative
Fusion dreier Faktoren (Bayes-Produkt, Skript-Kapitel 12):

    P(obj = o | .)  proportional zu  P_sem(o) * P_point(o) * P_temp(o)

  P_point(o) : kam der Zeiger o raeumlich nahe?      exp(-d_min(o)^2 / 2 sigma^2)
  P_temp(o)  : geschah das zur richtigen ZEIT (um t_word, Geste fuehrt leicht)?
               exp(-(delta_o - mu)^2 / 2 tau^2),  delta_o = t_naehe(o) - t_word
  P_sem(o)   : passt der Typ zum gehoerten Nomen?    ASR-Verteilung q[typ_o]

Der Generator ist vollstaendig offengelegt und mit Seed reproduzierbar. Er baut gezielt
zwei Arten von Mehrdeutigkeit ein:
  - ein "Decoy"-Objekt, ueber das der Zeiger auf dem WEG zum Ziel frueh hinwegstreicht
    (raeumlich nah, aber zur FALSCHEN Zeit) -> nur der TEMPORALE Faktor loest das.
  - mit 50 % ein raeumlicher "Zwilling" direkt neben dem Ziel, anderen Typs
    (raeumlich + zeitlich fast identisch) -> nur der SEMANTISCHE Faktor loest das.
"""
import numpy as np

TYPES = ["button", "slider", "text", "image"]


# ==========================================================================
# Szene & Kommando-Generator  (offengelegt, reproduzierbar)
# ==========================================================================
def make_scene(n_objects=7, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.1, 0.9, size=(n_objects, 2))
    types = rng.integers(0, len(TYPES), n_objects)
    return dict(pos=pos, types=types)


def make_command(scene, seed,
                 mu_true=-0.15,      # Geste FUEHRT das Wort um 150 ms (delta = t_nah - t_word < 0)
                 sigma_point=0.035,  # raeumliches Zeige-Rauschen
                 tau_true=0.18,      # zeitliche Streuung
                 t_word=1.0,
                 dt=0.02, t_max=2.0,
                 asr_correct_prob=0.75,  # P(ASR-Nomen == wahrer Typ)
                 noun_prob=0.8,          # P(ueberhaupt ein Typ-Nomen gesprochen)
                 twin_prob=0.5):         # P(raeumlicher Zwilling anderen Typs)
    """Erzeugt EIN Kommando auf 'scene'. Gibt ein dict mit ground truth + Stroemen zurueck.

    Der Zeiger startet AUF einem Decoy-Objekt und wandert zum Ziel, wo er ab t_arrive
    verweilt. So hat der Decoy eine kleine Minimaldistanz, aber zur falschen (fruehen) Zeit.
    """
    rng = np.random.default_rng(seed)
    pos = scene["pos"].copy()
    types = scene["types"].copy()
    n = pos.shape[0]

    target = int(rng.integers(n))
    # Decoy != target: der Zeiger streift ihn auf dem Weg
    decoy = int(rng.choice([i for i in range(n) if i != target]))

    # optionaler raeumlicher Zwilling: neues Objekt dicht neben dem Ziel, anderer Typ
    added_twin = False
    if rng.random() < twin_prob:
        direction = rng.normal(size=2); direction /= np.linalg.norm(direction)
        twin_pos = pos[target] + direction * (1.6 * sigma_point + 0.02)
        twin_pos = np.clip(twin_pos, 0.02, 0.98)
        # Typ != Zieltyp, damit Semantik ihn trennen kann
        twin_type = int(rng.choice([t for t in range(len(TYPES)) if t != types[target]]))
        pos = np.vstack([pos, twin_pos])
        types = np.append(types, twin_type)
        added_twin = True
    n = pos.shape[0]

    # ----- Gesten-Strom: Start(Decoy) -> Ziel (Wegpunkt) -> "there"-Ziel -----
    # Der Zeiger STREIFT den Decoy am Anfang (t~0), erreicht das Ziel als scharfen
    # Wegpunkt bei t_arrive (= t_word + mu_true, Geste fuehrt), und wandert dann zu
    # einer "there"-Position weiter. So ist der Ziel-Zeitpunkt sauber definiert und
    # Zeigen-allein wird mehrdeutig (Decoy und Ziel beide raeumlich nah).
    t = np.arange(0.0, t_max, dt)
    t_arrive = t_word + mu_true               # Ankunft am Ziel (vor dem Wort)
    t_there = t_arrive + 0.5                   # danach weiter zur "there"-Position
    start = pos[decoy] + rng.normal(0, sigma_point, 2)
    there = rng.uniform(0.1, 0.9, 2)           # freie Zielposition der Bewegung
    ptr = np.empty((t.size, 2))
    for i, ti in enumerate(t):
        if ti <= t_arrive:                     # Start -> Ziel
            frac = ti / max(t_arrive, 1e-6)
            ptr[i] = (1 - frac) * start + frac * pos[target]
        elif ti <= t_there:                    # Ziel -> there
            frac = (ti - t_arrive) / (t_there - t_arrive)
            ptr[i] = (1 - frac) * pos[target] + frac * there
        else:                                  # bei there verweilen
            ptr[i] = there
    ptr = ptr + rng.normal(0, sigma_point, ptr.shape)

    # ----- Sprach-Strom: ASR-Verteilung ueber Typ-Nomen (semantischer Hinweis) -----
    q = np.full(len(TYPES), 1.0 / len(TYPES))  # uniform = kein Nomen gehoert
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
# Interpreter — die multimodale Fusion
# ==========================================================================
def _object_evidence(cmd, o):
    """Fuer Objekt o: minimale Zeigerdistanz d_min und der Zeitpunkt t_nah, zu dem
    sie auftrat. Basis fuer P_point und P_temp."""
    d = np.linalg.norm(cmd["ptr_xy"] - cmd["scene_pos"][o], axis=1)
    i = int(d.argmin())
    return d[i], cmd["ptr_t"][i]


def resolve(cmd, mu_hat=-0.15, sigma_hat=0.035, tau_hat=0.18,
            use_sem=True, use_point=True, use_temp=True):
    """Loest die deiktische Referenz auf. Gibt (posterior ueber Objekte, argmax) zurueck.
    Ueber die use_*-Flags lassen sich einzelne Faktoren fuer Ablationen abschalten."""
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
# Naive Baseline: das Objekt, dem der Zeiger GENAU zum Wortzeitpunkt am naechsten war
# ==========================================================================
def resolve_naive_at_word(cmd):
    i = int(np.argmin(np.abs(cmd["ptr_t"] - cmd["t_word"])))
    ptr_now = cmd["ptr_xy"][i]
    d = np.linalg.norm(cmd["scene_pos"] - ptr_now, axis=1)
    return int(d.argmin())
