"""Petri nets + supervisory control: model a resource-sharing cell, find deadlocks, synthesise a
maximally permissive safe supervisor.  Module 23 — Advanced Automation.

>>> YOUR TASK <<<
The net data structure, firing semantics and the example cell are given. Implement the three
marked functions: reachability_graph, find_deadlocks and synthesize_supervisor. Solution in solution/.

A Petri net is stored as a dict with integer Pre/Post matrices (n places x m transitions), the
initial marking M0, names, a marked (goal) marking, and the set of UNCONTROLLABLE transitions.
Markings are tuples of ints. Everything from scratch (numpy + graph search).
"""
import numpy as np
from collections import deque


# ==========================================================================
# Petri-net data structure and firing semantics
# ==========================================================================
def petri_net(places, transitions, pre, post, m0, marked, uncontrollable):
    """pre/post: dicts {(place, transition): weight}. Returns the net as a dict."""
    n, m = len(places), len(transitions)
    pidx = {p: i for i, p in enumerate(places)}
    tidx = {t: j for j, t in enumerate(transitions)}
    Pre = np.zeros((n, m), int)
    Post = np.zeros((n, m), int)
    for (p, t), w in pre.items():
        Pre[pidx[p], tidx[t]] = w
    for (p, t), w in post.items():
        Post[pidx[p], tidx[t]] = w
    return dict(places=list(places), transitions=list(transitions),
                Pre=Pre, Post=Post, C=Post - Pre, M0=np.array(m0, int),
                marked=tuple(marked), uncontrollable=set(uncontrollable))


def enabled(net, M):
    """Indices of transitions enabled at marking M (each input place has enough tokens)."""
    M = np.asarray(M)
    return [t for t in range(net["Pre"].shape[1]) if np.all(M >= net["Pre"][:, t])]


def fire(net, M, t):
    """Fire transition t: the state equation M' = M + C[:, t] (as a tuple)."""
    return tuple(np.asarray(M) + net["C"][:, t])


# ==========================================================================
# Reachability graph
# ==========================================================================
def reachability_graph(net):
    """BFS from M0 firing every enabled transition. Returns (states set, edges list of (m, t, n))."""
    # TODO: BFS from tuple(net["M0"]). For each state, for each enabled(net, m) transition t,
    #       compute n = fire(net, m, t), record the edge (m, t, n), enqueue n if new.
    #       Return (states set, edges list).
    raise NotImplementedError


# ==========================================================================
# Deadlocks and blocking states
# ==========================================================================
def find_deadlocks(net, states):
    """Reachable markings with no enabled transition, other than the marked (goal) marking."""
    # TODO: return the set of reachable markings with no enabled transition, excluding net["marked"].
    raise NotImplementedError


def blocking_states(net, states, edges):
    """States from which the marked (goal) marking is unreachable (dead ends and livelocks)."""
    succ = {}
    for (m, t, n) in edges:
        succ.setdefault(m, []).append(n)

    def can_reach_goal(start):
        seen = {start}; q = deque([start])
        while q:
            x = q.popleft()
            if x == net["marked"]:
                return True
            for y in succ.get(x, []):
                if y not in seen:
                    seen.add(y); q.append(y)
        return False

    return {m for m in states if not can_reach_goal(m)}


# ==========================================================================
# Supervisor synthesis (forbidden-state, Ramadge-Wonham)
# ==========================================================================
def synthesize_supervisor(net, states, edges, forbidden):
    """Maximally permissive safe supervisor by a backward fixed-point over uncontrollable edges.

    forbidden : the initially-bad states (deadlocks + blocking).
    Returns (bad_set, disabled) where `disabled` is the list of (marking, transition_name) pairs the
    supervisor forbids — exactly the controllable edges from a safe state into the bad set.
    """
    tname = net["transitions"]
    unc = net["uncontrollable"]
    bad = set(forbidden)
    # TODO 1: grow `bad` to a fixed point: add any state m that has an UNCONTROLLABLE transition
    #         (tname[t] in unc) with successor n already in `bad` (from m, entering bad is
    #         unavoidable). Repeat over all edges until `bad` stops changing.
    # TODO 2: `disabled` = the CONTROLLABLE edges (tname[t] not in unc) from a safe state m
    #         (m not in bad) into `bad` (n in bad). Return (bad, disabled).
    raise NotImplementedError


def supervised_reachable(net, edges, bad):
    """States reachable from M0 when every edge into `bad` is forbidden."""
    start = tuple(net["M0"])
    seen = {start}; q = deque([start])
    while q:
        m = q.popleft()
        for (mm, t, n) in edges:
            if mm == m and n not in bad and n not in seen:
                seen.add(n); q.append(n)
    return seen


# ==========================================================================
# The example plant: two jobs, two shared resources (circular-wait deadlock)
# ==========================================================================
def make_cell(uncontrollable=("tA2", "tB2")):
    """Two jobs A, B and two shared resources R1, R2.

    Job A grabs R1 (tA1) then R2 (tA2, completing and releasing both). Job B grabs R2 (tB1) then R1
    (tB2). If A holds R1 and B holds R2, neither can proceed -> the classic circular-wait DEADLOCK.
    Acquire transitions tA1/tB1 are controllable (we schedule starts); pass a different
    `uncontrollable` set to explore the Ramadge-Wonham subtlety.
    """
    places = ["A_idle", "A_r1", "A_done", "B_idle", "B_r2", "B_done", "R1", "R2"]
    transitions = ["tA1", "tA2", "tB1", "tB2"]
    pre = {("A_idle", "tA1"): 1, ("R1", "tA1"): 1,
           ("A_r1", "tA2"): 1, ("R2", "tA2"): 1,
           ("B_idle", "tB1"): 1, ("R2", "tB1"): 1,
           ("B_r2", "tB2"): 1, ("R1", "tB2"): 1}
    post = {("A_r1", "tA1"): 1,
            ("A_done", "tA2"): 1, ("R1", "tA2"): 1, ("R2", "tA2"): 1,
            ("B_r2", "tB1"): 1,
            ("B_done", "tB2"): 1, ("R1", "tB2"): 1, ("R2", "tB2"): 1}
    m0 = [1, 0, 0, 1, 0, 0, 1, 1]
    marked = [0, 0, 1, 0, 0, 1, 1, 1]      # both jobs done, both resources returned
    return petri_net(places, transitions, pre, post, m0, marked, uncontrollable)
