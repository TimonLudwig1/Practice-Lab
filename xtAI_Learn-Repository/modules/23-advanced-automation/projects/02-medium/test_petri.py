"""Test suite P02-medium (Petri nets + supervisory control). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_petri.py
"""
import numpy as np
from petri import (make_cell, petri_net, enabled, fire, reachability_graph, find_deadlocks,
                   blocking_states, synthesize_supervisor, supervised_reachable)


def test_state_equation():
    # firing must follow M' = M + C[:, t]
    net = make_cell()
    M = tuple(net["M0"])
    en = enabled(net, M)
    assert len(en) > 0
    t = en[0]
    M2 = fire(net, M, t)
    assert np.array_equal(np.array(M2), np.array(M) + net["C"][:, t])
    # a transition needing an absent token is not enabled
    empty = tuple(np.zeros(len(net["places"]), int))
    assert enabled(net, empty) == []


def test_reachability_graph():
    net = make_cell()
    states, edges = reachability_graph(net)
    assert tuple(net["M0"]) in states
    assert net["marked"] in states, "the goal marking must be reachable"
    assert len(states) == 9 and len(edges) == 10


def test_finds_the_deadlock():
    net = make_cell()
    states, edges = reachability_graph(net)
    dead = find_deadlocks(net, states)
    assert len(dead) == 1
    d = next(iter(dead))
    # the deadlock is: A holds R1, B holds R2, both resources gone, no transition enabled
    named = {p: v for p, v in zip(net["places"], d)}
    assert named["A_r1"] == 1 and named["B_r2"] == 1 and named["R1"] == 0 and named["R2"] == 0
    assert enabled(net, d) == []


def test_supervisor_removes_deadlock_and_is_permissive():
    net = make_cell()
    states, edges = reachability_graph(net)
    forbidden = find_deadlocks(net, states) | blocking_states(net, states, edges)
    bad, disabled = synthesize_supervisor(net, states, edges, forbidden)
    sup = supervised_reachable(net, edges, bad)
    # no deadlock remains, goal still reachable, and only the unsafe states were removed
    assert find_deadlocks(net, sup) - bad == set()
    assert net["marked"] in sup
    assert len(sup) == 8 and len(disabled) == 2      # maximally permissive: removes only the deadlock


def test_supervisor_only_disables_controllable():
    net = make_cell()
    states, edges = reachability_graph(net)
    forbidden = find_deadlocks(net, states) | blocking_states(net, states, edges)
    _, disabled = synthesize_supervisor(net, states, edges, forbidden)
    for (m, t) in disabled:
        assert t not in net["uncontrollable"], "a supervisor may never disable an uncontrollable event"


def test_uncontrollable_forces_more_conservative_supervisor():
    # tB1 uncontrollable -> the bad set grows backward and fewer states stay reachable
    net_c = make_cell(uncontrollable=("tA2", "tB2"))
    net_u = make_cell(uncontrollable=("tA2", "tB2", "tB1"))
    out = {}
    for key, net in [("c", net_c), ("u", net_u)]:
        states, edges = reachability_graph(net)
        forbidden = find_deadlocks(net, states) | blocking_states(net, states, edges)
        bad, _ = synthesize_supervisor(net, states, edges, forbidden)
        out[key] = (len(bad), len(supervised_reachable(net, edges, bad)))
    assert out["u"][0] > out["c"][0], "uncontrollable tB1 must enlarge the bad set"
    assert out["u"][1] < out["c"][1], "and shrink the safely-reachable state set"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
