"""Petri-net + supervisory-control experiments: reachability, deadlock, supervisor synthesis.
/Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from petri import (make_cell, reachability_graph, find_deadlocks, blocking_states,
                   synthesize_supervisor, supervised_reachable, enabled)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def show(net, m):
    """Compact marking print: only the non-zero places."""
    return {p: int(v) for p, v in zip(net["places"], m) if v}


def exp_reachability_deadlock():
    print("=" * 72)
    print("EXPERIMENT 1 — the plant: reachability graph and the circular-wait deadlock")
    print("=" * 72)
    net = make_cell()
    states, edges = reachability_graph(net)
    dead = find_deadlocks(net, states)
    blk = blocking_states(net, states, edges)
    print(f"  two jobs, two shared resources R1/R2")
    print(f"  reachable markings: {len(states)}, transitions (edges): {len(edges)}")
    print(f"  goal marking reachable: {net['marked'] in states}")
    print(f"  DEADLOCKS: {len(dead)}")
    for d in dead:
        print(f"    {show(net, d)}  <- A holds R1, B holds R2: neither can finish")
    print(f"  blocking states (cannot reach the goal): {len(blk)}")
    print("  => The classic circular-wait deadlock: once A has grabbed R1 and B has grabbed R2,")
    print("     tA2 needs R2 and tB2 needs R1 — both held by the other job. The cell is stuck.")
    return net, states, edges


def exp_supervisor(net, states, edges):
    print("\n" + "=" * 72)
    print("EXPERIMENT 2 — supervisor synthesis: prevent the deadlock, permissively")
    print("=" * 72)
    forbidden = find_deadlocks(net, states) | blocking_states(net, states, edges)
    bad, disabled = synthesize_supervisor(net, states, edges, forbidden)
    sup = supervised_reachable(net, edges, bad)
    dead_after = find_deadlocks(net, sup) - bad
    print(f"  forbidden (deadlock+blocking) states: {len(forbidden)}")
    print(f"  bad set after the uncontrollable backward fixed-point: {len(bad)}")
    print(f"  supervisor disables {len(disabled)} controllable transition(s):")
    for (m, t) in disabled:
        print(f"    in {show(net, m)}: disable {t}")
    print(f"  supervised reachable states: {len(sup)} of {len(states)}  "
          f"(only the unsafe ones removed)")
    print(f"  deadlocks under supervision: {len(dead_after)}  |  goal still reachable: "
          f"{net['marked'] in sup}")
    print("  => The supervisor forbids exactly the two 'second grab' moves that close the circular")
    print("     wait (tB1 when A holds R1; tA1 when B holds R2). It is MAXIMALLY PERMISSIVE: it")
    print("     removes only the deadlock itself and keeps all concurrency that is safe.")
    return bad, disabled, sup


def exp_uncontrollable(net_states_edges):
    print("\n" + "=" * 72)
    print("EXPERIMENT 3 — why uncontrollable events make it harder (Ramadge-Wonham)")
    print("=" * 72)
    print("If job B grabs R2 on its own (tB1 uncontrollable, e.g. a part arrives and is taken")
    print("automatically), the supervisor can no longer stop B at the last moment — it must act")
    print("earlier. We re-synthesise with tB1 uncontrollable and compare.\n")
    rows = []
    for label, unc in [("all acquires controllable", ("tA2", "tB2")),
                       ("tB1 uncontrollable", ("tA2", "tB2", "tB1"))]:
        net = make_cell(uncontrollable=unc)
        states, edges = reachability_graph(net)
        forbidden = find_deadlocks(net, states) | blocking_states(net, states, edges)
        bad, disabled = synthesize_supervisor(net, states, edges, forbidden)
        sup = supervised_reachable(net, edges, bad)
        rows.append((label, len(bad), len(disabled), len(sup), len(states)))
        print(f"  {label:26s}: bad set {len(bad)}, disabled {len(disabled)}, "
              f"supervised states {len(sup)}/{len(states)}")
    print("  => With tB1 uncontrollable the bad set grows backward (the state where A can still")
    print("     grab R1 becomes unsafe, because B might then autonomously deadlock it), so the")
    print("     supervisor is FORCED to be more conservative — far fewer states remain reachable.")
    print("     This is the core Ramadge-Wonham lesson: uncontrollable events shrink what is safe.")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    labels = [r[0] for r in rows]
    supervised = [r[3] for r in rows]
    total = rows[0][4]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, supervised, 0.4, label="reachable under supervision", color="steelblue")
    ax.bar(x + 0.2, [total] * len(rows), 0.4, label="reachable without supervision", color="lightgray")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("number of reachable markings")
    ax.set_title("uncontrollable events force a more conservative supervisor")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "supervisor.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    net, states, edges = exp_reachability_deadlock()
    exp_supervisor(net, states, edges)
    exp_uncontrollable((net, states, edges))
    print("\nPlot in results/. Done.")
