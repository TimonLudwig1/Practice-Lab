"""The acceptance test for the MDP agent (the reference solution).

    python test_mdp.py    ->  "All tests passed."
"""
from gridworld import GridworldMDP
from mdp import value_iteration, greedy_policy, policy_iteration


# The reference utilities (Russell & Norvig, 4x3, R=-0.04, gamma=1)
REF_V = {
    (0, 2): 0.812, (1, 2): 0.868, (2, 2): 0.918, (3, 2): 1.000,
    (0, 1): 0.762,                (2, 1): 0.660, (3, 1): -1.000,
    (0, 0): 0.705, (1, 0): 0.655, (2, 0): 0.611, (3, 0): 0.388,
}
# The reference policy
REF_POLICY = {
    (0, 2): "E", (1, 2): "E", (2, 2): "E",
    (0, 1): "N",              (2, 1): "N",
    (0, 0): "N", (1, 0): "W", (2, 0): "W", (3, 0): "W",
}


def test_value_iteration_utilities():
    mdp = GridworldMDP()
    V, it, _ = value_iteration(mdp)
    for s, ref in REF_V.items():
        assert abs(V[s] - ref) < 0.005, (s, V[s], ref)
    print(f"  VI utilities == the AIMA reference ... OK ({it} iterations)")


def test_value_iteration_policy():
    mdp = GridworldMDP()
    V, _, _ = value_iteration(mdp)
    pol = greedy_policy(mdp, V)
    for s, a in REF_POLICY.items():
        assert pol[s] == a, (s, pol[s], a)
    print("  VI policy == the AIMA reference ...... OK")


def test_policy_iteration_matches_vi():
    mdp = GridworldMDP()
    V_vi, _, _ = value_iteration(mdp)
    V_pi, pol_pi, it = policy_iteration(mdp)
    assert all(pol_pi[s] == REF_POLICY[s] for s in REF_POLICY), "the PI policy deviates"
    assert max(abs(V_vi[s] - V_pi[s]) for s in mdp.states) < 1e-4
    print(f"  PI == VI (policy & values) .......... OK ({it} improvements)")


def test_gamma_changes_policy():
    # With strong discounting at least one action changes.
    p1 = greedy_policy(GridworldMDP(gamma=1.0), value_iteration(GridworldMDP(gamma=1.0))[0])
    m2 = GridworldMDP(gamma=0.5)
    p2 = greedy_policy(m2, value_iteration(m2)[0])
    assert any(p1[s] != p2[s] for s in p1 if p1[s]), "gamma should influence the policy"
    print("  gamma influences the policy ......... OK")


if __name__ == "__main__":
    print("Tests:")
    test_value_iteration_utilities()
    test_value_iteration_policy()
    test_policy_iteration_matches_vi()
    test_gamma_changes_policy()
    print("\nAll tests passed.")
