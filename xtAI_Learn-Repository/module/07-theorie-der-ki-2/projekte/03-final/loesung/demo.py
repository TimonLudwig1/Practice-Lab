"""Demonstration: Value Iteration & Policy Iteration auf der 4x3-Gridworld,
Vergleich, gamma-Studie."""
from gridworld import GridworldMDP, show_policy, show_values
from mdp import value_iteration, greedy_policy, policy_iteration, q_value


def main():
    mdp = GridworldMDP(living_reward=-0.04, gamma=1.0)

    print("=" * 52)
    print(" Value Iteration")
    print("=" * 52)
    V_vi, it_vi, deltas = value_iteration(mdp)
    pol_vi = greedy_policy(mdp, V_vi)
    print(f"konvergiert nach {it_vi} Iterationen (max. Bellman-Residuum < 1e-8)\n")
    print("Utilities V*(s):")
    print(show_values(mdp, V_vi))
    print("\nOptimale Policy:")
    print(show_policy(mdp, pol_vi))

    print("\n" + "=" * 52)
    print(" Policy Iteration")
    print("=" * 52)
    V_pi, pol_pi, it_pi = policy_iteration(mdp)
    print(f"konvergiert nach {it_pi} Policy-Verbesserungen\n")
    print("Optimale Policy:")
    print(show_policy(mdp, pol_pi))

    # Uebereinstimmung
    same = all(pol_vi[s] == pol_pi[s] for s in mdp.states)
    maxdiff = max(abs(V_vi[s] - V_pi[s]) for s in mdp.states)
    print(f"\nGleiche Policy wie Value Iteration: {same}")
    print(f"Max. Utility-Differenz VI vs PI: {maxdiff:.2e}")

    print("\n" + "=" * 52)
    print(" gamma-Studie: wie Diskontierung die Policy aendert")
    print("=" * 52)
    for g in (0.99, 0.9, 0.5, 0.2):
        m = GridworldMDP(living_reward=-0.04, gamma=g)
        V, it, _ = value_iteration(m)
        pol = greedy_policy(m, V)
        # Beispielzustand (2,0) unten: bei starker Diskontierung riskantere/kuerzere Wege
        print(f"\ngamma={g:<4}  (VI: {it} Iterationen)")
        print(show_policy(m, pol))

    print("\n" + "=" * 52)
    print(" living_reward-Studie: wie die Schrittkosten die Policy aendern")
    print("=" * 52)
    for r in (-0.04, -0.5, -2.0, 0.0):
        m = GridworldMDP(living_reward=r, gamma=1.0 if r < 0 else 0.99)
        V, _, _ = value_iteration(m)
        pol = greedy_policy(m, V)
        print(f"\nliving_reward={r}")
        print(show_policy(m, pol))


if __name__ == "__main__":
    main()
