"""A demonstration: value iteration and policy iteration on the 4x3 gridworld,
their comparison, and a gamma study."""
from gridworld import GridworldMDP, show_policy, show_values
from mdp import value_iteration, greedy_policy, policy_iteration, q_value


def main():
    mdp = GridworldMDP(living_reward=-0.04, gamma=1.0)

    print("=" * 52)
    print(" Value iteration")
    print("=" * 52)
    V_vi, it_vi, deltas = value_iteration(mdp)
    pol_vi = greedy_policy(mdp, V_vi)
    print(f"converged after {it_vi} iterations (max. Bellman residual < 1e-8)\n")
    print("Utilities V*(s):")
    print(show_values(mdp, V_vi))
    print("\nThe optimal policy:")
    print(show_policy(mdp, pol_vi))

    print("\n" + "=" * 52)
    print(" Policy iteration")
    print("=" * 52)
    V_pi, pol_pi, it_pi = policy_iteration(mdp)
    print(f"converged after {it_pi} policy improvements\n")
    print("The optimal policy:")
    print(show_policy(mdp, pol_pi))

    # agreement
    same = all(pol_vi[s] == pol_pi[s] for s in mdp.states)
    maxdiff = max(abs(V_vi[s] - V_pi[s]) for s in mdp.states)
    print(f"\nThe same policy as value iteration: {same}")
    print(f"Max. utility difference VI vs PI: {maxdiff:.2e}")

    print("\n" + "=" * 52)
    print(" The gamma study: how discounting changes the policy")
    print("=" * 52)
    for g in (0.99, 0.9, 0.5, 0.2):
        m = GridworldMDP(living_reward=-0.04, gamma=g)
        V, it, _ = value_iteration(m)
        pol = greedy_policy(m, V)
        # an example state (2,0) at the bottom: with strong discounting, riskier/shorter routes
        print(f"\ngamma={g:<4}  (VI: {it} iterations)")
        print(show_policy(m, pol))

    print("\n" + "=" * 52)
    print(" The living_reward study: how the step costs change the policy")
    print("=" * 52)
    for r in (-0.04, -0.5, -2.0, 0.0):
        m = GridworldMDP(living_reward=r, gamma=1.0 if r < 0 else 0.99)
        V, _, _ = value_iteration(m)
        pol = greedy_policy(m, V)
        print(f"\nliving_reward={r}")
        print(show_policy(m, pol))


if __name__ == "__main__":
    main()
