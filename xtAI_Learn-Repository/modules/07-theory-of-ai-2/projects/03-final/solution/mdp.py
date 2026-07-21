"""Solution procedures for MDPs: value iteration and policy iteration.

Both implement the Bellman equations from part 3 of the script.
"""


def q_value(mdp, s, a, V):
    """Q(s,a) = sum_{s'} P(s'|s,a) * V(s')   (the expectation term)."""
    return sum(p * V[s2] for p, s2 in mdp.transitions(s, a))


def bellman_update(mdp, s, V):
    """(BV)(s) = R(s) + gamma * max_a Q(s,a).   Terminals: only R(s)."""
    if mdp.is_terminal(s):
        return mdp.reward(s)
    return mdp.reward(s) + mdp.gamma * max(q_value(mdp, s, a, V) for a in mdp.actions(s))


# ====================================================================
#  Value iteration
# ====================================================================
def value_iteration(mdp, epsilon=1e-8, max_iter=10000):
    """Iterates the Bellman optimality operator to convergence.
    Returns: (V, iterations, the history of deltas)."""
    V = {s: 0.0 for s in mdp.states}
    for s in mdp.terminals:
        V[s] = mdp.reward(s)
    deltas = []
    for it in range(1, max_iter + 1):
        V_new = {}
        delta = 0.0
        for s in mdp.states:
            V_new[s] = bellman_update(mdp, s, V)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        deltas.append(delta)
        if delta < epsilon:
            return V, it, deltas
    return V, max_iter, deltas


def greedy_policy(mdp, V):
    """The policy read off greedily from V: argmax_a Q(s,a)."""
    policy = {}
    for s in mdp.states:
        if mdp.is_terminal(s):
            policy[s] = None
        else:
            policy[s] = max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, V))
    return policy


# ====================================================================
#  Policy iteration
# ====================================================================
def policy_evaluation(mdp, policy, V=None, k=200, epsilon=1e-10):
    """Iterative policy evaluation: solves V^pi(s)=R(s)+gamma*Q(s,pi(s))
    approximately by repeated application (up to k sweeps or convergence)."""
    if V is None:
        V = {s: 0.0 for s in mdp.states}
        for s in mdp.terminals:
            V[s] = mdp.reward(s)
    for _ in range(k):
        delta = 0.0
        V_new = {}
        for s in mdp.states:
            if mdp.is_terminal(s):
                V_new[s] = mdp.reward(s)
            else:
                V_new[s] = mdp.reward(s) + mdp.gamma * q_value(mdp, s, policy[s], V)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < epsilon:
            break
    return V


def policy_iteration(mdp, max_iter=1000):
    """Alternates policy evaluation and policy improvement until stability.
    Returns: (V, policy, iterations)."""
    # an arbitrary initial policy (the first action)
    policy = {s: (None if mdp.is_terminal(s) else mdp.actions(s)[0]) for s in mdp.states}
    V = {s: 0.0 for s in mdp.states}
    for s in mdp.terminals:
        V[s] = mdp.reward(s)
    for it in range(1, max_iter + 1):
        V = policy_evaluation(mdp, policy, V)
        stable = True
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
            best = max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, V))
            if best != policy[s] and \
               q_value(mdp, s, best, V) > q_value(mdp, s, policy[s], V) + 1e-12:
                policy[s] = best
                stable = False
        if stable:
            return V, policy, it
    return V, policy, max_iter
