# 12 — Dynamic Decision Making: Inventory Control as an MDP `[from your lectures]`

Difficulty: 🟠 Medium-Hard | Topic: Sequential Decision Making / Dynamic Programming

## 🎯 Project Goal
Formulate a single-product inventory problem as a Markov Decision Process, solve it exactly with dynamic programming, and show that the optimal policy beats sensible heuristics in simulation.

## 📊 Dataset + Evaluation Metric
- **Dataset:** None — you define the system. Suggested setting: a shop sells one product. Each day: observe stock level (0–20), decide how much to order (delivered next morning, capacity-capped), then random daily demand arrives (e.g., Poisson(4)). Economics: purchase cost 2€/unit, sale price 5€/unit, holding cost 0.1€/unit/night, unmet demand is lost (and costs goodwill, e.g., 1€/unit). Horizon: maximize long-run average or discounted profit.
- **Evaluation metric:** Average daily profit over ≥10,000 simulated days, with confidence intervals, compared across policies.

## 🏁 Success Criteria
- The problem written down formally: state space, action space, transition probabilities, reward function
- An exact DP solution (value iteration or policy iteration) with a convergence check
- ≥2 heuristic baselines (e.g., order-up-to-S, fixed order quantity) tuned fairly
- Simulation showing optimal vs heuristic performance with CIs; visualization of the optimal policy as a function of stock level
- A sensitivity analysis: how does the optimal policy shift when holding cost or demand rate changes?

Relevant techniques (look them up yourself): Markov Decision Processes, Bellman equation, value iteration, policy iteration, base-stock/(s,S) policies, Monte Carlo policy evaluation.
