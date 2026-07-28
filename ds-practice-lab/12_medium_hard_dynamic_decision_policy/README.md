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

---

# Deutsche Übersetzung

# 12 — Dynamische Entscheidungen: Bestandssteuerung als MDP `[aus deinen Vorlesungen]`

Schwierigkeit: 🟠 Mittel bis anspruchsvoll | Thema: Sequenzielle Entscheidungen und dynamische Programmierung

## 🎯 Projektziel
Formuliere ein Bestandsproblem für ein einzelnes Produkt als Markov-Entscheidungsprozess, löse es exakt mit dynamischer Programmierung und zeige in einer Simulation, dass die optimale Strategie sinnvolle Heuristiken übertrifft.

## 📊 Datensatz und Bewertungsmetrik
- **Datensatz:** Kein externer Datensatz; du definierst das System. Beispiel: Ein Geschäft verkauft ein Produkt. Jeden Tag wird der Bestand von 0 bis 20 beobachtet, eine Bestellmenge gewählt, die am nächsten Morgen bis zur Kapazitätsgrenze eintrifft, und anschließend eine zufällige Nachfrage, etwa Poisson(4), realisiert. Annahmen: Einkauf 2 € je Einheit, Verkauf 5 €, Lagerkosten 0,10 € je Einheit und Nacht sowie Kosten von beispielsweise 1 € je nicht erfüllter Einheit. Maximiere den langfristigen durchschnittlichen oder abgezinsten Gewinn.
- **Bewertungsmetrik:** Mittlerer Tagesgewinn über mindestens 10.000 simulierte Tage mit Konfidenzintervallen und Vergleich mehrerer Strategien.

## 🏁 Erfolgskriterien
- Formale Beschreibung von Zustandsraum, Aktionsraum, Übergangswahrscheinlichkeiten und Belohnungsfunktion
- Exakte DP-Lösung durch Wert- oder Strategieiteration mit Konvergenzprüfung
- Mindestens zwei fair abgestimmte heuristische Baselines, beispielsweise Auffüllen bis S oder eine feste Bestellmenge
- Simulationsvergleich der optimalen und heuristischen Strategien mit Konfidenzintervallen sowie Visualisierung der optimalen Aktion in Abhängigkeit vom Lagerbestand
- Sensitivitätsanalyse dazu, wie sich die optimale Strategie bei veränderten Lagerkosten oder Nachfrageraten verschiebt

Relevante Verfahren zum selbstständigen Nachschlagen: Markov-Entscheidungsprozesse, Bellman-Gleichung, Wertiteration, Strategieiteration, Base-Stock- und (s,S)-Strategien sowie Monte-Carlo-Strategiebewertung.
