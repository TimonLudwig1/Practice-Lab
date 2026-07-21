"""Generates the deliberately dirty order data set for the cleaning project.

Run (from the folder 02-medium, venv active):
    python generate_data.py        ->  datasets/orders_raw.csv

What the data represent: 500 orders of a fictional online shop
(id, date, city, category, price, quantity, customer age).

Built-in problems (all deliberate, all realistic):
  1. price as text with a comma and a currency word ("49,99 EUR")  -> type conversion
  2. 12 missing prices (empty field)                               -> NaN handling
  3. 3 price outliers (decimal-point error: factor 100)            -> IQR rule
  4. special code -999 in the customer age (= "no answer", 25x)    -> special codes
  5. 1 impossible age (234)                                        -> plausibility
  6. inconsistent city names (berlin / Berlin /  Berlin  /
     Munich / München ...)                                         -> text normalisation
  7. two date formats mixed (2024-03-14 and 14.03.2024)            -> to_datetime
  8. 15 exact duplicate rows                                       -> drop_duplicates

Why synthetic? So that every problem occurs EXACTLY ONCE in a controlled form and
you can check at the end whether you found them all (the "truth" is in this
script). Fixed seed -> reproducible.

Note on the numbers: the random sequence depends on the *lengths* of the lists
below, not on the strings themselves. Keep the list lengths as they are, otherwise
the reference figures in the README (12 missing prices, median about 71.6, 26 NaN
ages) no longer hold.
"""
import csv
import os
import random

random.seed(42)

CITIES = {
    "Berlin":   ["Berlin", "berlin", " Berlin ", "BERLIN"],
    "Munich":   ["Munich", "München", "munich"],
    "Hamburg":  ["Hamburg", "hamburg", "Hamburg "],
    "Cologne":  ["Cologne", "Köln", "cologne"],
    "Leipzig":  ["Leipzig", "leipzig"],
}
CATEGORIES = ["Electronics", "Books", "Clothing", "Household"]
PRICE_RANGE = {"Electronics": (20, 400), "Books": (5, 60), "Clothing": (10, 120), "Household": (8, 200)}

rows = []
for i in range(500):
    category = random.choice(CATEGORIES)
    lo, hi = PRICE_RANGE[category]
    price = round(random.uniform(lo, hi), 2)
    city_norm = random.choice(list(CITIES))
    city = random.choice(CITIES[city_norm])
    month, day = random.randint(1, 12), random.randint(1, 28)
    if random.random() < 0.5:
        date = f"2024-{month:02d}-{day:02d}"
    else:
        date = f"{day:02d}.{month:02d}.2024"
    age = random.randint(18, 79)
    rows.append({
        "order_id": 10000 + i,
        "date": date,
        "city": city,
        "category": category,
        "price": f"{price:.2f}".replace(".", ",") + " EUR",
        "quantity": random.randint(1, 5),
        "customer_age": age,
    })

# Problem 2: 12 missing prices
for idx in random.sample(range(500), 12):
    rows[idx]["price"] = ""

# Problem 3: 3 outliers (decimal-point error, factor 100) — only where a price exists
candidates = [i for i, r in enumerate(rows) if r["price"]]
for idx in random.sample(candidates, 3):
    value = float(rows[idx]["price"].replace(" EUR", "").replace(",", "."))
    rows[idx]["price"] = f"{value * 100:.2f}".replace(".", ",") + " EUR"

# Problem 4: special code -999 (no answer) in the age, 25x
for idx in random.sample(range(500), 25):
    rows[idx]["customer_age"] = -999

# Problem 5: one impossible age
rows[random.randrange(500)]["customer_age"] = 234

# Problem 8: append 15 exact duplicates
rows += [dict(rows[idx]) for idx in random.sample(range(500), 15)]
random.shuffle(rows)

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, "orders_raw.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} rows written to {path}")
