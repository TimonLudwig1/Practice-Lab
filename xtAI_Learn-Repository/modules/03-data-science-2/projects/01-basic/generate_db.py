"""Generates the exercise database datasets/shop.db (SQLite) for the SQL project.

Run (from the folder 01-basic, venv active):
    python generate_db.py

Content: a fictional online shop with three tables —
    customers(customer_id, name, city, registered_on)
    products(product_id, name, category, price)
    orders(order_id, customer_id, product_id, quantity, ordered_on)

Why synthetic? For SQL basics you need a small, manageable database with known
answers (the mini checks!) and a clean schema with foreign keys. Fixed seed 7 ->
reproducible. Deliberately built in:
  - 5 customers with NO orders (for the LEFT JOIN lesson)
  - skewed order quantities and monthly patterns (interesting for GROUP BY)

Note on the numbers: the random sequence depends on the *lengths* of the lists
below, not on the strings. Keep the list lengths as they are, otherwise the
reference answers in the README (6 products above 50 EUR, "Emma Schulz" as the top
customer, 5 customers without orders) no longer hold. The customer names stay
German on purpose — they are the data of a German shop, not code.
"""
import os
import random
import sqlite3

random.seed(7)

CITIES = ["Berlin", "Hamburg", "Munich", "Cologne", "Leipzig"]
FIRST_NAMES = ["Anna", "Ben", "Clara", "David", "Emma", "Felix", "Greta", "Hannes",
               "Ida", "Jonas", "Klara", "Leon", "Mia", "Noah", "Olivia", "Paul",
               "Quirin", "Rosa", "Samuel", "Tilda"]
LAST_NAMES = ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer",
              "Wagner", "Becker", "Schulz", "Hoffmann"]
PRODUCTS = {
    "Electronics": [("Headphones", 79.99), ("Mouse", 24.99), ("Keyboard", 49.99),
                    ("Monitor", 219.00), ("Webcam", 59.99), ("USB hub", 19.99),
                    ("Charger", 29.99), ("Speakers", 89.99)],
    "Books":       [("Python Handbook", 39.95), ("Statistics Primer", 29.95),
                    ("SQL for Beginners", 24.95), ("Data Science in Practice", 44.95),
                    ("AI Overview", 19.95), ("Bestselling Novel", 12.95)],
    "Household":   [("Kettle", 34.99), ("Toaster", 44.99), ("Blender", 59.99),
                    ("Frying pan", 39.99), ("Knife set", 69.99), ("Coffee grinder", 49.99)],
    "Sports":      [("Yoga mat", 24.99), ("Dumbbells 2x5kg", 34.99), ("Skipping rope", 9.99),
                    ("Water bottle", 14.99), ("Running shirt", 29.99)],
}

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, "shop.db")
if os.path.exists(path):
    os.remove(path)

con = sqlite3.connect(path)
cur = con.cursor()
cur.executescript("""
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    city          TEXT NOT NULL,
    registered_on TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    price      REAL NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    quantity    INTEGER NOT NULL,
    ordered_on  TEXT NOT NULL
);
""")

# 60 customers
for cid in range(1, 61):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    month, day = random.randint(1, 12), random.randint(1, 28)
    cur.execute("INSERT INTO customers VALUES (?, ?, ?, ?)",
                (cid, name, random.choice(CITIES), f"2023-{month:02d}-{day:02d}"))

# 25 products
pid = 0
for category, items in PRODUCTS.items():
    for name, price in items:
        pid += 1
        cur.execute("INSERT INTO products VALUES (?, ?, ?, ?)", (pid, name, category, price))

# 800 orders in 2024 — customers 56-60 NEVER order (the LEFT JOIN lesson)
ordering_customers = list(range(1, 56))
for oid in range(1, 801):
    cid = random.choice(ordering_customers)
    product = random.randint(1, pid)
    quantity = random.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]
    # build in a December peak
    month = random.choices(range(1, 13), weights=[6, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11, 16])[0]
    day = random.randint(1, 28)
    cur.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                (oid, cid, product, quantity, f"2024-{month:02d}-{day:02d}"))

con.commit()
for table in ("customers", "products", "orders"):
    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n} rows")
con.close()
print(f"Database written: {path}")
