"""Erzeugt die Uebungs-Datenbank daten/shop.db (SQLite) fuer das SQL-Projekt.

Ausfuehren (aus dem Ordner 01-basic, venv aktiv):
    python generate_db.py

Inhalt: ein fiktiver Online-Shop mit drei Tabellen —
    kunden(kunden_id, name, stadt, registriert_am)
    produkte(produkt_id, name, kategorie, preis)
    bestellungen(bestell_id, kunden_id, produkt_id, menge, bestellt_am)

Warum synthetisch? Fuer SQL-Grundlagen braucht man eine kleine, ueberschaubare
Datenbank mit bekannten Antworten (Mini-Checks!) und einem sauberen Schema mit
Fremdschluesseln. Fester Seed 7 -> reproduzierbar. Eingebaut sind gezielt:
  - 5 Kunden OHNE Bestellungen (fuer die LEFT-JOIN-Lektion)
  - schiefe Bestellmengen und Monatsmuster (fuer GROUP BY interessant)
"""
import os
import random
import sqlite3

random.seed(7)

STAEDTE = ["Berlin", "Hamburg", "Muenchen", "Koeln", "Leipzig"]
VORNAMEN = ["Anna", "Ben", "Clara", "David", "Emma", "Felix", "Greta", "Hannes",
            "Ida", "Jonas", "Klara", "Leon", "Mia", "Noah", "Olivia", "Paul",
            "Quirin", "Rosa", "Samuel", "Tilda"]
NACHNAMEN = ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer",
             "Wagner", "Becker", "Schulz", "Hoffmann"]
PRODUKTE = {
    "Elektronik": [("Kopfhoerer", 79.99), ("Maus", 24.99), ("Tastatur", 49.99),
                   ("Monitor", 219.00), ("Webcam", 59.99), ("USB-Hub", 19.99),
                   ("Ladegeraet", 29.99), ("Lautsprecher", 89.99)],
    "Buecher":    [("Python-Handbuch", 39.95), ("Statistik-Grundkurs", 29.95),
                   ("SQL fuer Einsteiger", 24.95), ("Data-Science-Praxis", 44.95),
                   ("KI-Ueberblick", 19.95), ("Roman Bestseller", 12.95)],
    "Haushalt":   [("Wasserkocher", 34.99), ("Toaster", 44.99), ("Mixer", 59.99),
                   ("Pfanne", 39.99), ("Messerset", 69.99), ("Kaffeemuehle", 49.99)],
    "Sport":      [("Yogamatte", 24.99), ("Hanteln 2x5kg", 34.99), ("Springseil", 9.99),
                   ("Trinkflasche", 14.99), ("Laufshirt", 29.99)],
}

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daten")
os.makedirs(ordner, exist_ok=True)
pfad = os.path.join(ordner, "shop.db")
if os.path.exists(pfad):
    os.remove(pfad)

con = sqlite3.connect(pfad)
cur = con.cursor()
cur.executescript("""
CREATE TABLE kunden (
    kunden_id      INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    stadt          TEXT NOT NULL,
    registriert_am TEXT NOT NULL
);
CREATE TABLE produkte (
    produkt_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    kategorie  TEXT NOT NULL,
    preis      REAL NOT NULL
);
CREATE TABLE bestellungen (
    bestell_id  INTEGER PRIMARY KEY,
    kunden_id   INTEGER NOT NULL REFERENCES kunden(kunden_id),
    produkt_id  INTEGER NOT NULL REFERENCES produkte(produkt_id),
    menge       INTEGER NOT NULL,
    bestellt_am TEXT NOT NULL
);
""")

# 60 Kunden
for kid in range(1, 61):
    name = f"{random.choice(VORNAMEN)} {random.choice(NACHNAMEN)}"
    monat, tag = random.randint(1, 12), random.randint(1, 28)
    cur.execute("INSERT INTO kunden VALUES (?, ?, ?, ?)",
                (kid, name, random.choice(STAEDTE), f"2023-{monat:02d}-{tag:02d}"))

# 25 Produkte
pid = 0
for kategorie, artikel in PRODUKTE.items():
    for name, preis in artikel:
        pid += 1
        cur.execute("INSERT INTO produkte VALUES (?, ?, ?, ?)", (pid, name, kategorie, preis))

# 800 Bestellungen in 2024 — Kunden 56-60 bestellen NIE (LEFT-JOIN-Lektion)
bestellende_kunden = list(range(1, 56))
for bid in range(1, 801):
    kid = random.choice(bestellende_kunden)
    produkt = random.randint(1, pid)
    menge = random.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]
    # Dezember-Peak einbauen
    monat = random.choices(range(1, 13), weights=[6, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11, 16])[0]
    tag = random.randint(1, 28)
    cur.execute("INSERT INTO bestellungen VALUES (?, ?, ?, ?, ?)",
                (bid, kid, produkt, menge, f"2024-{monat:02d}-{tag:02d}"))

con.commit()
for tabelle in ("kunden", "produkte", "bestellungen"):
    n = cur.execute(f"SELECT COUNT(*) FROM {tabelle}").fetchone()[0]
    print(f"{tabelle}: {n} Zeilen")
con.close()
print(f"Datenbank geschrieben: {pfad}")
