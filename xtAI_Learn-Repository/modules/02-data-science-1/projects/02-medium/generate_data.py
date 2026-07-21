"""Erzeugt den absichtlich verschmutzten Bestell-Datensatz fuer das Bereinigungsprojekt.

Ausfuehren (aus dem Ordner 02-medium, venv aktiv):
    python generate_data.py        ->  datasets/bestellungen_roh.csv

Was die Daten darstellen: 500 Bestellungen eines fiktiven Online-Shops
(ID, Datum, Stadt, Kategorie, Preis, Menge, Kundenalter).

Eingebaute Probleme (alle absichtlich, alle realistisch):
  1. Preis als Text mit Komma und Euro-Zeichen ("49,99 EUR")     -> Typkonvertierung
  2. 12 fehlende Preise (leeres Feld)                            -> NaN-Behandlung
  3. 3 Preis-Ausreisser (Kommafehler: Faktor 100)                -> IQR-Regel
  4. Sondercode -999 im Kundenalter (= "keine Angabe", 25x)      -> Sondercodes
  5. 1 unmoegliches Alter (234)                                  -> Plausibilitaet
  6. Staedtenamen inkonsistent (berlin / Berlin /  Berlin  /
     Muenchen / München ...)                                     -> Textnormalisierung
  7. Zwei Datumsformate gemischt (2024-03-14 und 14.03.2024)     -> to_datetime
  8. 15 exakte Duplikat-Zeilen                                   -> drop_duplicates

Warum synthetisch? Damit jedes Problem GENAU EINMAL in kontrollierter Form
vorkommt und du am Ende pruefen kannst, ob du alle gefunden hast (die "Wahrheit"
steht in diesem Skript). Fester Seed -> reproduzierbar.
"""
import csv
import os
import random

random.seed(42)

STAEDTE = {
    "Berlin":    ["Berlin", "berlin", " Berlin ", "BERLIN"],
    "Muenchen":  ["Muenchen", "München", "muenchen"],
    "Hamburg":   ["Hamburg", "hamburg", "Hamburg "],
    "Koeln":     ["Koeln", "Köln", "koeln"],
    "Leipzig":   ["Leipzig", "leipzig"],
}
KATEGORIEN = ["Elektronik", "Buecher", "Kleidung", "Haushalt"]
PREIS_BEREICH = {"Elektronik": (20, 400), "Buecher": (5, 60), "Kleidung": (10, 120), "Haushalt": (8, 200)}

zeilen = []
for i in range(500):
    kategorie = random.choice(KATEGORIEN)
    lo, hi = PREIS_BEREICH[kategorie]
    preis = round(random.uniform(lo, hi), 2)
    stadt_norm = random.choice(list(STAEDTE))
    stadt = random.choice(STAEDTE[stadt_norm])
    monat, tag = random.randint(1, 12), random.randint(1, 28)
    if random.random() < 0.5:
        datum = f"2024-{monat:02d}-{tag:02d}"
    else:
        datum = f"{tag:02d}.{monat:02d}.2024"
    alter = random.randint(18, 79)
    zeilen.append({
        "bestell_id": 10000 + i,
        "datum": datum,
        "stadt": stadt,
        "kategorie": kategorie,
        "preis": f"{preis:.2f}".replace(".", ",") + " EUR",
        "menge": random.randint(1, 5),
        "kunden_alter": alter,
    })

# Problem 2: 12 fehlende Preise
for idx in random.sample(range(500), 12):
    zeilen[idx]["preis"] = ""

# Problem 3: 3 Ausreisser (Kommafehler, Faktor 100) — nur bei vorhandenen Preisen
kandidaten = [i for i, z in enumerate(zeilen) if z["preis"]]
for idx in random.sample(kandidaten, 3):
    wert = float(zeilen[idx]["preis"].replace(" EUR", "").replace(",", "."))
    zeilen[idx]["preis"] = f"{wert * 100:.2f}".replace(".", ",") + " EUR"

# Problem 4: Sondercode -999 (keine Angabe) im Alter, 25x
for idx in random.sample(range(500), 25):
    zeilen[idx]["kunden_alter"] = -999

# Problem 5: ein unmoegliches Alter
zeilen[random.randrange(500)]["kunden_alter"] = 234

# Problem 8: 15 exakte Duplikate anhaengen
zeilen += [dict(zeilen[idx]) for idx in random.sample(range(500), 15)]
random.shuffle(zeilen)

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(ordner, exist_ok=True)
pfad = os.path.join(ordner, "bestellungen_roh.csv")
with open(pfad, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()))
    writer.writeheader()
    writer.writerows(zeilen)

print(f"{len(zeilen)} Zeilen geschrieben nach {pfad}")
