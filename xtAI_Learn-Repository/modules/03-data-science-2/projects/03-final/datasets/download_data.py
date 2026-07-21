"""Laedt den Bike-Sharing-Datensatz (UCI) herunter — hier brauchen wir day.csv.

Ausfuehren (venv aktiv, aus dem Ordner 03-final):
    python datasets/download_data.py

(Derselbe Datensatz wie im Final-Projekt von Modul 02 — falls du den schon geladen
hast, kannst du auch einfach die day.csv von dort hierher kopieren.)
"""
import io
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ZIEL_ORDNER = os.path.dirname(os.path.abspath(__file__))
ZIEL = os.path.join(ZIEL_ORDNER, "day.csv")

if os.path.exists(ZIEL):
    print("Schon vorhanden — nichts zu tun.")
else:
    print(f"Lade {URL} ...")
    with urllib.request.urlopen(URL) as antwort:
        archiv = zipfile.ZipFile(io.BytesIO(antwort.read()))
    archiv.extract("day.csv", ZIEL_ORDNER)
    print(f"Gespeichert: {ZIEL}")

with open(ZIEL) as f:
    print(f"day.csv: {sum(1 for _ in f) - 1} Datenzeilen (Tage)")
