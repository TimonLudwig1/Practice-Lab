"""Laedt den Bike-Sharing-Datensatz (UCI Machine Learning Repository) herunter.

Ausfuehren (venv aktiv, aus dem Ordner 03-final):
    python daten/download_daten.py

Erzeugt: daten/hour.csv (17.379 Stundenwerte) und daten/day.csv (731 Tageswerte)
Inhalt:  Fahrradverleih-Zahlen des Capital-Bikeshare-Systems, Washington D.C.,
         2011-2012, mit Wetter- und Kalendermerkmalen.
Quelle:  Fanaee-T & Gama (2013), https://archive.ics.uci.edu/dataset/275

WICHTIG (steht auch im Dataset-Readme): temp/atemp/hum/windspeed sind normiert —
temp*41 = Grad Celsius, atemp*50 = gefuehlte Temp., hum*100 = %, windspeed*67 = km/h.
"""
import io
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ZIEL_ORDNER = os.path.dirname(os.path.abspath(__file__))

if all(os.path.exists(os.path.join(ZIEL_ORDNER, f)) for f in ("hour.csv", "day.csv")):
    print("Schon vorhanden — nichts zu tun.")
else:
    print(f"Lade {URL} ...")
    with urllib.request.urlopen(URL) as antwort:
        archiv = zipfile.ZipFile(io.BytesIO(antwort.read()))
    for name in ("hour.csv", "day.csv"):
        archiv.extract(name, ZIEL_ORDNER)
        print(f"Gespeichert: {os.path.join(ZIEL_ORDNER, name)}")

for name in ("hour.csv", "day.csv"):
    with open(os.path.join(ZIEL_ORDNER, name)) as f:
        print(f"{name}: {sum(1 for _ in f) - 1} Datenzeilen")
