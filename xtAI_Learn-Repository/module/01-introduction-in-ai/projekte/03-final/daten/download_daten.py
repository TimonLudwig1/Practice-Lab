"""Laedt die SMS Spam Collection (UCI Machine Learning Repository) herunter.

Ausfuehren (venv aktiv, aus dem Ordner 03-final):
    python daten/download_daten.py

Erzeugt: daten/SMSSpamCollection  (~470 KB, 5.574 SMS, Format: label<TAB>text)
Quelle:  Almeida & Hidalgo (2011), https://archive.ics.uci.edu/dataset/228
"""
import io
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIEL_ORDNER = os.path.dirname(os.path.abspath(__file__))
ZIEL_DATEI = os.path.join(ZIEL_ORDNER, "SMSSpamCollection")

if os.path.exists(ZIEL_DATEI):
    print(f"Schon vorhanden: {ZIEL_DATEI}")
else:
    print(f"Lade {URL} ...")
    with urllib.request.urlopen(URL) as antwort:
        archiv = zipfile.ZipFile(io.BytesIO(antwort.read()))
    archiv.extract("SMSSpamCollection", ZIEL_ORDNER)
    print(f"Gespeichert: {ZIEL_DATEI}")

with open(ZIEL_DATEI, encoding="utf-8") as f:
    zeilen = f.readlines()
print(f"{len(zeilen)} SMS geladen. Erste Zeile: {zeilen[0][:60]}...")
