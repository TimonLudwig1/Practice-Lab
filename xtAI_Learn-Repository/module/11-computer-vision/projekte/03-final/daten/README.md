# daten/

Nimmt den EuroSAT-Datensatz auf (nicht eingecheckt, siehe `.gitignore`).

**EuroSAT** (27 000 Sentinel-2-Satellitenbilder, 64×64 RGB, 10 Landnutzungsklassen) wird
von `data.py` beim ersten Lauf automatisch via torchvision nach `eurosat/` geladen
(~90 MB, schnell).
