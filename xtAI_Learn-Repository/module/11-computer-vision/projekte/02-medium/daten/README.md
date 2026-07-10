# daten/

Nimmt den EuroSAT-Datensatz und den Feature-Cache auf (nicht eingecheckt, siehe
`.gitignore`).

- `eurosat/` — EuroSAT (27 000 Sentinel-2-Satellitenbilder, 64×64 RGB, 10 Klassen),
  automatischer Download durch `data.py` beim ersten Lauf (~90 MB).
- `feat_cache.npz` — die extrahierten pretrained Features (Neuaufbau mit
  `python run.py --rebuild`).
