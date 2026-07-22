"""Load Tatoeba DE-EN + tokenization (given)."""
import os
import io
import re
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")
RAW = os.path.join(DATA_DIR, "deu.txt")
URL = "https://www.manythings.org/anki/deu-eng.zip"


def load_pairs():
    """List of (german, english) — we translate DE -> EN."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW):
        print("Downloading Tatoeba DE-EN ...")
        hdr = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.9",
               "Referer": "https://www.manythings.org/anki/"}
        raw = urllib.request.urlopen(urllib.request.Request(URL, headers=hdr), timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open("deu.txt") as src, open(RAW, "wb") as dst:
                dst.write(src.read())
    lines = open(RAW, encoding="utf-8").read().strip().split("\n")
    pairs = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[1], parts[0]))   # (german, english)
    return pairs


def tokenize(s):
    """Lowercase, words + punctuation as tokens."""
    return re.findall(r"[a-zäöüßA-ZÄÖÜ]+|[.!?,]", s.lower())
