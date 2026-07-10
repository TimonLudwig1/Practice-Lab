# PROGRESS — Statistik-Lernbegleiter

> Fortschritts-Tracker für alle Projekte aus `STATISTIK_PROJEKTE.md`.
> Status-Legende: ⬜ offen · 🟨 in Arbeit · ✅ abgeschlossen
> "Abgeschlossen" heißt: Notebook fertig **und** eigene schriftliche Interpretation vorhanden (Regel Nr. 2).
>
> **Aktueller Modus (ab 2026-07-09):** Bau-Phase — Timon lässt zuerst alle Projekt-Notebooks als vollständige Musterlösungen bauen und arbeitet sie danach durch. Ein 🟨 heißt hier "Notebook gebaut & ausgeführt, wartet auf Durcharbeiten (Interpretation + Quiz)". Erst nach dem Durcharbeiten wird ✅ gesetzt.

## Setup-Status

| Element | Status | Notiz |
|---|---|---|
| `PROGRESS.md` | ✅ | Diese Datei — angelegt 2026-07-09 |
| `requirements.txt` | ✅ | Echte Versionen (Python 3.13.9); pymc/lifelines optional, später |
| `.gitignore` | ✅ | Python/Jupyter/OS; Notebook-Ausgaben bewusst versioniert |
| `utils/` (gemeinsame Funktionen) | ✅ | `make_rng`, `apply_house_style`, `add_truth_line`, `plot_null_distribution`, `coverage_simulation` — Smoke-Test bestanden |
| `projekte/` (leerer Ordner) | ✅ | Bereit für erstes Projekt |

---

## Phase 1 — Deskriptive Statistik & EDA
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 1.1 Lügendetektor für Mittelwerte ⭐ | 🟨 | 2026-07-09 | Notebook fertig & ausgeführt; wartet auf deine Interpretation in NOTIZEN.md |
| 1.2 Streuung zum Anfassen ⭐ | 🟨 | 2026-07-09 | Notebook gebaut & ausgeführt (0 Fehler); wartet auf Durcharbeiten |
| 1.3 Anscombe & Datasaurus ⭐⭐ | ⬜ | — | — |
| 1.4 EDA schmutziger Datensatz ⭐⭐ | ⬜ | — | — |
| 1.5 Quantile & Boxplots von Hand ⭐⭐ | ⬜ | — | — |
| 1.6 z-Score-Portal ⭐⭐ | ⬜ | — | — |

## Phase 2 — Wahrscheinlichkeit & Simulation
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 2.1 Monte-Carlo: π schätzen ⭐ | ⬜ | — | — |
| 2.2 Geburtstagsparadoxon ⭐⭐ | ⬜ | — | — |
| 2.3 Monty Hall ⭐⭐ | ⬜ | — | — |
| 2.4 Bayes: Medizintest ⭐⭐ | ⬜ | — | — |
| 2.5 Zufallsvariablen-Zoo ⭐⭐ | ⬜ | — | — |
| 2.6 Gambler's Ruin & Random Walks ⭐⭐⭐ | ⬜ | — | — |
| 2.7 Simpson's Paradox ⭐⭐⭐ | ⬜ | — | — |

## Phase 3 — Verteilungen
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 3.1 Verteilungs-Bestiarium ⭐⭐ | ⬜ | — | — |
| 3.2 Poisson in der echten Welt ⭐⭐ | ⬜ | — | — |
| 3.3 Normalverteilung entzaubern ⭐⭐ | ⬜ | — | — |
| 3.4 Fat Tails ⭐⭐⭐ | ⬜ | — | — |
| 3.5 Maximum-Likelihood von Hand ⭐⭐⭐ | ⬜ | — | — |
| 3.6 Ordnungsstatistiken & Extremwerte ⭐⭐⭐ | ⬜ | — | — |

## Phase 4 — Stichproben & ZGS
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 4.1 ZGS zum Anfassen ⭐⭐ | ⬜ | — | — |
| 4.2 SE ≠ SD ⭐⭐ | ⬜ | — | — |
| 4.3 Stichprobenziehung / Selbstbelügen ⭐⭐⭐ | ⬜ | — | — |
| 4.4 Survivorship Bias / Bomber ⭐⭐ | ⬜ | — | — |

## Phase 5 — Konfidenzintervalle & Schätzung
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 5.1 Was ein 95%-KI wirklich bedeutet ⭐⭐ | ⬜ | — | — |
| 5.2 KI-Kochbuch ⭐⭐⭐ | ⬜ | — | — |
| 5.3 Wahlumfragen nachbauen ⭐⭐⭐ | ⬜ | — | — |

## Phase 6 — Hypothesentests
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 6.1 Dame mit dem Tee ⭐⭐ | ⬜ | — | — |
| 6.2 Permutationstests ⭐⭐⭐ | ⬜ | — | — |
| 6.3 t-Test-Trilogie ⭐⭐⭐ | ⬜ | — | — |
| 6.4 Fehler-Fabrik (α, β, Power) ⭐⭐⭐ | ⬜ | — | — |
| 6.5 Power-Analyse ⭐⭐⭐ | ⬜ | — | — |
| 6.6 Chi-Quadrat ⭐⭐⭐ | ⬜ | — | — |
| 6.7 ANOVA ⭐⭐⭐ | ⬜ | — | — |
| 6.8 Nichtparametrische Tests ⭐⭐⭐ | ⬜ | — | — |
| 6.9 Multiple Testing ⭐⭐⭐ | ⬜ | — | — |
| 6.10 p-Hacking-Simulator ⭐⭐⭐⭐ | ⬜ | — | — |

## Phase 7 — A/B-Testing & Experimentaldesign
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 7.1 A/B-Test-Framework ⭐⭐⭐⭐ | ⬜ | — | — |
| 7.2 Peeking ⭐⭐⭐ | ⬜ | — | — |
| 7.3 Metriken-Weisheit ⭐⭐⭐ | ⬜ | — | — |
| 7.4 Bayesianisches A/B-Testing ⭐⭐⭐⭐ | ⬜ | — | — |
| 7.5 Randomisierung & Blockdesign ⭐⭐⭐ | ⬜ | — | — |

## Phase 8 — Korrelation, Regression & Kausalität
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 8.1 Korrelations-Werkstatt ⭐⭐ | ⬜ | — | — |
| 8.2 Einfache lineare Regression ⭐⭐⭐ | ⬜ | — | — |
| 8.3 Regressionsdiagnostik ⭐⭐⭐ | ⬜ | — | — |
| 8.4 Multiple Regression & Confounder ⭐⭐⭐⭐ | ⬜ | — | — |
| 8.5 Multikollinearität ⭐⭐⭐ | ⬜ | — | — |
| 8.6 Logistische Regression ⭐⭐⭐⭐ | ⬜ | — | — |
| 8.7 Regression zur Mitte ⭐⭐⭐ | ⬜ | — | — |
| 8.8 Kausal-Inferenz-Einstieg ⭐⭐⭐⭐⭐ | ⬜ | — | — |
| 8.9 Instrumentvariablen ⭐⭐⭐⭐⭐ | ⬜ | — | — |

## Phase 9 — Bootstrap, Resampling & Bayes
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 9.1 Bootstrap von Grund auf ⭐⭐⭐ | ⬜ | — | — |
| 9.2 Bootstrap in der Praxis ⭐⭐⭐⭐ | ⬜ | — | — |
| 9.3 Bayes mit Gittern ⭐⭐⭐ | ⬜ | — | — |
| 9.4 Bayes vs. Frequentismus ⭐⭐⭐⭐ | ⬜ | — | — |
| 9.5 Naive Bayes Spam-Filter ⭐⭐⭐ | ⬜ | — | — |
| 9.6 MCMC / Metropolis von Hand ⭐⭐⭐⭐⭐ | ⬜ | — | — |

## Phase 10 — Zeitreihen, Survival & Capstones
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| 10.1 Zeitreihen-Anatomie ⭐⭐⭐ | ⬜ | — | — |
| 10.2 Forecasting-Basics ⭐⭐⭐⭐ | ⬜ | — | — |
| 10.3 Survival Analysis ⭐⭐⭐⭐ | ⬜ | — | — |
| 10.4 MCAR/MAR/MNAR ⭐⭐⭐⭐ | ⬜ | — | — |
| 10.5 Bias-Varianz für ML ⭐⭐⭐⭐ | ⬜ | — | — |
| 10.6 Unsicherheit in ML-Metriken ⭐⭐⭐⭐ | ⬜ | — | — |

## Capstones
| Projekt | Status | Datum | Größte Erkenntnis (1 Satz) |
|---|---|---|---|
| C1 Kompletter Analyse-Zyklus ⭐⭐⭐⭐⭐ | ⬜ | — | — |
| C2 Statistik-Mythbusters ⭐⭐⭐⭐⭐ | ⬜ | — | — |
| C3 Simulations-Bibliothek `statlab` ⭐⭐⭐⭐⭐ | ⬜ | — | — |
| C4 "Erklär's mir"-Sammlung ⭐⭐⭐⭐ | ⬜ | — | — |
| C5 Echte Daten, echtes Chaos ⭐⭐⭐⭐⭐ | ⬜ | — | — |

---

## Session-Log
- **2026-07-09** — Projekt-Setup gestartet. `CLAUDE.md` und `STATISTIK_PROJEKTE.md` gelesen, `PROGRESS.md` angelegt.
- **2026-07-09** — Setup abgeschlossen: `requirements.txt`, `.gitignore`, `utils/` (+ `projekte/`). Umgebung geprüft (Python 3.13.9, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, statsmodels 0.14.5, seaborn 0.13.2, scikit-learn 1.8.0, jupyter). `pymc`/`lifelines` noch nicht installiert (erst Phase 9 bzw. 10.3). utils-Smoke-Test bestanden.
- **2026-07-09** — Projekt 1.1 gebaut. `projekte/1_1_luegendetektor_mittelwerte/` mit `data/generate_data.py` (3 Gehalts-DGPs), `notebook.ipynb` (30 Zellen, ausgeführt, 0 Fehler), `NOTIZEN.md`. Hand-mean/median/mode/skew gegen NumPy/SciPy verifiziert; Stretch `honest_measure()`.
- **2026-07-09** — Bau-Modus vereinbart: erst alle Notebooks bauen, dann durcharbeiten. Projekt 1.2 gebaut. `projekte/1_2_streuung_zum_anfassen/` mit `data/generate_data.py` (2 Kaffeemaschinen, gleicher Mittelwert, σ=5 vs. 15), `notebook.ipynb` (26 Zellen, ausgeführt, 0 Fehler), `NOTIZEN.md`. Hand-range/var/std/IQR/MAD verifiziert; Überlauf simuliert vs. Formel (A 0,17% vs. B 17%); Stretch Ausreißer-Robustheit.
- **2026-07-09 — SESSION-ENDE.** Für heute Schluss nach 1.2. Stand: Setup komplett (requirements.txt, .gitignore, utils/, projekte/); Projekte **1.1 und 1.2 gebaut & ausgeführt** (Status 🟨 = gebaut, noch nicht durchgearbeitet). Keine offenen Fehler, keine halbfertigen Dateien.

---

## 👉 HIER WEITERMACHEN (nächste Session)

**Modus:** Bau-Phase — vollständige Musterlösungs-Notebooks der Reihe nach bauen (siehe Muster von 1.1/1.2). Durcharbeiten (Interpretation in NOTIZEN.md + Quiz) kommt später separat.

**Nächstes Projekt: 1.3 Anscombe & der Datasaurus** (Phase 1). Ablauf pro Projekt bewährt:
1. Ordner `projekte/1_3_anscombe_datasaurus/data/` anlegen
2. `data/generate_data.py` (bzw. Daten laden) → kurz testen
3. Notebook via Build-Skript in `$CLAUDE_JOB_DIR/tmp` mit nbformat bauen
4. `python3 -m nbconvert --to notebook --execute --inplace notebook.ipynb` (jupyter-CLI ist NICHT im PATH → immer `python3 -m nbconvert` nutzen)
5. Auf 0 Fehlerzellen prüfen, `NOTIZEN.md` anlegen, PROGRESS.md updaten

**⚠️ Offener Punkt für 1.3:** `seaborn.load_dataset('anscombe')` und der Datasaurus-Datensatz brauchen ggf. Internet. Beim letzten Versuch wurde der Konnektivitäts-Test abgebrochen (nicht wg. Fehler). Zuerst prüfen, ob die Downloads offline gehen — falls nicht, laut CLAUDE.md realistischen Ersatz generieren (Anscombe-Werte sind bekannt/hartcodierbar; Datasaurus notfalls per simulated annealing nachbauen — ist ohnehin das Stretch Goal) und das im Notebook klar kennzeichnen.

**Konventionen (etabliert):** Deutsch als Arbeitssprache, Code/Docstrings englisch. Seed 42 via `utils.make_rng`. Plot-Titel formulieren die Aussage. Statistik-Kern immer von Hand + Verifikation gegen Bibliothek. utils/ nur Infrastruktur, keine Stat-Kernkonzepte.
