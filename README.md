# Practice Labs

Übergeordnetes Repository für meine praxisorientierten Lern-Labs. Es bündelt mehrere in sich geschlossene Lab-Sammlungen zu einzelnen Themen- und Studienbereichen und dient als gemeinsamer Einstiegspunkt, gemeinsame Konvention und langfristiges Archiv des Lernfortschritts.

Ziel des Repos ist es, Theorie konsequent in eigenständig lauffähigen, nachvollziehbaren Übungen zu verankern – vom Grundkonzept bis zur praxisnahen Anwendung.

---

## Inhalt

Jede Lab-Sammlung liegt in einem eigenen Unterverzeichnis und ist unabhängig von den übrigen nutzbar. Es gibt zwei Bauweisen (siehe Abschnitt „Aufbau einer Lab-Sammlung"). Die Übersicht wird beim Hinzufügen neuer Sammlungen fortgeschrieben.

| Sammlung | Verzeichnis | Bauweise | Schwerpunkt | Umfang / Stand |
|---|---|---|---|---|
| Statistik-Intuition durch Projekte | `statistics_practice_lab/` | katalogbasiert | Statistik von Grund auf, simulationsgetrieben | 10 Phasen |
| ML/DS Practice Projects | `ds-practice-lab/` | katalogbasiert | End-to-end-ML-Projekte, Data-Science-Handwerk | 25 Projekte (2 Batches), noch nicht begonnen |
| Data Science / Machine Learning / IT-Operations | `ML_DS_ITOP_Learn-Repository/` | modulbasiert | Datenmanagement, ML, BI, IT-Operations | 21 Module, noch nicht begonnen |
| eXtended Artificial Intelligence (xtAI) | `xtAI_Learn-Repository/` | modulbasiert | KI-Methoden, NLP, Computer Vision, XR | 41 Module, in Arbeit (Modul 11) |

> Der detaillierte Stand jeder Sammlung steht in deren eigenem Fortschritts-Tracker (`PROGRESS.md`).

---

## Aufbau einer Lab-Sammlung

Beide Bauweisen teilen dieselben übergeordneten Prinzipien (siehe „Konventionen"), unterscheiden sich aber in der Gliederung.

### Katalogbasiert

Eine flache, thematisch geordnete Sammlung vieler eigenständiger Projekte. Geeignet für breites, selbstgesteuertes Üben.

- Ein zentraler **Projektkatalog** listet alle Projekte mit Schwierigkeitsgrad, Kernkonzepten, Aufgabenstellung und Datenquelle.
- Die Projekte sind in **Phasen** (bzw. Batches) organisiert, die aufeinander aufbauen; ein empfohlener **Lernpfad** gibt die Reihenfolge vor.
- Jedes Projekt liegt in einem eigenen Ordner mit eigener `README.md`.

```
<sammlung>/
├── CLAUDE.md              # Arbeits- und Tutor-Anweisungen
├── <KATALOG>.md           # Projektkatalog (Phasen, Schwierigkeit, Aufgaben)
├── PROGRESS.md            # Fortschritts-Tracker
└── <projekt-slug>/
    ├── README.md          # Aufgabenstellung und Interpretation
    └── ...                # Notebook / Skript / Daten
```

### Modulbasiert

Eine nach Studienmodulen gegliederte Sammlung. Pro Modul entstehen eine Lektion und drei aufeinander aufbauende Projekte:

- **Lektion** (`modules/NN-slug/README.md`) – progressive Erklärung von den Grundlagen bis zu fortgeschrittenen Konzepten.
- **Projekt `01-basic`** – Einstieg mit minimalem Umfang zur Festigung der Kernidee.
- **Projekt `02-medium`** – Vertiefung mit realistischeren Anforderungen.
- **Projekt `03-final`** – abschließende, praxisnahe Aufgabe, die die Modulinhalte integriert (ohne vorgegebenen Code).

```
<sammlung>/
├── CLAUDE.md              # Bau-/Arbeitsanweisungen für die Content-Erstellung
├── PROGRESS.md            # strikter Fortschritts-Tracker (einzige Wahrheit)
├── <Modulübersicht>.md    # inhaltliche Quelle (Inhalte + Qualifikationsziele)
└── modules/
    └── NN-modul-slug/
        ├── README.md      # die Lektion
        ├── daten/         # generierte Datensätze (falls nötig)
        │   └── generate_data.py
        └── projects/
            ├── 01-basic/
            ├── 02-medium/
            └── 03-final/
```

Das Format je Projekt wird bewusst gewählt und in der jeweiligen README begründet – Jupyter Notebook, Python-Skript, SQL oder ein geführter Markdown-Walkthrough, je nachdem, was das Thema am besten trägt.

---

## Konventionen

Diese Prinzipien gelten einheitlich für alle Sammlungen:

- **Lokal lauffähig zuerst.** Bevorzugt werden Standardbibliotheken und verbreitete Pakete (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `statsmodels`). Schwere oder optionale Abhängigkeiten werden in der jeweiligen README bzw. `requirements.txt` klar ausgewiesen.
- **CPU-freundlich.** Projekte sind so zugeschnitten, dass sie ohne dedizierte GPU in Sekunden bis wenigen Minuten laufen. Rechenintensive Trainings werden vermieden; wo nötig, kommen kleine Modelle, kleine Datensätze oder vortrainierte Feature-Extraktion zum Einsatz.
- **Datensätze reproduzierbar.** Wo kein passender, frei verfügbarer Datensatz sinnvoll ist, wird ein synthetischer Datensatz per Skript (`generate_data.py`, fester Seed) erzeugt und kurz dokumentiert. Große Dateien werden nicht eingecheckt (nur der Generator bzw. eine Download-Anleitung).
- **Verstehen vor Erledigen.** Die Projekte sind didaktisch aufgebaut; jedes endet mit einer schriftlichen Interpretation in eigenen Worten. Code ohne Interpretation gilt nicht als abgeschlossen.
- **Fortschritt zentral.** Der Stand einer Sammlung steht ausschließlich in ihrer `PROGRESS.md`; sie ist die einzige Fortschrittsquelle. Fertige Einheiten werden nicht ohne konkreten Anlass erneut geöffnet oder überarbeitet.
- **Sprache.** Lektionen und Erklärungen auf Deutsch, Fachbegriffe und Code auf Englisch – wie im Feld üblich.

---

## Nutzung

Repository klonen:

```bash
git clone <repository-url>
cd practice-labs
```

Eine Sammlung öffnen:

```bash
cd xtAI_Learn-Repository
```

Der Einstieg in eine Sammlung erfolgt über deren **`PROGRESS.md`** (bei modulbasierten Sammlungen über den Zeiger „Nächste Aktion"; bei katalogbasierten über den empfohlenen Lernpfad) sowie über die **`CLAUDE.md`**, die den Arbeitskontext beschreibt. Für die Ausführung wird eine Python-Umgebung mit Jupyter empfohlen; sammlungs- oder projektspezifische Abhängigkeiten sind in der zugehörigen README bzw. `requirements.txt` festgehalten.

---

## Neue Lab-Sammlung hinzufügen

1. Ein neues Unterverzeichnis mit sprechendem Namen anlegen (z. B. `web-engineering/`).
2. Bauweise wählen (katalog- oder modulbasiert) und die entsprechenden Dateien anlegen: `CLAUDE.md`, `PROGRESS.md` sowie den Projektkatalog bzw. die Modulübersicht.
3. Projekte bzw. Module gemäß der oben beschriebenen Struktur schrittweise aufbauen.
4. Die Sammlung in der Übersichtstabelle dieser README eintragen.

---

## Voraussetzungen

- Git
- Python 3 mit Jupyter
- Weitere Pakete je Sammlung/Projekt gemäß der jeweiligen README

---

## Hinweis

Dieses Repository entsteht zu Lern- und Übungszwecken. Inhalte werden fortlaufend ergänzt und sind bewusst didaktisch aufgebaut.
