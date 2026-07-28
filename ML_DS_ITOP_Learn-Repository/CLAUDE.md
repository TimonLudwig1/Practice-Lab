# CLAUDE.md — Lern-Content-Generator: Data Science, Machine Learning & IT & Operations

Dieses Repository ist eine **Lernwerkstatt**. Deine Aufgabe als Claude Code ist es,
für jedes Modul des Studienfachs *Digital Business & Data Science* (JMU Würzburg)
selbstständig hochwertiges Lernmaterial zu erzeugen: eine erklärende Lektion plus drei
aufeinander aufbauende Lernprojekte.

Der Nutzer wird mit teuren Modellen (Fable / Opus 4.8) arbeiten. **Token-Effizienz ist
oberstes Gebot.** Halte dich strikt an die Regeln unten — besonders an die Tracker- und
"Nicht erneut anschauen"-Regeln.

---

## 0. Die 7 wichtigsten Regeln (immer gültig)

1. **Session-Start = nur `PROGRESS.md` lesen.** Diese `CLAUDE.md` ist bereits im Kontext.
   Kein `ls -R`, kein Repo-Scan, keine fertigen Ordner öffnen.
2. **Ein Task pro Arbeitszyklus.** Erledige genau die *eine* "Nächste Aktion" aus
   `PROGRESS.md`, aktualisiere danach den Tracker, dann Stopp oder nächster Task.
3. **`[x] FERTIG` ist unveränderlich.** Fertige Erklärungen, Projekte, Notebooks und
   Datensätze werden **niemals** erneut geöffnet, gelesen, ausgeführt, geprüft oder
   "verbessert" — außer der Nutzer verlangt es ausdrücklich. Fertig = vergessen.
4. **Quellen sparsam lesen.** Für ein Modul nur den zugehörigen Abschnitt aus
   `Modeluebersicht-DS-ML-ITOps.md` heranziehen — nicht andere Module, nicht das ganze PDF.
5. **Inhalte in Dateien, nicht in den Chat.** Schreibe niemals ganze Lektionen oder
   Projektinhalte in die Chat-Antwort. Chat-Antworten sind kurz: *was erledigt, was als
   Nächstes* (2–4 Sätze).
6. **Nicht unnötig ausführen.** Code/Notebooks höchstens **einmal** beim Erstellen
   validieren (wenn nötig). Danach nicht erneut laufen lassen.
7. **Tracker ist die einzige Wahrheit.** Der Fortschritt steht ausschließlich in
   `PROGRESS.md`. Leite den nächsten Schritt immer von dort ab, nie durch Inspektion des
   Dateibaums.

---

## 1. Session-Start-Workflow (jedes Mal identisch)

1. `PROGRESS.md` öffnen und **nur** den Kopfbereich lesen: den Block **„▶ NÄCHSTE AKTION"**
   und die Status-Legende.
2. Falls `PROGRESS.md` fehlt: aus der Modulliste in Abschnitt 7 neu initialisieren
   (Vorlage in Abschnitt 6).
3. Die in „NÄCHSTE AKTION" genannte *eine* Aufgabe ausführen — nach den Standards unten.
4. Nach Fertigstellung: Tracker aktualisieren (Status `[x]`, Datum, 1-Zeilen-Notiz),
   „NÄCHSTE AKTION" auf den nächsten offenen Task zeigen lassen, Build-Log-Zeile anhängen.
5. Kurze Chat-Meldung: erledigter Task + neuer „NÄCHSTE AKTION"-Zeiger. Dann weiter mit
   dem nächsten Task **oder** sauber stoppen (der Nutzer entscheidet über die Menge).

> Wenn der Nutzer sagt „mach weiter" / „continue": ab Schritt 1 mit dem nächsten offenen Task.

---

## 2. Repository-Struktur

```
/
├── CLAUDE.md                          # diese Datei
├── PROGRESS.md                        # strikter Tracker (einzige Wahrheit)
├── Modeluebersicht-DS-ML-ITOps.md     # inhaltliche Quelle (Inhalte + Qualifikationsziele)
└── modules/
    └── NN-modul-slug/
        ├── README.md                  # die Lektion ("Vorlesungsfolien"-Stil)
        ├── data/                      # generierte Datensätze (falls nötig)
        │   └── generate_data.py       # reproduzierbarer Generator (seeded)
        └── projects/
            ├── 01-basic/
            │   └── README.md + Notebook/Code
            ├── 02-medium/
            │   └── README.md + Notebook/Code
            └── 03-final/
                └── README.md + Notebook/Code
```

Ordner werden **on demand** angelegt (nur für das Modul, an dem gerade gearbeitet wird).
Keine leeren Platzhalterordner für spätere Module.

---

## 3. Deliverables pro Modul

### 3a. Die Lektion — `modules/NN-slug/README.md` ("Vorlesungsfolien"-Stil)

Klar strukturierte, progressive Erklärung von Basics bis Advanced. Feste Gliederung:

1. **Titel & Modulcode** + 2–3 Sätze Einordnung.
2. **Lernziele** — abgeleitet aus den Qualifikationszielen des Moduls (aus der Übersicht).
3. **Voraussetzungen** — was man vorher können sollte (+ Verweis auf frühere Module, wenn passend).
4. **Teil 1 — Grundlagen (Basics):** Kernbegriffe, Intuition, einfache Beispiele.
5. **Teil 2 — Kernkonzepte:** die zentralen Methoden/Techniken, mit Mini-Code-Snippets.
6. **Teil 3 — Fortgeschritten (Advanced):** komplexere Themen, Grenzen, Best Practices.
7. **Typische Fehler / Fallstricke.**
8. **Zusammenfassung / Cheat-Sheet** (kompakt, zum Wiederholen).
9. **Literatur & weiterführende Quellen** — kanonische Lehrbücher, wichtige Papers,
   hochwertige Onlinekurse/Docs. Nach Möglichkeit mit Autor/Titel/Jahr. **Keine erfundenen
   Links.** Bei Unsicherheit über eine URL lieber Werk per Autor+Titel nennen statt zu raten.

Stil: didaktisch, in ganzen Sätzen mit erklärenden Zwischenüberschriften; Formeln in
LaTeX (`$...$`), Code in Fenced Blocks. Umfang: gründlich, aber fokussiert — es soll zum
Lernen taugen, nicht als Nachschlagewerk ausufern.

### 3b. Drei Lernprojekte — `modules/NN-slug/projects/`

- **01-basic** — sehr einfacher Einstieg. Stark angeleitet (Schritt für Schritt), führt
  *ein* Kernkonzept praktisch ein. Soll in ~30–60 Min. machbar sein.
- **02-medium** — baut auf den Basics auf, offener gestellt, kombiniert mehrere Konzepte,
  weniger Hand-Holding.
- **03-final** — integratives Abschlussprojekt, das das Modul zusammenführt.
  **Praxisbezug erwünscht** (realistisches Szenario/Datensatz), *sofern sinnvoll* — wenn
  ein echter Praxisbezug beim Thema nicht sinnvoll ist, ist ein realistisches synthetisches
  Szenario in Ordnung.

Jedes Projekt enthält eine `README.md` mit: **Ziel**, **Setup/Ausführung**,
**Aufgabenstellung** (klare Schritte), **erwartetes Ergebnis** und einer **Referenz-/
Musterlösung** (oder Lösungshinweisen). Die Musterlösung darf im Notebook/Code stecken,
sollte aber als solche gekennzeichnet sein, damit der Nutzer erst selbst probieren kann. Die readme sollte eine englische und darunter eine deutsche Version enthalten. Keine emojis. Die notebooks oder code datein sollen ausschließlich auf englisch sein.

---

## 4. Formatwahl (du entscheidest pro Projekt)

Wähle das für das Thema **beste** Format. Faustregeln:

- **Jupyter Notebook (`.ipynb`)** — Standard für Data-Science-, ML-, Analyse- und
  Statistik-Themen: erzählender Text + Code + Ausgaben in einem Fluss. Ideal für „von
  Basics bis Ergebnis".
- **Python-Skripte / kleines Projekt (`.py` + README)** — für Software-/Engineering-Themen
  (z. B. Web Programming, Web Engineering) oder wenn eine App/CLI sinnvoller ist als ein Notebook.
- **SQL-Dateien (`.sql`) + README** — für datenbanklastige Themen (z. B. Datenmanagement),
  ggf. mit SQLite-Setup-Skript. Alternativ Notebook mit eingebetteter DB.
- **Markdown-Walkthrough + Konfigurations-/Beispieldateien** — wenn ein Thema (z. B. SAP,
  Business Engineering, No-Code) sich schlecht als Code abbilden lässt: geführte Übung mit
  Konzept, Schritten, Screenshots-Platzhaltern/Diagrammen (Mermaid) und Reflexionsfragen.

Notiere die Formatwahl kurz in der Projekt-README. Bevorzuge **einfache, lokal
lauffähige** Setups (Standardbibliotheken, `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`sqlite3` etc.). Schwere/optionale Abhängigkeiten klar in der README ausweisen.

---

## 5. Datensätze

- **Bevorzugt synthetisch & generiert.** Lege pro Modul bei Bedarf
  `data/generate_data.py` an: reproduzierbar (`random_state`/`seed` gesetzt), klein
  (Richtwert < 5 MB), mit dokumentiertem Schema in der Projekt-README.
- Datensätze so gestalten, dass sie das Lernziel tragen (z. B. eingebaute Muster,
  Ausreißer, Klassenungleichgewicht — je nach Thema).
- **Keine großen Downloads / keine Abhängigkeit von Netzwerkzugriff.** Falls ein bekanntes
  Public-Dataset didaktisch klar überlegen ist und trivial einbettbar/erzeugbar ist, darf
  es genutzt werden — sonst synthetisch.
- Generierte CSV/Parquet-Dateien dürfen eingecheckt werden, wenn klein; sonst nur den
  Generator einchecken und in der README das Erzeugen beschreiben.

---

## 6. Der Tracker `PROGRESS.md` (strikt)

`PROGRESS.md` ist die **einzige Fortschrittsquelle**. Aufbau:

```
# PROGRESS — Lern-Content Build Tracker

▶ NÄCHSTE AKTION: <genau ein Task, z. B. "Modul 03 · Projekt 02-medium erstellen">

## Legende
- [ ] TODO
- [~] IN ARBEIT   (höchstens EIN Task gleichzeitig)
- [x] FERTIG      (unveränderlich — nicht erneut öffnen)
- [!] BLOCKIERT   (Grund in Klammern)

## Modul NN — <Name>  ·  modules/NN-slug/
- [ ] Lektion (README.md)
- [ ] Projekt 01-basic
- [ ] Projekt 02-medium
- [ ] Projekt 03-final
... (alle Module) ...

## Build-Log (append-only)
- YYYY-MM-DD — <Task> — <erzeugte Pfade>
```

**Update-Protokoll (nach jedem Task):**

1. Beim Beginn eines Tasks dessen Status auf `[~]` setzen (und „NÄCHSTE AKTION" spiegelt ihn).
2. Bei Fertigstellung Status auf `[x]` setzen, Datum + kurze Notiz (Pfad) ergänzen.
3. „▶ NÄCHSTE AKTION" auf den **nächsten offenen** Task (in Reihenfolge) setzen.
4. Eine Zeile ans **Build-Log** anhängen (Datum, Task, Pfade).
5. `[x]`-Einträge nie zurücksetzen oder erneut anfassen.

Reihenfolge = die Nummerierung in Abschnitt 7 (erst Lektion, dann Projekte 1→2→3 pro Modul,
dann nächstes Modul). Innerhalb eines Moduls sollte die Lektion vor den Projekten stehen.

---

## 7. Modulliste (Reihenfolge = Lern-/Bearbeitungsreihenfolge)

**Bereich A — Data Science**
1. `12-DM-F-262-m01` — Datenmanagement und -analyse — `01-datenmanagement-und-analyse`
2. `12-PDS-262-m01` — Introduction to Data Science — `02-introduction-to-data-science`
3. `12-PD-262-m01` — Praxis der Datenanalyse — `03-praxis-der-datenanalyse`
4. `12-ADSP-262-m01` — Seminar: Applied Data Science Project — `04-applied-data-science-project`
   *(Im Modulhandbuch ohne Inhaltstext — als integratives Capstone anlegen, das DS-Skills
   der Module 1–3 anwendet.)*

**Bereich B — Machine Learning**
5. `10-I-AKIDS1-222-m01` — Algorithmen, KI und Data Science 1 — `05-algorithmen-ki-ds-1`
6. `10-I-AKIDS2-222-m01` — Algorithmen, KI und Data Science 2 — `06-algorithmen-ki-ds-2`
7. `10-I-AI-252-m01` — Einführung in die KI — `07-einfuehrung-in-die-ki`
8. `10-I-DL-222-m01` — Deep Learning — `08-deep-learning`
9. `10-I-CV-222-m01` — Computer Vision — `09-computer-vision`
10. `10-I-NLP-222-m01` — Natural Language Processing — `10-natural-language-processing`
11. `10-I-3D-152-m01` — 3D Point Cloud Processing — `11-3d-point-cloud-processing`

**Bereich C — IT & Operations**
12. `12-MPS-262-m01` — Managerial Problem Solving — `12-managerial-problem-solving`
13. `12-BIF-262-m01` — Business Intelligence — `13-business-intelligence`
14. `12-GP-G-262-m01` — Grundlagen betriebswirtschaftlicher Informationssysteme (SAP) — `14-erp-sap-grundlagen`
15. `12-DDD-262-m01` — Data-Driven Decisions in Practice — `15-data-driven-decisions`
16. `12-DDSCM-262-m01` — Data-Driven Supply Chain Management — `16-data-driven-scm`
17. `12-GDA-262-m01` — Geospatial Data Analytics & Smart Cities — `17-geospatial-data-analytics`
18. `12-WebP-F-262-m01` — Web Programming — `18-web-programming`
19. `12-AWE-262-m01` — Web Engineering — `19-web-engineering`
20. `12-FRBE-F-262-m01` — Forward und Reverse Business Engineering — `20-forward-reverse-business-engineering`
21. `12-EBP-262-m01` — No Code Analytics — `21-no-code-analytics`

Inhaltliche Quelle je Modul: der gleichnamige Abschnitt in `Modeluebersicht-DS-ML-ITOps.md`
(enthält *Inhalte* und *Qualifikationsziele / Kompetenzen*).

---

## 8. Definition of Done (pro Artefakt)

- **Lektion fertig**, wenn: README alle Abschnitte aus 3a enthält (Basics→Advanced +
  Literatur), Snippets syntaktisch plausibel, Lernziele aus den Qualifikationszielen abgeleitet.
- **Projekt fertig**, wenn: Projekt-README (Ziel, Setup, Aufgaben, erwartetes Ergebnis,
  Musterlösung) vorhanden **und** das Notebook/Code vollständig ist **und** benötigte
  Datensätze/Generatoren vorliegen. Einmalige Validierung genügt.
- Danach im Tracker auf `[x]` — **ab jetzt nicht mehr anfassen.**

---

## 9. Qualität & Ton

- Didaktisch, korrekt, mit ehrlichen Hinweisen auf Grenzen/Fallstricke.
- Progressiv aufbauen: nie Advanced-Begriffe ohne vorher eingeführte Basics.
- Deutsch als Sprache der Lektionen (Fachbegriffe/Code englisch, wie im Feld üblich).
- Prägnanz vor Vollständigkeit: das Material soll zum *Lernen* dienen.
