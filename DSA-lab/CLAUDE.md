# CLAUDE.md — DSA Practice Lab (Data Structures & Algorithms)

Dieses Repository ist eine **Lernwerkstatt**. Deine Aufgabe als Claude Code ist es,
für jedes der 17 Module aus `Moduluebersicht_DSA.md` selbstständig hochwertiges
Lernmaterial zu erzeugen: ein umfassendes Theorie-Skript (`THEORIE.md`) plus drei
aufeinander aufbauende Projekte (01-basic, 02-medium, 03-final).

Der Nutzer arbeitet mit teuren Modellen. **Token-Effizienz ist oberstes Gebot.**

---

## Rollenverteilung der Steuerdateien

- **`PROGRESS.md`** — einzige Wahrheit über den Build-Fortschritt. Enthält den
  Block „NÄCHSTE AKTION", den Modul-Tracker und die verbindlichen Arbeitsregeln.
- **`Moduluebersicht_DSA.md`** — inhaltliche Quelle und Leitfaden. Enthält die
  Arbeitsanweisungen (Sprache, Didaktik, Projektstruktur) sowie pro Modul:
  Theorieinhalte, Qualifikationsziele und die drei Projektvorgaben.
- **Diese Datei** — nur Einstiegspunkt und Kurzreferenz. Keine eigenen Inhalte,
  die anderswo stehen.

---

## Ablauf jeder Session

1. `PROGRESS.md` lesen. Sonst nichts — kein `ls -R`, kein Repo-Scan, keine
   fertigen Ordner öffnen.
2. Aus `Moduluebersicht_DSA.md` **nur** lesen: den Abschnitt
   „Arbeitsanweisungen" (einmal pro Session) und den Abschnitt des Moduls, das
   in „NÄCHSTE AKTION" genannt ist. Keine anderen Modulabschnitte.
3. Genau den einen Task aus „NÄCHSTE AKTION" erledigen (Theorie ODER ein
   einzelnes Projekt ODER Test-Durchlauf).
4. `PROGRESS.md` aktualisieren (Tracker-Zelle, neuer NÄCHSTE-AKTION-Block,
   Session-Log), dann `git commit` mit sprechender Message
   (z. B. `modul-04: 02-medium fertig`).
5. Weiter mit dem nächsten Task oder sauber stoppen — der Tracker macht jede
   Unterbrechung resume-fähig.

---

## Die wichtigsten Regeln (Details in PROGRESS.md)

1. **FERTIG ist unveränderlich.** Abgeschlossene Inhalte niemals erneut öffnen,
   lesen, ausführen oder „verbessern" — außer auf ausdrückliche Anweisung.
2. **Reihenfolge strikt:** Module 01 bis 17; innerhalb eines Moduls Theorie →
   01-basic → 02-medium → 03-final → Test-Durchlauf.
3. **Immer erst ein Modul komplett fertigstellen**, dann das nächste. Nicht bei
   jedem Zwischenschritt um Erlaubnis fragen — selbstständig arbeiten, Tracker
   aktuell halten.
4. **Inhalte gehören in Dateien, nicht in Chat-Antworten.**
5. **Didaktik:** Jedes Theorie-Skript folgt der Struktur Intuition → Simulation →
   Formalisierung. Erklärend statt auflistend — Ziel ist Verständnis.
6. **Sprache:** Dokumentation auf Deutsch mit englischen Fachbegriffen, Code und
   Docstrings auf Englisch, keine Emojis.
7. **Code läuft:** Projekte enthalten getesteten Code; Datensätze werden mit
   fixem Seed generiert (`data/generate_data.py`), Begründung als Kommentar.
8. **Bei Unsicherheit über Tiefe:** lieber gründlich und einsteigerfreundlich als
   knapp. Echte Design-Entscheidungen (z. B. Notebook vs. Skript) selbst treffen
   und kurz in der Projekt-README begründen.

---

## Erster Start

Falls noch nicht vorhanden: `modules/`-Verzeichnis anlegen, dann direkt mit der
in `PROGRESS.md` genannten nächsten Aktion beginnen. Keine weitere
Initialisierung nötig — beide Steuerdateien sind bereits vollständig vorbereitet.
