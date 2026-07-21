# CLAUDE.md — xtAI Lern-Repository

Dies ist ein persönliches Lern-Repository. Ziel: Ich (der Nutzer) will die Themen des Masterstudiengangs **eXtended Artificial Intelligence (xtAI)** der Uni Würzburg selbstständig durcharbeiten. Du (Claude Code) baust dafür strukturiertes Lernmaterial und Lernprojekte zu jedem Modul.

Ich bin **Anfänger bezüglich der Tooling-Details** (Jupyter, Umgebungen etc.), aber lernbereit. Erkläre Setup-Schritte, wenn sie das erste Mal auftauchen, und triff sinnvolle Standardentscheidungen für mich.

---

## Was du bauen sollst

Für **jedes Modul** aus `module-liste.md` (bzw. der Modulübersicht) erstellst du einen eigenen Ordner mit:

1. **Einer Erklärungs-Datei** (`README.md` oder `skript.md`) — wie Vorlesungsfolien in Textform: von den Basics bis zu Advanced-Themen, am Ende Literatur und weiterführende Quellen. Die Erklärungsdatei, und oder readme.d sollen zuerst auf englisch verfasst und darunter eine deutsch übersetzte Version enthalten. Keine emojis verwenden.
2. **Drei Lernprojekten**, aufsteigend in Schwierigkeit (basic → medium → final). Format wählst du pro Projekt selbst (siehe unten). Die Sprache in den projekten selbst, soll ausschließlich englisch sein. 

### Sprachregel (gilt für alles, verbindlich)

Diese Regel präzisiert die beiden Punkte oben und hat Vorrang, falls weiter unten etwas anderes steht:

- **Jede `README.md`** — sowohl die Modul-Erklärungsdatei als auch die READMEs der einzelnen Projekte — ist **zweisprachig**: zuerst der vollständige englische Text, darunter durch einen Trenner abgesetzt die deutsche Fassung. Der englische Block ist eine **vollständige Spiegelung**, keine Kurzfassung.
- **Alles in den Projekten selbst ist ausschließlich englisch**: Notebooks (Markdown-Zellen und Code), `.py`-Dateien, Kommentare, Docstrings, `print`-Ausgaben, Testnamen — und auch die **Bezeichner** (Variablen, Funktionen, Konstanten). Kein deutsches Wort im Code.
- **Datei- und Ordnernamen sind englisch** (`modules/`, `projects/`, `solution/`, `datasets/`, `results/`, `search.ipynb` …). Wer den deutschen Fachbegriff sucht, findet ihn im deutschen README-Block.
- Der **deutsche Block bleibt deutsch** — mit englischen Fachbegriffen, wo sie in der Praxis Standard sind (*overfitting*, *policy*, *frontier*). Nicht eindeutschen, was niemand eindeutscht.

Arbeite **ein Modul nach dem anderen vollständig ab**, bevor du zum nächsten übergehst. Nach jedem fertigen Modul: kurze Statusmeldung an mich und weiter, außer ich sage etwas anderes.

---

## Ordnerstruktur

Lege pro Modul einen Ordner unter `modules/` an. Benenne ihn mit laufender Nummer + sprechendem Kürzel, z. B.:

```
modules/
  01-introduction-in-ai/
    README.md              <- das Lernskript
    projects/
      01-basic/
      02-medium/
      03-final/
    ressourcen.md          <- optional, wenn Literaturliste lang wird
  02-data-science-1/
  ...
progress.md                <- Fortschritts-Tracker (du pflegst ihn)
module-liste.md            <- die Liste aller Module (Referenz)
```

Nummeriere die Module in einer sinnvollen **Lernreihenfolge**, nicht alphabetisch (siehe Abschnitt „Reihenfolge").

---

## Aufbau des Lernskripts (README.md pro Modul)

Schreibe es so, dass ich es allein durcharbeiten kann, ohne Vorlesung. Struktur:

1. **Titelblock**: Modulname, worum es geht in 2–3 Sätzen, welche Vorkenntnisse hilfreich sind, welche anderen Module man idealerweise vorher gemacht hat.
2. **Lernziele**: Was ich nach dem Modul können soll (in eigenen Worten, Stichpunkte).
3. **Grundlagen (Basics)**: Die Kernkonzepte von Null erklärt. Intuition zuerst, dann Formalismus. Nutze Beispiele, Analogien, kleine durchgerechnete Beispiele.
4. **Aufbau (Intermediate)**: Die wichtigsten Methoden/Algorithmen/Modelle des Gebiets, jeweils erklärt mit *wann nutzt man das* und *warum funktioniert es*.
5. **Advanced-Themen**: Aktuelle bzw. anspruchsvollere Inhalte des Moduls. Hier darf es dichter werden, aber immer noch erklärend.
6. **Zusammenfassung / Cheat-Sheet**: Kompakte Übersicht der wichtigsten Begriffe und Formeln zum Nachschlagen.
7. **Selbsttest**: 5–10 Verständnisfragen (mit ausklappbaren Antworten oder Antworten am Ende), damit ich prüfen kann, ob ich es verstanden habe.
8. **Literatur & Quellen**: 
   - Lehrbücher (mit konkreten Kapitelempfehlungen wenn möglich)
   - frei verfügbare Onlinekurse / Vorlesungen / Papers
   - gute Blogposts / interaktive Visualisierungen
   Kennzeichne, was **einsteigerfreundlich** und was **vertiefend** ist. Kennzeichne kostenlose Ressourcen.

**Stil**: Die obige Struktur gilt für **beide** Sprachblöcke — erst komplett auf Englisch, darunter dieselbe Gliederung auf Deutsch (siehe Sprachregel). Im deutschen Block: Fachbegriffe auf Englisch, wo üblich (die sind so in der Praxis). Formeln als LaTeX in Markdown (`$...$` / `$$...$$`). Erklärend, geduldig, aber nicht geschwätzig. Nutze Diagramme/ASCII-Skizzen oder Mermaid, wo es hilft.

---

## Die drei Lernprojekte pro Modul

Für jedes Modul drei Projekte, jeweils in eigenem Unterordner mit einer eigenen `README.md`, die enthält: Ziel, Vorwissen, Schritt-für-Schritt-Aufgabenstellung, was am Ende funktionieren soll, und **eine Musterlösung oder Lösungshinweise** (getrennt, damit ich erst selbst probieren kann — z. B. in `solution/` oder als ausklappbarer Abschnitt).

- **01 – basic**: Sanfter Einstieg. Führt das zentrale Konzept praktisch ein. Klein, in sich abgeschlossen, in unter ~1–2 Stunden machbar. Viel Anleitung.
- **02 – medium**: Baut auf dem Basic-Projekt auf, kombiniert mehrere Konzepte des Moduls. Weniger Anleitung, mehr Eigenleistung.
- **03 – final**: Abschlussprojekt, das das Modul konsolidiert. **Praxisbezug erwünscht** (echte/realistische Daten, ein reales Anwendungsszenario), *sofern das für das Thema sinnvoll geht*. Wenn ein echter Praxisbezug nicht sinnvoll ist (z. B. bei sehr theoretischen Modulen), ist ein realistisches, gut motiviertes künstliches Szenario in Ordnung — begründe die Wahl kurz. Hier kaum bis keine code Vorgabe.

### Format der Projekte — du entscheidest pro Projekt
Wähle jeweils das didaktisch beste Format und **begründe kurz in der Projekt-README, warum**. Richtlinien:

- **Jupyter Notebook** (`.ipynb`): ideal für alles mit Daten, Schrittweise-Exploration, Visualisierung, ML/Data Science, wo Code + Erklärung + Output zusammengehören. Wahrscheinlich das häufigste Format hier.
- **Python-Skript(e) / kleines Projekt** (`.py` + `README`): für Dinge mit richtiger Struktur (Robotik-Logik, Algorithmen, Simulationen, Software-Sicherheit, wo eine echte Codebasis Sinn ergibt).
- **Reines Markdown-Aufgabenblatt**: für theorielastige Module (z. B. Theorie der KI, Deduktive Datenbanken), wo „Projekt" eher Beweise/Herleitungen/Konzeptaufgaben bedeutet. Dann mit Aufgaben + Lösungsweg.
- Andere Formate (HTML/JS-Visualisierung, Notebooks in anderer Sprache) nur, wenn es klar besser passt.

## progress.md — Build-Tracker & nahtloses Weiterarbeiten

`progress.md` ist die **einzige Quelle der Wahrheit** über den Baufortschritt. Sie muss so geführt werden, dass du in einer **neuen Session sofort weiterarbeiten kannst, ohne bereits fertige Module oder Projekte erneut zu lesen** (das würde nur Tokens verschwenden).

### Aufbau von progress.md
Führe ganz oben einen **„NEXT ACTION"-Block**, den du bei jeder Änderung aktualisierst:

```
## NEXT ACTION
Als Nächstes zu bauen: Modul 03 "Machine Learning 1" — Schritt: Skript begonnen, fehlt: Advanced-Abschnitt + alle 3 Projekte.
Letzte abgeschlossene Einheit: Modul 02 "Data Science 1" (komplett, getestet).
```

Darunter die Fortschrittstabelle:

| Nr | Modul | Status | Formate | Getestet | Notizen |
|----|-------|--------|---------|----------|---------|

Status-Werte: `offen` · `skript-in-arbeit` · `skript-fertig` · `projekte-in-arbeit` · `fertig-getestet`.

### Regeln fürs Weiterarbeiten (wichtig — Token sparen)
- **Beim Start einer neuen Session liest du NUR `progress.md`** (und bei Bedarf `CLAUDE.md`). Du liest **nicht** die bereits als `fertig-getestet` markierten Modulordner ein. Vertraue dem Tracker.
- Du springst direkt zur unter „NEXT ACTION" genannten Einheit und arbeitest dort weiter.
- Öffne fertige Dateien nur dann erneut, wenn ein neues Modul **inhaltlich zwingend** darauf aufbaut (z. B. Machine Learning 2 referenziert Machine Learning 1) — und dann nur die eine relevante Datei, nicht ganze Ordner.
- Aktualisiere `progress.md` **nach jedem abgeschlossenen Teilschritt** (nicht erst am Modulende), damit ein Sessionabbruch nie mehr als einen Teilschritt Verlust bedeutet. Ein Teilschritt ist z. B. „Skript fertig" oder „Projekt 02 fertig".
- Wenn du ein Modul beginnst, trage es **sofort** als `skript-in-arbeit` ein, bevor du schreibst — falls die Session mitten drin endet, ist der Zustand trotzdem korrekt vermerkt.
- Nutze **Git-Commits als zweite Absicherung**: committe nach jedem abgeschlossenen Teilschritt mit klarer Nachricht (z. B. `Modul 03: Skript fertig` / `Modul 03: Projekt basic fertig`). So spiegelt die Git-History den Fortschritt und du kannst dich notfalls daran orientieren.

### Session-Limit-Verhalten
- Gehe davon aus, dass eine Session **jederzeit** enden kann (Token-/Zeitlimit). Baue deshalb immer in **kleinen, abgeschlossenen Einheiten** und schreibe nach jeder Einheit erst `progress.md`, dann committe, dann die nächste Einheit.
- Beginne **keine** Einheit, ohne dass die vorige im Tracker als fertig steht.
- Wenn du merkst, dass eine Einheit zu groß für den verbleibenden Rest ist, teile sie und markiere den erreichten Zwischenstand präzise in „NEXT ACTION" (z. B. „Modul 05 Skript: Basics + Intermediate fertig, Advanced + Selbsttest offen").

Da ich Anfänger bin: Wenn ein Projekt ein bestimmtes Setup braucht (Notebook-Kernel, Bibliotheken), schreibe die genauen Schritte in die README und lege eine `requirements.txt` (oder `environment.yml`) im Modulordner an.

### Datensätze
Viele Projekte brauchen Daten. Gehe so vor:

- **Bevorzugt echte, frei verfügbare Datensätze**, wenn es sie gibt und sie ohne Hürden nutzbar sind (z. B. über `scikit-learn`/`torchvision`/`seaborn` mitgelieferte Datensätze, oder bekannte offene Datensätze). Wenn ein Download nötig ist, schreibe die genauen Schritte in die README oder ein kleines Download-Skript und **committe keine großen Dateien** ins Repo.
- **Wenn kein passender echter Datensatz sinnvoll verfügbar ist, generiere selbst einen sinnvollen synthetischen Datensatz.** Das ist ausdrücklich erwünscht. Der generierte Datensatz soll:
  - das zu lernende Konzept sauber illustrieren (z. B. klar trennbare vs. verrauschte Klassen, gewünschte Verteilung, passende Struktur/Größe),
  - **reproduzierbar** sein (festen Random-Seed setzen),
  - per Skript oder Notebook-Zelle erzeugt werden, das im Projekt liegt (`generate_data.py` o. Ä.), damit ich sehe und ändern kann, wie die Daten entstehen,
  - kurz dokumentiert sein: was die Daten darstellen, wie sie erzeugt wurden und warum sie so gewählt sind.
- Wähle pro Projekt bewusst und **begründe in der README kurz**, ob echte oder synthetische Daten genutzt werden.
- Für das jeweilige **final-Projekt** mit Praxisbezug: bemühe dich bevorzugt um realistische bzw. echte Daten; nur wenn das nicht sinnvoll geht, nutze einen realistisch modellierten synthetischen Datensatz und erkläre die Wahl.

---

## Technisches Setup (bitte einmal zu Beginn einrichten)

Bevor du mit dem ersten Modul startest:

1. Lege eine kurze **`SETUP.md`** im Wurzelverzeichnis an, die mir für meinen Rechner erklärt: wie ich Python/eine virtuelle Umgebung einrichte, wie ich Jupyter starte, und wie ich die Projekte öffne. Frage mich einmal nach meinem Betriebssystem, falls du es nicht sicher weißt, und passe die Anleitung an.
2. Nutze eine gemeinsame virtuelle Umgebung (z. B. `venv`) für das ganze Repo, es sei denn ein Modul braucht etwas Spezielles. Halte die Abhängigkeiten schlank und dokumentiert.
3. Bevorzuge **weit verbreitete, gut dokumentierte Bibliotheken** (numpy, pandas, scikit-learn, matplotlib, pytorch, o. Ä.) — ich will Werkzeuge lernen, die in der Praxis zählen.
4. Prüfe Code, den du schreibst, wenn möglich durch Ausführen, bevor du ihn als fertig meldest.

---

## Arbeitsweise & Reihenfolge

**Lernreihenfolge** (didaktisch, nicht wie im Modulhandbuch sortiert). Grober Vorschlag — passe an, wenn du eine bessere Reihenfolge siehst, und halte sie in `progress.md` fest:

1. Introduction in AI  *(Fundament)*
2. Data Science 1 → 2
3. Machine Learning 1 → 2
4. Theorie der Künstlichen Intelligenz 1 → 2
5. Natural Language Processing 1 → 2 → Multilingual NLP
6. Computer Vision → Image Processing and Computational Photography
7. Reinforcement Learning and Computational Decision-Making → Deep Reinforcement Learning for Optimal Control
8. Machine Learning for Networks 1 → 2
9. Core XR: Principles of Interactive Systems → Multimodal Interfaces → 3D User Interfaces
10. Anwendungen: 3D Point Cloud Processing, Robotics 1 → 2, Advanced Automation, Self-aware Computing, Interaktive Computergraphik, Music Information Retrieval, Remote Sensing, Maschinelles Lernen in der Bioinformatik, Datenbanken 2
11. Computer Science: Simulationstechnik, Sicherheit von Softwaresystemen, Deduktive Datenbanken, Logische Programmierung, Systems Benchmarking, Fortgeschrittenes Programmieren
12. „Selected Topics" / „Ausgewählte Kapitel"-Module: nur ein kompaktes Skript, das erklärt, dass es sich um wechselnde Vertiefungsthemen handelt, plus Vorschläge, wie man sich selbst ein Thema erarbeitet. Keine drei vollen Projekte nötig — ein Leitfaden reicht. (Ebenso für die reinen Struktur-/Organisationsmodule wie *xtAI Lab 1–3*, *Seminare*, *Wissenschaftliches Praktikum*, *Summer School*, *Master's Thesis*, *Concluding Colloquium*: dafür statt Projekten je eine kurze `README.md` mit Erklärung, was das Modul im Studium bedeutet und wie man es angeht — keine Lerninhalte im engeren Sinn.)

**Umfang-Hinweis**: Das ist viel Material. Baue **immer erst ein Modul komplett fertig** (Skript + 3 Projekte + getestet), committe/melde es, dann das nächste. Frag mich nicht bei jedem Zwischenschritt um Erlaubnis — arbeite selbstständig, aber halte `progress.md` aktuell, damit ich jederzeit sehe, was fertig ist und was als Nächstes kommt.

**Wenn du unsicher bist**, welche Tiefe ich will: lieber gründlich und einsteigerfreundlich als knapp. Wenn eine echte inhaltliche Design-Entscheidung ansteht (z. B. „soll Modul X PyTorch oder scikit-learn nutzen"), triff sie selbst und notiere die Begründung kurz in der README.

> Avoid implementing computationally heavy tasks (like local deep learning/NMT training runs) that overload a single laptop CPU; instead, use lightweight workarounds (e.g., highly reduced epochs/data pairs), use MPS for hardware acceleration if possible, or explain complex architectures theoretically in the text by noting: "In practice, this is done using [X], but for a single laptop, it is far too computationally expensive."

---

## progress.md

Siehe die ausführliche Sektion **„progress.md — Build-Tracker & nahtloses Weiterarbeiten"** weiter oben. Kurzfassung: eine Tabelle (Modul | Status | Formate | getestet | Notizen) plus ein „NEXT ACTION"-Block ganz oben, aktualisiert nach jedem Teilschritt.

---

## Qualitätsansprüche (kurz)

- Inhaltlich korrekt. Wenn du dir bei einem Fakt/einer Formel unsicher bist, sag es offen im Text, statt zu raten.
- Erklärend statt nur auflistend — ich will *verstehen*, nicht nur Stichworte sehen.
- Lauffähiger, getesteter Code in den Projekten.
- Konsistente Struktur über alle Module, damit ich mich schnell zurechtfinde.
- Sprache nach der **Sprachregel** oben: READMEs zweisprachig (Englisch zuerst, Deutsch darunter), Projektinhalte und Dateinamen ausschließlich englisch. Im deutschen Block englische Fachbegriffe, wo Standard.
- **Strikte Master-Niveau-Garantie (Keine Abkürzungen):** Überspringe oder vereinfache unter keinen Umständen mathematische, statistische oder algorithmische Komplexität. Sätze wie „das sprengt den Rahmen“ sind strikt untersagt. Wenn ein Konzept komplex ist, liefere das **vollständige, exakte und komplexe Modell in formaler Notation**. Kompensiere die Komplexität niemals durch inhaltliche Reduktion, sondern ausschließlich durch eine herausragende, präzise und geduldige Erklärung. Die Abschlussprojekte müssen dem Niveau einer echten Master-Prüfungsleistung entsprechen.
- **Wichtig:** Es ist ok, wenn das beginner Projekt viel code vorgabe hat. Das medium projekt sollte weniger code vorgabe haben, nur vereinzelt bei den schwereren Ansätzen eine kleine Inspiration, das final Projekt sollte keinen schon vorgeschriebenen code haben. Es soll dazu dienen die gelernten Inhalte final anzuwenden. 

Leg zuerst `SETUP.md`, `progress.md` und `module-liste.md` an (letztere aus der beigefügten Modulübersicht), frag mich nach dem Betriebssystem, und beginne dann mit Modul 1 beziehungsweise dem Modul in progress.
