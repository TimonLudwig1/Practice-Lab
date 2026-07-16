# Lösung — Performance-Audit

## 1. Baseline-Analyse

Seien:

- \(n\): Anzahl aller Ereignisse,
- \(u\): Anzahl unterschiedlicher Kunden mit mindestens einem
  abgeschlossenen Ereignis.

Die Baseline materialisiert die Eingabe zunächst als Liste in \(O(n)\). Beim
Aufbau von **customer_ids** wird für jedes abgeschlossene Ereignis ein
Membership-Test auf einer Liste ausgeführt. Dieser kostet bis zu \(O(u)\).
Allein diese Phase benötigt damit \(O(nu)\).

Anschließend scannt die Pipeline für jeden der \(u\) Kunden erneut alle \(n\)
Ereignisse. Das erzeugt nochmals \(O(un)\). Aggregationen innerhalb der
kundenspezifischen Trefferlisten verarbeiten über alle Kunden zusammen \(O(n)\),
und die Ausgabesortierung kostet \(O(u\log u)\).

Insgesamt:

\[
O(nu + u\log u).
\]

Da \(u\le n\) und im Worst Case jeder Kunde nur ein Ereignis besitzt, folgt
\(O(n^2)\). Der gleichzeitig belegte Zusatzspeicher ist \(O(n+u)\), weil die
Eingabeliste, Kunden-IDs, Ausgaben und jeweils eine temporäre Trefferliste
gehalten werden.

## 2. Optimierungsstrategie

Die optimierte Pipeline verwendet eine Hash Map:

\[
\text{customer_id} \rightarrow \text{Accumulator}.
\]

Jedes abgeschlossene Ereignis aktualisiert genau einen Akkumulator:

- Zähler erhöhen,
- Nettobetrag addieren,
- Kategorie einem Set hinzufügen,
- maximalen Zeitstempel aktualisieren.

Unter üblichen Hashing-Annahmen sind Zugriff und Update erwartbar konstant.
Der einmalige Durchlauf kostet daher erwartete \(O(n)\)-Zeit. Danach werden
\(u\) Schlüssel für eine deterministische Ausgabe sortiert:

\[
O(n + u\log u).
\]

Der Zusatzspeicher beträgt \(O(u+c)\), wobei \(c\) die insgesamt in den
kundenspezifischen Sets gespeicherten Kategorieeinträge bezeichnet. Da die
Kategorienmenge begrenzt ist, ist dies im Projekt \(O(u)\).

## 3. Korrektheit des Refactorings

Ein Performance-Refactoring darf fachliche Ergebnisse nicht verändern. Der
Benchmark vergleicht daher vor jeder Zeitmessung die vollständigen Listen von
**CustomerSummary**-Objekten. Die unveränderlichen Dataclasses und die sortierte
Ausgabe machen den Vergleich streng und reproduzierbar.

Die Gleichheit wird zusätzlich auf handkonstruierten Kantenfällen und einem
synthetischen Datensatz getestet. Dazu gehören ignorierte Statuswerte, mehrere
Kategorien, Rabatte und verschiedene Ereigniszeitpunkte.

## 4. Messdesign

Datengenerierung und CSV-Laden werden außerhalb der Messung ausgeführt. So misst
das Experiment die untersuchte Aggregationslogik statt Dateisystem- und
Parserkosten. Beide Varianten erhalten dieselben Event-Objekte und verändern sie
nicht.

Mehrere wachsende Präfixe zeigen das Skalierungsverhalten. Der Median mehrerer
Wiederholungen reduziert den Einfluss kurzer Systemstörungen. Der Speedup wird
für jede Größe separat berechnet:

\[
\text{Speedup}(n)=
\frac{T_{\text{Baseline}}(n)}{T_{\text{optimiert}}(n)}.
\]

Da die Baseline asymptotisch stärker wächst, sollte der Speedup mit zunehmendem
\(n\) tendenziell steigen. Einzelne Messpunkte dürfen durch Cache,
Speicherallokation, CPU-Takt und Systemlast abweichen.

## 5. Grenzen

Der Benchmark isoliert bewusst einen Algorithmus-Engpass. Eine produktive
Pipeline kann zusätzlich durch Netzwerk, Datenträger, Datenbankabfragen,
Serialisierung oder Parallelität begrenzt sein. Der synthetische Datensatz
bildet außerdem keine realen Schieflagen oder saisonalen Muster vollständig ab.

Trotzdem ist das Audit belastbar für seine konkrete Aussage: Wiederholte
Vollscans und lineare Membership-Tests skalieren schlechter als eine
Single-Pass-Aggregation mit direktem Hash-Lookup. Absolute Sekunden sind
rechnerspezifisch; Ergebnisgleichheit und Wachstumsverhalten sind die
übertragbaren Erkenntnisse.
