# Ergebnisbericht: Hash-basierte Log-Analyse

Die Engine verarbeitet 4000 chronologisch sortierte Logzeilen.
Nach Event-ID-Deduplizierung bleiben 3848 eindeutige Ereignisse;
152 Zeilen wurden als Wiederholung entfernt.

Bei einem Inaktivitätsfenster von 30 Sekunden entstehen
1505 Sessions. Die häufigste IP ist `10.0.0.1` mit 766
eindeutigen Ereignissen.

## Vergleich mit der sortierbasierten Referenz

Beide Engines liefern für jeden Benchmark-Präfix exakt dieselben Top-K-Werte,
Duplikate und Session-Zuordnungen. Für 4000 Eingabezeilen benötigt die
Hash-Engine 1.727 ms, die sortierbasierte Referenz
3.896 ms. Das entspricht in diesem Lauf einem Faktor von
2.26 zugunsten der Hash-Variante.

Absolute Zeiten sind systemabhängig. Der strukturelle Unterschied ist stabil:
Die Hash-Engine aggregiert und sessionisiert in einem erwarteten O(n)-Durchlauf;
die Referenz sortiert mehrfach und benötigt O(n log n). Die abschließende
Rangfolge der unterschiedlichen IPs wird in beiden Fällen sortiert.
