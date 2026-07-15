# Modul 01 — Komplexitätsanalyse und Big-O

Komplexitätsanalyse beantwortet nicht zuerst die Frage „Wie viele Millisekunden
braucht mein Programm?“, sondern die robustere Frage: **Wie verändert sich sein
Ressourcenbedarf, wenn die Eingabe wächst?** Diese Perspektive macht Algorithmen
vergleichbar, obwohl Rechner, Programmiersprachen und konkrete Eingaben
unterschiedlich sind.

Dieses Skript folgt drei Ebenen:

1. **Intuition:** Warum Wachstum wichtiger ist als eine einzelne Laufzeit.
2. **Simulation:** Wie sich Wachstum am Code und in Messreihen beobachten lässt.
3. **Formalisierung:** Wie Big-O, Ω, Θ sowie Zeit- und Speicherkomplexität präzise
   beschrieben werden.

Am Ende sollst du unbekannten Python-Code systematisch analysieren, Messungen
kritisch interpretieren und erklären können, warum ein asymptotisch besserer
Algorithmus für große Eingaben gewinnt.

---

## 1. Intuition: Nicht die Stoppuhr, sondern das Wachstum entscheidet

### 1.1 Dasselbe Problem, drei Lösungswege

Angenommen, eine Liste enthält Messwerte und wir möchten wissen, ob ein Wert
doppelt vorkommt. Drei plausible Lösungen sind:

~~~python
def contains_duplicate_pairs(values):
    """Compare every pair."""
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                return True
    return False


def contains_duplicate_sorted(values):
    """Sort a copy and compare neighbors."""
    ordered = sorted(values)
    for index in range(1, len(ordered)):
        if ordered[index - 1] == ordered[index]:
            return True
    return False


def contains_duplicate_set(values):
    """Remember values that have already appeared."""
    seen = set()
    for value in values:
        if value in seen:
            return True
        seen.add(value)
    return False
~~~

Alle drei Funktionen liefern dasselbe logische Ergebnis. Ihre Arbeit wächst
jedoch unterschiedlich:

- Die Paarvergleichslösung prüft im ungünstigsten Fall ungefähr
  \(n(n-1)/2\) Paare. Verdoppelt sich \(n\), vervierfacht sich die Arbeit
  annähernd.
- Die Sortierlösung zahlt typischerweise \(n \log n\) für das Sortieren und
  danach \(n\) für den Durchlauf.
- Die Set-Lösung besucht jedes Element einmal. Unter der üblichen Annahme
  konstanter durchschnittlicher Set-Operationen wächst die Arbeit linear.

Für zehn Werte können alle drei Varianten augenblicklich wirken. Für zehn
Millionen Werte trennt das Wachstumsverhalten sie drastisch. Genau dafür
abstrahiert die Komplexitätsanalyse von einzelnen Millisekunden.

### 1.2 Warum eine schnelle Kleinmessung täuschen kann

Eine reale Laufzeit lässt sich grob als

\[
T(n) = \text{Anzahl elementarer Schritte} \times \text{Kosten pro Schritt}
\]

verstehen. Die Kosten pro Schritt hängen unter anderem von Hardware,
Python-Version, Cache-Verhalten, Betriebssystem und Implementierungsdetails ab.
Ein quadratischer Algorithmus in optimiertem C kann bei kleinen Eingaben einen
linearen Algorithmus in Python schlagen. Das ändert nicht, welcher Algorithmus
bei hinreichend großem \(n\) besser skaliert.

Komplexitätsklassen ignorieren deshalb konstante Faktoren und weniger stark
wachsende Terme. Aus

\[
T(n) = 3n^2 + 20n + 400
\]

wird asymptotisch \(\Theta(n^2)\), weil der quadratische Term für große \(n\)
dominiert.

### 1.3 Das Verdopplungsexperiment als Denkwerkzeug

Ein nützliches mentales Modell lautet: „Was geschieht ungefähr, wenn ich die
Eingabe verdopple?“

| Wachstum | Arbeit bei \(n\) | Arbeit bei \(2n\) | ungefährer Faktor |
|---|---:|---:|---:|
| konstant | \(1\) | \(1\) | \(1\) |
| logarithmisch | \(\log_2 n\) | \(\log_2 n + 1\) | nahe \(1\) |
| linear | \(n\) | \(2n\) | \(2\) |
| linearithmisch | \(n\log_2 n\) | \(2n(\log_2 n+1)\) | etwas über \(2\) |
| quadratisch | \(n^2\) | \(4n^2\) | \(4\) |
| exponentiell | \(2^n\) | \(2^{2n}\) | \(2^n\) |

Die Tabelle ist kein Ersatz für eine Analyse, aber ein starker Plausibilitätstest
für Messdaten.

---

## 2. Simulation: Arbeit sichtbar machen

### 2.1 Erst Operationen zählen

Zeitmessungen rauschen. Ein Operationszähler macht das strukturelle Wachstum
zunächst ohne Hardwareeinfluss sichtbar:

~~~python
def count_pair_checks(size):
    """Return how many unordered index pairs exist."""
    checks = 0
    for left in range(size):
        for right in range(left + 1, size):
            checks += 1
    return checks


for size in (10, 20, 40, 80):
    print(size, count_pair_checks(size))
~~~

Erwartete Werte:

| \(n\) | Paarprüfungen \(n(n-1)/2\) | Faktor zur vorherigen Zeile |
|---:|---:|---:|
| 10 | 45 | — |
| 20 | 190 | 4,22 |
| 40 | 780 | 4,11 |
| 80 | 3.160 | 4,05 |

Der Faktor nähert sich bei Verdopplung der Eingabe dem Wert vier. Der lineare
Anteil in \(n(n-1)/2 = (n^2-n)/2\) wird relativ immer unbedeutender.

### 2.2 Ein Algorithmus von Hand: binäre Suche

Logarithmisches Wachstum entsteht häufig, wenn ein Algorithmus den verbleibenden
Suchraum pro Schritt mit einem konstanten Faktor verkleinert. Bei binärer Suche
wird ein sortiertes Intervall halbiert.

Gesucht sei 31 in:

\[
[3, 7, 12, 18, 24, 31, 42, 56, 63]
\]

| Schritt | linke Grenze | rechte Grenze | Mitte | Wert | Entscheidung |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8 | 4 | 24 | rechts weitersuchen |
| 2 | 5 | 8 | 6 | 42 | links weitersuchen |
| 3 | 5 | 5 | 5 | 31 | gefunden |

Selbst bei rund einer Million sortierter Elemente sind höchstens ungefähr
\(\log_2(1.000.000) \approx 20\) Halbierungen nötig. Eine lineare Suche kann
dagegen eine Million Vergleiche benötigen.

~~~python
def binary_search(values, target):
    """Return the target index or -1 if it is absent."""
    left = 0
    right = len(values) - 1

    while left <= right:
        middle = (left + right) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return -1
~~~

### 2.3 Laufzeiten mit time.perf_counter messen

Eine einzelne Messung ist selten vertrauenswürdig. Sinnvolle Messreihen beachten
mindestens diese Punkte:

- Eingaben werden außerhalb des gemessenen Bereichs erzeugt.
- Jede Messung wird wiederholt; der Median dämpft Ausreißer.
- Die Funktion läuft einmal vorab, damit einmalige Startkosten weniger stören.
- Ergebnisse werden verwendet oder überprüft, damit tatsächlich dieselbe Arbeit
  ausgeführt wird.
- Eingabegrößen decken mehrere Verdopplungen ab.
- Sehr schnelle Funktionen werden pro Messpunkt mehrfach aufgerufen.

Das folgende Gerüst misst bewusst die drei Duplikat-Varianten aus Abschnitt 1:

~~~python
from statistics import median
from time import perf_counter


def benchmark(function, values, repeats=7):
    """Return the median runtime in seconds."""
    function(values)
    durations = []

    for _ in range(repeats):
        start = perf_counter()
        result = function(values)
        durations.append(perf_counter() - start)
        assert result is False

    return median(durations)


sizes = [250, 500, 1_000, 2_000, 4_000]
functions = [
    contains_duplicate_pairs,
    contains_duplicate_sorted,
    contains_duplicate_set,
]

measurements = {function.__name__: [] for function in functions}

for size in sizes:
    values = list(range(size))
    for function in functions:
        elapsed = benchmark(function, values)
        measurements[function.__name__].append(elapsed)
~~~

Die Liste enthält absichtlich keine Duplikate. So muss jede Funktion ihren
Worst Case für diese konkrete Aufgabe durchlaufen. Würde das Duplikat an den
ersten beiden Positionen liegen, könnten die Paar- und Set-Variante sofort
abbrechen. Wir würden dann nicht mehr dasselbe Szenario vergleichen.

### 2.4 Wachstumskurven plotten

Ein Log-Log-Plot eignet sich besonders für polynomial wachsende Funktionen. Beide
Achsen werden logarithmisch dargestellt:

~~~python
import matplotlib.pyplot as plt


for name, durations in measurements.items():
    plt.plot(sizes, durations, marker="o", label=name)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Input size n")
plt.ylabel("Median runtime in seconds")
plt.title("Duplicate detection: empirical growth")
plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()
~~~

Warum ist das hilfreich? Für \(T(n) = c n^k\) gilt:

\[
\log T(n) = \log c + k \log n
\]

Im Log-Log-Plot wird daraus annähernd eine Gerade mit Steigung \(k\). Eine
lineare Funktion hat also ungefähr Steigung 1, eine quadratische ungefähr
Steigung 2. \(n\log n\) ist keine perfekte Potenz und krümmt leicht, liegt aber
typischerweise zwischen linear und quadratisch.

### 2.5 Verdopplungsquotienten aus Messdaten

Ein Plot kann täuschen, wenn Achsen oder Größenbereiche ungünstig sind. Ergänzend
lässt sich der Quotient benachbarter Messwerte untersuchen:

~~~python
def doubling_ratios(durations):
    """Return ratios between consecutive runtimes."""
    return [
        current / previous
        for previous, current in zip(durations, durations[1:])
        if previous > 0
    ]


for name, durations in measurements.items():
    print(name, doubling_ratios(durations))
~~~

Erwartungswerte bei Verdopplung von \(n\):

- \(O(1)\): ungefähr 1,
- \(O(\log n)\): knapp über 1,
- \(O(n)\): ungefähr 2,
- \(O(n\log n)\): etwas über 2,
- \(O(n^2)\): ungefähr 4.

In echten Messungen entstehen keine perfekten Faktoren. Bei kleinen Eingaben
dominieren Messauflösung und Konstanten; bei großen Eingaben können Cache,
Speicherallokation, Garbage Collection oder Systemlast eingreifen. Empirie
liefert Evidenz, keinen mathematischen Beweis.

### 2.6 Theorie und Messung sauber abgleichen

Eine belastbare Interpretation trennt drei Aussagen:

1. **Theoretische Aussage:** Der Code führt im Worst Case quadratisch viele
   Vergleiche aus.
2. **Experimentelle Beobachtung:** Im untersuchten Größenbereich nähert sich der
   Verdopplungsquotient vier.
3. **Schlussfolgerung:** Die Messung ist mit \(\Theta(n^2)\) vereinbar.

„Der Plot beweist Big-O“ wäre zu stark. Ein begrenztes Experiment kann nicht
ausschließen, dass sich das Verhalten für noch größere Eingaben ändert.

---

## 3. Die wichtigsten Wachstumsklassen

### 3.1 O(1): konstanter Aufwand

Der Aufwand hängt nicht von der Länge der Liste ab:

~~~python
def first_or_none(values):
    """Return the first value if available."""
    return values[0] if values else None
~~~

Ein Indexzugriff auf eine Python-Liste ist konstant. \(O(1)\) bedeutet nicht
„genau eine Operation“ und auch nicht „immer schnell“. Eine Operation kann einen
großen konstanten Faktor haben. Entscheidend ist, dass ihr Aufwand nicht mit
\(n\) wächst.

### 3.2 O(log n): logarithmischer Aufwand

Binäre Suche ist das Standardbeispiel. Nach \(k\) Halbierungen bleiben von \(n\)
Kandidaten ungefähr \(n/2^k\). Die Suche endet, wenn

\[
\frac{n}{2^k} \le 1
\]

und damit \(k \ge \log_2 n\). Die Basis des Logarithmus wird in Big-O meist
weggelassen, denn Logarithmen verschiedener Basen unterscheiden sich nur um
einen konstanten Faktor.

### 3.3 O(n): linearer Aufwand

~~~python
def maximum(values):
    """Return the largest value."""
    if not values:
        raise ValueError("values must not be empty")

    largest = values[0]
    for value in values[1:]:
        if value > largest:
            largest = value
    return largest
~~~

Jedes Element muss im Worst Case betrachtet werden. Die Schleife ist linear.
Allerdings erzeugt values[1:] zusätzlich eine lineare Kopie. Zeitlich bleibt die
Funktion \(\Theta(n)\), benötigt dadurch aber unnötig \(\Theta(n)\) zusätzlichen
Speicher. Eine Iteration über Indexpositionen oder einen Iterator würde diese
Kopie vermeiden.

### 3.4 O(n log n): linearithmischer Aufwand

Effiziente vergleichsbasierte Sortierverfahren wie Merge Sort haben typischerweise
diese Klasse. Intuitiv entstehen \(\log_2 n\) Teilungsebenen. Auf jeder Ebene
werden insgesamt \(n\) Elemente verarbeitet:

\[
n + n + \dots + n
\quad \text{über } \log_2 n \text{ Ebenen}
= n\log_2 n
\]

Auch Pythons sorted liegt im Worst Case in \(O(n\log n)\); bei bereits
strukturierten Daten kann Timsort vorhandene Ordnung ausnutzen.

### 3.5 O(n²): quadratischer Aufwand

~~~python
def all_ordered_pairs(values):
    """Return every ordered pair."""
    pairs = []
    for left in values:
        for right in values:
            pairs.append((left, right))
    return pairs
~~~

Für jedes der \(n\) äußeren Elemente läuft die innere Schleife \(n\)-mal:
\(n \cdot n = n^2\). Zusätzlich enthält das Ergebnis \(n^2\) Paare. Hier ist
quadratische Zeit unvermeidbar, wenn wirklich alle Paare materialisiert werden
sollen, denn schon die Ausgabe ist quadratisch groß.

### 3.6 O(2ⁿ): exponentieller Aufwand

Die naive rekursive Berechnung der Fibonacci-Zahlen verzweigt mehrfach:

~~~python
def fibonacci(number):
    """Return a Fibonacci number using naive recursion."""
    if number <= 1:
        return number
    return fibonacci(number - 1) + fibonacci(number - 2)
~~~

Viele Teilprobleme werden wiederholt berechnet. Die genaue Laufzeit wächst näher
an \(\varphi^n\) als an \(2^n\), wird aber häufig mit der einfacheren oberen
Schranke \(O(2^n)\) beschrieben. Schon eine Erhöhung von \(n\) um einen kleinen
Betrag kann die Arbeit vervielfachen. Memoization reduziert dieses konkrete
Problem auf lineare Zeit, weil jedes Teilproblem nur einmal gelöst wird.

### 3.7 O(n!): faktorieller Aufwand

Wer alle Reihenfolgen von \(n\) Elementen erzeugt, erhält \(n!\) Permutationen:

~~~python
def permutations(values):
    """Return all permutations without library shortcuts."""
    if len(values) <= 1:
        return [values[:]]

    result = []
    for index, value in enumerate(values):
        rest = values[:index] + values[index + 1 :]
        for suffix in permutations(rest):
            result.append([value] + suffix)
    return result
~~~

Für \(n=10\) existieren bereits \(3.628.800\) Permutationen. Wenn alle
Permutationen tatsächlich ausgegeben werden müssen, kann kein Algorithmus die
Größe der Ausgabe umgehen. Häufig besteht der eigentliche Fortschritt deshalb
darin, den Suchraum durch Bedingungen früh zu beschneiden oder gar nicht erst
vollständig zu materialisieren.

### 3.8 Größenordnungen im direkten Vergleich

Die folgenden Werte zählen abstrakte Arbeitseinheiten und ignorieren Konstanten:

| \(n\) | \(\log_2 n\) | \(n\) | \(n\log_2 n\) | \(n^2\) | \(2^n\) |
|---:|---:|---:|---:|---:|---:|
| 10 | 3,3 | 10 | 33 | 100 | 1.024 |
| 100 | 6,6 | 100 | 664 | 10.000 | ungefähr \(1{,}27 \cdot 10^{30}\) |
| 1.000 | 10,0 | 1.000 | 9.966 | 1.000.000 | astronomisch |
| 1.000.000 | 19,9 | 1.000.000 | 19.931.569 | \(10^{12}\) | nicht praktikabel |

Hier wird sichtbar, warum \(O(n\log n)\) langfristig \(O(n^2)\) schlägt.
Konstanten können den Schnittpunkt verschieben, aber nicht verhindern.
Beispielsweise vergleichen wir:

\[
1000n\log_2 n \quad \text{und} \quad n^2
\]

Für kleinere \(n\) kann die quadratische Variante wegen ihres kleineren
konstanten Faktors schneller sein. Sobald \(n > 1000\log_2 n\), gewinnt die
linearithmische Variante und ihr Vorsprung wächst weiter.

---

## 4. Formalisierung: Was O, Ω und Θ wirklich sagen

### 4.1 Big-O als asymptotische obere Schranke

Seien \(f(n)\) und \(g(n)\) nichtnegative Funktionen für hinreichend große
\(n\). Formal gilt

\[
f(n) \in O(g(n)),
\]

wenn positive Konstanten \(c\) und \(n_0\) existieren, sodass

\[
0 \le f(n) \le c \cdot g(n)
\quad \text{für alle } n \ge n_0.
\]

Die Konstanten erlauben, Implementierungsdetails und das Verhalten kleiner
Eingaben auszublenden.

Beispiel: \(f(n)=3n^2+20n+400\). Für \(n\ge 1\) gilt

\[
3n^2+20n+400 \le 3n^2+20n^2+400n^2 = 423n^2.
\]

Mit \(c=423\) und \(n_0=1\) folgt \(f(n)\in O(n^2)\). Die Konstanten müssen
nicht eng sein; ihre Existenz genügt.

Wichtig: \(O(n^2)\) ist nur eine obere Schranke. Auch eine lineare Funktion liegt
formal in \(O(n^2)\). In der Praxis nennt man meist die engste übliche Schranke,
um möglichst viel Information zu vermitteln.

### 4.2 Ω als untere Schranke

\[
f(n) \in \Omega(g(n))
\]

gilt, wenn positive Konstanten \(c\) und \(n_0\) existieren, sodass

\[
f(n) \ge c \cdot g(n)
\quad \text{für alle } n \ge n_0.
\]

Ω beschreibt, dass \(f\) asymptotisch mindestens so schnell wächst wie \(g\).

### 4.3 Θ als enge Schranke

\[
f(n) \in \Theta(g(n))
\]

gilt, wenn \(f(n)\) sowohl in \(O(g(n))\) als auch in \(\Omega(g(n))\) liegt.
Dann wächst \(f\) bis auf konstante Faktoren genau in derselben Größenordnung
wie \(g\). Für \(3n^2+20n+400\) ist \(\Theta(n^2)\) die informative Aussage.

### 4.4 Terme systematisch vereinfachen

Beim Ableiten einer asymptotischen Klasse helfen drei Regeln:

1. **Konstante Faktoren entfallen:** \(7n \rightarrow \Theta(n)\).
2. **Dominanter Term bleibt:** \(n^2+50n+2 \rightarrow \Theta(n^2)\).
3. **Aufeinanderfolgende Blöcke addieren, geschachtelte Arbeit multiplizieren.**

Beispiel:

~~~python
def example(values):
    """Perform one linear and one quadratic phase."""
    total = 0

    for value in values:
        total += value

    for left in values:
        for right in values:
            total += left * right

    return total
~~~

Die Arbeit ist \(n+n^2\), also \(\Theta(n^2)\). Die Schleifen folgen
aufeinander; deshalb werden ihre Kosten addiert, nicht multipliziert.

### 4.5 Mehrere Eingabegrößen nicht vorschnell zusammenwerfen

~~~python
def common_values(left_values, right_values):
    """Return matching pairs from two independent inputs."""
    matches = []
    for left in left_values:
        for right in right_values:
            if left == right:
                matches.append(left)
    return matches
~~~

Sind die Eingabelängen \(n\) und \(m\), lautet die Zeitkomplexität
\(\Theta(nm)\), nicht automatisch \(\Theta(n^2)\). Erst wenn ausdrücklich
\(n=m\) angenommen wird, ergibt sich \(\Theta(n^2)\).

### 4.6 Best, Average und Worst Case

Die Wachstumsklasse benötigt ein Szenario:

~~~python
def linear_search(values, target):
    """Return the first target index or -1."""
    for index, value in enumerate(values):
        if value == target:
            return index
    return -1
~~~

- **Best Case:** Das Ziel steht vorne: \(\Theta(1)\).
- **Worst Case:** Das Ziel steht hinten oder fehlt: \(\Theta(n)\).
- **Average Case:** Unter einer konkret festgelegten Verteilung der Zielpositionen
  werden im Mittel proportional viele Elemente geprüft: \(\Theta(n)\).

„Average Case“ ist ohne Wahrscheinlichkeitsmodell unvollständig. Sind Treffer
häufig? Ist jede Position gleich wahrscheinlich? Darf das Ziel fehlen? Je nach
Annahme ändert sich der Erwartungswert.

Big-O und Worst Case sind außerdem keine Synonyme. O ist eine Art von Schranke;
Worst Case beschreibt, welche Eingabe derselben Größe betrachtet wird. Man kann
beispielsweise sagen: „Die Worst-Case-Laufzeit liegt in \(\Theta(n^2)\).“

---

## 5. Space Complexity

### 5.1 Gesamtspeicher und Auxiliary Space

Space Complexity beschreibt, wie der Speicherbedarf mit der Eingabe wächst.
Dabei sollte klar benannt werden, ob

- der **Gesamtspeicher** inklusive Eingabe und Ausgabe oder
- nur der **zusätzliche Speicher** des Algorithmus, der Auxiliary Space,

gemeint ist.

~~~python
def doubled_copy(values):
    """Return a new list with doubled values."""
    return [value * 2 for value in values]


def double_in_place(values):
    """Double all values inside the existing list."""
    for index in range(len(values)):
        values[index] *= 2
~~~

Beide Funktionen brauchen \(\Theta(n)\) Zeit. doubled_copy erzeugt jedoch eine
neue Liste mit \(\Theta(n)\) zusätzlichem Speicher. double_in_place verwendet
abgesehen von wenigen Variablen \(\Theta(1)\) Auxiliary Space.

### 5.2 Die Ausgabe kann die Untergrenze bestimmen

all_ordered_pairs aus Abschnitt 3 gibt \(n^2\) Paare zurück. Selbst wenn ihre
Berechnung trickreich optimiert würde, benötigt das materialisierte Ergebnis
\(\Theta(n^2)\) Speicher. Eine Generator-Lösung könnte den gleichzeitig belegten
Zusatzspeicher reduzieren, ändert aber nicht die Anzahl ausgegebener Paare.

### 5.3 Rekursion belegt Stack-Speicher

Jeder noch nicht abgeschlossene rekursive Aufruf belegt einen Stack Frame.
Eine Rekursionstiefe von \(n\) verursacht daher typischerweise \(O(n)\)
zusätzlichen Stack-Speicher, selbst wenn keine Liste angelegt wird. Bei einer
balancierten Divide-and-Conquer-Rekursion kann die Tiefe dagegen \(O(\log n)\)
sein.

### 5.4 Time-Space Trade-offs

Die Set-Variante der Duplikatsuche spart gegenüber dem Paarvergleich Zeit, zahlt
aber mit \(\Theta(n)\) zusätzlichem Speicher. Solche Trade-offs sind zentral:

- Eine Lookup-Struktur kann wiederholte Suche beschleunigen.
- Caching oder Memoization speichert Ergebnisse, um Neuberechnung zu vermeiden.
- In-Place-Verfahren sparen Speicher, können aber komplexer oder langsamer sein.

Es gibt nicht allgemein „den besten“ Algorithmus. Die richtige Wahl hängt von
Zeitbudget, Speichergrenze, Eingabegröße und Anforderungen an die Ausgabe ab.

---

## 6. Amortisierte Analyse: Warum append trotz Resize O(1) ist

### 6.1 Das scheinbare Problem dynamischer Arrays

Eine Python-Liste verhält sich konzeptionell wie ein dynamisches Array. Die
Elemente liegen in einem zusammenhängenden Puffer mit einer bestimmten
Kapazität. Ist der Puffer voll, muss ein größerer reserviert und der bisherige
Inhalt kopiert werden.

Ein einzelnes append kann deshalb \(\Theta(n)\) kosten. Trotzdem wird append
üblicherweise mit **amortisiert \(O(1)\)** angegeben. Das ist kein Widerspruch:
Amortisierte Analyse verteilt seltene teure Operationen über eine ganze Folge
von Operationen.

### 6.2 Simulation mit vereinfachter Verdopplungsstrategie

Nehmen wir eine Startkapazität von 1 an und verdoppeln bei jedem vollen Puffer:

| append Nr. | Kapazität vorher | Resize? | Kopien | Kapazität nachher |
|---:|---:|:---:|---:|---:|
| 1 | 1 | nein | 0 | 1 |
| 2 | 1 | ja | 1 | 2 |
| 3 | 2 | ja | 2 | 4 |
| 4 | 4 | nein | 0 | 4 |
| 5 | 4 | ja | 4 | 8 |
| 6 | 8 | nein | 0 | 8 |
| 7 | 8 | nein | 0 | 8 |
| 8 | 8 | nein | 0 | 8 |

Bis \(n\) appends werden bei Resizes ungefähr

\[
1 + 2 + 4 + 8 + \dots < 2n
\]

alte Elemente kopiert. Zusätzlich fallen \(n\) normale Schreiboperationen an.
Die gesamte Arbeit für die komplette Folge ist damit \(O(n)\). Geteilt durch
\(n\) appends ergibt das konstante amortisierte Kosten pro append.

### 6.3 Was „amortisiert“ nicht bedeutet

Amortisiert ist nicht dasselbe wie Average Case:

- Average-Case-Analyse mittelt über eine angenommene Verteilung möglicher
  Eingaben.
- Amortisierte Analyse garantiert eine Schranke für jede ausreichend lange
  Operationsfolge, ohne Wahrscheinlichkeitsannahme.

Ein bestimmtes append kann weiterhin linear teuer sein. Für latenzkritische
Systeme ist dieser einzelne Ausschlag relevant; für den Gesamtdurchsatz einer
langen Folge ist die amortisierte Sicht oft passender.

Python verwendet intern nicht exakt die hier simulierte Verdopplungsstrategie.
Die konkrete Over-Allocation ist eine Implementierungsentscheidung. Das
didaktische Argument bleibt: zusätzlicher freier Platz macht Resizes selten
genug, um append amortisiert konstant zu halten.

---

## 7. Typische Analysefehler in Python

### 7.1 „Zwei Schleifen bedeuten immer O(n²)“

Falsch. Entscheidend ist, wie oft die Schleifen insgesamt laufen.

~~~python
def two_passes(values):
    """Run two consecutive linear passes."""
    for value in values:
        print(value)
    for value in values:
        print(value)
~~~

Die Kosten sind \(n+n=2n\), also \(\Theta(n)\).

Auch eine verschachtelte Schleife muss nicht quadratisch sein:

~~~python
def shrinking_work(size):
    """Halve the remaining work after each outer iteration."""
    count = 0
    current = size

    while current > 0:
        for _ in range(current):
            count += 1
        current //= 2

    return count
~~~

Die Arbeit ist

\[
n + n/2 + n/4 + \dots < 2n,
\]

also \(\Theta(n)\), nicht \(\Theta(n\log n)\).

### 7.2 Versteckte Kosten von Slicing

List Slicing kopiert die ausgewählten Referenzen:

~~~python
def recursive_sum(values):
    """Sum values while copying a suffix in every call."""
    if not values:
        return 0
    return values[0] + recursive_sum(values[1:])
~~~

Es gibt zwar \(n\) Aufrufe, aber die Slices haben Größen
\(n-1,n-2,\dots,1\). Ihre gesamten Kopierkosten sind quadratisch. Zusätzlich
werden viele temporäre Listen erzeugt. Eine Variante mit einem weitergereichten
Index vermeidet die Slices und erreicht lineare Zeit.

### 7.3 in hängt vom Container ab

~~~python
target in values_list
target in values_set
~~~

Bei einer Liste ist die Mitgliedschaftssuche im Worst Case \(O(n)\). Bei einem
Set ist sie durchschnittlich \(O(1)\), im pathologischen Worst Case jedoch
\(O(n)\). Der Operator allein verrät die Kosten nicht; der Datentyp gehört zur
Analyse.

Ein verbreiteter quadratischer Fehler ist:

~~~python
def retain_allowed(values, allowed_list):
    """Filter using repeated linear membership checks."""
    return [value for value in values if value in allowed_list]
~~~

Bei \(n\) Werten und \(m\) erlaubten Werten kostet dies \(O(nm)\). Wird
allowed_list einmal in ein Set umgewandelt, entstehen durchschnittlich
\(O(m+n)\) Zeit und \(O(m)\) zusätzlicher Speicher.

### 7.4 String-Konkatenation in Schleifen

Strings sind immutable. Eine Konkatenation kann deshalb einen neuen String
erzeugen und bisherigen Inhalt kopieren:

~~~python
def join_words_slow(words):
    """Build a string through repeated concatenation."""
    result = ""
    for word in words:
        result += word
    return result
~~~

Implementierungen dürfen manche Fälle optimieren, doch darauf sollte eine
allgemeine Analyse nicht bauen. Das robuste Muster sammelt Teile und verbindet
sie einmal:

~~~python
def join_words(words):
    """Join all words in one operation."""
    return "".join(words)
~~~

Wenn \(L\) die Gesamtlänge aller Zeichen ist, arbeitet join in \(\Theta(L)\).
Wichtig ist hier die passende Eingabegröße: Nicht nur die Anzahl der Wörter,
sondern auch ihre Gesamtlänge bestimmt die Arbeit.

### 7.5 Bibliotheksaufrufe sind keine kostenlosen Einzeloperationen

Eine Codezeile kann intern viel Arbeit ausführen:

~~~python
ordered = sorted(values)
copy = values[:]
smallest = min(values)
~~~

Typische Kosten sind \(O(n\log n)\), \(O(n)\) und \(O(n)\). Die Anzahl der
Quellcodezeilen ist kein Komplexitätsmaß.

### 7.6 Frühes return falsch verallgemeinern

Ein return in einer Schleife macht die Funktion nicht konstant. Bei linearer
Suche ist der Best Case konstant, der Worst Case bleibt linear. Für eine
vollständige Aussage müssen Szenario und Eingabeannahmen genannt werden.

### 7.7 Durchschnittliche Hash-Kosten als absolute Garantie behandeln

dict und set bieten unter üblichen Bedingungen durchschnittlich konstante
Operationen. Kollisionen und ungünstige Fälle können die Kosten verschlechtern.
Eine saubere Formulierung lautet deshalb beispielsweise:

> Die Suche benötigt erwartete beziehungsweise durchschnittliche \(O(1)\)-Zeit
> unter den üblichen Hashing-Annahmen; der Worst Case ist \(O(n)\).

---

## 8. Ein systematisches Vorgehen bei unbekanntem Code

### Schritt 1: Eingabegrößen benennen

Lege fest, wofür \(n\), \(m\) oder andere Variablen stehen. Unterschiedliche
Container können unabhängig groß sein.

### Schritt 2: Elementare und versteckte Kosten markieren

Prüfe insbesondere Schleifen, Rekursion, Slicing, Sortieren, Kopieren,
Mitgliedschaftstests, String-Aufbau und Größe der Ausgabe.

### Schritt 3: Ausführungshäufigkeiten bestimmen

Frage nicht nur „Ist hier eine Schleife?“, sondern „Wie oft läuft dieser Block
insgesamt?“ Nutze bei Bedarf Summen:

\[
\sum_{i=1}^{n} i = \frac{n(n+1)}{2} \in \Theta(n^2)
\]

oder geometrische Reihen:

\[
n+n/2+n/4+\dots \in \Theta(n).
\]

### Schritt 4: Kosten kombinieren

Aufeinanderfolgende Phasen werden addiert, geschachtelte unabhängige
Wiederholungen multipliziert. Danach bleiben dominante Terme und relevante
Eingabevariablen übrig.

### Schritt 5: Szenario benennen

Ist die Aussage Best, Average, Worst oder amortisiert? Welche Annahmen gelten
für Hashing, Verteilung oder Vorsortierung?

### Schritt 6: Speicher separat analysieren

Berücksichtige neue Container, Ausgaben, Kopien durch Slicing und den
Rekursions-Stack. Sage, ob Gesamt- oder Auxiliary Space gemeint ist.

### Schritt 7: Empirisch plausibilisieren

Erzeuge kontrollierte Eingaben, miss mehrere Größen und Wiederholungen und
vergleiche Verdopplungsquotienten. Nutze die Messung als Kontrolle deiner
Herleitung, nicht als Ersatz.

---

## 9. Analysebeispiele

### 9.1 Dreieckige Schleife

~~~python
def triangular(values):
    """Count pairs whose right index is not smaller."""
    count = 0
    for left in range(len(values)):
        for right in range(left, len(values)):
            count += 1
    return count
~~~

Die innere Schleife läuft \(n,n-1,\dots,1\)-mal. Insgesamt:

\[
n+(n-1)+\dots+1 = \frac{n(n+1)}{2} \in \Theta(n^2).
\]

Dass nicht jede innere Schleife genau \(n\)-mal läuft, ändert die Klasse nicht.

### 9.2 Halbierung mit linearer Nacharbeit

~~~python
def levels_with_scan(values):
    """Scan prefixes whose sizes are repeatedly halved."""
    total = 0
    size = len(values)

    while size > 0:
        for index in range(size):
            total += values[index]
        size //= 2

    return total
~~~

Oberflächlich sieht man eine logarithmische äußere und eine lineare innere
Schleife. Die pauschale Multiplikation \(O(n\log n)\) wäre zu grob. Die konkrete
Summe \(n+n/2+n/4+\dots\) ist \(\Theta(n)\).

### 9.3 Sortieren und danach mehrfach suchen

Angenommen, \(m\) Suchanfragen werden gegen dieselben \(n\) Werte ausgeführt:

1. Einmal sortieren: \(O(n\log n)\).
2. Jede Anfrage per binärer Suche: \(O(\log n)\).
3. Insgesamt: \(O(n\log n + m\log n)\).

Lineare Suche ohne Vorbereitung kostet \(O(mn)\). Ob sich das Sortieren lohnt,
hängt von \(m\), \(n\), Mutationen der Daten und konstanten Faktoren ab. Für nur
eine kleine Suche kann der einfache lineare Durchlauf schneller sein; bei vielen
Anfragen amortisiert sich die Vorbereitung.

### 9.4 Ausgabe-sensitive Analyse

Eine Funktion findet alle Trefferpaare in zwei Listen. Selbst mit schnellen
Lookups kann sie im Fall vieler Duplikate \(k\) Treffer ausgeben. Eine
informative Schranke lautet dann etwa \(O(n+m+k)\), wobei \(k\) die Ausgabegröße
ist. Nur \(O(n+m)\) anzugeben würde die Materialisierung der Ausgabe ignorieren.

---

## 10. Selbstkontrolle

### Aufgabe 1

Welche Zeit- und zusätzliche Speicherkomplexität hat die Funktion?

~~~python
def reverse_copy(values):
    result = []
    for index in range(len(values) - 1, -1, -1):
        result.append(values[index])
    return result
~~~

### Aufgabe 2

Warum ist dieser Code nicht \(O(n)\), obwohl nur eine sichtbare Schleife
existiert?

~~~python
def prefixes(values):
    result = []
    for index in range(len(values)):
        result.append(values[: index + 1])
    return result
~~~

### Aufgabe 3

Zwei Algorithmen benötigen \(50n\log_2 n\) beziehungsweise \(n^2\) abstrakte
Schritte. Erkläre ohne pauschal „Big-O ignoriert Konstanten“ zu sagen, weshalb
der erste langfristig gewinnt und weshalb der zweite für kleine Eingaben
trotzdem schneller sein kann.

### Aufgabe 4

Ordne die Szenarien den Begriffen Best Case, Worst Case oder amortisiert zu:

1. Die Kosten eines einzelnen append, bei dem gerade ein Resize nötig ist.
2. Die durchschnittlichen Kosten pro append über eine lange Folge.
3. Lineare Suche, wenn das Ziel an Position 0 steht.
4. Lineare Suche, wenn das Ziel nicht vorkommt.

### Lösungen

1. reverse_copy benötigt \(\Theta(n)\) Zeit und \(\Theta(n)\) zusätzlichen
   Speicher für die Ergebnisliste.
2. Jeder Slice kopiert einen wachsenden Prefix. Die kopierten Längen summieren
   sich zu \(1+2+\dots+n=\Theta(n^2)\). Auch die materialisierte Ausgabe hat
   quadratische Gesamtgröße.
3. Das Verhältnis
   \[
   \frac{n^2}{50n\log_2 n} = \frac{n}{50\log_2 n}
   \]
   wächst langfristig ohne Grenze. Daher wird der linearithmische Algorithmus
   ab einem Schnittpunkt schneller. Vor diesem Schnittpunkt kann sein Faktor 50
   den asymptotischen Vorteil überwiegen.
4. Ein Resize-append ist der teure Einzelfall; die Kosten über die Folge werden
   amortisiert betrachtet. Position 0 ist der Best Case, ein fehlendes Ziel der
   Worst Case.

---

## 11. Zusammenfassung

Komplexitätsanalyse beschreibt das Wachstum von Zeit und Speicher in Abhängigkeit
von klar benannten Eingabegrößen. Sie ersetzt Messungen nicht, sondern ergänzt
sie: Die Code-Analyse liefert eine asymptotische Begründung, während kontrollierte
Messreihen zeigen, ob das beobachtete Verhalten im relevanten Größenbereich dazu
passt.

Die zentralen Gedanken dieses Moduls sind:

- Konstanten und kleine Terme können praktisch wichtig sein, ändern aber nicht
  die langfristige Wachstumsklasse.
- \(O\) ist eine obere, \(\Omega\) eine untere und \(\Theta\) eine enge
  asymptotische Schranke.
- Best, Average, Worst und amortisiert beschreiben unterschiedliche
  Betrachtungsweisen und müssen explizit benannt werden.
- Versteckte Python-Kosten wie Slicing, Listenmitgliedschaft, String-Kopien und
  Bibliotheksaufrufe gehören vollständig in die Analyse.
- Zeit und Speicher sind getrennt zu untersuchen; häufig wird das eine gegen das
  andere getauscht.
- Bei Verdopplung der Eingabe wächst \(n\log n\) nur etwas stärker als um den
  Faktor zwei, \(n^2\) dagegen ungefähr um den Faktor vier. Darum schlägt
  \(O(n\log n)\) für hinreichend große \(n\) \(O(n^2)\), unabhängig von festen
  konstanten Faktoren.

Wenn du eine Laufzeitklasse aus Code herleiten, sie durch eine saubere Messreihe
plausibilisieren und die getroffenen Annahmen erklären kannst, sind die
Qualifikationsziele dieses Theorieabschnitts erreicht.
