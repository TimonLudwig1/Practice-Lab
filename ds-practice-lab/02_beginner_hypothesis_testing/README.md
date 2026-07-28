# 02 — Hypothesis Testing: Do Smokers Tip Differently? 🧪 `[from your lectures]`

Difficulty: 🟢 Beginner | Topic: Statistics — Hypothesis Testing

## 🎯 Project Goal
Run real hypothesis tests (t-test, Mann-Whitney U, chi-square) on the classic `tips` restaurant dataset and learn to interpret p-values *correctly*.

## 📚 What You'll Learn
- The logic of null hypothesis significance testing: H₀, H₁, test statistic, p-value, α
- What a p-value actually means (and the 3 most common misinterpretations)
- Two-sample t-test with `scipy.stats.ttest_ind` — and when it's valid (assumptions!)
- Checking assumptions: normality (histogram + Shapiro-Wilk) and equal variance (Levene)
- Non-parametric fallback: Mann-Whitney U test
- Chi-square test of independence for two categorical variables
- Effect size (Cohen's d) — why "significant" ≠ "important"

## 🗂️ Dataset Description
**Tips** — 244 restaurant bills: total bill, tip, sex, smoker (yes/no), day, time, party size. Ships with seaborn:

```python
import seaborn as sns
df = sns.load_dataset("tips")
```

## 🚀 Getting Started
```bash
cd 02_beginner_hypothesis_testing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Step-by-Step Guide
1. **Load & explore** — look at the data, compute the *tip percentage* (`tip / total_bill * 100`) as a new column. *Why:* raw tip amounts are confounded by bill size; the percentage is the fair comparison unit.
2. **State your hypotheses BEFORE looking at group means.** H₀: smokers and non-smokers have the same mean tip percentage. H₁: they differ (two-sided). *Why:* deciding the hypothesis after peeking at the data is p-hacking — exactly what your lectures warn about.
3. **Visualize the two groups** — histograms / box plots of tip pct for smokers vs non-smokers (matplotlib practice!). *Why:* a test should never surprise you if you've seen the picture first.
4. **Check assumptions** — the t-test assumes roughly normal group distributions (or large n) and comparable variances. Use Shapiro-Wilk (`scipy.stats.shapiro`) and Levene (`scipy.stats.levene`). *Why:* running a test whose assumptions are violated gives you a precise answer to the wrong question.
5. **Run the t-test** — `ttest_ind(group_a, group_b, equal_var=...)`. Interpret: if p < 0.05 we reject H₀. Write one sentence in plain English about what that does and does NOT tell you.
6. **Run the non-parametric cousin** — Mann-Whitney U (`scipy.stats.mannwhitneyu`). *Why:* tip percentages are right-skewed; comparing both tests teaches you how much assumptions matter.
7. **Compute Cohen's d** — effect size = (mean₁ − mean₂) / pooled std. *Why:* with huge samples, microscopic differences become "significant"; effect size tells you if anyone should care.
8. **Chi-square test** — is smoking status independent of the day of the week? Build a contingency table with `pd.crosstab`, then `scipy.stats.chi2_contingency`. *Why:* t-tests handle numeric vs binary; chi-square handles categorical vs categorical.
9. **Write a conclusion cell** — 3-4 sentences summarizing your findings like you'd report them to a non-statistician.

## ✅ Completion Checklist
- [ ] I created the tip-percentage column and can explain why
- [ ] I wrote H₀ and H₁ down *before* computing group means
- [ ] I checked normality and variance assumptions
- [ ] I ran and interpreted a t-test, Mann-Whitney U, and chi-square test
- [ ] I computed Cohen's d and interpreted it (small/medium/large)
- [ ] I can state what a p-value is in one correct sentence
- [ ] I can name 2 wrong interpretations of "p = 0.03"

## 💡 Hints & Tips
- Splitting groups: `smokers = df.loc[df["smoker"] == "Yes", "tip_pct"]` and similarly for `"No"`.
- p-value, one correct sentence: *"The probability of observing a difference at least this extreme, assuming H₀ is true."* It is NOT the probability that H₀ is true.
- Cohen's d rule of thumb: 0.2 small, 0.5 medium, 0.8 large.
- `chi2_contingency` returns four things — the p-value is the **second** one.
- If Shapiro-Wilk rejects normality, don't panic: with n > 30 per group the t-test is fairly robust (Central Limit Theorem). Mention this trade-off in your conclusion.

## 🔗 Further Reading
- [scipy.stats tutorial](https://docs.scipy.org/doc/scipy/tutorial/stats.html)
- [ASA Statement on p-values](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf) — short and worth it
- [Seeing Theory — Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/index.html) (interactive)

---

# Deutsche Übersetzung

# 02 — Hypothesentests: Geben Raucher anders Trinkgeld? 🧪 `[aus deinen Vorlesungen]`

Schwierigkeit: 🟢 Einsteiger | Thema: Statistik — Hypothesentests

## 🎯 Projektziel
Führe mit dem klassischen Restaurantdatensatz `tips` echte Hypothesentests durch – t-Test, Mann-Whitney-U-Test und Chi-Quadrat-Test – und lerne, p-Werte *korrekt* zu interpretieren.

## 📚 Das lernst du
- Die Logik von Signifikanztests: H₀, H₁, Teststatistik, p-Wert und α
- Was ein p-Wert tatsächlich bedeutet und welche drei Fehlinterpretationen besonders häufig sind
- Den Zweistichproben-t-Test mit `scipy.stats.ttest_ind` und seine Voraussetzungen
- Prüfung der Voraussetzungen: Normalverteilung mit Histogramm und Shapiro-Wilk-Test sowie Varianzgleichheit mit dem Levene-Test
- Den nichtparametrischen Mann-Whitney-U-Test als Alternative
- Den Chi-Quadrat-Unabhängigkeitstest für zwei kategoriale Variablen
- Die Effektstärke Cohen's d und den Unterschied zwischen statistischer Signifikanz und praktischer Bedeutung

## 🗂️ Beschreibung des Datensatzes
**Tips** enthält 244 Restaurantrechnungen mit Rechnungsbetrag, Trinkgeld, Geschlecht, Raucherstatus, Wochentag, Tageszeit und Gruppengröße. Der Datensatz wird mit seaborn ausgeliefert:

```python
import seaborn as sns
df = sns.load_dataset("tips")
```

## 🚀 Erste Schritte
```bash
cd 02_beginner_hypothesis_testing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Schritt-für-Schritt-Anleitung
1. **Laden und untersuchen** — Sieh dir die Daten an und berechne den *prozentualen Trinkgeldanteil* (`tip / total_bill * 100`) als neue Spalte. *Warum:* Der absolute Trinkgeldbetrag hängt von der Rechnungshöhe ab; der Anteil ermöglicht einen faireren Vergleich.
2. **Formuliere deine Hypothesen, bevor du die Gruppenmittelwerte ansiehst.** H₀: Raucher und Nichtraucher geben im Mittel denselben prozentualen Trinkgeldanteil. H₁: Die Werte unterscheiden sich. *Warum:* Eine Hypothese erst nach der Betrachtung der Daten festzulegen ist p-Hacking.
3. **Visualisiere die beiden Gruppen** — Erstelle Histogramme oder Boxplots für Raucher und Nichtraucher. *Warum:* Ein statistischer Test sollte dich nicht überraschen, wenn du die Verteilungen vorher betrachtet hast.
4. **Prüfe die Voraussetzungen** — Der t-Test setzt ungefähr normalverteilte Gruppen oder ausreichend große Stichproben sowie vergleichbare Varianzen voraus. Verwende Shapiro-Wilk (`scipy.stats.shapiro`) und Levene (`scipy.stats.levene`). *Warum:* Bei verletzten Voraussetzungen beantwortet der Test möglicherweise nicht die beabsichtigte Frage.
5. **Führe den t-Test durch** — Verwende `ttest_ind(group_a, group_b, equal_var=...)`. Bei p < 0,05 wird H₀ verworfen. Formuliere in einem einfachen Satz, was das Ergebnis aussagt und was nicht.
6. **Führe die nichtparametrische Alternative durch** — Verwende den Mann-Whitney-U-Test (`scipy.stats.mannwhitneyu`). *Warum:* Trinkgeldanteile sind rechtsschief; der Vergleich beider Tests zeigt den Einfluss der Voraussetzungen.
7. **Berechne Cohen's d** — Effektstärke = (Mittelwert₁ − Mittelwert₂) / gepoolte Standardabweichung. *Warum:* Bei sehr großen Stichproben können winzige Unterschiede signifikant werden; die Effektstärke zeigt ihre praktische Bedeutung.
8. **Chi-Quadrat-Test** — Untersuche, ob Raucherstatus und Wochentag unabhängig sind. Erstelle mit `pd.crosstab` eine Kontingenztafel und verwende `scipy.stats.chi2_contingency`. *Warum:* t-Tests vergleichen numerische Werte zwischen Gruppen; der Chi-Quadrat-Test untersucht zwei kategoriale Variablen.
9. **Schreibe eine Schlussfolgerung** — Fasse deine Ergebnisse in drei bis vier Sätzen für eine Person ohne Statistikkenntnisse zusammen.

## ✅ Checkliste zum Abschluss
- [ ] Ich habe den prozentualen Trinkgeldanteil berechnet und kann erklären, warum.
- [ ] Ich habe H₀ und H₁ vor der Berechnung der Gruppenmittelwerte formuliert.
- [ ] Ich habe Normalverteilung und Varianzgleichheit geprüft.
- [ ] Ich habe einen t-Test, einen Mann-Whitney-U-Test und einen Chi-Quadrat-Test durchgeführt und interpretiert.
- [ ] Ich habe Cohen's d berechnet und als kleinen, mittleren oder großen Effekt eingeordnet.
- [ ] Ich kann die Bedeutung eines p-Werts in einem korrekten Satz erklären.
- [ ] Ich kann zwei falsche Interpretationen von „p = 0,03“ nennen.

## 💡 Hinweise und Tipps
- Teile die Gruppen beispielsweise mit `smokers = df.loc[df["smoker"] == "Yes", "tip_pct"]` und entsprechend für `"No"` auf.
- Eine korrekte Formulierung lautet: *„Der p-Wert ist die Wahrscheinlichkeit, unter Annahme von H₀ einen mindestens so extremen Unterschied zu beobachten.“* Er ist nicht die Wahrscheinlichkeit dafür, dass H₀ wahr ist.
- Faustregel für Cohen's d: 0,2 klein, 0,5 mittel und 0,8 groß.
- `chi2_contingency` liefert vier Werte zurück; der p-Wert ist der **zweite**.
- Wenn Shapiro-Wilk die Normalverteilung verwirft, ist das nicht automatisch ein Problem: Bei mehr als 30 Beobachtungen pro Gruppe ist der t-Test aufgrund des zentralen Grenzwertsatzes recht robust. Erwähne diese Abwägung in deinem Fazit.

## 🔗 Weiterführende Informationen
- [Tutorial zu scipy.stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html)
- [ASA-Erklärung zu p-Werten](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf)
- [Seeing Theory — Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/index.html) (interaktiv)
