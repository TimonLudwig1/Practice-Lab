# 01 — Matplotlib Fundamentals: Penguins EDA 🐧

Difficulty: 🟢 Beginner | Topic: EDA & Visualization

## 🎯 Project Goal
Learn matplotlib from the ground up by exploring the Palmer Penguins dataset — and build a mental "translation table" from the ggplot/lets-plot world you already know.

## 📚 What You'll Learn
- The matplotlib mental model: **Figure → Axes → plot calls** (very different from ggplot's grammar of graphics!)
- The two APIs: `pyplot` (quick) vs the **object-oriented API** (`fig, ax = plt.subplots()`) — you'll use the OO one, it's the professional standard
- Core chart types: histogram, scatter, bar, box plot
- Subplots (matplotlib's version of `facet_wrap`)
- Styling: labels, titles, legends, colors
- Saving figures to disk
- Basic pandas EDA: `.info()`, `.describe()`, `.value_counts()`, `.groupby()`

## 🗂️ Dataset Description
**Palmer Penguins** — 344 penguins from 3 species (Adelie, Chinstrap, Gentoo) observed in Antarctica. Columns: species, island, bill length/depth (mm), flipper length (mm), body mass (g), sex. It ships with seaborn, so there is nothing to download:

```python
import seaborn as sns
df = sns.load_dataset("penguins")
```

## 🚀 Getting Started
```bash
cd 01_beginner_matplotlib_eda
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Step-by-Step Guide

### Coming from ggplot? Read this first.
In ggplot you describe *what* you want (`aes(x=..., y=...) + geom_point()`) and the library figures out how. In matplotlib you build the picture *imperatively*: you create a canvas (Figure), place coordinate systems on it (Axes), and call drawing methods on those Axes. The standard pattern you will use in **every single cell**:

```python
fig, ax = plt.subplots(figsize=(8, 5))   # canvas + one axes
ax.<some_plot_method>(...)                # draw on it
ax.set_xlabel("..."); ax.set_ylabel("..."); ax.set_title("...")
plt.show()
```

Translation cheat sheet:

| ggplot / lets-plot | matplotlib |
|---|---|
| `geom_point()` | `ax.scatter(x, y)` |
| `geom_histogram()` | `ax.hist(values)` |
| `geom_bar()` | `ax.bar(categories, heights)` |
| `geom_boxplot()` | `ax.boxplot(...)` |
| `facet_wrap(~var)` | `fig, axes = plt.subplots(nrows, ncols)` |
| `labs(title=, x=, y=)` | `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()` |
| `aes(color=species)` | you loop over groups and plot each with a `label=` + `ax.legend()` |

That last row is the biggest mindset shift: matplotlib has no `aes()` mapping. To color by category, you filter the DataFrame per category and call `ax.scatter()` once per group.

### The steps (mirrored in the notebook)
1. **Load & inspect** — load penguins, run `.info()`, `.describe()`, `.isna().sum()`. *Why:* you never plot data you haven't looked at; you need to know dtypes and missing values first.
2. **Clean** — drop rows with missing values (`.dropna()`). *Why:* matplotlib will choke or silently skip NaNs; for an EDA project dropping 11 rows is acceptable.
3. **First plot: histogram** of body mass. *Why:* histograms are the fastest way to understand a single numeric variable's distribution.
4. **Scatter plot** of flipper length vs body mass, colored by species (loop over species!). *Why:* this teaches the group-loop pattern AND reveals the dataset's clearest relationship.
5. **Bar chart** of penguin counts per island (use `.value_counts()` first). *Why:* bar charts need pre-aggregated data in matplotlib — another difference from ggplot, which aggregates for you.
6. **Box plots** of bill length per species. *Why:* box plots compare distributions across groups at a glance.
7. **Subplots** — a 2×2 figure combining four views. *Why:* `plt.subplots(2, 2)` returns an *array* of Axes; mastering this unlocks faceting.
8. **Polish & save** — pick your best figure, add proper labels/title/legend, save with `fig.savefig("../outputs/figures/penguins_final.png", dpi=150, bbox_inches="tight")`. *Why:* a figure nobody can read or find is a figure that doesn't exist.

## ✅ Completion Checklist
- [ ] I can explain the difference between a Figure and an Axes
- [ ] I used `fig, ax = plt.subplots()` in every plot (no bare `plt.plot()`)
- [ ] I made a histogram, scatter, bar chart, and box plot
- [ ] I colored a scatter plot by category using the group-loop pattern
- [ ] I built a 2×2 subplot grid
- [ ] I saved at least one figure as PNG into `outputs/figures/`
- [ ] I can name 3 differences between ggplot and matplotlib

## 💡 Hints & Tips
- `plt.subplots(2, 2)` returns `(fig, axes)` where `axes` is a 2D numpy array — access with `axes[0, 0]`, `axes[0, 1]`, …
- If your labels overlap, call `fig.tight_layout()` before showing/saving.
- `ax.hist(df["body_mass_g"], bins=20)` — always experiment with `bins`.
- For the box plot, `ax.boxplot()` wants a *list of arrays*, one per group: `[df.loc[df.species == s, "bill_length_mm"] for s in species_list]`, then `ax.set_xticklabels(species_list)`.
- Stuck on coloring by group? Pattern: `for species, group in df.groupby("species"): ax.scatter(group[x], group[y], label=species)` then `ax.legend()`.

## 🔗 Further Reading
- [Matplotlib Quick Start Guide](https://matplotlib.org/stable/users/explain/quick_start.html) — read the "Parts of a Figure" diagram, it's gold
- [Anatomy of a Figure](https://matplotlib.org/stable/gallery/showcase/anatomy.html)
- [Lifecycle of a Plot](https://matplotlib.org/stable/tutorials/lifecycle.html) — pyplot vs OO API explained
- [Palmer Penguins background](https://allisonhorst.github.io/palmerpenguins/)
