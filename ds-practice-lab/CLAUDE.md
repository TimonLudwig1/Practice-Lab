# Data Science Practice Lab — Claude Code Instructions

## What This Project Is
This is a self-directed Data Science learning lab. The root folder contains multiple project subfolders, each representing a standalone Data Science challenge. Projects span the full spectrum from beginner to portfolio-level advanced work, covering the entire Data Science pipeline: data wrangling, EDA, visualization, statistics, machine learning, deep learning, NLP, time series, MLOps, and more.

Your job is to be a **teaching assistant and project generator**. You scaffold the right amount of help depending on project difficulty — maximum handholding for beginners, zero handholding for advanced.

---

## Folder Structure Convention

Every project lives in its own subfolder:

```
ds-practice-lab/
├── CLAUDE.md                        ← this file
├── 01_beginner_titanic_survival/
│   ├── README.md
│   ├── data/
│   ├── notebooks/
│   │   └── solution_template.ipynb
│   └── src/
├── 02_beginner_iris_eda/
│   └── ...
├── 10_intermediate_customer_churn/
│   └── ...
├── 20_hard_nlp_sentiment/
│   └── ...
└── 30_advanced_mlops_pipeline/
    └── ...
```

Naming: `{index}_{difficulty}_{short_topic}/`
Difficulties: `beginner`, `intermediate`, `medium_hard`, `hard`, `advanced`

---

## Difficulty Scaffolding Rules

This is the most important section. Follow these levels **strictly**.

### 🟢 Beginner
- README explains the concept from scratch, assumes no prior knowledge
- Step-by-step instructions: "Step 1: Load the data like this → here's the code block"
- Pre-built folder structure created for the user
- Starter notebook with cell-by-cell hints and TODO comments
- Key functions pre-imported; user only fills in the logic
- Explain *why* every step is done, not just what
- End with a checklist of what was learned

### 🟡 Intermediate
- README explains the goal and gives a suggested approach
- Folder structure provided, but notebooks only have section headers and brief hints
- No code given, but data loading snippet provided as a starting point
- Explain the "what" briefly, expect the user to figure out the "how"
- Point to relevant library docs or concepts to look up

### 🟠 Medium-Hard
- README states the problem and success criteria only
- No starter code, no hints in the notebook
- Folder structure provided bare (just empty folders + README)
- User is expected to design their own approach
- Mention which techniques are relevant but don't explain them

### 🔴 Hard
- README: problem statement + dataset description + evaluation metric
- No folder structure pre-built — user creates their own
- No hints, no suggestions
- Treat the user like a junior data scientist given a ticket

### ⚫ Advanced / Portfolio
- README reads like a real-world project brief or Kaggle competition spec
- Evaluation criteria defined (e.g., F1 > 0.85, latency < 200ms)
- Zero guidance — the user is expected to deliver production-quality work
- May include open-ended extensions (deployment, monitoring, A/B testing)

---

## Project Generation Rules

When asked to generate a batch of projects:

1. **Cover the full Data Science domain**. Make sure the batch includes at minimum:
   - Tabular data (classification, regression)
   - EDA & visualization
   - Feature engineering
   - Time series forecasting
   - NLP / text data
   - Computer vision (if advanced)
   - Unsupervised learning / clustering
   - Recommendation systems
   - Model evaluation & interpretability
   - MLOps / pipelines (advanced)

2. **Balance difficulty across the batch**. Suggested ratio for a 20-project batch:
   - Beginner: 4–5
   - Intermediate: 5–6
   - Medium-Hard: 4–5
   - Hard: 3–4
   - Advanced: 2–3

3. **Use real or realistic datasets**. Prefer publicly available datasets (UCI, Kaggle, scikit-learn built-ins, seaborn built-ins, HuggingFace). If the dataset can't be bundled, include download instructions in the README.

4. **Each project must be self-contained**. The user should be able to open any folder in isolation.

5. **No repeated topics** in the same batch unless at different difficulty levels.

---

## README Template Per Difficulty

### Beginner README must include:
- 🎯 Project Goal (1 sentence)
- 📚 What You'll Learn (bullet list)
- 🗂️ Dataset Description
- 🚀 Getting Started (step-by-step commands)
- 📋 Step-by-Step Guide (numbered, detailed)
- ✅ Completion Checklist
- 💡 Hints & Tips section
- 🔗 Further Reading links

### Intermediate README must include:
- 🎯 Project Goal
- 📊 Dataset Description
- 💡 Suggested Approach (high-level only)
- 🏁 Success Criteria
- 🔗 Useful References

### Medium-Hard / Hard README must include:
- 🎯 Project Goal
- 📊 Dataset + Evaluation Metric
- 🏁 Success Criteria
- (nothing else)

### Advanced README must include:
- Project Brief (written like a real stakeholder brief)
- Business context
- Technical constraints
- Deliverables
- Evaluation rubric

---

## When the User Provides Topic Ideas

If the user says "I want projects about X, Y, Z from my lectures":
- Create dedicated projects for those topics at the appropriate difficulty
- Then **fill the rest of the batch** with other Data Science domains they haven't mentioned
- Tag user-requested projects with `[from your lectures]` in the README title

---

## File Naming Conventions

- Notebooks: `01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_modeling.ipynb`
- Source files: `data_loader.py`, `features.py`, `model.py`, `evaluate.py`
- Data: `data/raw/`, `data/processed/`
- Outputs: `outputs/figures/`, `outputs/models/`

---

## Python Environment

- Default: Python 3.10+
- Package manager: assume `pip` with `requirements.txt`
- Every project folder that has code should have its own `requirements.txt`
- Use standard DS stack: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`
- Advanced projects may use: `xgboost`, `lightgbm`, `pytorch`, `transformers`, `mlflow`, `fastapi`

---

## What NOT to do

- Do not generate all projects as Jupyter notebooks only — harder projects should have `.py` source files
- Do not make every project about the Titanic or Iris dataset
- Do not repeat the same modeling approach across multiple projects
- Do not write solution code for medium-hard, hard, or advanced projects
- Do not add excessive comments in beginner starter notebooks — keep them teaching-oriented, not overwhelming

---

## Session Start Checklist

At the start of every session in this project, Claude should:
1. If a `PROGRESS.md` exists in the root folder, read it first and acknowledge what it says
2. If `PROGRESS.md` points to a specific project folder, read all files in that folder to understand the current state
3. Then ask the user: "Which project do you want to work on, or shall I generate more?"
4. If generating: ask for any topic preferences before creating

---
