# Concrete Compressive Strength Prediction | Applied Data Science

A small reproducible Python research project built as interview preparation for an AI & Data Science working-student role in applied construction research. **Independent learning project; not commissioned by or affiliated with IAB Weimar.**

## Research question

Can an interpretable linear model or a non-linear random forest estimate measured concrete compressive strength (MPa) from mixture components and curing age?

## Dataset and attribution

- **I-Cheng Yeh (1998), Concrete Compressive Strength**, UCI Machine Learning Repository, https://doi.org/10.24432/C5PK67
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
- **1,030 observations; 8 numeric features; one regression target.** Features: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate (all kg/m³), and curing age (days). Target: measured compressive strength in MPa.
- The original dataset is downloaded using `ucimlrepo` and locally cached; it is not committed to this repository. Internet required for the first download.

## Setup (Windows PowerShell)

```powershell
cd iab-concrete-strength-ml
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.load_data
python -m src.train_model
python -m pytest -q
```

If PowerShell blocks activation, use `./.venv/Scripts/python.exe -m pip install -r requirements.txt` and that same Python executable for subsequent commands.

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.load_data
python -m src.train_model
python -m pytest -q
```

Execute commands from the **repository root**, not from the `src/` folder.

## Workflow

1. **Data acquisition:** Download the official UCI dataset (ID 165), standardize column names by documented feature order, and cache as CSV.
2. **Quality checks:** Inspect and validate missing values, non-numeric values, nonfinite values, plausible nonnegative ingredient values, and positive age/strength. Report duplicates rather than automatically deleting them: repeated recipes can be real observations.
3. **Exploratory annotation:** Create relative lower/middle/upper strength bands using dataset tertiles. These are **not structural grades** and **not model inputs**.
4. **Train/test:** Split 80% training and 20% held-out test (`random_state=42`).
5. **Model selection:** Five-fold shuffled cross-validation on **training only**, comparing Linear Regression and Random Forest by mean absolute error (MAE). Preprocessing is inside each ML pipeline to avoid test-to-train leakage.
6. **Final evaluation:** Fit only the selected model on all training rows, then measure MAE, RMSE and R² once on the held-out test set.
7. **Research communication:** Export plots, metrics, fitted model, holdout predictions and a short generated report.

## Repository map

```text
src/load_data.py           Official UCI acquisition and local caching
src/preprocess.py          Checks, exploratory labels and quality diagnostics
src/train_model.py         CV model selection, training and evaluation
notebooks/                 Optional exploratory notebooks
tests/test_preprocess.py   Validation and end-to-end smoke tests
data/README.md             Citation and data sourcing
reports/README.md          Explanation of generated outputs
```

## Results

**Results must be generated locally on the actual UCI dataset.** This repository does not invent a model accuracy score. Run `python -m src.train_model` and inspect `reports/results.md`, `reports/metrics.json` and the generated graphs. You can then add your measured results here.

| Model | 5-fold training CV MAE (MPa) |
|---|---|
| Linear Regression | Run the experiment |
| Random Forest | Run the experiment |

**Held-out test MAE / RMSE / R²:** Run the experiment and report the actual numbers.

## Limitations

- Correlations are not causal evidence.
- Small historical dataset; generalization to new materials, plants and methods is uncertain.
- Same-recipe or near-duplicate observations might occur across random splits and inflate reported performance for novel mixtures; stronger research would consider a grouped split if recipe identifiers are available.
- Model is **not suitable for structural design, certification or construction safety decisions**.
- Exploratory strength bands depend on dataset distribution and are not certified material classes.

## Next improvements

- Compare feature-importance methods and uncertainty intervals.
- Add group-aware evaluation for repeated formulations when unique mixture IDs exist.
- Add an external validation dataset and data-quality monitoring.

## Interview summary (Deutsch)

> Ich habe die Betondruckfestigkeit mithilfe von Python und Machine Learning untersucht. Dafür habe ich einen offenen Datensatz geladen, überprüft und analysiert. Anschließend habe ich lineare Regression und Random Forest verglichen. Die Modellauswahl erfolgte mit Kreuzvalidierung auf den Trainingsdaten. Die endgültige Leistung habe ich auf separaten Testdaten gemessen. Wichtig waren mir reproduzierbare Ergebnisse, verständliche Dokumentation und die Grenzen des Modells.
