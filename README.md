# Freight Rate Prediction

Predict each freight load's posted rate in dollars from its route, distance, equipment, weight, and date. The assessment provides labeled January-October loads and unlabeled November-December loads, plus a fixed December lane chart.

## Setup

Use Python 3.10 or newer. From the repository root:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-pipeline.txt
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-pipeline.txt
```

## Data files

The assessment CSVs are inputs and are not included in this repository. Put them in `data/`:

- `train_test.csv` (48,000 labeled rows)
- `validation.csv` (12,000 unlabeled rows)
- `validation_predictions_template.csv` (required output IDs)
- `december_chart_inputs.csv` (31 fixed chart dates)

The pipeline uses underscores in its filenames. If the supplied files use hyphens, rename them after copying into `data/`.

**Windows PowerShell**

```powershell
Rename-Item .\data\train-test.csv train_test.csv
Rename-Item .\data\validation-predictions-template.csv validation_predictions_template.csv
Rename-Item .\data\december-chart-inputs.csv december_chart_inputs.csv
```

**macOS or Linux**

```bash
mv data/train-test.csv data/train_test.csv
mv data/validation-predictions-template.csv data/validation_predictions_template.csv
mv data/december-chart-inputs.csv data/december_chart_inputs.csv
```

`validation.csv` already has the expected name. If a supplied file already has underscores, leave that filename unchanged. The input data stays local under the gitignored `data/` directory.

## Train, predict, score, and test

Run from the repository root. The commands are the same in PowerShell and macOS/Linux:

```text
python -m src.train --config configs/config.yaml
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python -m unittest discover -s tests -v
```

To reproduce the exploratory data summary and figures separately:

```text
python scripts/eda.py --output-dir reports/eda_run --findings reports/eda_run/EDA_FINDINGS.md
```

On a four-thread machine, training and final prediction take about 6-8 minutes. Scoring and tests take a few seconds; EDA usually takes under a minute.

Training writes `validation_predictions.csv`, the completed `data/december_chart_inputs.csv`, a fitted model under ignored `artifacts/`, and diagnostic metrics under `reports/`. Scoring checks the 12,000 validation rows and 31 December rows and writes `scorer_results/candidate_december.png`. The submission copy of the chart is `reports/figures/december_chart.png`; the Word report is `reports/report.docx`.

## Approach

- Five expanding-window folds score May through September; every fold trains on earlier dates only.
- October is evaluated after the CV comparisons as a later-period check.
- A fold-local leave-one-out rate-per-mile screen removes ratios below 0.4 or above 2.5 from model fitting.
- Numeric imputation and categorical handling are fitted on the current training rows; validation rows are used only for prediction.
- The median rate-per-mile baseline uses the same cleaned fit rows.
- Two CatBoost candidates are compared by pooled CV MAE; the selected model uses a log target to reduce sensitivity to the right-skewed rate scale.

## Repository layout

```text
src/                   pipeline modules
configs/config.yaml    final model and split settings
tests/                 core behavior checks
scripts/eda.py         reproducible exploratory analysis
reports/report.docx    final assessment report
reports/figures/       report figure
score.py               supplied output validator
```

## Limitations

- The ratio screen is a heuristic: it can remove legitimate loads and miss incorrect labels.
- Validation labels are unavailable and may contain similar target corruption; local future accuracy cannot be measured.
- October informed the decision to use cleaning, so October is no longer a pure holdout; the five CV folds are the primary performance evidence.
- Training covers only ten months, and validation contains market-index shifts and unseen cities and lanes.


