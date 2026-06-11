# Model Instability Analysis

A Python pipeline for assessing **prediction instability** in clinical risk models via bootstrap resampling. This repository implements analyses for Logistic Regression and Artificial Neural Network (MLP) models across varying sample sizes using the GUSTO-I trial dataset.

---

## Background

When a clinical prediction model is retrained on resampled data derived from the same population, how much do its predicted probabilities change? This pipeline formalises that question using:

| Metric | What it captures |
|--------|-----------------|
| **C-statistic optimism** | Over-fitting bias — gap between apparent AUC and bootstrap AUC |
| **MAPE** (Mean Absolute Prediction Error) | Average absolute deviation of bootstrapped probabilities from the original model |
| **CII** (Classification Instability Index) | Proportion of bootstrap models that would flip a patient across a clinical threshold |
| **Calibration slope / CITL** | Whether the model systematically over- or under-predicts after resampling |

Analyses are repeated across six stratified subsamples of the full dataset (`fractions = [1, 0.50, 0.25, 0.125, 0.05, 0.0125]`) to reveal how sample size drives instability.

---

## Project Structure

The repository has been structured cleanly to keep core analysis scripts tracked in version control, while data and generated results are ignored:

```
model_instability/
├── combine_charts.py               # Combined cross-sample size plotting code
├── logistic.py                     # Standalone script for Logistic Regression instability analysis
├── mlp.py                          # Standalone script for MLP instability analysis
├── requirements.txt                # Python dependencies
├── LICENSE                         # Project license
├── .gitignore                      # Git ignore patterns (ignores local dataset/ and results/)
└── README.md                       # This file
```

> [!NOTE]
> Local execution generates a `dataset/` directory (for data subsamples) and a `results/` directory (for generated plots, metrics, and logs). These directories are ignored by Git to keep the repository clean.

---

## Quickstart

### 1. Prerequisites

- Python ≥ 3.10
- A virtual environment is strongly recommended

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

Run the scripts directly from the project root:

```bash
# Run Logistic Regression instability analysis
python logistic.py

# Run MLP instability analysis
python mlp.py

# Generate combined charts comparing the two models
python combine_charts.py
```

---

## Dataset

The pipeline uses the **GUSTO-I** trial dataset (`dataset/gusto_dataset(Sheet1).csv`), a landmark cardiovascular trial examining 30-day mortality (`day30`) following myocardial infarction.

**Features used:**

| Column | Description |
|--------|-------------|
| `age` | Age in years |
| `sex` | Sex (`male`/`female` → encoded to 1/0) |
| `hyp` | Hypotension |
| `htn` | Hypertension |
| `hrt` | Heart rate |
| `ste` | ST-elevation |
| `pmi` | Prior myocardial infarction (`yes`/`no` → 1/0) |
| `sysbp` | Systolic blood pressure |
| `day30` | **Outcome** — 30-day mortality (binary) |

---

## How the Pipeline Works

```
Load CSV  →  Preprocess  →  For each sample fraction:
                                │
                                ├── GridSearchCV (tune hyperparams)
                                ├── Fit original model  →  Apparent AUC
                                │
                                ├── C-statistic bootstrap (n=200)
                                │       └── Optimism = |C_bootstrap − C_apparent|
                                │
                                ├── Prediction stability bootstrap (n=200)
                                │       ├── MAPE per patient
                                │       ├── LOWESS confidence band
                                │       ├── Calibration curves
                                │       └── CII per patient
                                │
                                └── Save figures + output.json
                            │
                            Cross-sample comparison plots
                            Cross-sample summary CSV
```

---

## Output Files

For each model, results are saved under the `results/<ModelName>/` folder. For each sample size fraction `n`, a folder `df<n>/` is created containing:

```
df<n>/
├── output.json              # All numeric metrics (AUC, MAPE, CII, optimism, CIs)
├── output.txt               # Human-readable C-statistic summary
├── full_predictions.csv     # Original + 200 bootstrap predicted probabilities
├── bootstrap_probs.npy      # Raw bootstrap probability matrix [200 × n]
└── figure/
    ├── optimism.png          # Histogram of bootstrap C-statistics
    ├── probability_comparison.png  # Original vs. bootstrap predicted probs
    ├── calibration_comparison.png  # Calibration instability
    ├── mape.png              # MAPE scatter per patient
    └── classification.png   # CII scatter per patient
```

Cross-sample comparison outputs are written to `results/<ModelName>/cross_sample_figures/`:

```
cross_sample_figures/
├── optimism_vs_mape.png
├── optimism_mape_correlation.png
├── optimism_vs_cii.png
├── optimism_cii_correlation.png
└── cross_sample_summary.csv
```

---

## Declarations & Author Details

### Author Affiliations

**Natthanaphop Isaradech** $^{1,2}$, **Phichayut Phinyo** $^{2}$, **Wuttipat Kiratipaisarl** $^{1}$, **Pakpoom Wongyikul** $^{1}$, **Noraworn Jirattikanwong** $^{2}$, **Wachiranun Sirikul** $^{1,2,*}$

1Department of Community Medicine, Faculty of Medicine, Chiang Mai University, Thailand, 2Department of Biomedical Informatics and Clinical Epidemiology (BioCE), Faculty of Medicine, Chiang Mai University, Thailand

$^*$ **Corresponding Author:** Wachiranun Sirikul

### Author Contributions

**N.I.** and **P.P.** initiated the project. **W.K.**, **P.W.**, and **N.J.** contributed to the study design and manuscript writing. **N.I.** carried out the modeling work, wrote the original draft of the manuscript, and conducted manuscript editing. **W.S.** conceptualized the study, contributed to the modeling framework, provided overall supervision, and reviewed/edited the manuscript. All authors have read and agreed to the published version of the manuscript.

### Funding
This research received no external funding.

### Conflicts of Interest
The authors declare no conflict of interest.

### Acknowledgments
The authors thank the GUSTO-I trial investigators and participants for making the dataset publicly available for research. We also thank our respective departments and faculties for providing the research facilities and resources that supported this study.

---

## License

See [LICENSE](LICENSE).
