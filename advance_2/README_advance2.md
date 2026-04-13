# Advance 2 — Complete Inferential Analysis

## Objective
Implementation of complete inferential analysis with statistical modeling, interpreted visualizations, and conclusions coherent with literature.

**Research Question:** Does medical procedure intensity vary significantly among public hospitals for patients with neoplasm or sepsis diagnoses, and is this variation associated with differences in in-hospital mortality and length of stay?

---

## Project Structure

```
advance_2/
├── config.py                     # Centralized parameters (α, seeds, paths)
├── utils.py                      # Reusable functions (load, clean, derive)
├── README_advance2.md            # This file
├── scripts/
│   ├── 01_eda_profundo.py        # Deep EDA + 6 visualizations
│   ├── 02_preparacion_datos.py   # Cleaning decisions + documentation
│   ├── 03_tests_normalidad.py    # Shapiro-Wilk normality tests
│   ├── 04_kruskal_wallis.py      # H1: Kruskal-Wallis + Dunn-Bonferroni
│   ├── 05_regresion_logistica.py # H2: Logistic regression (mortality)
│   ├── 06_regresion_ols.py       # H3: OLS regression (days stay)
│   └── 07_sintesis_resultados.py # Final visualizations + summary table
├── notebooks/
│   └── Advance2_Analisis_Completo.ipynb  # End-to-end executable notebook
└── outputs/
    ├── tablas/                   # CSV result tables
    ├── graficos/                 # PNG visualizations (300 dpi)
    └── modelos/                  # Model summaries + metrics JSON
```

---

## Hypotheses

| # | Hypothesis | Test | Target Variable |
|---|-----------|------|----------------|
| H1 | Significant inter-hospital variability in procedure intensity | Shapiro-Wilk → Kruskal-Wallis → Dunn-Bonferroni | n_procedures, days_stay, n_unique_proc |
| H2 | Procedure intensity associated with in-hospital mortality | Logistic regression + hospital fixed effects | mortality (TIPOALTA == 'FALLECIDO') |
| H3 | Procedure intensity predicts length of stay independently of severity | OLS + hospital fixed effects + VIF | days_stay |

---

## Diagnostic Groups

| Group | ICD-10 Codes | Type | Model |
|-------|-------------|------|-------|
| Neoplasm | C50, C18, C19, C20, C53, C34 | Scheduled | Elective care paradigm |
| Sepsis | A40, A41 | Emergency | Critical care paradigm |

---

## Dependencies

```
pandas>=1.5
numpy>=1.23
scipy>=1.9
statsmodels>=0.13
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.1
openpyxl
scikit-posthocs  # optional, for Dunn post-hoc (falls back to manual implementation)
```

Install:
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn openpyxl scikit-posthocs
```

---

## How to Run

### Option 1: Individual scripts (from `advance_2/` directory)
```bash
cd advance_2/

python scripts/01_eda_profundo.py            # EDA visualizations → outputs/graficos/
python scripts/02_preparacion_datos.py       # Data cleaning documentation
python scripts/03_tests_normalidad.py        # Shapiro-Wilk → outputs/tablas/
python scripts/04_kruskal_wallis.py          # H1 → outputs/tablas/
python scripts/05_regresion_logistica.py     # H2 → outputs/tablas/ + modelos/
python scripts/06_regresion_ols.py           # H3 → outputs/tablas/ + modelos/
python scripts/07_sintesis_resultados.py     # Final summary → outputs/
```

### Option 2: Complete notebook
```bash
jupyter notebook notebooks/Advance2_Analisis_Completo.ipynb
```
Run all cells from top to bottom. The notebook is end-to-end executable and produces all outputs.

---

## Output Files

### Tables (`outputs/tablas/`)
| File | Content |
|------|---------|
| `completeness_after_cleaning.csv` | % completeness by variable + group |
| `resultados_shapiro_wilk.csv` | W-stat, p-value, conclusion per group |
| `resultados_kruskal_wallis.csv` | H-stat, p-value, N hospitals per variable |
| `dunn_posthoc_neoplasm.csv` | Pairwise Bonferroni-corrected p-values |
| `dunn_posthoc_sepsis.csv` | Pairwise Bonferroni-corrected p-values |
| `coeficientes_regresion_logistica.csv` | OR, CI95%, p-values (H2) |
| `coeficientes_regresion_ols.csv` | β, CI95%, p-values (H3) |
| `vif_diagnostics.csv` | VIF per predictor per group |
| `tabla_comparativa_neoplasia_vs_sepsis.csv` | Cross-hypothesis summary |

### Graphics (`outputs/graficos/`)
| File | Content |
|------|---------|
| `01_distributions.png` | Histograms + KDE: days_stay, n_procedures |
| `02_qqplots.png` | Q-Q plots for normality assessment |
| `03_boxplot_severity.png` | Boxplots by GRD severity |
| `04_violin_hospitals.png` | Inter-hospital violin plots (top 15) |
| `05_barplot_procedures.png` | Mean ± CI95% n_procedures by hospital |
| `06_scatter_weight_stay.png` | GRD weight vs. days_stay scatter |
| `07_coef_comparison.png` | β comparison: neoplasm vs sepsis |
| `08_residual_diagnostics.png` | OLS residual histogram + Q-Q |
| `09_pred_vs_obs.png` | Predicted vs. observed days_stay |
| `10_hospital_heatmap.png` | Hospital fixed effects heatmap |

---

## Statistical Parameters

| Parameter | Value | Justification |
|-----------|-------|--------------|
| α (significance level) | 0.05 | Standard in biomedical research |
| Bonferroni correction | Yes (Dunn post-hoc) | Multiple pairwise comparisons |
| Min cases per hospital | 30 | Adequate within-group variance for fixed effects |
| P99 cutoff (days_stay) | 99th percentile per group | Removes extreme outliers while retaining severe cases |
| Shapiro-Wilk sample | 5000 (seed=42) | Test valid for n ≤ 5000; reproducible |

---

## Key Findings (to be updated after running)

Results will appear in `outputs/tablas/tabla_comparativa_neoplasia_vs_sepsis.csv`.

Interpretation framework:
- **H1 rejected** (p < 0.05): Hospitals do NOT treat similar patients uniformly
- **H2 β_proc > 0**: More procedures → higher mortality odds (severity confounding)
- **H2 β_proc < 0**: More procedures → lower mortality odds (better treatment quality)
- **H3 β_proc > 0**: More procedures → longer stays (expected in both groups)

---

## Data Source

GRD Público FONASA 2019–2024, located in `../DATASET-PROBLEMA8/`:
- `GRD_PUBLICO_2019.csv` through `GRD_PUBLICO_2024.csv`
- Delimiter: `|` | Encoding: UTF-8
- ~5.8 million records total
