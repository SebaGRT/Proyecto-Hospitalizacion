"""
config.py — Centralized configuration parameters for Advance 2.
All statistical thresholds, paths and seeds are defined here.
"""

import os

# ── Statistical parameters ──────────────────────────────────────────────────
ALPHA = 0.05          # Significance level for all tests
P99_CUTOFF = 99       # Percentile for days_stay outlier removal
MIN_CASES = 30        # Minimum unique patients per hospital-diagnosis group
SEED = 42             # Fixed seed for reproducibility (Shapiro-Wilk subsample, etc.)
SHAPIRO_N = 5000      # Subsample size for Shapiro-Wilk test

# ── Diagnostic groups ────────────────────────────────────────────────────────
NEOPLASM_CODES = ['C50', 'C18', 'C19', 'C20', 'C53', 'C34']
SEPSIS_CODES   = ['A40', 'A41']

# ── Variable names in raw data ────────────────────────────────────────────────
COL_HOSPITAL   = 'COD_HOSPITAL'
COL_BIRTHDATE  = 'FECHA_NACIMIENTO'
COL_ADMISSION  = 'FECHA_INGRESO'
COL_DISCHARGE  = 'FECHAALTA'
COL_TIPOALTA   = 'TIPOALTA'
COL_DIAG1      = 'DIAGNOSTICO1'
COL_SEVERITY   = 'IR_29301_SEVERIDAD'
COL_WEIGHT     = 'IR_29301_PESO'
PROC_COLS      = [f'PROCEDIMIENTO{i}' for i in range(1, 31)]

MORTALITY_VALUE = 'FALLECIDO'

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, '..', 'DATASET-PROBLEMA8')
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')
TABLAS_DIR   = os.path.join(OUTPUTS_DIR, 'tablas')
GRAFICOS_DIR = os.path.join(OUTPUTS_DIR, 'graficos')
MODELOS_DIR  = os.path.join(OUTPUTS_DIR, 'modelos')

# GRD source files
DATA_FILES = [
    'GRD_PUBLICO_2019.csv',
    'GRD_PUBLICO_2020.csv',
    'GRD_PUBLICO_2021.csv',
    'GRD_PUBLICO_EXTERNO_2022.csv',
    'GRD_PUBLICO_2023.csv',
    'GRD_PUBLICO_2024.csv',
]

# ── Plot style ────────────────────────────────────────────────────────────────
FIGURE_DPI   = 300
FIGURE_STYLE = 'seaborn-v0_8-whitegrid'

# ── Significance labeling ─────────────────────────────────────────────────────
def sig_label(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.10:
        return '.'
    return 'ns'
