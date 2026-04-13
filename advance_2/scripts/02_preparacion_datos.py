"""
02_preparacion_datos.py — Data Cleaning and Variable Derivation

This script documents and applies all data preparation decisions:
1. Load raw GRD data (all years)
2. Derive analytical variables (age, days_stay, n_procedures, mortality)
3. Filter diagnostic groups (neoplasms / sepsis)
4. Apply data-quality cleaning rules
5. Save cleaned datasets and a decision log

Decision log (printed and saved):
  - Empty hospital → removed (can't assign hospital effect)
  - Empty GRD severity → removed (severity is a model covariate)
  - days_stay < 0 → removed (data error: discharge before admission)
  - days_stay > p99 → removed (extreme outliers distort models; use p99 per group)
  - Severity 'DESCONOCIDO' → removed (non-numeric, unusable)
  - Hospital with < MIN_CASES records → removed (insufficient power for fixed effects)

Outputs:
  - Logs printed to stdout
  - (DataFrames are returned for downstream scripts; not saved as files by default)
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COL_HOSPITAL, COL_SEVERITY, COL_WEIGHT, MIN_CASES,
    NEOPLASM_CODES, P99_CUTOFF, SEED, SEPSIS_CODES, TABLAS_DIR,
)
from utils import (
    clean_data, completeness_table, derive_variables,
    filter_diagnostic_groups, free_memory, load_grd_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

COLUMNS_NEEDED = [
    COL_HOSPITAL, 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


def document_decisions() -> None:
    """Print documented cleaning decisions to stdout."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║               DATA PREPARATION — DECISION DOCUMENTATION                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STEP 1 — Empty COD_HOSPITAL                                                 ║
║   Reason: Hospital code is the main grouping variable for hypothesis H1     ║
║           and the fixed effect in H2/H3 models. Without it, the record      ║
║           cannot be attributed to any hospital.                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STEP 2 — Empty/missing IR_29301_SEVERIDAD (GRD severity)                    ║
║   Reason: Severity is a key confounder in logistic and OLS models.           ║
║           Records without severity cannot be adequately controlled for.      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STEP 3 — days_stay < 0                                                       ║
║   Reason: Discharge date before admission date is a data entry error.        ║
║           Negative values are logically impossible.                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STEP 4 — days_stay > P99 (99th percentile per diagnostic group)             ║
║   Reason: Very long stays (e.g., ICU months) are clinically distinct         ║
║           from typical hospitalizations. They heavily skew regression        ║
║           coefficients and inflate variance. P99 is chosen to retain         ║
║           extreme but plausible values while removing clear outliers.        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STEP 5 — Severity == 'DESCONOCIDO' (after numeric coercion → NaN)           ║
║   Reason: Non-numeric severity cannot be encoded as an ordinal covariate.   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STEP 6 — Hospital with < 30 unique records (MIN_CASES)                       ║
║   Reason: Fixed-effects estimation requires adequate within-group variance.  ║
║           Hospitals with very few cases produce unreliable coefficient        ║
║           estimates and inflate Type I error in post-hoc comparisons.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def prepare_data() -> tuple:
    """Run the full data preparation pipeline.

    Returns
    -------
    tuple : (df_neoplasm, df_sepsis)
        Both DataFrames cleaned and ready for analysis.
    """
    document_decisions()

    # Load
    logger.info("Loading GRD data …")
    raw = load_grd_data(usecols=COLUMNS_NEEDED)
    logger.info("Raw rows loaded: {:,}".format(len(raw)))

    # Derive variables
    raw = derive_variables(raw)

    # Filter groups
    df_neo, df_sep = filter_diagnostic_groups(raw)
    raw_neo_n = len(df_neo)
    raw_sep_n = len(df_sep)
    free_memory(raw)

    # Clean
    df_neo = clean_data(df_neo, 'neoplasm')
    df_sep = clean_data(df_sep, 'sepsis')

    # Summary
    print("\n=== PREPARATION SUMMARY ===")
    print(f"{'Group':<12} {'Raw records':>12} {'After cleaning':>15} {'Retained %':>12}")
    print("-" * 55)
    print(f"{'Neoplasm':<12} {raw_neo_n:>12,} {len(df_neo):>15,} {100*len(df_neo)/raw_neo_n:>11.1f}%")
    print(f"{'Sepsis':<12} {raw_sep_n:>12,} {len(df_sep):>15,} {100*len(df_sep)/raw_sep_n:>11.1f}%")

    # Diagnostic group value counts
    print("\nNeoplasm — top DIAGNOSTICO1 codes:")
    print(df_neo['DIAGNOSTICO1'].value_counts().head(10).to_string())
    print("\nSepsis — top DIAGNOSTICO1 codes:")
    print(df_sep['DIAGNOSTICO1'].value_counts().head(10).to_string())

    # Save completeness table
    os.makedirs(TABLAS_DIR, exist_ok=True)
    key_cols = [COL_HOSPITAL, 'age', 'days_stay', 'n_procedures', 'n_unique_proc',
                'mortality', COL_SEVERITY, COL_WEIGHT]
    comp = pd.concat([
        completeness_table(df_neo[[c for c in key_cols if c in df_neo.columns]], 'neoplasm'),
        completeness_table(df_sep[[c for c in key_cols if c in df_sep.columns]], 'sepsis'),
    ])
    comp.to_csv(os.path.join(TABLAS_DIR, 'completeness_after_cleaning.csv'), index=False)
    logger.info("Completeness table saved.")

    return df_neo, df_sep


def main():
    logger.info("=== 02_preparacion_datos.py START ===")
    df_neo, df_sep = prepare_data()
    logger.info("Neoplasm: {:,} rows | Sepsis: {:,} rows".format(len(df_neo), len(df_sep)))
    logger.info("=== 02_preparacion_datos.py COMPLETE ===")


if __name__ == '__main__':
    main()
