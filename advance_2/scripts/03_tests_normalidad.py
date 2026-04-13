"""
03_tests_normalidad.py — Normality Tests (Shapiro-Wilk)

Justification for non-parametric path (H1):
  The Shapiro-Wilk test is applied on a subsample (n=5000, seed=42) of days_stay
  for each diagnostic group.  If p < alpha, the distribution is non-normal and
  Kruskal-Wallis (non-parametric ANOVA) is appropriate for H1.

Outputs:
  - outputs/tablas/resultados_shapiro_wilk.csv
  - Printed interpretation
"""

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALPHA, FIGURE_DPI, FIGURE_STYLE, GRAFICOS_DIR, SEED,
    SHAPIRO_N, TABLAS_DIR,
)
# Data loading is handled inline in main()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def shapiro_test(series: pd.Series, group_label: str, var_name: str) -> dict:
    """Run Shapiro-Wilk on a subsample of `series`.

    Parameters
    ----------
    series : pd.Series
        Numeric values to test.
    group_label : str
        Diagnostic group name (for display).
    var_name : str
        Variable name (for display).

    Returns
    -------
    dict with keys: group, variable, n_sample, W, p_value, conclusion
    """
    rng = np.random.default_rng(SEED)
    vals = series.dropna().values
    n = min(SHAPIRO_N, len(vals))
    sample = rng.choice(vals, size=n, replace=False)

    W, p = stats.shapiro(sample)
    conclusion = 'Not normal' if p < ALPHA else 'Normal (cannot reject H0)'

    return {
        'group':      group_label,
        'variable':   var_name,
        'n_sample':   n,
        'W_stat':     round(W, 4),
        'p_value':    p,
        'p_display':  f'<0.001' if p < 0.001 else f'{p:.4f}',
        'conclusion': conclusion,
    }


def run_normality_tests(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> pd.DataFrame:
    """Run Shapiro-Wilk for all key variables and both groups.

    Parameters
    ----------
    df_neo, df_sep : cleaned DataFrames

    Returns
    -------
    pd.DataFrame results table
    """
    variables = ['days_stay', 'n_procedures', 'n_unique_proc']
    results = []

    for group_label, df in [('Neoplasm', df_neo), ('Sepsis', df_sep)]:
        for var in variables:
            if var not in df.columns:
                continue
            row = shapiro_test(df[var], group_label, var)
            results.append(row)
            logger.info(
                "[%s | %s] W=%.4f  p=%s  → %s",
                group_label, var, row['W_stat'], row['p_display'], row['conclusion'],
            )

    return pd.DataFrame(results)


def print_interpretation(results: pd.DataFrame) -> None:
    """Print a textual interpretation of normality results."""
    print("\n" + "=" * 70)
    print("NORMALITY TEST RESULTS — Shapiro-Wilk")
    print("=" * 70)
    print(f"{'Group':<12} {'Variable':<18} {'n':<6} {'W-stat':<8} {'p-value':<10} {'Conclusion'}")
    print("-" * 70)
    for _, row in results.iterrows():
        print(
            f"{row['group']:<12} {row['variable']:<18} {row['n_sample']:<6} "
            f"{row['W_stat']:<8.4f} {row['p_display']:<10} {row['conclusion']}"
        )
    print("=" * 70)

    not_normal = results[results['conclusion'] == 'Not normal']
    if len(not_normal) == len(results):
        print("\nINTERPRETATION: ALL variables in BOTH groups reject normality (p < 0.05).")
        print("→ Non-parametric tests (Kruskal-Wallis) are appropriate for H1.")
        print("→ This is consistent with count/length-of-stay data being right-skewed.")
    else:
        normal_ones = results[results['conclusion'] != 'Not normal']
        print(f"\nINTERPRETATION: {len(not_normal)} of {len(results)} variables non-normal.")
        print("Normal variables:", normal_ones[['group', 'variable']].to_string(index=False))
        print("→ Kruskal-Wallis remains appropriate as a conservative choice.")


def main():
    logger.info("=== 03_tests_normalidad.py START ===")

    os.makedirs(TABLAS_DIR, exist_ok=True)

    # Load data via shared pipeline
    from utils import (
        clean_data, derive_variables, filter_diagnostic_groups,
        free_memory, load_grd_data,
    )
    from config import COL_HOSPITAL, COL_SEVERITY, COL_WEIGHT

    COLUMNS_NEEDED = [
        COL_HOSPITAL, 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
        'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
    ] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]

    raw = load_grd_data(usecols=COLUMNS_NEEDED)
    raw = derive_variables(raw)
    df_neo, df_sep = filter_diagnostic_groups(raw)
    free_memory(raw)
    df_neo = clean_data(df_neo, 'neoplasm')
    df_sep = clean_data(df_sep, 'sepsis')

    # Run tests
    results = run_normality_tests(df_neo, df_sep)

    # Output
    print_interpretation(results)

    out_csv = os.path.join(TABLAS_DIR, 'resultados_shapiro_wilk.csv')
    results.to_csv(out_csv, index=False)
    logger.info("Results saved: %s", out_csv)

    logger.info("=== 03_tests_normalidad.py COMPLETE ===")


if __name__ == '__main__':
    main()
