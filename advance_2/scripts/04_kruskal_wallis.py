"""
04_kruskal_wallis.py — Hypothesis H1: Inter-Hospital Variability

Tests whether n_procedures, n_unique_proc, and days_stay differ significantly
across hospitals for each diagnostic group.

Method:
  1. Filter hospitals with >= MIN_CASES records (sufficient statistical power).
  2. Kruskal-Wallis H-test (non-parametric ANOVA) for each variable + group.
  3. If p < alpha → Dunn post-hoc with Bonferroni correction.
  4. Save results tables.

Outputs:
  - outputs/tablas/resultados_kruskal_wallis.csv
  - outputs/tablas/dunn_posthoc_neoplasm.csv
  - outputs/tablas/dunn_posthoc_sepsis.csv
"""

import logging
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALPHA, COL_HOSPITAL, FIGURE_DPI, GRAFICOS_DIR,
    MIN_CASES, TABLAS_DIR,
)
from config import sig_label

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def kruskal_wallis_test(df: pd.DataFrame, variable: str, group_label: str) -> dict:
    """Run Kruskal-Wallis on `variable` grouped by hospital.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data for one diagnostic group.
    variable : str
        Column to test.
    group_label : str
        Label for the diagnostic group.

    Returns
    -------
    dict with H-stat, p-value, n_hospitals, conclusion.
    """
    groups = [
        grp[variable].dropna().values
        for _, grp in df.groupby(COL_HOSPITAL)
        if len(grp) >= MIN_CASES
    ]
    valid_hospitals = [
        h for h, grp in df.groupby(COL_HOSPITAL) if len(grp) >= MIN_CASES
    ]
    n_hospitals = len(groups)

    if n_hospitals < 2:
        return {
            'diagnostic_group': group_label,
            'variable': variable,
            'H_stat': np.nan,
            'p_value': np.nan,
            'p_display': 'n/a',
            'n_hospitals': n_hospitals,
            'sig': 'n/a',
            'conclusion': 'Not enough hospitals',
        }

    H, p = stats.kruskal(*groups)
    from config import sig_label as _sig
    sig = _sig(p)
    conclusion = 'REJECT H0' if p < ALPHA else 'FAIL TO REJECT H0'

    logger.info(
        "[%s | %s] H=%.2f  p=%.4e  n_hospitals=%d  → %s %s",
        group_label, variable, H, p, n_hospitals, conclusion, sig,
    )

    return {
        'diagnostic_group': group_label,
        'variable': variable,
        'H_stat': round(H, 4),
        'p_value': p,
        'p_display': '<0.001' if p < 0.001 else f'{p:.4f}',
        'n_hospitals': n_hospitals,
        'sig': sig,
        'conclusion': conclusion,
    }


def dunn_bonferroni(df: pd.DataFrame, variable: str, group_label: str) -> pd.DataFrame:
    """Pairwise Dunn test with Bonferroni correction.

    Uses the z-score approximation of the Dunn statistic (manual implementation
    compatible with scipy, without requiring scikit-posthocs for all environments).

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
    group_label : str

    Returns
    -------
    pd.DataFrame with pairwise comparisons.
    """
    try:
        from scikit_posthocs import posthoc_dunn
        _use_skp = True
    except ImportError:
        _use_skp = False

    valid_hospitals = [
        h for h, grp in df.groupby(COL_HOSPITAL) if len(grp) >= MIN_CASES
    ]
    df_valid = df[df[COL_HOSPITAL].isin(valid_hospitals)]

    if _use_skp:
        # scikit-posthocs implementation
        try:
            from scikit_posthocs import posthoc_dunn
            result = posthoc_dunn(
                df_valid, val_col=variable, group_col=COL_HOSPITAL, p_adjust='bonferroni'
            )
            rows = []
            hospitals = result.columns.tolist()
            for i, h1 in enumerate(hospitals):
                for h2 in hospitals[i+1:]:
                    p_adj = result.loc[h1, h2]
                    from config import sig_label as _sig
                    rows.append({
                        'diagnostic_group': group_label,
                        'variable': variable,
                        'hospital_A': h1,
                        'hospital_B': h2,
                        'p_bonferroni': round(p_adj, 6),
                        'sig': _sig(p_adj),
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.warning("scikit_posthocs failed (%s), using manual Dunn.", e)

    # Manual Dunn z-score approximation
    groups_data = {
        h: df_valid[df_valid[COL_HOSPITAL] == h][variable].dropna().values
        for h in valid_hospitals
    }

    all_vals = np.concatenate(list(groups_data.values()))
    N = len(all_vals)
    ranks = stats.rankdata(all_vals)

    # Map back to group ranks
    idx = 0
    group_ranks = {}
    for h, vals in groups_data.items():
        n = len(vals)
        group_ranks[h] = ranks[idx:idx + n]
        idx += n

    # Tie correction
    _, counts = np.unique(all_vals, return_counts=True)
    tie_sum = np.sum(counts ** 3 - counts)
    tie_correction = 1 - tie_sum / (N ** 3 - N)

    rows = []
    pairs = list(combinations(valid_hospitals, 2))
    n_pairs = len(pairs)

    for h1, h2 in pairs:
        ni = len(group_ranks[h1])
        nj = len(group_ranks[h2])
        Ri = group_ranks[h1].mean()
        Rj = group_ranks[h2].mean()

        se = np.sqrt(
            (N * (N + 1) / 12 - tie_sum / (12 * (N - 1)))
            * (1 / ni + 1 / nj)
        )
        if se == 0:
            continue
        z = (Ri - Rj) / se
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_adj = min(p_raw * n_pairs, 1.0)  # Bonferroni

        from config import sig_label as _sig
        rows.append({
            'diagnostic_group': group_label,
            'variable': variable,
            'hospital_A': h1,
            'hospital_B': h2,
            'z_stat': round(z, 4),
            'p_raw': round(p_raw, 6),
            'p_bonferroni': round(p_adj, 6),
            'sig': _sig(p_adj),
        })

    return pd.DataFrame(rows)


def run_all_tests(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> pd.DataFrame:
    """Run Kruskal-Wallis for all variable × group combinations.

    Returns
    -------
    pd.DataFrame with all KW results.
    """
    variables = ['days_stay', 'n_procedures', 'n_unique_proc']
    rows = []

    for group_label, df in [('Neoplasm', df_neo), ('Sepsis', df_sep)]:
        for var in variables:
            if var not in df.columns:
                continue
            row = kruskal_wallis_test(df, var, group_label)
            rows.append(row)

    return pd.DataFrame(rows)


def run_posthoc(df_neo: pd.DataFrame, df_sep: pd.DataFrame, kw_results: pd.DataFrame) -> dict:
    """Run Dunn post-hoc for significant KW results.

    Returns
    -------
    dict {'neoplasm': pd.DataFrame, 'sepsis': pd.DataFrame}
    """
    posthoc = {}

    for group_label, df in [('Neoplasm', df_neo), ('Sepsis', df_sep)]:
        sig_vars = kw_results[
            (kw_results['diagnostic_group'] == group_label) &
            (kw_results['p_value'] < ALPHA)
        ]['variable'].tolist()

        if not sig_vars:
            logger.info("[%s] No significant KW results → skipping Dunn post-hoc", group_label)
            continue

        frames = []
        for var in sig_vars:
            logger.info("[%s | %s] Running Dunn-Bonferroni post-hoc …", group_label, var)
            dunn_df = dunn_bonferroni(df, var, group_label)
            frames.append(dunn_df)

        if frames:
            posthoc[group_label.lower()] = pd.concat(frames, ignore_index=True)

    return posthoc


def print_kw_summary(kw_results: pd.DataFrame) -> None:
    """Print Kruskal-Wallis results table."""
    print("\n" + "=" * 80)
    print("KRUSKAL-WALLIS RESULTS")
    print("=" * 80)
    print(f"{'Group':<12} {'Variable':<18} {'H-stat':>8} {'p-value':>10} {'N hosp':>7} {'Sig':>5} {'Conclusion'}")
    print("-" * 80)
    for _, r in kw_results.iterrows():
        print(
            f"{r['diagnostic_group']:<12} {r['variable']:<18} {r['H_stat']:>8.2f} "
            f"{r['p_display']:>10} {r['n_hospitals']:>7} {r['sig']:>5} {r['conclusion']}"
        )
    print("=" * 80)


def main():
    logger.info("=== 04_kruskal_wallis.py START ===")

    os.makedirs(TABLAS_DIR, exist_ok=True)

    # Load cleaned data
    from utils import (
        clean_data, derive_variables, filter_diagnostic_groups,
        free_memory, load_grd_data,
    )
    from config import COL_WEIGHT

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

    # Kruskal-Wallis
    kw_results = run_all_tests(df_neo, df_sep)
    print_kw_summary(kw_results)

    out_kw = os.path.join(TABLAS_DIR, 'resultados_kruskal_wallis.csv')
    kw_results.to_csv(out_kw, index=False)
    logger.info("KW results saved: %s", out_kw)

    # Dunn post-hoc
    posthoc = run_posthoc(df_neo, df_sep, kw_results)

    for group_key, dunn_df in posthoc.items():
        out_dunn = os.path.join(TABLAS_DIR, f'dunn_posthoc_{group_key}.csv')
        dunn_df.to_csv(out_dunn, index=False)
        logger.info("Dunn post-hoc saved: %s", out_dunn)

        print(f"\n--- Dunn-Bonferroni Post-hoc: {group_key} (significant pairs only) ---")
        sig_pairs = dunn_df[dunn_df['p_bonferroni'] < ALPHA]
        print(f"Significant pairs: {len(sig_pairs)} of {len(dunn_df)}")
        if not sig_pairs.empty:
            print(sig_pairs[['variable', 'hospital_A', 'hospital_B', 'p_bonferroni', 'sig']].head(20).to_string(index=False))

    logger.info("=== 04_kruskal_wallis.py COMPLETE ===")


if __name__ == '__main__':
    main()
