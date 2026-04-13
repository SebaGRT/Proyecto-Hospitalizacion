"""
06_regresion_ols.py — Hypothesis H3: Length of Stay (OLS + Fixed Effects)

Tests whether procedure intensity independently predicts days_stay after
controlling for GRD severity, age, and hospital fixed effects.

Model (per diagnostic group):
  days_stay = α + β_proc*n_proc + β_sev*C(severity) +
              β_age*age + Σ γ_h*C(hospital) + ε

Technical specifications:
  - statsmodels.formula.api.ols
  - Hospital fixed effects via C(COD_HOSPITAL)
  - VIF diagnosis for multicollinearity
  - Residual normality assessment

Outputs:
  - outputs/tablas/coeficientes_regresion_ols.csv
  - outputs/modelos/ols_neoplasm_summary.txt
  - outputs/modelos/ols_sepsis_summary.txt
  - outputs/tablas/vif_diagnostics.csv
"""

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALPHA, COL_HOSPITAL, COL_SEVERITY, COL_WEIGHT,
    MODELOS_DIR, TABLAS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def compute_vif(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """Compute Variance Inflation Factor for numerical predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Should include n_procedures, age, IR_29301_SEVERIDAD (numeric).
    group_label : str

    Returns
    -------
    pd.DataFrame with VIF values.
    """
    numeric_cols = ['n_procedures', 'age', COL_SEVERITY]
    vif_data = df[numeric_cols].dropna().copy()
    vif_data[COL_SEVERITY] = pd.to_numeric(vif_data[COL_SEVERITY], errors='coerce')
    vif_data = vif_data.dropna()

    if len(vif_data) < 10:
        return pd.DataFrame()

    # Add constant
    from statsmodels.tools.tools import add_constant
    X = add_constant(vif_data)

    vif_rows = []
    for i, col in enumerate(X.columns):
        try:
            vif_val = variance_inflation_factor(X.values, i)
            vif_rows.append({
                'diagnostic_group': group_label,
                'variable': col,
                'VIF': round(vif_val, 4),
                'interpretation': (
                    'OK (< 5)' if vif_val < 5 else
                    'Moderate (5-10)' if vif_val < 10 else
                    'High (> 10) — multicollinearity concern'
                )
            })
        except Exception:
            pass

    return pd.DataFrame(vif_rows)


def fit_ols(df: pd.DataFrame, group_label: str) -> dict:
    """Fit OLS regression model for one diagnostic group.

    Parameters
    ----------
    df : pd.DataFrame
    group_label : str

    Returns
    -------
    dict with model_result, coef_table, metrics, residuals
    """
    required = ['days_stay', 'n_procedures', COL_SEVERITY, 'age', COL_HOSPITAL]
    df_model = df[required].dropna().copy()
    df_model[COL_SEVERITY] = pd.to_numeric(df_model[COL_SEVERITY], errors='coerce')
    df_model = df_model.dropna()
    df_model[COL_SEVERITY] = df_model[COL_SEVERITY].astype(int)

    # Drop hospitals with insufficient variance in days_stay
    std_by_hosp = df_model.groupby(COL_HOSPITAL)['days_stay'].std()
    valid_hosps = std_by_hosp[std_by_hosp > 0].index
    df_model = df_model[df_model[COL_HOSPITAL].isin(valid_hosps)]

    n_obs = len(df_model)
    logger.info("[%s] OLS model — N observations: {:,}".format(n_obs) % group_label)

    if n_obs < 100:
        logger.warning("[%s] Too few observations for OLS.", group_label)
        return None

    formula = (
        f"days_stay ~ n_procedures + C({COL_SEVERITY}) + age + C({COL_HOSPITAL})"
    )

    logger.info("[%s] Fitting OLS model …", group_label)
    try:
        model = smf.ols(formula, data=df_model)
        result = model.fit()
    except Exception as e:
        logger.error("[%s] OLS fitting failed: %s", group_label, e)
        return None

    # Coefficient table (excluding individual hospital fixed effects)
    params = result.params
    conf  = result.conf_int()
    pvals = result.pvalues

    rows = []
    from config import sig_label as _sig
    for var in params.index:
        if var.startswith(f'C({COL_HOSPITAL})'):
            continue
        coef  = params[var]
        ci_lo = conf.loc[var, 0]
        ci_hi = conf.loc[var, 1]
        p     = pvals[var]
        rows.append({
            'diagnostic_group': group_label,
            'variable':   var,
            'coef':       round(coef, 4),
            'CI95_lower': round(ci_lo, 4),
            'CI95_upper': round(ci_hi, 4),
            'p_value':    p,
            'p_display':  '<0.001' if p < 0.001 else f'{p:.4f}',
            'sig':        _sig(p),
        })

    coef_table = pd.DataFrame(rows)

    # VIF
    vif_table = compute_vif(df_model, group_label)

    metrics = {
        'diagnostic_group': group_label,
        'n_observations': n_obs,
        'adj_r_squared': round(result.rsquared_adj, 4),
        'f_statistic': round(result.fvalue, 4),
        'f_pvalue': result.f_pvalue,
        'aic': round(result.aic, 2),
        'bic': round(result.bic, 2),
    }

    logger.info(
        "[%s] OLS — Adj-R²=%.4f  F=%.2f  p(F)=%.4e  n=%d",
        group_label, metrics['adj_r_squared'], metrics['f_statistic'],
        metrics['f_pvalue'], n_obs,
    )

    residuals = pd.Series(result.resid, name=f'residuals_{group_label.lower()}')

    return {
        'model_result': result,
        'coef_table':   coef_table,
        'metrics':      metrics,
        'vif_table':    vif_table,
        'residuals':    residuals,
        'fitted':       pd.Series(result.fittedvalues, name='fitted'),
        'observed':     df_model['days_stay'].reset_index(drop=True),
    }


def print_ols_summary(coef_table: pd.DataFrame, metrics: dict, group_label: str) -> None:
    """Print formatted OLS results."""
    print(f"\n=== OLS Regression — {group_label} ===")
    print(f"{'Variable':<35} {'Coef':>8} {'CI95% lo':>10} {'CI95% hi':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 82)
    for _, r in coef_table.iterrows():
        print(
            f"{r['variable']:<35} {r['coef']:>+8.4f} "
            f"{r['CI95_lower']:>10.4f} {r['CI95_upper']:>10.4f} "
            f"{r['p_display']:>10} {r['sig']:>5}"
        )
    print(f"\n  Adjusted R²:  {metrics['adj_r_squared']:.4f}")
    print(f"  F-statistic:  {metrics['f_statistic']:.2f}  (p={'<0.001' if metrics['f_pvalue'] < 0.001 else f\"{metrics['f_pvalue']:.4f}\"})")
    print(f"  N observations: {metrics['n_observations']:,}")


def main():
    logger.info("=== 06_regresion_ols.py START ===")

    os.makedirs(TABLAS_DIR, exist_ok=True)
    os.makedirs(MODELOS_DIR, exist_ok=True)

    # Load data
    from utils import (
        clean_data, derive_variables, filter_diagnostic_groups,
        free_memory, load_grd_data,
    )

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

    all_coef_tables = []
    all_vif_tables  = []
    all_metrics     = []
    ols_results_store = {}

    for group_label, df in [('Neoplasm', df_neo), ('Sepsis', df_sep)]:
        result_dict = fit_ols(df, group_label)

        if result_dict is None:
            continue

        coef_table = result_dict['coef_table']
        metrics    = result_dict['metrics']
        model_res  = result_dict['model_result']
        vif_table  = result_dict['vif_table']

        print_ols_summary(coef_table, metrics, group_label)

        if not vif_table.empty:
            print(f"\n  VIF Diagnostics ({group_label}):")
            print(vif_table[['variable', 'VIF', 'interpretation']].to_string(index=False))

        all_coef_tables.append(coef_table)
        all_metrics.append(metrics)
        if not vif_table.empty:
            all_vif_tables.append(vif_table)

        ols_results_store[group_label.lower()] = result_dict

        # Save full model summary
        summary_path = os.path.join(MODELOS_DIR, f'ols_{group_label.lower()}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(model_res.summary().as_text())
        logger.info("OLS summary saved: %s", summary_path)

        # Save metrics JSON
        metrics_copy = dict(metrics)
        metrics_copy['f_pvalue'] = float(metrics_copy['f_pvalue'])
        metrics_path = os.path.join(MODELOS_DIR, f'ols_{group_label.lower()}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_copy, f, indent=2)

    # Save combined tables
    if all_coef_tables:
        combined = pd.concat(all_coef_tables, ignore_index=True)
        out = os.path.join(TABLAS_DIR, 'coeficientes_regresion_ols.csv')
        combined.to_csv(out, index=False)
        logger.info("OLS coefficients saved: %s", out)

    if all_vif_tables:
        vif_combined = pd.concat(all_vif_tables, ignore_index=True)
        vif_path = os.path.join(TABLAS_DIR, 'vif_diagnostics.csv')
        vif_combined.to_csv(vif_path, index=False)
        logger.info("VIF table saved: %s", vif_path)

    print("\n=== METRICS SUMMARY ===")
    for m in all_metrics:
        print(f"  {m['diagnostic_group']}: Adj-R²={m['adj_r_squared']:.4f}, "
              f"F={m['f_statistic']:.2f}, N={m['n_observations']:,}")

    logger.info("=== 06_regresion_ols.py COMPLETE ===")


if __name__ == '__main__':
    main()
