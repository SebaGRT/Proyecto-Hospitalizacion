"""
05_regresion_logistica.py — Hypothesis H2: In-Hospital Mortality

Tests whether procedure intensity (n_procedures) is associated with mortality
after controlling for GRD severity, age, and hospital fixed effects.

Model (per diagnostic group):
  logit(P(mortality=1)) = α + β_proc*n_proc + β_sev*C(severity) +
                          β_age*age + Σ γ_h*C(hospital)

Technical specifications:
  - statsmodels.formula.api.logit
  - Hospital fixed effects via C(COD_HOSPITAL)
  - Report: Coefficients, Odds Ratios (exp(β)), CI95%, p-values, pseudo-R²

Outputs:
  - outputs/tablas/coeficientes_regresion_logistica.csv
  - outputs/modelos/logit_neoplasm_summary.txt
  - outputs/modelos/logit_sepsis_summary.txt
"""

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALPHA, COL_HOSPITAL, COL_SEVERITY, COL_WEIGHT,
    MODELOS_DIR, TABLAS_DIR,
)
from config import sig_label

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


def fit_logit(df: pd.DataFrame, group_label: str) -> dict:
    """Fit logistic regression model for one diagnostic group.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with: mortality, n_procedures, IR_29301_SEVERIDAD, age, COD_HOSPITAL
    group_label : str

    Returns
    -------
    dict with keys: model_result, coef_table, metrics
    """
    required = ['mortality', 'n_procedures', COL_SEVERITY, 'age', COL_HOSPITAL]
    df_model = df[required].dropna().copy()
    df_model[COL_SEVERITY] = df_model[COL_SEVERITY].astype(int)

    # Filter hospitals with variance in mortality (needed for convergence)
    hosp_mort = df_model.groupby(COL_HOSPITAL)['mortality'].mean()
    valid_hosps = hosp_mort[(hosp_mort > 0) & (hosp_mort < 1)].index
    df_model = df_model[df_model[COL_HOSPITAL].isin(valid_hosps)]

    n_obs = len(df_model)
    logger.info("[%s] Logit model — N observations: {:,}".format(n_obs) % group_label)

    if n_obs < 100:
        logger.warning("[%s] Too few observations for logit.", group_label)
        return None

    formula = (
        f"mortality ~ n_procedures + C({COL_SEVERITY}) + age + C({COL_HOSPITAL})"
    )

    logger.info("[%s] Fitting logit model …", group_label)
    try:
        model = smf.logit(formula, data=df_model)
        result = model.fit(
            method='bfgs', maxiter=200,
            disp=False, warn_convergence=False,
        )
    except Exception as e:
        logger.error("[%s] Model fitting failed: %s", group_label, e)
        return None

    # Build coefficient table
    params = result.params
    conf  = result.conf_int()
    pvals = result.pvalues

    rows = []
    for var in params.index:
        # Skip individual hospital fixed effects from detailed output (too many)
        if var.startswith(f'C({COL_HOSPITAL})'):
            continue
        coef  = params[var]
        ci_lo = conf.loc[var, 0]
        ci_hi = conf.loc[var, 1]
        p     = pvals[var]
        or_   = np.exp(coef)
        or_lo = np.exp(ci_lo)
        or_hi = np.exp(ci_hi)
        from config import sig_label as _sig
        rows.append({
            'diagnostic_group': group_label,
            'variable':   var,
            'coef':       round(coef, 4),
            'OR':         round(or_, 4),
            'CI95_lower': round(or_lo, 4),
            'CI95_upper': round(or_hi, 4),
            'p_value':    p,
            'p_display':  '<0.001' if p < 0.001 else f'{p:.4f}',
            'sig':        _sig(p),
        })

    coef_table = pd.DataFrame(rows)

    # Pseudo-R² (McFadden)
    pseudo_r2 = result.prsquared

    metrics = {
        'diagnostic_group': group_label,
        'n_observations': n_obs,
        'pseudo_r2_mcfadden': round(pseudo_r2, 4),
        'log_likelihood': round(result.llf, 2),
        'AIC': round(result.aic, 2),
        'BIC': round(result.bic, 2),
        'converged': bool(result.mle_retvals.get('converged', True)),
    }

    logger.info(
        "[%s] Logit — Pseudo-R²=%.4f  AIC=%.1f  n=%d",
        group_label, pseudo_r2, result.aic, n_obs,
    )

    return {
        'model_result': result,
        'coef_table':   coef_table,
        'metrics':      metrics,
    }


def print_coef_table(coef_table: pd.DataFrame, group_label: str) -> None:
    """Print a formatted coefficient table."""
    print(f"\n=== Logistic Regression — {group_label} ===")
    print(f"{'Variable':<35} {'Coef':>8} {'OR':>8} {'CI95% lo':>10} {'CI95% hi':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 90)
    for _, r in coef_table.iterrows():
        print(
            f"{r['variable']:<35} {r['coef']:>+8.4f} {r['OR']:>8.4f} "
            f"{r['CI95_lower']:>10.4f} {r['CI95_upper']:>10.4f} "
            f"{r['p_display']:>10} {r['sig']:>5}"
        )


def main():
    logger.info("=== 05_regresion_logistica.py START ===")

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
    all_metrics = []

    for group_label, df in [('Neoplasm', df_neo), ('Sepsis', df_sep)]:
        result_dict = fit_logit(df, group_label)

        if result_dict is None:
            logger.warning("[%s] Skipping — model failed.", group_label)
            continue

        coef_table = result_dict['coef_table']
        metrics    = result_dict['metrics']
        model_res  = result_dict['model_result']

        print_coef_table(coef_table, group_label)
        print(f"\n  Pseudo-R² (McFadden): {metrics['pseudo_r2_mcfadden']:.4f}")
        print(f"  N observations:       {metrics['n_observations']:,}")

        all_coef_tables.append(coef_table)
        all_metrics.append(metrics)

        # Save full model summary
        summary_path = os.path.join(MODELOS_DIR, f'logit_{group_label.lower()}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(model_res.summary().as_text())
        logger.info("Model summary saved: %s", summary_path)

        # Save metrics JSON
        metrics_path = os.path.join(MODELOS_DIR, f'logit_{group_label.lower()}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    # Save combined coefficient table
    if all_coef_tables:
        combined = pd.concat(all_coef_tables, ignore_index=True)
        out = os.path.join(TABLAS_DIR, 'coeficientes_regresion_logistica.csv')
        combined.to_csv(out, index=False)
        logger.info("Coefficients table saved: %s", out)

        print("\n=== METRICS SUMMARY ===")
        for m in all_metrics:
            print(f"  {m['diagnostic_group']}: Pseudo-R²={m['pseudo_r2_mcfadden']:.4f}, "
                  f"N={m['n_observations']:,}, AIC={m['AIC']:.1f}")

    logger.info("=== 05_regresion_logistica.py COMPLETE ===")


if __name__ == '__main__':
    main()
