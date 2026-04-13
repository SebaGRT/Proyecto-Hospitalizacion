"""
01_eda_profundo.py — Deep Exploratory Data Analysis

Generates interpreted visualizations for neoplasm and sepsis groups:
- Histograms of days_stay and n_procedures
- Q-Q plots for normality assessment
- Boxplots by GRD severity
- Violin plots (top 15 hospitals by volume)
- Bar chart: mean ± CI95% of n_procedures by hospital
- Scatter: GRD weight vs. days_stay by hospital
- Completeness table (CSV output)

Outputs: advance_2/outputs/graficos/, advance_2/outputs/tablas/
"""

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COL_HOSPITAL, COL_SEVERITY, COL_WEIGHT, FIGURE_DPI, FIGURE_STYLE,
    GRAFICOS_DIR, SEED, TABLAS_DIR,
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

try:
    plt.style.use(FIGURE_STYLE)
except OSError:
    plt.style.use('seaborn-whitegrid')

os.makedirs(GRAFICOS_DIR, exist_ok=True)
os.makedirs(TABLAS_DIR, exist_ok=True)

COLUMNS_NEEDED = [
    COL_HOSPITAL, 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_distributions(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Histograms + KDE for days_stay and n_procedures, both groups side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distributions by Diagnostic Group', fontsize=14, fontweight='bold')

    pairs = [
        (df_neo, 'Neoplasm', 'days_stay', 'Length of Stay (days)', axes[0, 0]),
        (df_sep, 'Sepsis',   'days_stay', 'Length of Stay (days)', axes[0, 1]),
        (df_neo, 'Neoplasm', 'n_procedures', 'N Procedures',        axes[1, 0]),
        (df_sep, 'Sepsis',   'n_procedures', 'N Procedures',        axes[1, 1]),
    ]
    for df, label, col, xlabel, ax in pairs:
        vals = df[col].dropna()
        ax.hist(vals, bins=50, density=True, alpha=0.6, color='steelblue' if 'Neo' in label else 'coral')
        vals.plot.kde(ax=ax, color='navy' if 'Neo' in label else 'darkred', linewidth=2)
        ax.set_title(f'{label}: {col}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.axvline(vals.median(), color='green', linestyle='--', linewidth=1.2, label=f'Median={vals.median():.1f}')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, '01_distributions.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_qqplots(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Q-Q plots to visually assess normality."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Q-Q Plots — Normality Assessment', fontsize=14, fontweight='bold')

    pairs = [
        (df_neo, 'Neoplasm', 'days_stay',    axes[0, 0]),
        (df_sep, 'Sepsis',   'days_stay',    axes[0, 1]),
        (df_neo, 'Neoplasm', 'n_procedures', axes[1, 0]),
        (df_sep, 'Sepsis',   'n_procedures', axes[1, 1]),
    ]
    for df, label, col, ax in pairs:
        vals = df[col].dropna().sample(min(5000, len(df)), random_state=SEED)
        stats.probplot(vals, dist='norm', plot=ax)
        ax.set_title(f'{label}: {col}')
        ax.get_lines()[0].set(markersize=2, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, '02_qqplots.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_boxplots_severity(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Boxplots of days_stay and n_procedures by GRD severity level."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution by GRD Severity', fontsize=14, fontweight='bold')

    pairs = [
        (df_neo, 'Neoplasm', 'days_stay',    'Length of Stay (days)', axes[0, 0]),
        (df_sep, 'Sepsis',   'days_stay',    'Length of Stay (days)', axes[0, 1]),
        (df_neo, 'Neoplasm', 'n_procedures', 'N Procedures',           axes[1, 0]),
        (df_sep, 'Sepsis',   'n_procedures', 'N Procedures',           axes[1, 1]),
    ]
    palette = {1: '#4CAF50', 2: '#FFC107', 3: '#FF5722', 4: '#9C27B0'}

    for df, label, col, ylabel, ax in pairs:
        tmp = df[[COL_SEVERITY, col]].dropna()
        tmp[COL_SEVERITY] = tmp[COL_SEVERITY].astype(int)
        order = sorted(tmp[COL_SEVERITY].unique())
        sns.boxplot(
            data=tmp, x=COL_SEVERITY, y=col, order=order,
            palette=palette, ax=ax, showfliers=False,
        )
        ax.set_title(f'{label}: {col} by Severity')
        ax.set_xlabel('GRD Severity Level')
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, '03_boxplot_severity.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_violin_hospitals(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Violin plots of days_stay for top 15 hospitals by volume."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Inter-Hospital Variability in Length of Stay (top 15 hospitals)',
                 fontsize=13, fontweight='bold')

    for ax, df, label in [(axes[0], df_neo, 'Neoplasm'), (axes[1], df_sep, 'Sepsis')]:
        top15 = (
            df[COL_HOSPITAL].value_counts().head(15).index.tolist()
        )
        tmp = df[df[COL_HOSPITAL].isin(top15)][[COL_HOSPITAL, 'days_stay']].dropna()
        order = (
            tmp.groupby(COL_HOSPITAL)['days_stay'].median()
            .sort_values(ascending=False).index.tolist()
        )
        sns.violinplot(
            data=tmp, x=COL_HOSPITAL, y='days_stay', order=order,
            palette='tab20', ax=ax, inner='quartile', linewidth=0.8,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{label}: days_stay by Hospital')
        ax.set_xlabel('Hospital Code')
        ax.set_ylabel('Length of Stay (days)')

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, '04_violin_hospitals.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_barplot_procedures_hospital(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Bar chart: mean ± CI95% of n_procedures by hospital (top 15)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Mean ± CI95% of N Procedures by Hospital (top 15)', fontsize=13, fontweight='bold')

    for ax, df, label in [(axes[0], df_neo, 'Neoplasm'), (axes[1], df_sep, 'Sepsis')]:
        top15 = df[COL_HOSPITAL].value_counts().head(15).index.tolist()
        tmp = df[df[COL_HOSPITAL].isin(top15)]

        stats_df = (
            tmp.groupby(COL_HOSPITAL)['n_procedures']
            .agg(['mean', 'sem', 'count'])
            .reset_index()
        )
        stats_df['ci95'] = 1.96 * stats_df['sem']
        stats_df = stats_df.sort_values('mean', ascending=False)

        ax.bar(
            range(len(stats_df)),
            stats_df['mean'],
            yerr=stats_df['ci95'],
            capsize=4,
            color='steelblue' if 'Neo' in label else 'coral',
            alpha=0.8,
            error_kw={'elinewidth': 1.2},
        )
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels(stats_df[COL_HOSPITAL].tolist(), rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{label}: Mean N Procedures')
        ax.set_xlabel('Hospital Code')
        ax.set_ylabel('Mean N Procedures ± CI95%')

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, '05_barplot_procedures.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_scatter_weight_stay(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Scatter plot: GRD weight vs. days_stay, colored by hospital (top 5)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('GRD Weight vs. Length of Stay by Hospital (top 5)', fontsize=13, fontweight='bold')

    for ax, df, label in [(axes[0], df_neo, 'Neoplasm'), (axes[1], df_sep, 'Sepsis')]:
        top5 = df[COL_HOSPITAL].value_counts().head(5).index.tolist()
        tmp = df[df[COL_HOSPITAL].isin(top5)][[COL_HOSPITAL, COL_WEIGHT, 'days_stay']].dropna()
        tmp[COL_WEIGHT] = pd.to_numeric(tmp[COL_WEIGHT], errors='coerce')
        tmp = tmp.dropna()

        palette = dict(zip(top5, sns.color_palette('tab10', 5)))
        for hosp in top5:
            sub = tmp[tmp[COL_HOSPITAL] == hosp]
            ax.scatter(sub[COL_WEIGHT], sub['days_stay'], label=hosp,
                       alpha=0.3, s=10, color=palette[hosp])

        ax.set_title(f'{label}: GRD Weight vs. Days Stay')
        ax.set_xlabel('GRD Relative Weight')
        ax.set_ylabel('Length of Stay (days)')
        ax.legend(title='Hospital', fontsize=8, markerscale=2)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, '06_scatter_weight_stay.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", path)


def save_completeness(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Save completeness table to CSV."""
    key_cols = [
        COL_HOSPITAL, 'age', 'days_stay', 'n_procedures', 'n_unique_proc',
        'mortality', COL_SEVERITY, COL_WEIGHT,
    ]
    neo_comp = completeness_table(df_neo[[c for c in key_cols if c in df_neo.columns]], 'neoplasm')
    sep_comp = completeness_table(df_sep[[c for c in key_cols if c in df_sep.columns]], 'sepsis')
    combined = pd.concat([neo_comp, sep_comp], ignore_index=True)

    out = os.path.join(TABLAS_DIR, 'completeness_table.csv')
    combined.to_csv(out, index=False)
    logger.info("Completeness table saved: %s", out)
    print("\n=== Completeness Table ===")
    print(combined.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== 01_eda_profundo.py START ===")

    # Load data
    raw = load_grd_data(usecols=COLUMNS_NEEDED)

    # Derive variables
    raw = derive_variables(raw)

    # Filter diagnostic groups
    df_neo, df_sep = filter_diagnostic_groups(raw)
    free_memory(raw)

    # Clean each group
    df_neo = clean_data(df_neo, 'neoplasm')
    df_sep = clean_data(df_sep, 'sepsis')

    logger.info("Neoplasm records: {:,}  |  Sepsis records: {:,}".format(len(df_neo), len(df_sep)))

    # Generate all visualizations
    plot_distributions(df_neo, df_sep)
    plot_qqplots(df_neo, df_sep)
    plot_boxplots_severity(df_neo, df_sep)
    plot_violin_hospitals(df_neo, df_sep)
    plot_barplot_procedures_hospital(df_neo, df_sep)
    plot_scatter_weight_stay(df_neo, df_sep)
    save_completeness(df_neo, df_sep)

    logger.info("=== 01_eda_profundo.py COMPLETE ===")


if __name__ == '__main__':
    main()
