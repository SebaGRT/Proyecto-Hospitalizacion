"""
utils.py — Reusable utility functions for Advance 2.

Covers: data loading, variable derivation, filtering, and helper tests.
"""

import gc
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ALPHA, COL_ADMISSION, COL_BIRTHDATE, COL_DIAG1, COL_DISCHARGE,
    COL_HOSPITAL, COL_SEVERITY, COL_TIPOALTA, COL_WEIGHT,
    DATA_DIR, DATA_FILES, MIN_CASES, MORTALITY_VALUE,
    NEOPLASM_CODES, P99_CUTOFF, PROC_COLS, SEED, SEPSIS_CODES,
)

logger = logging.getLogger(__name__)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_grd_data(
    years: Optional[list] = None,
    usecols: Optional[list] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load GRD CSV files into a single DataFrame.

    Parameters
    ----------
    years : list of int, optional
        Subset of years to load (e.g. [2021, 2022]).  None loads all.
    usecols : list of str, optional
        Column names to read.  None reads all.
    nrows : int, optional
        Maximum rows per file (for quick testing).

    Returns
    -------
    pd.DataFrame
        Concatenated raw data with dtype=str for all columns.
    """
    files = DATA_FILES
    if years:
        files = [f for f in files if any(str(y) in f for y in years)]

    frames = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            logger.warning("File not found: %s — skipping", path)
            continue
        logger.info("Loading %s …", fname)
        df = pd.read_csv(
            path,
            sep='|',
            dtype=str,
            usecols=usecols,
            nrows=nrows,
            low_memory=False,
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No GRD files found in %s" % DATA_DIR)

    result = pd.concat(frames, ignore_index=True)
    logger.info("Total rows loaded: {:,}".format(len(result)))
    return result


# ── Variable derivation ───────────────────────────────────────────────────────

def derive_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Derive analytical variables from raw GRD data.

    Adds columns:
    - age           : approximate age at admission
    - days_stay     : length of stay in days
    - n_procedures  : count of non-null procedure codes
    - n_unique_proc : count of unique non-null procedure codes
    - mortality     : 1 if TIPOALTA == 'FALLECIDO', else 0

    Parameters
    ----------
    df : pd.DataFrame
        Raw GRD data with at minimum the expected columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new derived columns appended.
    """
    df = df.copy()

    # Dates
    df[COL_ADMISSION] = pd.to_datetime(df[COL_ADMISSION], errors='coerce')
    df[COL_DISCHARGE] = pd.to_datetime(df[COL_DISCHARGE], errors='coerce')
    df[COL_BIRTHDATE] = pd.to_datetime(df[COL_BIRTHDATE], errors='coerce')

    # Age (approximate years at admission)
    df['age'] = df[COL_ADMISSION].dt.year - df[COL_BIRTHDATE].dt.year

    # Length of stay
    df['days_stay'] = (df[COL_DISCHARGE] - df[COL_ADMISSION]).dt.days

    # Procedure counts — only columns present in the DataFrame
    proc_present = [c for c in PROC_COLS if c in df.columns]
    proc_df = df[proc_present].replace('', np.nan)
    df['n_procedures']    = proc_df.notna().sum(axis=1)
    df['n_unique_proc']   = proc_df.apply(lambda row: row.dropna().nunique(), axis=1)

    # Mortality
    df['mortality'] = (df[COL_TIPOALTA].str.strip() == MORTALITY_VALUE).astype(int)

    logger.info("Variables derived: age, days_stay, n_procedures, n_unique_proc, mortality")
    return df


# ── Filtering / cleaning ──────────────────────────────────────────────────────

def filter_diagnostic_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into neoplasm and sepsis diagnostic groups.

    Filters on DIAGNOSTICO1 starting with the defined ICD-10 prefixes.
    Creates a 'diagnostic_group' column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_neoplasm, df_sepsis)
    """
    diag = df[COL_DIAG1].str.strip().str.upper()

    neo_mask = diag.str.startswith(tuple(NEOPLASM_CODES))
    sep_mask = diag.str.startswith(tuple(SEPSIS_CODES))

    df_neo = df[neo_mask].copy()
    df_neo['diagnostic_group'] = 'neoplasm'

    df_sep = df[sep_mask].copy()
    df_sep['diagnostic_group'] = 'sepsis'

    logger.info(
        "Diagnostic split — Neoplasm: {:,} | Sepsis: {:,}".format(len(df_neo), len(df_sep))
    )
    return df_neo, df_sep


def clean_data(df: pd.DataFrame, group_label: str = '') -> pd.DataFrame:
    """Apply standard data-quality filters.

    Steps (with row counts logged after each):
    1. Drop records with empty COD_HOSPITAL.
    2. Drop records with empty IR_29301_SEVERIDAD (GRD severity).
    3. Drop records with days_stay < 0.
    4. Remove outliers: days_stay > 99th percentile within this group.
    5. Convert severity and weight to numeric.
    6. Keep only hospital-diagnosis groups with >= MIN_CASES unique records.

    Parameters
    ----------
    df : pd.DataFrame
    group_label : str
        Label for logging (e.g. 'neoplasm').

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    n0 = len(df)
    lbl = group_label or 'data'

    # 1. Empty hospital
    df = df[df[COL_HOSPITAL].notna() & (df[COL_HOSPITAL].str.strip() != '')]
    logger.info("[%s] After drop empty hospital: {:,} (removed {:,})".format(len(df), n0 - len(df)) % lbl)
    n1 = len(df)

    # 2. Empty GRD severity
    df = df[df[COL_SEVERITY].notna() & (df[COL_SEVERITY].str.strip() != '')]
    logger.info("[%s] After drop empty severity: {:,} (removed {:,})".format(len(df), n1 - len(df)) % lbl)
    n2 = len(df)

    # 3. Negative days_stay
    df = df[df['days_stay'] >= 0]
    logger.info("[%s] After drop days_stay<0: {:,} (removed {:,})".format(len(df), n2 - len(df)) % lbl)
    n3 = len(df)

    # 4. Outlier removal: days_stay > p99
    p99 = np.percentile(df['days_stay'].dropna(), P99_CUTOFF)
    df = df[df['days_stay'] <= p99]
    logger.info("[%s] After drop days_stay>p99 (%.1f): {:,} (removed {:,})".format(len(df), n3 - len(df)) % (lbl, p99))

    # 5. Numeric conversions
    df[COL_SEVERITY] = pd.to_numeric(df[COL_SEVERITY], errors='coerce')
    df[COL_WEIGHT]   = pd.to_numeric(df[COL_WEIGHT],   errors='coerce')
    df['age']        = pd.to_numeric(df['age'],         errors='coerce')

    # Drop rows where severity became NaN after coercion (e.g. 'DESCONOCIDO')
    df = df[df[COL_SEVERITY].notna()]

    # 6. Minimum cases per hospital
    counts = df.groupby(COL_HOSPITAL)['days_stay'].count()
    valid_hospitals = counts[counts >= MIN_CASES].index
    n_before = len(df)
    df = df[df[COL_HOSPITAL].isin(valid_hospitals)]
    logger.info(
        "[%s] After min_cases filter (>=%d): {:,} (removed {:,}), hospitals: %d".format(
            len(df), n_before - len(df)
        ) % (lbl, MIN_CASES, len(valid_hospitals))
    )

    # Use category dtype for hospital to save memory
    df[COL_HOSPITAL] = df[COL_HOSPITAL].astype('category')
    df[COL_SEVERITY] = df[COL_SEVERITY].astype('category')

    return df.reset_index(drop=True)


# ── Completeness ──────────────────────────────────────────────────────────────

def completeness_table(df: pd.DataFrame, group_label: str = '') -> pd.DataFrame:
    """Compute % completeness for each column.

    Parameters
    ----------
    df : pd.DataFrame
    group_label : str

    Returns
    -------
    pd.DataFrame
        Table with columns: variable, n_total, n_missing, pct_complete.
    """
    n = len(df)
    records = []
    for col in df.columns:
        missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == '').sum()
        missing = min(missing, n)  # safety cap
        records.append({
            'variable':    col,
            'group':       group_label,
            'n_total':     n,
            'n_missing':   int(missing),
            'pct_complete': round(100 * (n - missing) / n, 1),
        })
    return pd.DataFrame(records)


# ── Memory helpers ─────────────────────────────────────────────────────────────

def free_memory(*dfs) -> None:
    """Delete DataFrames and call gc.collect().

    Parameters
    ----------
    *dfs : DataFrames to delete.
    """
    for df in dfs:
        del df
    gc.collect()
