# =============================================================================
# 06_regresion_ols.py — Hipótesis H3: Días de Estadía (OLS + Efectos Fijos)
# =============================================================================
# Prueba si la intensidad de procedimientos predice independientemente
# la duración de la hospitalización, controlando severidad, edad y hospital.
#
# Modelo (por grupo diagnóstico):
#   dias_estadia = α
#                + β_proc × n_procedimientos
#                + β_sev  × C(severidad_grd)
#                + β_edad × edad
#                + Σ γ_h  × C(hospital)
#                + ε
#
# Interpretación del coeficiente β_proc:
#   β > 0 → cada procedimiento adicional agrega β días de estadía
#            (esperado: más procedimientos = estancia más larga)
#   β ≈ 0 → los procedimientos no predicen independientemente la estadía
#            una vez controlada la severidad
#
# Especificación técnica:
#   - statsmodels.formula.api.ols
#   - Efectos fijos por hospital vía C(COD_HOSPITAL)
#   - Diagnóstico de multicolinealidad: Factor de Inflación de Varianza (VIF)
#   - Se reportan: coeficientes, IC95%, p-valores, R² ajustado, F-estadístico
#
# Salidas:
#   - outputs/tablas/coeficientes_regresion_ols.csv
#   - outputs/tablas/diagnostico_vif.csv
#   - outputs/modelos/ols_neoplasia_summary.txt
#   - outputs/modelos/ols_sepsis_summary.txt
# =============================================================================

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALPHA, COL_HOSPITAL, COL_SEVERIDAD, COL_PESO,
    DIR_MODELOS, DIR_TABLAS, sig_etiqueta,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

COLUMNAS_NECESARIAS = [
    'COD_HOSPITAL', 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


# =============================================================================
# DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)
# =============================================================================

def calcular_vif(df_modelo: pd.DataFrame, etiqueta: str) -> pd.DataFrame:
    """Calcula el Factor de Inflación de Varianza (VIF) para predictores numéricos.

    Interpretación del VIF:
      VIF < 5    → Baja multicolinealidad (aceptable)
      VIF 5-10   → Multicolinealidad moderada (revisar)
      VIF > 10   → Alta multicolinealidad (problema serio, coeficientes inestables)

    Se excluyen las dummies de hospital porque generan VIF artificialmente altos.

    Parámetros
    ----------
    df_modelo : pd.DataFrame
    etiqueta : str

    Retorna
    -------
    pd.DataFrame con VIF por variable.
    """
    # Solo variables numéricas continuas (no las dummies de hospital)
    cols_num = ['n_procedimientos', 'edad', COL_SEVERIDAD]
    X = df_modelo[cols_num].dropna().copy()
    X[COL_SEVERIDAD] = pd.to_numeric(X[COL_SEVERIDAD], errors='coerce')
    X = X.dropna()

    if len(X) < 10:
        return pd.DataFrame()

    # Agregar constante para el VIF (statsmodels requiere esto)
    from statsmodels.tools.tools import add_constant
    Xc = add_constant(X)

    filas = []
    for i, col in enumerate(Xc.columns):
        try:
            vif_val = variance_inflation_factor(Xc.values, i)
            filas.append({
                'grupo':      etiqueta,
                'variable':   col,
                'VIF':        round(vif_val, 4),
                'interpretacion': (
                    'OK (< 5)' if vif_val < 5 else
                    'Moderado (5-10)' if vif_val < 10 else
                    'Alto (> 10) — multicolinealidad'
                )
            })
        except Exception:
            pass

    return pd.DataFrame(filas)


# =============================================================================
# AJUSTE DEL MODELO OLS
# =============================================================================

def ajustar_ols(df: pd.DataFrame, etiqueta: str) -> dict:
    """Ajusta OLS con efectos fijos por hospital para un grupo diagnóstico.

    Parámetros
    ----------
    df : pd.DataFrame
        Datos limpios.
    etiqueta : str
        Nombre del grupo ('Neoplasia' o 'Sepsis').

    Retorna
    -------
    dict con resultado_modelo, tabla_coeficientes, metricas, vif, residuos, predichos
        o None si hay error.
    """
    columnas_req = ['dias_estadia', 'n_procedimientos', COL_SEVERIDAD, 'edad', COL_HOSPITAL]
    df_modelo = df[columnas_req].dropna().copy()
    df_modelo[COL_SEVERIDAD] = pd.to_numeric(df_modelo[COL_SEVERIDAD], errors='coerce')
    df_modelo = df_modelo.dropna()
    df_modelo[COL_SEVERIDAD] = df_modelo[COL_SEVERIDAD].astype(int)

    # Excluir hospitales sin varianza en la variable dependiente (OLS no puede ajustar)
    desv_est_por_hosp = df_modelo.groupby(COL_HOSPITAL)['dias_estadia'].std()
    hosps_validos = desv_est_por_hosp[desv_est_por_hosp > 0].index
    df_modelo = df_modelo[df_modelo[COL_HOSPITAL].isin(hosps_validos)]

    n_obs = len(df_modelo)
    logger.info("[%s] Regresión OLS — N observaciones: {:,}".format(n_obs) % etiqueta)

    if n_obs < 100:
        logger.warning("[%s] Muy pocas observaciones para OLS.", etiqueta)
        return None

    formula = f"dias_estadia ~ n_procedimientos + C({COL_SEVERIDAD}) + edad + C({COL_HOSPITAL})"

    logger.info("[%s] Ajustando modelo OLS...", etiqueta)
    try:
        modelo = smf.ols(formula, data=df_modelo)
        resultado = modelo.fit()
    except Exception as e:
        logger.error("[%s] Error al ajustar OLS: %s", etiqueta, e)
        return None

    # --- Tabla de coeficientes ---
    params = resultado.params
    conf   = resultado.conf_int()
    pvals  = resultado.pvalues

    filas = []
    for var in params.index:
        # Omitir dummies de hospital individualmente (son decenas de coeficientes)
        if var.startswith(f'C({COL_HOSPITAL})'):
            continue

        coef  = params[var]
        ic_lo = conf.loc[var, 0]
        ic_hi = conf.loc[var, 1]
        p     = pvals[var]

        filas.append({
            'grupo_diagnostico': etiqueta,
            'variable':  var,
            'coef':      round(coef, 4),
            'IC95_inf':  round(ic_lo, 4),
            'IC95_sup':  round(ic_hi, 4),
            'p_valor':   p,
            'p_display': '<0.001' if p < 0.001 else f'{p:.4f}',
            'sig':       sig_etiqueta(p),
        })

    tabla_coef = pd.DataFrame(filas)

    # --- VIF para diagnóstico de multicolinealidad ---
    tabla_vif = calcular_vif(df_modelo, etiqueta)

    # --- Métricas globales ---
    metricas = {
        'grupo_diagnostico':  etiqueta,
        'n_observaciones':    n_obs,
        'r2_ajustado':        round(resultado.rsquared_adj, 4),
        'f_estadistico':      round(resultado.fvalue, 4),
        'f_pvalor':           float(resultado.f_pvalue),
        'aic':                round(resultado.aic, 2),
        'bic':                round(resultado.bic, 2),
    }

    logger.info(
        "[%s] R² ajustado=%.4f  F=%.2f  N=%d",
        etiqueta, metricas['r2_ajustado'], metricas['f_estadistico'], n_obs,
    )

    return {
        'resultado_modelo':   resultado,
        'tabla_coeficientes': tabla_coef,
        'metricas':           metricas,
        'tabla_vif':          tabla_vif,
        'residuos':           pd.Series(resultado.resid, name=f'residuos_{etiqueta.lower()}'),
        'predichos':          pd.Series(resultado.fittedvalues, name='predichos'),
        'observados':         df_modelo['dias_estadia'].reset_index(drop=True),
    }


# =============================================================================
# IMPRESIÓN DE RESULTADOS
# =============================================================================

def imprimir_tabla_ols(tabla: pd.DataFrame, metricas: dict, etiqueta: str) -> None:
    """Imprime tabla de coeficientes OLS con métricas globales."""
    print(f"\n=== Regresión OLS — {etiqueta} ===")
    print(f"{'Variable':<35} {'Coef':>8} {'IC95% inf':>10} {'IC95% sup':>10} {'p-valor':>10} {'Sig':>5}")
    print("-" * 84)
    for _, r in tabla.iterrows():
        print(
            f"{r['variable']:<35} {r['coef']:>+8.4f} "
            f"{r['IC95_inf']:>10.4f} {r['IC95_sup']:>10.4f} "
            f"{r['p_display']:>10} {r['sig']:>5}"
        )
    p_f = metricas['f_pvalor']
    print(f"\n  R² ajustado: {metricas['r2_ajustado']:.4f}")
    print(f"  F-estadístico: {metricas['f_estadistico']:.2f}  (p={'<0.001' if p_f < 0.001 else f'{p_f:.4f}'})")
    print(f"  N observaciones: {metricas['n_observaciones']:,}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 06_regresion_ols.py — INICIO ===")

    os.makedirs(DIR_TABLAS, exist_ok=True)
    os.makedirs(DIR_MODELOS, exist_ok=True)

    # Cargar y preparar datos
    from utils import (
        limpiar_datos, derivar_variables, filtrar_grupos_diagnosticos,
        liberar_memoria, cargar_datos_grd,
    )
    datos_crudos = cargar_datos_grd(columnas=COLUMNAS_NECESARIAS)
    datos_crudos = derivar_variables(datos_crudos)
    df_neo, df_sep = filtrar_grupos_diagnosticos(datos_crudos)
    liberar_memoria(datos_crudos)
    df_neo = limpiar_datos(df_neo, 'neoplasia')
    df_sep = limpiar_datos(df_sep, 'sepsis')

    tablas_coef = []
    tablas_vif  = []
    metricas_all = []

    for etiqueta, df in [('Neoplasia', df_neo), ('Sepsis', df_sep)]:
        resultado = ajustar_ols(df, etiqueta)

        if resultado is None:
            continue

        tabla_coef = resultado['tabla_coeficientes']
        metricas   = resultado['metricas']
        modelo_res = resultado['resultado_modelo']
        tabla_vif  = resultado['tabla_vif']

        imprimir_tabla_ols(tabla_coef, metricas, etiqueta)

        # Mostrar VIF si está disponible
        if not tabla_vif.empty:
            print(f"\n  Diagnóstico VIF ({etiqueta}):")
            print(tabla_vif[['variable', 'VIF', 'interpretacion']].to_string(index=False))

        tablas_coef.append(tabla_coef)
        metricas_all.append(metricas)
        if not tabla_vif.empty:
            tablas_vif.append(tabla_vif)

        # Guardar summary completo del modelo
        ruta_summary = os.path.join(DIR_MODELOS, f'ols_{etiqueta.lower()}_summary.txt')
        with open(ruta_summary, 'w') as f:
            f.write(modelo_res.summary().as_text())
        logger.info("Summary OLS guardado: %s", ruta_summary)

        # Guardar métricas en JSON
        ruta_metricas = os.path.join(DIR_MODELOS, f'ols_{etiqueta.lower()}_metricas.json')
        with open(ruta_metricas, 'w') as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)

    # Guardar tablas combinadas
    if tablas_coef:
        combinada = pd.concat(tablas_coef, ignore_index=True)
        ruta = os.path.join(DIR_TABLAS, 'coeficientes_regresion_ols.csv')
        combinada.to_csv(ruta, index=False)
        logger.info("Coeficientes OLS guardados: %s", ruta)

    if tablas_vif:
        vif_combinada = pd.concat(tablas_vif, ignore_index=True)
        ruta_vif = os.path.join(DIR_TABLAS, 'diagnostico_vif.csv')
        vif_combinada.to_csv(ruta_vif, index=False)
        logger.info("Tabla VIF guardada: %s", ruta_vif)

    print("\n=== RESUMEN DE MÉTRICAS ===")
    for m in metricas_all:
        print(f"  {m['grupo_diagnostico']}: R² ajustado={m['r2_ajustado']:.4f}, "
              f"F={m['f_estadistico']:.2f}, N={m['n_observaciones']:,}")

    logger.info("=== 06_regresion_ols.py — COMPLETO ===")


if __name__ == '__main__':
    main()
