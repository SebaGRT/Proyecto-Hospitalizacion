# =============================================================================
# 05_regresion_logistica.py — Hipótesis H2: Mortalidad Intrahospitalaria
# =============================================================================
# Prueba si la intensidad de procedimientos (n_procedimientos) está asociada
# con la mortalidad, controlando por severidad GRD, edad y efectos fijos
# por hospital.
#
# Modelo (por grupo diagnóstico):
#   logit(P(mortalidad=1)) = α
#                          + β_proc × n_procedimientos
#                          + β_sev  × C(severidad_grd)
#                          + β_edad × edad
#                          + Σ γ_h  × C(hospital)
#
# Interpretación del coeficiente β_proc:
#   β > 0 → más procedimientos asociados a MAYOR riesgo de muerte
#            (los procedimientos reflejan gravedad del caso)
#   β < 0 → más procedimientos asociados a MENOR riesgo de muerte
#            (el tratamiento más intensivo es protector)
#
# Especificación técnica:
#   - statsmodels.formula.api.logit
#   - Efectos fijos por hospital vía C(COD_HOSPITAL)
#   - Se reportan: Coeficientes, Odds Ratio (exp(β)), IC95%, p-valores
#   - Pseudo-R² de McFadden como medida de bondad de ajuste
#
# Salidas:
#   - outputs/tablas/coeficientes_regresion_logistica.csv
#   - outputs/modelos/logit_neoplasia_summary.txt
#   - outputs/modelos/logit_sepsis_summary.txt
#   - outputs/modelos/logit_neoplasia_metricas.json
# =============================================================================

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

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

# Suprimir advertencias de convergencia numéricas (comunes con muchas variables dummy)
warnings.filterwarnings('ignore', category=RuntimeWarning)

COLUMNAS_NECESARIAS = [
    'COD_HOSPITAL', 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


# =============================================================================
# AJUSTE DEL MODELO LOGÍSTICO
# =============================================================================

def ajustar_logit(df: pd.DataFrame, etiqueta: str) -> dict:
    """Ajusta el modelo de regresión logística para un grupo diagnóstico.

    Parámetros
    ----------
    df : pd.DataFrame
        Datos limpios con columnas: mortalidad, n_procedimientos,
        IR_29301_SEVERIDAD, edad, COD_HOSPITAL.
    etiqueta : str
        Nombre del grupo ('Neoplasia' o 'Sepsis').

    Retorna
    -------
    dict con claves: resultado_modelo, tabla_coeficientes, metricas
        o None si el modelo no puede ajustarse.
    """
    columnas_req = ['mortalidad', 'n_procedimientos', COL_SEVERIDAD, 'edad', COL_HOSPITAL]
    df_modelo = df[columnas_req].dropna().copy()
    df_modelo[COL_SEVERIDAD] = df_modelo[COL_SEVERIDAD].astype(int)

    # Excluir hospitales donde toda la población tiene el mismo resultado
    # (todos vivos o todos muertos) — el logit no puede converger en esos casos
    tasa_mort = df_modelo.groupby(COL_HOSPITAL)['mortalidad'].mean()
    hosps_validos = tasa_mort[(tasa_mort > 0) & (tasa_mort < 1)].index
    df_modelo = df_modelo[df_modelo[COL_HOSPITAL].isin(hosps_validos)]

    n_obs = len(df_modelo)
    logger.info("[%s] Regresión logística — N observaciones: {:,}".format(n_obs) % etiqueta)

    if n_obs < 100:
        logger.warning("[%s] Muy pocas observaciones para logit (%d).", etiqueta, n_obs)
        return None

    # C() en statsmodels: trata la variable como categórica (genera dummies)
    formula = f"mortalidad ~ n_procedimientos + C({COL_SEVERIDAD}) + edad + C({COL_HOSPITAL})"

    logger.info("[%s] Ajustando modelo logístico...", etiqueta)
    try:
        modelo = smf.logit(formula, data=df_modelo)
        resultado = modelo.fit(
            method='bfgs',      # BFGS es más estable que Newton con muchas dummies
            maxiter=200,
            disp=False,
            warn_convergence=False,
        )
    except Exception as e:
        logger.error("[%s] Error al ajustar el modelo: %s", etiqueta, e)
        return None

    # --- Tabla de coeficientes ---
    params = resultado.params
    conf   = resultado.conf_int()
    pvals  = resultado.pvalues

    filas = []
    for var in params.index:
        # Omitir efectos fijos de hospitales individuales (son muchos y dilatan la tabla)
        if var.startswith(f'C({COL_HOSPITAL})'):
            continue

        coef  = params[var]
        ic_lo = conf.loc[var, 0]
        ic_hi = conf.loc[var, 1]
        p     = pvals[var]

        # Odds Ratio: exp(β). OR > 1 = factor de riesgo; OR < 1 = factor protector
        or_   = np.exp(coef)
        or_lo = np.exp(ic_lo)
        or_hi = np.exp(ic_hi)

        filas.append({
            'grupo_diagnostico': etiqueta,
            'variable':  var,
            'coef':      round(coef, 4),
            'OR':        round(or_, 4),
            'IC95_inf':  round(or_lo, 4),
            'IC95_sup':  round(or_hi, 4),
            'p_valor':   p,
            'p_display': '<0.001' if p < 0.001 else f'{p:.4f}',
            'sig':       sig_etiqueta(p),
        })

    tabla_coef = pd.DataFrame(filas)

    # --- Métricas del modelo ---
    metricas = {
        'grupo_diagnostico':   etiqueta,
        'n_observaciones':     n_obs,
        'pseudo_r2_mcfadden':  round(resultado.prsquared, 4),
        'log_verosimilitud':   round(resultado.llf, 2),
        'AIC': round(resultado.aic, 2),
        'BIC': round(resultado.bic, 2),
        'convergencia': bool(resultado.mle_retvals.get('converged', True)),
    }

    logger.info(
        "[%s] Pseudo-R²=%.4f  AIC=%.1f  N=%d",
        etiqueta, metricas['pseudo_r2_mcfadden'], metricas['AIC'], n_obs,
    )

    return {
        'resultado_modelo':    resultado,
        'tabla_coeficientes':  tabla_coef,
        'metricas':            metricas,
    }


# =============================================================================
# IMPRESIÓN DE RESULTADOS
# =============================================================================

def imprimir_tabla_coef(tabla: pd.DataFrame, etiqueta: str) -> None:
    """Imprime la tabla de coeficientes con OR e IC95% en la consola."""
    print(f"\n=== Regresión Logística — {etiqueta} ===")
    print(f"{'Variable':<35} {'Coef':>8} {'OR':>8} {'IC95% inf':>10} {'IC95% sup':>10} {'p-valor':>10} {'Sig':>5}")
    print("-" * 92)
    for _, r in tabla.iterrows():
        print(
            f"{r['variable']:<35} {r['coef']:>+8.4f} {r['OR']:>8.4f} "
            f"{r['IC95_inf']:>10.4f} {r['IC95_sup']:>10.4f} "
            f"{r['p_display']:>10} {r['sig']:>5}"
        )


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 05_regresion_logistica.py — INICIO ===")

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
    metricas_all = []

    for etiqueta, df in [('Neoplasia', df_neo), ('Sepsis', df_sep)]:
        resultado = ajustar_logit(df, etiqueta)

        if resultado is None:
            logger.warning("[%s] Modelo no disponible, se omite.", etiqueta)
            continue

        tabla_coef = resultado['tabla_coeficientes']
        metricas   = resultado['metricas']
        modelo_res = resultado['resultado_modelo']

        imprimir_tabla_coef(tabla_coef, etiqueta)
        print(f"\n  Pseudo-R² (McFadden): {metricas['pseudo_r2_mcfadden']:.4f}")
        print(f"  N observaciones:      {metricas['n_observaciones']:,}")

        tablas_coef.append(tabla_coef)
        metricas_all.append(metricas)

        # Guardar resumen completo (incluye todos los efectos fijos por hospital)
        ruta_summary = os.path.join(DIR_MODELOS, f'logit_{etiqueta.lower()}_summary.txt')
        with open(ruta_summary, 'w') as f:
            f.write(modelo_res.summary().as_text())
        logger.info("Summary guardado: %s", ruta_summary)

        # Guardar métricas en JSON (útil para importar en otros scripts)
        ruta_metricas = os.path.join(DIR_MODELOS, f'logit_{etiqueta.lower()}_metricas.json')
        with open(ruta_metricas, 'w') as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)

    # Guardar tabla combinada de coeficientes
    if tablas_coef:
        combinada = pd.concat(tablas_coef, ignore_index=True)
        ruta = os.path.join(DIR_TABLAS, 'coeficientes_regresion_logistica.csv')
        combinada.to_csv(ruta, index=False)
        logger.info("Tabla de coeficientes guardada: %s", ruta)

        print("\n=== RESUMEN DE MÉTRICAS ===")
        for m in metricas_all:
            print(f"  {m['grupo_diagnostico']}: Pseudo-R²={m['pseudo_r2_mcfadden']:.4f}, "
                  f"N={m['n_observaciones']:,}, AIC={m['AIC']:.1f}")

    logger.info("=== 05_regresion_logistica.py — COMPLETO ===")


if __name__ == '__main__':
    main()
