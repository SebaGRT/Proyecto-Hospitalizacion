# =============================================================================
# 03_tests_normalidad.py — Pruebas de Normalidad (Shapiro-Wilk)
# =============================================================================
# Justificación para elegir una ruta no paramétrica (H1):
#
#   Antes de aplicar Kruskal-Wallis (H1), debemos verificar que las variables
#   de interés NO siguen una distribución normal. Si no son normales, el
#   uso de ANOVA paramétrico estaría invalidado y Kruskal-Wallis es apropiado.
#
#   Se usa Shapiro-Wilk sobre una submuestra de n=5000 (semilla=42) porque:
#   - El test Shapiro-Wilk solo es válido para n ≤ 5000
#   - Con millones de datos, cualquier prueba rechazaría H0 trivialmente
#   - La submuestra estratificada con semilla fija es reproducible
#
# Salidas:
#   - outputs/tablas/resultados_shapiro_wilk.csv
#   - Interpretación impresa en consola
# =============================================================================

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALPHA, DIR_TABLAS, SEMILLA, SHAPIRO_N
from utils import (
    limpiar_datos, derivar_variables, filtrar_grupos_diagnosticos,
    liberar_memoria, cargar_datos_grd,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

COLUMNAS_NECESARIAS = [
    'COD_HOSPITAL', 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


# =============================================================================
# PRUEBA SHAPIRO-WILK
# =============================================================================

def prueba_shapiro(serie: pd.Series, etiqueta_grupo: str, nombre_var: str) -> dict:
    """Ejecuta Shapiro-Wilk sobre una submuestra aleatoria de la serie.

    Parámetros
    ----------
    serie : pd.Series
        Valores numéricos a testear.
    etiqueta_grupo : str
        Nombre del grupo diagnóstico (para el reporte).
    nombre_var : str
        Nombre de la variable (para el reporte).

    Retorna
    -------
    dict con: grupo, variable, n_muestra, W_stat, p_valor, p_display, conclusion
    """
    rng = np.random.default_rng(SEMILLA)  # Generador reproducible
    valores = serie.dropna().values
    n = min(SHAPIRO_N, len(valores))

    # Muestreo sin reemplazo con semilla fija
    muestra = rng.choice(valores, size=n, replace=False)

    # Estadístico W: cercano a 1 = más normal; cercano a 0 = muy no-normal
    W, p = stats.shapiro(muestra)

    # Si p < alpha rechazamos H0 (normalidad)
    conclusion = 'No normal' if p < ALPHA else 'Normal (no se rechaza H0)'

    return {
        'grupo':      etiqueta_grupo,
        'variable':   nombre_var,
        'n_muestra':  n,
        'W_stat':     round(W, 4),
        'p_valor':    p,
        'p_display':  '<0.001' if p < 0.001 else f'{p:.4f}',
        'conclusion': conclusion,
    }


def ejecutar_tests_normalidad(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> pd.DataFrame:
    """Aplica Shapiro-Wilk a todas las variables clave para ambos grupos.

    Variables evaluadas: dias_estadia, n_procedimientos, n_proc_unicos

    Parámetros
    ----------
    df_neo, df_sep : DataFrames limpios

    Retorna
    -------
    pd.DataFrame con todos los resultados.
    """
    variables = ['dias_estadia', 'n_procedimientos', 'n_proc_unicos']
    resultados = []

    for etiqueta, df in [('Neoplasia', df_neo), ('Sepsis', df_sep)]:
        for var in variables:
            if var not in df.columns:
                logger.warning("Variable '%s' no encontrada en %s", var, etiqueta)
                continue

            fila = prueba_shapiro(df[var], etiqueta, var)
            resultados.append(fila)

            logger.info(
                "[%s | %s] W=%.4f  p=%s  → %s",
                etiqueta, var, fila['W_stat'], fila['p_display'], fila['conclusion'],
            )

    return pd.DataFrame(resultados)


# =============================================================================
# INTERPRETACIÓN TEXTUAL
# =============================================================================

def imprimir_interpretacion(resultados: pd.DataFrame) -> None:
    """Imprime una interpretación clara de los resultados en consola."""
    print("\n" + "=" * 72)
    print("RESULTADOS — PRUEBA DE NORMALIDAD SHAPIRO-WILK")
    print("=" * 72)
    print(f"{'Grupo':<12} {'Variable':<20} {'n':<6} {'W-stat':<8} {'p-valor':<10} {'Conclusión'}")
    print("-" * 72)
    for _, fila in resultados.iterrows():
        print(
            f"{fila['grupo']:<12} {fila['variable']:<20} {fila['n_muestra']:<6} "
            f"{fila['W_stat']:<8.4f} {fila['p_display']:<10} {fila['conclusion']}"
        )
    print("=" * 72)

    no_normales = resultados[resultados['conclusion'] == 'No normal']

    if len(no_normales) == len(resultados):
        print("\nINTERPRETACIÓN: TODAS las variables en AMBOS grupos rechazan")
        print("la normalidad (p < 0.05).")
        print("→ Las pruebas no paramétricas (Kruskal-Wallis) son apropiadas para H1.")
        print("→ Consistente con datos de conteo y días de estadía (sesgo a la derecha).")
    else:
        normales = resultados[resultados['conclusion'] != 'No normal']
        print(f"\nINTERPRETACIÓN: {len(no_normales)} de {len(resultados)} variables son no normales.")
        print("Variables normales:", normales[['grupo', 'variable']].to_string(index=False))
        print("→ Kruskal-Wallis sigue siendo apropiado como elección conservadora.")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 03_tests_normalidad.py — INICIO ===")

    os.makedirs(DIR_TABLAS, exist_ok=True)

    # Cargar y preparar datos
    datos_crudos = cargar_datos_grd(columnas=COLUMNAS_NECESARIAS)
    datos_crudos = derivar_variables(datos_crudos)
    df_neo, df_sep = filtrar_grupos_diagnosticos(datos_crudos)
    liberar_memoria(datos_crudos)
    df_neo = limpiar_datos(df_neo, 'neoplasia')
    df_sep = limpiar_datos(df_sep, 'sepsis')

    # Ejecutar pruebas
    resultados = ejecutar_tests_normalidad(df_neo, df_sep)

    # Mostrar interpretación
    imprimir_interpretacion(resultados)

    # Guardar tabla de resultados
    ruta = os.path.join(DIR_TABLAS, 'resultados_shapiro_wilk.csv')
    resultados.to_csv(ruta, index=False)
    logger.info("Resultados guardados: %s", ruta)

    logger.info("=== 03_tests_normalidad.py — COMPLETO ===")


if __name__ == '__main__':
    main()
