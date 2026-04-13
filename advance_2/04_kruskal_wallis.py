# =============================================================================
# 04_kruskal_wallis.py — Hipótesis H1: Variabilidad Inter-Hospital
# =============================================================================
# Prueba si n_procedimientos, n_proc_unicos y dias_estadia difieren
# significativamente entre hospitales para cada grupo diagnóstico.
#
# Método:
#   1. Filtrar hospitales con >= MIN_CASOS registros (poder estadístico suficiente)
#   2. Kruskal-Wallis H: ANOVA no paramétrico por rangos
#   3. Si p < alpha → Post-hoc Dunn con corrección de Bonferroni
#   4. Guardar tablas de resultados
#
# Interpretación:
#   H0: Las distribuciones de la variable son IGUALES en todos los hospitales
#   H1: Al menos un hospital difiere significativamente
#   Rechazar H0 → evidencia de variabilidad en la práctica clínica
#
# Salidas:
#   - outputs/tablas/resultados_kruskal_wallis.csv
#   - outputs/tablas/dunn_posthoc_neoplasia.csv
#   - outputs/tablas/dunn_posthoc_sepsis.csv
# =============================================================================

import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALPHA, COL_HOSPITAL, MIN_CASOS, DIR_TABLAS, sig_etiqueta

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
# PRUEBA KRUSKAL-WALLIS
# =============================================================================

def prueba_kruskal_wallis(df: pd.DataFrame, variable: str, etiqueta: str) -> dict:
    """Aplica Kruskal-Wallis agrupando por hospital.

    Parámetros
    ----------
    df : pd.DataFrame
        Datos limpios de un grupo diagnóstico.
    variable : str
        Columna a analizar (dias_estadia, n_procedimientos, etc.)
    etiqueta : str
        Nombre del grupo diagnóstico (para logging y tabla).

    Retorna
    -------
    dict con H-stat, p-valor, n_hospitales, conclusión.
    """
    # Agrupar solo los hospitales que cumplan el mínimo de casos
    grupos = [
        grp[variable].dropna().values
        for _, grp in df.groupby(COL_HOSPITAL)
        if len(grp) >= MIN_CASOS
    ]
    n_hospitales = len(grupos)

    # Necesitamos al menos 2 grupos para el test
    if n_hospitales < 2:
        return {
            'grupo_diagnostico': etiqueta,
            'variable': variable,
            'H_stat': np.nan, 'p_valor': np.nan, 'p_display': 'n/a',
            'n_hospitales': n_hospitales,
            'sig': 'n/a',
            'conclusion': 'Insuficientes hospitales',
        }

    H, p = stats.kruskal(*grupos)
    sig = sig_etiqueta(p)
    conclusion = 'RECHAZAR H0' if p < ALPHA else 'NO RECHAZAR H0'

    logger.info(
        "[%s | %s] H=%.2f  p=%.4e  hospitales=%d  → %s %s",
        etiqueta, variable, H, p, n_hospitales, conclusion, sig,
    )

    return {
        'grupo_diagnostico': etiqueta,
        'variable': variable,
        'H_stat': round(H, 4),
        'p_valor': p,
        'p_display': '<0.001' if p < 0.001 else f'{p:.4f}',
        'n_hospitales': n_hospitales,
        'sig': sig,
        'conclusion': conclusion,
    }


# =============================================================================
# POST-HOC DUNN CON CORRECCIÓN BONFERRONI
# =============================================================================

def dunn_bonferroni(df: pd.DataFrame, variable: str, etiqueta: str) -> pd.DataFrame:
    """Comparaciones pareadas entre hospitales con corrección de Bonferroni.

    Implementa el estadístico z de Dunn manualmente (compatible sin
    dependencias externas). Si scikit-posthocs está disponible, lo usa.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    etiqueta : str

    Retorna
    -------
    pd.DataFrame con p-valores pareados corregidos por Bonferroni.
    """
    # Intentar usar scikit-posthocs si está instalado (más robusto)
    try:
        from scikit_posthocs import posthoc_dunn
        hospitales_validos = [
            h for h, grp in df.groupby(COL_HOSPITAL) if len(grp) >= MIN_CASOS
        ]
        df_valido = df[df[COL_HOSPITAL].isin(hospitales_validos)]
        resultado = posthoc_dunn(
            df_valido, val_col=variable, group_col=COL_HOSPITAL, p_adjust='bonferroni'
        )
        filas = []
        hospitales = resultado.columns.tolist()
        for i, h1 in enumerate(hospitales):
            for h2 in hospitales[i+1:]:
                p_adj = resultado.loc[h1, h2]
                filas.append({
                    'grupo_diagnostico': etiqueta, 'variable': variable,
                    'hospital_A': h1, 'hospital_B': h2,
                    'p_bonferroni': round(p_adj, 6),
                    'sig': sig_etiqueta(p_adj),
                })
        return pd.DataFrame(filas)
    except ImportError:
        logger.info("scikit-posthocs no disponible, usando implementación manual de Dunn.")

    # --- Implementación manual del estadístico z de Dunn ---
    hospitales_validos = [
        h for h, grp in df.groupby(COL_HOSPITAL) if len(grp) >= MIN_CASOS
    ]
    datos_por_hosp = {
        h: df[df[COL_HOSPITAL] == h][variable].dropna().values
        for h in hospitales_validos
    }

    # Calcular rangos globales (todos los grupos juntos)
    todos_valores = np.concatenate(list(datos_por_hosp.values()))
    N = len(todos_valores)
    rangos_globales = stats.rankdata(todos_valores)

    # Asignar rangos de vuelta a cada grupo
    idx = 0
    rangos_por_hosp = {}
    for h, valores in datos_por_hosp.items():
        n = len(valores)
        rangos_por_hosp[h] = rangos_globales[idx:idx + n]
        idx += n

    # Corrección por empates (tie correction)
    _, conteos = np.unique(todos_valores, return_counts=True)
    suma_empates = np.sum(conteos ** 3 - conteos)

    filas = []
    pares = list(combinations(hospitales_validos, 2))
    n_comparaciones = len(pares)  # Para la corrección de Bonferroni

    for h1, h2 in pares:
        ni = len(rangos_por_hosp[h1])
        nj = len(rangos_por_hosp[h2])
        Ri = rangos_por_hosp[h1].mean()
        Rj = rangos_por_hosp[h2].mean()

        # Error estándar del estadístico de Dunn
        se = np.sqrt(
            (N * (N + 1) / 12 - suma_empates / (12 * (N - 1)))
            * (1 / ni + 1 / nj)
        )
        if se == 0:
            continue

        z = (Ri - Rj) / se
        p_crudo = 2 * (1 - stats.norm.cdf(abs(z)))  # p bilateral
        p_bonferroni = min(p_crudo * n_comparaciones, 1.0)  # Corrección de Bonferroni

        filas.append({
            'grupo_diagnostico': etiqueta, 'variable': variable,
            'hospital_A': h1, 'hospital_B': h2,
            'z_stat': round(z, 4),
            'p_crudo': round(p_crudo, 6),
            'p_bonferroni': round(p_bonferroni, 6),
            'sig': sig_etiqueta(p_bonferroni),
        })

    return pd.DataFrame(filas)


# =============================================================================
# EJECUCIÓN PARA TODOS LOS GRUPOS Y VARIABLES
# =============================================================================

def ejecutar_todos_los_tests(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> pd.DataFrame:
    """Aplica Kruskal-Wallis a todas las combinaciones variable × grupo.

    Retorna
    -------
    pd.DataFrame con todos los resultados de Kruskal-Wallis.
    """
    variables = ['dias_estadia', 'n_procedimientos', 'n_proc_unicos']
    filas = []

    for etiqueta, df in [('Neoplasia', df_neo), ('Sepsis', df_sep)]:
        for var in variables:
            if var not in df.columns:
                continue
            fila = prueba_kruskal_wallis(df, var, etiqueta)
            filas.append(fila)

    return pd.DataFrame(filas)


def ejecutar_posthoc(df_neo: pd.DataFrame, df_sep: pd.DataFrame, kw_resultados: pd.DataFrame) -> dict:
    """Ejecuta Dunn post-hoc solo para resultados KW significativos.

    Retorna
    -------
    dict {'neoplasia': pd.DataFrame, 'sepsis': pd.DataFrame}
    """
    posthoc = {}

    for etiqueta, df in [('Neoplasia', df_neo), ('Sepsis', df_sep)]:
        # Solo variables con KW significativo
        vars_sig = kw_resultados[
            (kw_resultados['grupo_diagnostico'] == etiqueta) &
            (kw_resultados['p_valor'] < ALPHA)
        ]['variable'].tolist()

        if not vars_sig:
            logger.info("[%s] Sin resultados significativos — omitiendo Dunn post-hoc", etiqueta)
            continue

        frames = []
        for var in vars_sig:
            logger.info("[%s | %s] Ejecutando Dunn-Bonferroni...", etiqueta, var)
            dunn_df = dunn_bonferroni(df, var, etiqueta)
            frames.append(dunn_df)

        if frames:
            posthoc[etiqueta.lower()] = pd.concat(frames, ignore_index=True)

    return posthoc


# =============================================================================
# IMPRIMIR TABLA RESUMEN
# =============================================================================

def imprimir_resumen_kw(kw_resultados: pd.DataFrame) -> None:
    """Imprime los resultados de Kruskal-Wallis en formato tabla."""
    print("\n" + "=" * 82)
    print("RESULTADOS — PRUEBA KRUSKAL-WALLIS (H1: Variabilidad Inter-Hospital)")
    print("=" * 82)
    print(f"{'Grupo':<12} {'Variable':<20} {'H-stat':>8} {'p-valor':>10} {'Hospitales':>11} {'Sig':>5} {'Conclusión'}")
    print("-" * 82)
    for _, r in kw_resultados.iterrows():
        print(
            f"{r['grupo_diagnostico']:<12} {r['variable']:<20} {r['H_stat']:>8.2f} "
            f"{r['p_display']:>10} {r['n_hospitales']:>11} {r['sig']:>5} {r['conclusion']}"
        )
    print("=" * 82)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 04_kruskal_wallis.py — INICIO ===")

    os.makedirs(DIR_TABLAS, exist_ok=True)

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

    # Kruskal-Wallis para todos los grupos y variables
    kw_resultados = ejecutar_todos_los_tests(df_neo, df_sep)
    imprimir_resumen_kw(kw_resultados)

    ruta_kw = os.path.join(DIR_TABLAS, 'resultados_kruskal_wallis.csv')
    kw_resultados.to_csv(ruta_kw, index=False)
    logger.info("Resultados KW guardados: %s", ruta_kw)

    # Post-hoc Dunn para resultados significativos
    posthoc = ejecutar_posthoc(df_neo, df_sep, kw_resultados)

    for clave, dunn_df in posthoc.items():
        ruta_dunn = os.path.join(DIR_TABLAS, f'dunn_posthoc_{clave}.csv')
        dunn_df.to_csv(ruta_dunn, index=False)
        logger.info("Dunn post-hoc guardado: %s", ruta_dunn)

        # Solo mostrar pares significativos para no saturar la consola
        pares_sig = dunn_df[dunn_df['p_bonferroni'] < ALPHA]
        print(f"\n--- Dunn-Bonferroni: {clave} ---")
        print(f"Pares significativos: {len(pares_sig)} de {len(dunn_df)} comparaciones")
        if not pares_sig.empty:
            print(pares_sig[['variable', 'hospital_A', 'hospital_B', 'p_bonferroni', 'sig']].head(20).to_string(index=False))

    logger.info("=== 04_kruskal_wallis.py — COMPLETO ===")


if __name__ == '__main__':
    main()
