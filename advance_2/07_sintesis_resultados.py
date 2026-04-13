# =============================================================================
# 07_sintesis_resultados.py — Síntesis de Resultados y Visualizaciones Finales
# =============================================================================
# Lee las tablas generadas por los scripts 04-06 y produce:
#
#   1. Gráfico de comparación de coeficientes: neoplasia vs. sepsis
#      (β de n_procedimientos en logit y OLS, con IC95%)
#   2. Diagnóstico de residuos OLS: histograma + KDE + Q-Q
#   3. Predichos vs. observados (scatter con línea de predicción perfecta)
#   4. Heatmap de efectos fijos por hospital (neoplasia y sepsis)
#
# Tabla final:
#   - tabla_comparativa_neoplasia_vs_sepsis.csv
#     Resumen cruzado de los tres tests y ambos grupos diagnósticos
#
# Entradas (de scripts anteriores):
#   - outputs/tablas/resultados_kruskal_wallis.csv
#   - outputs/tablas/coeficientes_regresion_logistica.csv
#   - outputs/tablas/coeficientes_regresion_ols.csv
#
# Salidas: outputs/graficos/ y outputs/tablas/
# =============================================================================

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALPHA, COL_HOSPITAL, COL_SEVERIDAD,
    DPI_FIGURA, ESTILO_FIGURA, DIR_GRAFICOS, DIR_TABLAS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

try:
    plt.style.use(ESTILO_FIGURA)
except OSError:
    plt.style.use('seaborn-whitegrid')


# =============================================================================
# CARGA DE TABLAS PREVIAS
# =============================================================================

def cargar_tablas() -> dict:
    """Lee los CSV generados por los scripts 04-06.

    Retorna
    -------
    dict con claves 'logit', 'ols', 'kruskal', 'shapiro'
    """
    archivos = {
        'logit':   os.path.join(DIR_TABLAS, 'coeficientes_regresion_logistica.csv'),
        'ols':     os.path.join(DIR_TABLAS, 'coeficientes_regresion_ols.csv'),
        'kruskal': os.path.join(DIR_TABLAS, 'resultados_kruskal_wallis.csv'),
        'shapiro': os.path.join(DIR_TABLAS, 'resultados_shapiro_wilk.csv'),
    }
    tablas = {}
    for clave, ruta in archivos.items():
        if os.path.exists(ruta):
            tablas[clave] = pd.read_csv(ruta)
            logger.info("Cargado: %s (%d filas)", ruta, len(tablas[clave]))
        else:
            logger.warning("Archivo no encontrado: %s", ruta)
            tablas[clave] = None
    return tablas


# =============================================================================
# FIGURA 1: Comparación de coeficientes neoplasia vs. sepsis
# =============================================================================

def graficar_comparacion_coeficientes(tablas: dict) -> None:
    """Barplot con IC95% del coeficiente de n_procedimientos en ambos grupos.

    Permite comparar visualmente si el efecto de los procedimientos sobre
    mortalidad y estadía va en la misma o diferente dirección según el
    tipo de diagnóstico (neoplasia vs. sepsis).
    """
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Efecto de n_procedimientos sobre Resultados Clínicos\nNeoplasia vs. Sepsis',
        fontsize=13, fontweight='bold',
    )

    for ax, clave_modelo, nombre_modelo in [
        (ejes[0], 'logit', 'Mortalidad (Regresión Logística)'),
        (ejes[1], 'ols',   'Días de Estadía (OLS)'),
    ]:
        df = tablas.get(clave_modelo)
        if df is None:
            ax.set_title(f'{nombre_modelo} — datos no disponibles')
            continue

        # Filtrar solo la fila de n_procedimientos para cada grupo
        fila_neo = df[(df['grupo_diagnostico'] == 'Neoplasia') & (df['variable'] == 'n_procedimientos')]
        fila_sep = df[(df['grupo_diagnostico'] == 'Sepsis')   & (df['variable'] == 'n_procedimientos')]

        grupos = ['Neoplasia', 'Sepsis']
        coefs  = []
        errores_lo = []
        errores_hi = []
        colores = ['#2196F3', '#FF5722']

        for fila in [fila_neo, fila_sep]:
            if fila.empty:
                coefs.append(0)
                errores_lo.append(0)
                errores_hi.append(0)
            else:
                coef = fila.iloc[0]['coef']
                lo   = fila.iloc[0]['IC95_inf']
                hi   = fila.iloc[0]['IC95_sup']
                coefs.append(coef)
                errores_lo.append(coef - lo)
                errores_hi.append(hi - coef)

        x = np.arange(len(grupos))
        ax.bar(x, coefs, color=colores, alpha=0.8, width=0.5)

        # Barras de error para IC95%
        for i in range(len(coefs)):
            ax.errorbar(
                x[i], coefs[i],
                yerr=[[errores_lo[i]], [errores_hi[i]]],
                fmt='none', color='black', capsize=6, linewidth=1.5,
            )

        # Línea horizontal en cero (referencia de "sin efecto")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(grupos, fontsize=11)
        ax.set_title(nombre_modelo, fontsize=11)
        ax.set_ylabel('Coeficiente (β) ± IC95%')
        ax.set_xlabel('Grupo Diagnóstico')

        # Anotaciones de significancia
        for i, fila in enumerate([fila_neo, fila_sep]):
            if not fila.empty:
                sig = fila.iloc[0].get('sig', '')
                offset = abs(coefs[i]) * 0.05 + 0.001
                ax.text(x[i], coefs[i] + (offset if coefs[i] >= 0 else -offset),
                        sig, ha='center', va='bottom' if coefs[i] >= 0 else 'top',
                        fontsize=12, fontweight='bold')

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '07_comparacion_coeficientes.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 2: Diagnóstico de residuos OLS
# =============================================================================

def graficar_residuos(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Histograma + KDE + Q-Q plot de los residuos del modelo OLS.

    La normalidad de los residuos es un supuesto de OLS.
    Si los residuos son muy no-normales, los intervalos de confianza
    pueden ser poco confiables y se debe considerar transformación o
    regresión robusta.
    """
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings('ignore')

    fig, ejes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Diagnóstico de Residuos OLS', fontsize=13, fontweight='bold')

    for i, (df, etiqueta) in enumerate([(df_neo, 'Neoplasia'), (df_sep, 'Sepsis')]):
        # Preparar datos mínimos para re-ajustar OLS
        cols_req = ['dias_estadia', 'n_procedimientos', COL_SEVERIDAD, 'edad', COL_HOSPITAL]
        df_m = df[cols_req].dropna().copy()
        df_m[COL_SEVERIDAD] = pd.to_numeric(df_m[COL_SEVERIDAD], errors='coerce')
        df_m = df_m.dropna()
        df_m[COL_SEVERIDAD] = df_m[COL_SEVERIDAD].astype(int)

        if len(df_m) < 50:
            continue

        try:
            formula = f"dias_estadia ~ n_procedimientos + C({COL_SEVERIDAD}) + edad + C({COL_HOSPITAL})"
            res = smf.ols(formula, data=df_m).fit()
            residuos = res.resid
        except Exception as e:
            logger.warning("[%s] No se pudo ajustar OLS para residuos: %s", etiqueta, e)
            continue

        # Histograma + KDE
        ax_hist = ejes[i, 0]
        ax_hist.hist(residuos, bins=60, density=True, alpha=0.6,
                     color='#2196F3' if i == 0 else '#FF5722')
        residuos.plot.kde(ax=ax_hist,
                          color='navy' if i == 0 else 'darkred', linewidth=2)
        ax_hist.set_title(f'{etiqueta}: Distribución de Residuos')
        ax_hist.set_xlabel('Residuos (días)')
        ax_hist.set_ylabel('Densidad')

        # Q-Q plot: si los puntos se alejan de la diagonal, los residuos no son normales
        ax_qq = ejes[i, 1]
        sp_stats.probplot(residuos, dist='norm', plot=ax_qq)
        ax_qq.set_title(f'{etiqueta}: Q-Q Plot de Residuos')
        ax_qq.get_lines()[0].set(markersize=2, alpha=0.4)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '08_diagnostico_residuos.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 3: Valores predichos vs. observados
# =============================================================================

def graficar_predichos_vs_observados(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Scatter de dias_estadia observados vs. predichos por OLS (top 5 hospitales).

    Una nube dispersada alrededor de la diagonal indica buen ajuste.
    Puntos sistemáticamente por encima o debajo sugieren sesgos del modelo.
    """
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings('ignore')

    fig, ejes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Días de Estadía: Observados vs. Predichos (OLS)', fontsize=13, fontweight='bold')

    for ax, df, etiqueta in [(ejes[0], df_neo, 'Neoplasia'), (ejes[1], df_sep, 'Sepsis')]:
        cols_req = ['dias_estadia', 'n_procedimientos', COL_SEVERIDAD, 'edad', COL_HOSPITAL]
        df_m = df[cols_req].dropna().copy()
        df_m[COL_SEVERIDAD] = pd.to_numeric(df_m[COL_SEVERIDAD], errors='coerce')
        df_m = df_m.dropna()
        df_m[COL_SEVERIDAD] = df_m[COL_SEVERIDAD].astype(int)

        # Limitar a top 5 hospitales para legibilidad del gráfico
        top5 = df_m[COL_HOSPITAL].value_counts().head(5).index.tolist()
        df_m = df_m[df_m[COL_HOSPITAL].isin(top5)]

        if len(df_m) < 50:
            continue

        try:
            formula = f"dias_estadia ~ n_procedimientos + C({COL_SEVERIDAD}) + edad + C({COL_HOSPITAL})"
            res = smf.ols(formula, data=df_m).fit()
            predichos = res.fittedvalues
        except Exception as e:
            logger.warning("[%s] OLS para predichos falló: %s", etiqueta, e)
            continue

        paleta = dict(zip(top5, sns.color_palette('tab10', 5)))
        for hosp in top5:
            mascara = df_m[COL_HOSPITAL] == hosp
            ax.scatter(
                df_m.loc[mascara, 'dias_estadia'],
                predichos[mascara],
                label=hosp, alpha=0.3, s=8, color=paleta[hosp],
            )

        # Línea diagonal = predicción perfecta (predicho == observado)
        lims = [
            min(df_m['dias_estadia'].min(), predichos.min()),
            max(df_m['dias_estadia'].max(), predichos.max()),
        ]
        ax.plot(lims, lims, 'k--', linewidth=1.2, label='Predicción perfecta')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(f'{etiqueta}')
        ax.set_xlabel('Observado (días de estadía)')
        ax.set_ylabel('Predicho (días de estadía)')
        ax.legend(title='Hospital', fontsize=8, markerscale=3)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '09_predichos_vs_observados.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 4: Heatmap de efectos fijos por hospital
# =============================================================================

def graficar_heatmap_hospitales(tablas: dict) -> None:
    """Heatmap de los efectos fijos por hospital del modelo OLS.

    Valores positivos (rojo): ese hospital tiende a tener estadías más largas
    que el hospital de referencia, incluso controlando severidad y edad.

    Valores negativos (azul): ese hospital tiende a tener estadías más cortas.

    Este gráfico identifica "hospitales atípicos" que merecen análisis cualitativo.
    """
    ols_df = tablas.get('ols')
    if ols_df is None:
        logger.warning("Tabla OLS no disponible para el heatmap.")
        return

    # Extraer solo las filas de efectos fijos de hospital
    filas_hosp = ols_df[ols_df['variable'].str.startswith(f'C({COL_HOSPITAL})')].copy()
    if filas_hosp.empty:
        logger.warning("No se encontraron efectos fijos de hospital en la tabla OLS.")
        return

    # Extraer el código del hospital del nombre de la variable
    filas_hosp['hospital'] = filas_hosp['variable'].str.extract(r'\[T\.(.+)\]')
    filas_hosp = filas_hosp.dropna(subset=['hospital'])

    # Crear tabla pivote: hospitales en filas, grupos diagnósticos en columnas
    pivote = filas_hosp.pivot_table(
        index='hospital', columns='grupo_diagnostico', values='coef',
    )

    if pivote.empty:
        logger.warning("Tabla pivote vacía para heatmap.")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivote) * 0.4)))
    sns.heatmap(
        pivote,
        cmap='RdBu_r',      # Rojo = más días, Azul = menos días
        center=0,            # Centrar la escala en 0 (hospital de referencia)
        annot=True, fmt='.2f',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Efecto fijo (días respecto al hospital de referencia)'},
    )
    ax.set_title(
        'Efectos Fijos por Hospital (OLS)\n'
        'Valores: días adicionales/menos respecto al hospital de referencia',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('Grupo Diagnóstico')
    ax.set_ylabel('Código Hospital')

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '10_heatmap_hospitales.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# TABLA COMPARATIVA FINAL
# =============================================================================

def construir_tabla_comparativa(tablas: dict) -> pd.DataFrame:
    """Construye una tabla resumen cruzando hipótesis y grupos diagnósticos.

    Esta tabla es el producto final del Avance 2: permite ver de un vistazo
    los resultados de H1, H2 y H3 para ambos grupos diagnósticos.
    """
    filas = []

    # H1 — Kruskal-Wallis
    kw = tablas.get('kruskal')
    if kw is not None:
        for _, r in kw.iterrows():
            filas.append({
                'hipotesis':  'H1 (Kruskal-Wallis)',
                'prueba':     'Kruskal-Wallis + Dunn-Bonferroni',
                'variable':   r['variable'],
                'grupo':      r['grupo_diagnostico'],
                'estadistico': r.get('H_stat', ''),
                'p_valor':    r.get('p_display', ''),
                'sig':        r.get('sig', ''),
                'conclusion': r.get('conclusion', ''),
            })

    # H2 — Regresión logística (solo n_procedimientos)
    logit = tablas.get('logit')
    if logit is not None:
        for _, r in logit[logit['variable'] == 'n_procedimientos'].iterrows():
            filas.append({
                'hipotesis':  'H2 (Logística)',
                'prueba':     'Logit + efectos fijos hospital',
                'variable':   'n_procedimientos → mortalidad',
                'grupo':      r['grupo_diagnostico'],
                'estadistico': r.get('coef', ''),
                'p_valor':    r.get('p_display', ''),
                'sig':        r.get('sig', ''),
                'conclusion': f"OR={r.get('OR', '')} IC95%=[{r.get('IC95_inf', '')},{r.get('IC95_sup', '')}]",
            })

    # H3 — OLS (solo n_procedimientos)
    ols = tablas.get('ols')
    if ols is not None:
        for _, r in ols[ols['variable'] == 'n_procedimientos'].iterrows():
            filas.append({
                'hipotesis':  'H3 (OLS)',
                'prueba':     'OLS + efectos fijos hospital',
                'variable':   'n_procedimientos → dias_estadia',
                'grupo':      r['grupo_diagnostico'],
                'estadistico': r.get('coef', ''),
                'p_valor':    r.get('p_display', ''),
                'sig':        r.get('sig', ''),
                'conclusion': f"β={r.get('coef', '')} IC95%=[{r.get('IC95_inf', '')},{r.get('IC95_sup', '')}]",
            })

    return pd.DataFrame(filas)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 07_sintesis_resultados.py — INICIO ===")

    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    os.makedirs(DIR_TABLAS, exist_ok=True)

    # Cargar tablas de resultados previos
    tablas = cargar_tablas()

    # Cargar datos limpios para re-ajustar OLS en los gráficos de diagnóstico
    from utils import (
        limpiar_datos, derivar_variables, filtrar_grupos_diagnosticos,
        liberar_memoria, cargar_datos_grd,
    )
    COLUMNAS_NECESARIAS = [
        'COD_HOSPITAL', 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
        'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
    ] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]

    datos_crudos = cargar_datos_grd(columnas=COLUMNAS_NECESARIAS)
    datos_crudos = derivar_variables(datos_crudos)
    df_neo, df_sep = filtrar_grupos_diagnosticos(datos_crudos)
    liberar_memoria(datos_crudos)
    df_neo = limpiar_datos(df_neo, 'neoplasia')
    df_sep = limpiar_datos(df_sep, 'sepsis')

    # Generar visualizaciones finales
    graficar_comparacion_coeficientes(tablas)
    graficar_residuos(df_neo, df_sep)
    graficar_predichos_vs_observados(df_neo, df_sep)
    graficar_heatmap_hospitales(tablas)

    # Tabla comparativa final
    tabla_comp = construir_tabla_comparativa(tablas)
    ruta_comp = os.path.join(DIR_TABLAS, 'tabla_comparativa_neoplasia_vs_sepsis.csv')
    tabla_comp.to_csv(ruta_comp, index=False)
    logger.info("Tabla comparativa guardada: %s", ruta_comp)

    print("\n=== TABLA COMPARATIVA FINAL ===")
    print(tabla_comp.to_string(index=False))

    logger.info("=== 07_sintesis_resultados.py — COMPLETO ===")


if __name__ == '__main__':
    main()
