# =============================================================================
# 01_eda_profundo.py — Análisis Exploratorio de Datos Profundo
# =============================================================================
# Genera visualizaciones interpretadas para los grupos neoplasia y sepsis:
#
#   1. Histogramas + KDE: distribución de dias_estadia y n_procedimientos
#   2. Gráficos Q-Q: evaluación visual de normalidad
#   3. Boxplots por nivel de severidad GRD
#   4. Violinplots: variabilidad inter-hospital (top 15 por volumen)
#   5. Barplot: media ± IC95% de n_procedimientos por hospital
#   6. Scatter: peso GRD vs. dias_estadia, coloreado por hospital
#   7. Tabla de completitud exportada a CSV
#
# Salidas: advance_2/outputs/graficos/ y advance_2/outputs/tablas/
# =============================================================================

import logging
import os
import sys

# Configurar backend no-interactivo (necesario en entornos sin display)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# El script está en advance_2/, por lo que config.py y utils.py están en el mismo directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COL_HOSPITAL, COL_SEVERIDAD, COL_PESO,
    DPI_FIGURA, ESTILO_FIGURA, DIR_GRAFICOS, SEMILLA, DIR_TABLAS,
)
from utils import (
    limpiar_datos, tabla_completitud, derivar_variables,
    filtrar_grupos_diagnosticos, liberar_memoria, cargar_datos_grd,
)

# Configurar sistema de logging para ver decisiones en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Aplicar estilo visual
try:
    plt.style.use(ESTILO_FIGURA)
except OSError:
    plt.style.use('seaborn-whitegrid')

# Crear directorios de salida si no existen
os.makedirs(DIR_GRAFICOS, exist_ok=True)
os.makedirs(DIR_TABLAS, exist_ok=True)

# Columnas que necesitamos leer (evitamos cargar todo el dataset para ahorrar memoria)
COLUMNAS_NECESARIAS = [
    COL_HOSPITAL, 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


# =============================================================================
# FIGURA 1: Distribuciones de variables clave
# =============================================================================

def graficar_distribuciones(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Histogramas + KDE para dias_estadia y n_procedimientos, ambos grupos.

    La asimetría hacia la derecha en ambas distribuciones justifica
    el uso de pruebas no paramétricas (Kruskal-Wallis) en H1.
    """
    fig, ejes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribución de Variables por Grupo Diagnóstico', fontsize=14, fontweight='bold')

    pares = [
        (df_neo, 'Neoplasia', 'dias_estadia',      'Días de Estadía',   '#2196F3', ejes[0, 0]),
        (df_sep, 'Sepsis',    'dias_estadia',      'Días de Estadía',   '#FF5722', ejes[0, 1]),
        (df_neo, 'Neoplasia', 'n_procedimientos',  'N° Procedimientos', '#2196F3', ejes[1, 0]),
        (df_sep, 'Sepsis',    'n_procedimientos',  'N° Procedimientos', '#FF5722', ejes[1, 1]),
    ]

    for df, etiqueta, col, xlabel, color, ax in pares:
        valores = df[col].dropna()

        # Histograma normalizado (densidad)
        ax.hist(valores, bins=50, density=True, alpha=0.6, color=color)

        # Curva KDE (estimación de densidad de kernel)
        valores.plot.kde(
            ax=ax,
            color='navy' if 'Neo' in etiqueta else 'darkred',
            linewidth=2,
        )

        # Línea vertical en la mediana (más robusta que la media para datos sesgados)
        ax.axvline(
            valores.median(), color='green', linestyle='--', linewidth=1.2,
            label=f'Mediana = {valores.median():.1f}',
        )

        ax.set_title(f'{etiqueta}: {col}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Densidad')
        ax.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '01_distribuciones.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 2: Gráficos Q-Q para evaluación de normalidad
# =============================================================================

def graficar_qqplots(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Compara los cuantiles de los datos con los de una distribución normal.

    Si los puntos se alejan de la diagonal, los datos NO son normales.
    Este resultado refuerza la elección de Kruskal-Wallis para H1.
    """
    fig, ejes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Gráficos Q-Q — Evaluación de Normalidad', fontsize=14, fontweight='bold')

    pares = [
        (df_neo, 'Neoplasia', 'dias_estadia',     ejes[0, 0]),
        (df_sep, 'Sepsis',    'dias_estadia',     ejes[0, 1]),
        (df_neo, 'Neoplasia', 'n_procedimientos', ejes[1, 0]),
        (df_sep, 'Sepsis',    'n_procedimientos', ejes[1, 1]),
    ]

    for df, etiqueta, col, ax in pares:
        # Submuestra con semilla fija para garantizar reproducibilidad
        valores = df[col].dropna().sample(min(5000, len(df)), random_state=SEMILLA)
        stats.probplot(valores, dist='norm', plot=ax)
        ax.set_title(f'{etiqueta}: {col}')
        ax.get_lines()[0].set(markersize=2, alpha=0.4)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '02_qqplots.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 3: Boxplots por severidad GRD
# =============================================================================

def graficar_boxplots_severidad(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Visualiza cómo varía dias_estadia y n_procedimientos según severidad GRD.

    Severidad: 1=Menor  2=Moderada  3=Severa  4=Extrema
    Se espera relación positiva: mayor severidad → más días y más procedimientos.
    """
    fig, ejes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribución por Nivel de Severidad GRD', fontsize=14, fontweight='bold')

    paleta = {1: '#4CAF50', 2: '#FFC107', 3: '#FF5722', 4: '#9C27B0'}

    pares = [
        (df_neo, 'Neoplasia', 'dias_estadia',     'Días de Estadía',   ejes[0, 0]),
        (df_sep, 'Sepsis',    'dias_estadia',     'Días de Estadía',   ejes[0, 1]),
        (df_neo, 'Neoplasia', 'n_procedimientos', 'N° Procedimientos', ejes[1, 0]),
        (df_sep, 'Sepsis',    'n_procedimientos', 'N° Procedimientos', ejes[1, 1]),
    ]

    for df, etiqueta, col, ylabel, ax in pares:
        tmp = df[[COL_SEVERIDAD, col]].dropna().copy()
        tmp[COL_SEVERIDAD] = tmp[COL_SEVERIDAD].astype(int)
        orden = sorted(tmp[COL_SEVERIDAD].unique())
        sns.boxplot(
            data=tmp, x=COL_SEVERIDAD, y=col, order=orden,
            palette=paleta, ax=ax, showfliers=False,
        )
        ax.set_title(f'{etiqueta}: {col} por Severidad')
        ax.set_xlabel('Nivel de Severidad GRD')
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '03_boxplot_severidad.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 4: Violinplots de variabilidad inter-hospital
# =============================================================================

def graficar_violin_hospitales(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Violinplot de dias_estadia para los 15 hospitales con mayor volumen.

    La variabilidad observada entre hospitales motiva visualmente H1:
    si todos trataran igual a sus pacientes, los violines serían similares.
    """
    fig, ejes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        'Variabilidad Inter-Hospital en Días de Estadía (Top 15 Hospitales)',
        fontsize=13, fontweight='bold',
    )

    for ax, df, etiqueta in [(ejes[0], df_neo, 'Neoplasia'), (ejes[1], df_sep, 'Sepsis')]:
        top15 = df[COL_HOSPITAL].value_counts().head(15).index.tolist()
        tmp = df[df[COL_HOSPITAL].isin(top15)][[COL_HOSPITAL, 'dias_estadia']].dropna()

        # Ordenar hospitales por mediana descendente
        orden = (
            tmp.groupby(COL_HOSPITAL)['dias_estadia']
            .median().sort_values(ascending=False).index.tolist()
        )
        sns.violinplot(
            data=tmp, x=COL_HOSPITAL, y='dias_estadia', order=orden,
            palette='tab20', ax=ax, inner='quartile', linewidth=0.8,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{etiqueta}: Variabilidad por Hospital')
        ax.set_xlabel('Código Hospital')
        ax.set_ylabel('Días de Estadía')

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '04_violin_hospitales.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 5: Barplot con IC95% de procedimientos por hospital
# =============================================================================

def graficar_barras_procedimientos(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Media ± IC95% de n_procedimientos para los top 15 hospitales."""
    fig, ejes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Media ± IC95% de N° Procedimientos por Hospital (Top 15)', fontsize=13, fontweight='bold')

    for ax, df, etiqueta in [(ejes[0], df_neo, 'Neoplasia'), (ejes[1], df_sep, 'Sepsis')]:
        top15 = df[COL_HOSPITAL].value_counts().head(15).index.tolist()
        tmp = df[df[COL_HOSPITAL].isin(top15)]
        estadisticas = (
            tmp.groupby(COL_HOSPITAL)['n_procedimientos']
            .agg(['mean', 'sem', 'count']).reset_index()
        )
        estadisticas['ic95'] = 1.96 * estadisticas['sem']
        estadisticas = estadisticas.sort_values('mean', ascending=False)

        ax.bar(
            range(len(estadisticas)), estadisticas['mean'],
            yerr=estadisticas['ic95'], capsize=4,
            color='#2196F3' if 'Neo' in etiqueta else '#FF5722',
            alpha=0.8, error_kw={'elinewidth': 1.2},
        )
        ax.set_xticks(range(len(estadisticas)))
        ax.set_xticklabels(estadisticas[COL_HOSPITAL].tolist(), rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{etiqueta}: Promedio N° Procedimientos')
        ax.set_xlabel('Código Hospital')
        ax.set_ylabel('Media N° Procedimientos ± IC95%')

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '05_barras_procedimientos.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# FIGURA 6: Scatter Peso GRD vs. Días de Estadía
# =============================================================================

def graficar_scatter_peso_estadia(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Dispersión entre peso relativo GRD y días de estadía (top 5 hospitales)."""
    fig, ejes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Peso GRD vs. Días de Estadía por Hospital (Top 5)', fontsize=13, fontweight='bold')

    for ax, df, etiqueta in [(ejes[0], df_neo, 'Neoplasia'), (ejes[1], df_sep, 'Sepsis')]:
        top5 = df[COL_HOSPITAL].value_counts().head(5).index.tolist()
        tmp = df[df[COL_HOSPITAL].isin(top5)][[COL_HOSPITAL, COL_PESO, 'dias_estadia']].dropna()
        tmp[COL_PESO] = pd.to_numeric(tmp[COL_PESO], errors='coerce')
        tmp = tmp.dropna()
        paleta = dict(zip(top5, sns.color_palette('tab10', 5)))

        for hosp in top5:
            sub = tmp[tmp[COL_HOSPITAL] == hosp]
            ax.scatter(sub[COL_PESO], sub['dias_estadia'],
                       label=hosp, alpha=0.3, s=10, color=paleta[hosp])

        ax.set_title(f'{etiqueta}')
        ax.set_xlabel('Peso Relativo GRD')
        ax.set_ylabel('Días de Estadía')
        ax.legend(title='Hospital', fontsize=8, markerscale=2)

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, '06_scatter_peso_estadia.png')
    fig.savefig(ruta, dpi=DPI_FIGURA, bbox_inches='tight')
    plt.close(fig)
    logger.info("Guardado: %s", ruta)


# =============================================================================
# TABLA DE COMPLETITUD
# =============================================================================

def guardar_completitud(df_neo: pd.DataFrame, df_sep: pd.DataFrame) -> None:
    """Calcula y guarda la tabla de completitud de variables clave."""
    cols_clave = [
        COL_HOSPITAL, 'edad', 'dias_estadia', 'n_procedimientos', 'n_proc_unicos',
        'mortalidad', COL_SEVERIDAD, COL_PESO,
    ]
    comp_neo = tabla_completitud(df_neo[[c for c in cols_clave if c in df_neo.columns]], 'neoplasia')
    comp_sep = tabla_completitud(df_sep[[c for c in cols_clave if c in df_sep.columns]], 'sepsis')
    combinada = pd.concat([comp_neo, comp_sep], ignore_index=True)
    ruta = os.path.join(DIR_TABLAS, 'tabla_completitud.csv')
    combinada.to_csv(ruta, index=False)
    logger.info("Tabla de completitud guardada: %s", ruta)
    print("\n=== TABLA DE COMPLETITUD ===")
    print(combinada.to_string(index=False))


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 01_eda_profundo.py — INICIO ===")

    datos_crudos = cargar_datos_grd(columnas=COLUMNAS_NECESARIAS)
    datos_crudos = derivar_variables(datos_crudos)
    df_neo, df_sep = filtrar_grupos_diagnosticos(datos_crudos)
    liberar_memoria(datos_crudos)

    df_neo = limpiar_datos(df_neo, 'neoplasia')
    df_sep = limpiar_datos(df_sep, 'sepsis')

    logger.info("Registros finales — Neoplasia: {:,} | Sepsis: {:,}".format(len(df_neo), len(df_sep)))

    graficar_distribuciones(df_neo, df_sep)
    graficar_qqplots(df_neo, df_sep)
    graficar_boxplots_severidad(df_neo, df_sep)
    graficar_violin_hospitales(df_neo, df_sep)
    graficar_barras_procedimientos(df_neo, df_sep)
    graficar_scatter_peso_estadia(df_neo, df_sep)
    guardar_completitud(df_neo, df_sep)

    logger.info("=== 01_eda_profundo.py — COMPLETO ===")


if __name__ == '__main__':
    main()
