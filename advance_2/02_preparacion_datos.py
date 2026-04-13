# =============================================================================
# 02_preparacion_datos.py — Limpieza y Preparación de Datos
# =============================================================================
# Documenta y aplica todas las decisiones de limpieza del Avance 2:
#
#   1. Carga datos crudos GRD (todos los años disponibles)
#   2. Deriva variables analíticas (edad, dias_estadia, n_procedimientos, mortalidad)
#   3. Filtra grupos diagnósticos (neoplasias / sepsis)
#   4. Aplica reglas de calidad de datos (documentadas con justificación)
#   5. Imprime resumen de impacto de cada filtro
#
# Este script puede ejecutarse de forma independiente para documentar
# el pipeline de preparación de datos ante revisores o docentes.
# =============================================================================

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COL_HOSPITAL, COL_SEVERIDAD, COL_PESO,
    MIN_CASOS, P99_CUTOFF, SEMILLA, DIR_TABLAS,
)
from utils import (
    limpiar_datos, tabla_completitud, derivar_variables,
    filtrar_grupos_diagnosticos, liberar_memoria, cargar_datos_grd,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Solo las columnas necesarias para este análisis
COLUMNAS_NECESARIAS = [
    COL_HOSPITAL, 'FECHA_NACIMIENTO', 'FECHA_INGRESO', 'FECHAALTA',
    'TIPOALTA', 'DIAGNOSTICO1', 'IR_29301_SEVERIDAD', 'IR_29301_PESO',
] + [f'PROCEDIMIENTO{i}' for i in range(1, 31)]


# =============================================================================
# DOCUMENTACIÓN DE DECISIONES DE LIMPIEZA
# =============================================================================

def documentar_decisiones() -> None:
    """Imprime en consola las decisiones de limpieza y su justificación."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           DOCUMENTACIÓN DE DECISIONES DE PREPARACIÓN DE DATOS          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PASO 1 — Eliminar registros sin COD_HOSPITAL                          ║
║    Razón: El hospital es la variable de agrupación principal (H1) y    ║
║           el efecto fijo en los modelos H2 y H3. Sin código de         ║
║           hospital, el registro no puede ser atribuido a ningún        ║
║           establecimiento.                                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PASO 2 — Eliminar registros sin IR_29301_SEVERIDAD                    ║
║    Razón: La severidad GRD es la covariable de control más importante  ║
║           en los modelos. Sin ella, no es posible controlar por la     ║
║           complejidad del caso, lo que introduciría confusión severa.  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PASO 3 — Eliminar registros con dias_estadia < 0                      ║
║    Razón: Una fecha de alta anterior al ingreso es un error de         ║
║           codificación. Los valores negativos son físicamente          ║
║           imposibles y distorsionan los modelos de regresión.          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PASO 4 — Eliminar outliers: dias_estadia > Percentil 99              ║
║    Razón: Estadías extremadamente largas (p.ej. meses en UCI)          ║
║           corresponden a casos clínicamente distintos de una           ║
║           hospitalización típica. Se usa el P99 por grupo para         ║
║           preservar las diferencias naturales entre neoplasias         ║
║           (más largas) y sepsis (más cortas pero más agudas).          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PASO 5 — Eliminar severidad 'DESCONOCIDO' (→ NaN tras coerción)      ║
║    Razón: Los valores no numéricos no pueden codificarse como          ║
║           covariable ordinal en los modelos.                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PASO 6 — Hospitales con < 30 casos únicos                             ║
║    Razón: Los efectos fijos por hospital requieren varianza            ║
║           intra-grupo adecuada. Hospitales con muy pocos casos         ║
║           producen estimaciones inestables (errores estándar           ║
║           inflados) y aumentan el error Tipo I en el post-hoc.        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# PIPELINE DE PREPARACIÓN
# =============================================================================

def preparar_datos() -> tuple:
    """Ejecuta el pipeline completo de preparación de datos.

    Retorna
    -------
    tuple : (df_neoplasia, df_sepsis)
        Ambos DataFrames limpios y listos para análisis.
    """
    documentar_decisiones()

    # --- Carga ---
    logger.info("Cargando datos GRD...")
    datos_crudos = cargar_datos_grd(columnas=COLUMNAS_NECESARIAS)
    logger.info("Filas cargadas: {:,}".format(len(datos_crudos)))

    # --- Derivación de variables ---
    datos_crudos = derivar_variables(datos_crudos)

    # --- Separación por grupo diagnóstico ---
    df_neo, df_sep = filtrar_grupos_diagnosticos(datos_crudos)
    n_neo_crudo = len(df_neo)
    n_sep_crudo = len(df_sep)
    liberar_memoria(datos_crudos)  # Liberar el DataFrame completo de ~5.8M filas

    # --- Limpieza por grupo ---
    df_neo = limpiar_datos(df_neo, 'neoplasia')
    df_sep = limpiar_datos(df_sep, 'sepsis')

    # --- Resumen de impacto ---
    print("\n=== RESUMEN DE PREPARACIÓN ===")
    print(f"{'Grupo':<12} {'Registros crudos':>18} {'Tras limpieza':>15} {'% Retenido':>12}")
    print("-" * 60)
    print(f"{'Neoplasia':<12} {n_neo_crudo:>18,} {len(df_neo):>15,} {100*len(df_neo)/n_neo_crudo:>11.1f}%")
    print(f"{'Sepsis':<12} {n_sep_crudo:>18,} {len(df_sep):>15,} {100*len(df_sep)/n_sep_crudo:>11.1f}%")

    # Distribución de códigos diagnósticos finales
    print("\nTop diagnósticos — Neoplasia:")
    print(df_neo['DIAGNOSTICO1'].value_counts().head(8).to_string())
    print("\nTop diagnósticos — Sepsis:")
    print(df_sep['DIAGNOSTICO1'].value_counts().head(5).to_string())

    # Guardar tabla de completitud post-limpieza
    os.makedirs(DIR_TABLAS, exist_ok=True)
    cols_clave = [COL_HOSPITAL, 'edad', 'dias_estadia', 'n_procedimientos',
                  'n_proc_unicos', 'mortalidad', COL_SEVERIDAD, COL_PESO]
    comp = pd.concat([
        tabla_completitud(df_neo[[c for c in cols_clave if c in df_neo.columns]], 'neoplasia'),
        tabla_completitud(df_sep[[c for c in cols_clave if c in df_sep.columns]], 'sepsis'),
    ])
    comp.to_csv(os.path.join(DIR_TABLAS, 'completitud_post_limpieza.csv'), index=False)
    logger.info("Tabla de completitud guardada.")

    return df_neo, df_sep


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    logger.info("=== 02_preparacion_datos.py — INICIO ===")
    df_neo, df_sep = preparar_datos()
    logger.info("Neoplasia: {:,} filas | Sepsis: {:,} filas".format(len(df_neo), len(df_sep)))
    logger.info("=== 02_preparacion_datos.py — COMPLETO ===")


if __name__ == '__main__':
    main()
