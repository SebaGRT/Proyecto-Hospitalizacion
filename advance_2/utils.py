# =============================================================================
# utils.py — Funciones reutilizables para el Avance 2
# =============================================================================
# Contiene toda la lógica compartida entre scripts:
#   - Carga de archivos GRD
#   - Derivación de variables analíticas
#   - Filtrado por grupo diagnóstico
#   - Limpieza y validación de datos
#   - Tabla de completitud
#   - Liberación de memoria
# =============================================================================

import gc
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Importamos parámetros centralizados desde config.py
from config import (
    ALPHA,
    COL_INGRESO, COL_NACIMIENTO, COL_DIAGNOSTICO, COL_ALTA,
    COL_HOSPITAL, COL_SEVERIDAD, COL_TIPOALTA, COL_PESO,
    DIR_DATOS, ARCHIVOS_GRD, MIN_CASOS, VALOR_FALLECIDO,
    CODIGOS_NEOPLASIA, P99_CUTOFF, COLS_PROCEDIMIENTO, SEMILLA, CODIGOS_SEPSIS,
)

# Logger para registrar decisiones importantes durante la ejecución
logger = logging.getLogger(__name__)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_datos_grd(
    años: Optional[list] = None,
    columnas: Optional[list] = None,
    nfilas: Optional[int] = None,
) -> pd.DataFrame:
    """Carga los archivos CSV del GRD Público y los une en un solo DataFrame.

    Se leen con dtype=str para evitar problemas de inferencia de tipos
    y minimizar el uso de memoria en la carga inicial.

    Parámetros
    ----------
    años : list de int, opcional
        Subconjunto de años a cargar (ej. [2021, 2022]). None carga todos.
    columnas : list de str, opcional
        Columnas a leer. None lee todas.
    nfilas : int, opcional
        Máximo de filas por archivo (útil para pruebas rápidas).

    Retorna
    -------
    pd.DataFrame
        Datos crudos concatenados con dtype=str en todas las columnas.
    """
    archivos = ARCHIVOS_GRD

    # Filtrar por años si se especifican
    if años:
        archivos = [f for f in archivos if any(str(a) in f for a in años)]

    frames = []
    for nombre in archivos:
        ruta = os.path.join(DIR_DATOS, nombre)

        # Verificar existencia antes de intentar leer
        if not os.path.exists(ruta):
            logger.warning("Archivo no encontrado: %s — omitiendo", ruta)
            continue

        logger.info("Cargando %s …", nombre)
        df = pd.read_csv(
            ruta,
            sep='|',          # Separador pipe del GRD
            dtype=str,        # Todo como string para no perder ceros iniciales
            usecols=columnas,
            nrows=nfilas,
            low_memory=False,
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No se encontraron archivos GRD en: %s" % DIR_DATOS)

    resultado = pd.concat(frames, ignore_index=True)
    logger.info("Total de filas cargadas: {:,}".format(len(resultado)))
    return resultado


# Alias en inglés para compatibilidad con el notebook
load_grd_data = cargar_datos_grd


# =============================================================================
# DERIVACIÓN DE VARIABLES
# =============================================================================

def derivar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las variables analíticas a partir de los datos crudos del GRD.

    Variables derivadas:
    - edad          : años cumplidos aproximados al momento del ingreso
    - dias_estadia  : duración de la hospitalización en días
    - n_procedimientos : cantidad de procedimientos CIE-9 registrados (no nulos)
    - n_proc_unicos : cantidad de procedimientos únicos (diversidad de intervención)
    - mortalidad    : 1 si TIPOALTA == 'FALLECIDO', 0 en caso contrario

    Parámetros
    ----------
    df : pd.DataFrame
        Datos crudos del GRD con las columnas esperadas.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas añadidas.
    """
    df = df.copy()

    # --- Conversión de fechas ---
    # Necesarias para calcular edad y días de estadía
    df[COL_INGRESO]    = pd.to_datetime(df[COL_INGRESO],    errors='coerce')
    df[COL_ALTA]       = pd.to_datetime(df[COL_ALTA],       errors='coerce')
    df[COL_NACIMIENTO] = pd.to_datetime(df[COL_NACIMIENTO], errors='coerce')

    # --- Edad aproximada (años) ---
    # Diferencia de años entre fecha de ingreso y fecha de nacimiento
    # Es una aproximación: no considera si ya pasó el cumpleaños en el año de ingreso
    df['edad'] = df[COL_INGRESO].dt.year - df[COL_NACIMIENTO].dt.year

    # --- Días de estadía ---
    # Alta - Ingreso en días naturales
    df['dias_estadia'] = (df[COL_ALTA] - df[COL_INGRESO]).dt.days

    # --- Número de procedimientos registrados ---
    # Solo se cuentan las columnas PROCEDIMIENTO1 a PROCEDIMIENTO30 que estén presentes
    cols_proc_presentes = [c for c in COLS_PROCEDIMIENTO if c in df.columns]
    df_proc = df[cols_proc_presentes].replace('', np.nan)  # Celdas vacías → NaN
    df['n_procedimientos'] = df_proc.notna().sum(axis=1)

    # --- Número de procedimientos únicos ---
    # Diversidad de tipos de procedimiento (evita contar duplicados)
    df['n_proc_unicos'] = df_proc.apply(
        lambda fila: fila.dropna().nunique(), axis=1
    )

    # --- Mortalidad (variable binaria 0/1) ---
    # 1 = el paciente falleció durante la hospitalización
    df['mortalidad'] = (
        df[COL_TIPOALTA].str.strip() == VALOR_FALLECIDO
    ).astype(int)

    logger.info("Variables derivadas: edad, dias_estadia, n_procedimientos, n_proc_unicos, mortalidad")
    return df


# Alias en inglés para compatibilidad con el notebook
derive_variables = derivar_variables


# =============================================================================
# FILTRADO POR GRUPO DIAGNÓSTICO
# =============================================================================

def filtrar_grupos_diagnosticos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el DataFrame en dos grupos según el diagnóstico principal (DIAGNOSTICO1).

    Criterio de selección:
    - Neoplasias: DIAGNOSTICO1 comienza con C50, C18, C19, C20, C53 o C34
    - Sepsis: DIAGNOSTICO1 comienza con A40 o A41

    Se agrega la columna 'grupo_diagnostico' para identificar el grupo.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna DIAGNOSTICO1 presente.

    Retorna
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_neoplasia, df_sepsis)
    """
    # Normalizar: quitar espacios y convertir a mayúsculas
    diagnostico = df[COL_DIAGNOSTICO].str.strip().str.upper()

    # Crear máscaras booleanas para cada grupo
    mascara_neo = diagnostico.str.startswith(tuple(CODIGOS_NEOPLASIA))
    mascara_sep = diagnostico.str.startswith(tuple(CODIGOS_SEPSIS))

    df_neo = df[mascara_neo].copy()
    df_neo['grupo_diagnostico'] = 'neoplasia'

    df_sep = df[mascara_sep].copy()
    df_sep['grupo_diagnostico'] = 'sepsis'

    logger.info(
        "Separación diagnóstica — Neoplasia: {:,} registros | Sepsis: {:,} registros".format(
            len(df_neo), len(df_sep)
        )
    )
    return df_neo, df_sep


# Alias en inglés para compatibilidad con el notebook
filter_diagnostic_groups = filtrar_grupos_diagnosticos


# =============================================================================
# LIMPIEZA Y VALIDACIÓN
# =============================================================================

def limpiar_datos(df: pd.DataFrame, etiqueta: str = '') -> pd.DataFrame:
    """Aplica los filtros de calidad documentados para el Avance 2.

    Pasos de limpieza (con registro del impacto en número de filas):
    1. Eliminar registros sin COD_HOSPITAL (no asignables a ningún hospital)
    2. Eliminar registros sin IR_29301_SEVERIDAD (covariate clave en los modelos)
    3. Eliminar registros con dias_estadia < 0 (error de datos: alta antes de ingreso)
    4. Eliminar outliers: dias_estadia > P99 (valores extremos distorsionan modelos)
    5. Convertir severidad y peso a numérico; descartar 'DESCONOCIDO' tras coerción
    6. Descartar hospitales con < MIN_CASOS registros (poder estadístico insuficiente)

    Parámetros
    ----------
    df : pd.DataFrame
    etiqueta : str
        Nombre del grupo (ej. 'neoplasia') para el registro de logs.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio y listo para análisis.
    """
    n0 = len(df)
    lbl = etiqueta or 'datos'

    # PASO 1: Eliminar registros sin hospital
    # Razón: el hospital es la variable de agrupación principal (H1) y el
    # efecto fijo en los modelos H2 y H3. Sin él, el registro es inutilizable.
    df = df[df[COL_HOSPITAL].notna() & (df[COL_HOSPITAL].str.strip() != '')]
    n1 = len(df)
    logger.info("[%s] Paso 1 — Sin hospital: {:,} filas (eliminadas: {:,})".format(n1, n0 - n1) % lbl)

    # PASO 2: Eliminar registros sin severidad GRD
    # Razón: IR_29301_SEVERIDAD es covariable de control en todos los modelos.
    # Sin ella no se puede controlar por complejidad del caso.
    df = df[df[COL_SEVERIDAD].notna() & (df[COL_SEVERIDAD].str.strip() != '')]
    n2 = len(df)
    logger.info("[%s] Paso 2 — Sin severidad: {:,} filas (eliminadas: {:,})".format(n2, n1 - n2) % lbl)

    # PASO 3: Eliminar dias_estadia negativos
    # Razón: alta antes del ingreso es un error de codificación, no un caso real.
    df = df[df['dias_estadia'] >= 0]
    n3 = len(df)
    logger.info("[%s] Paso 3 — dias_estadia < 0: {:,} filas (eliminadas: {:,})".format(n3, n2 - n3) % lbl)

    # PASO 4: Eliminar outliers de estadía (percentil 99)
    # Razón: estadías muy prolongadas (p.ej. meses en UCI) corresponden a casos
    # clínicamente distintos de una hospitalización típica y distorsionan los
    # coeficientes de regresión. Se usa P99 por grupo para respetar diferencias
    # entre neoplasias (electivas, estadías más largas) y sepsis (urgencias).
    p99 = np.percentile(df['dias_estadia'].dropna(), P99_CUTOFF)
    df = df[df['dias_estadia'] <= p99]
    n4 = len(df)
    logger.info(
        "[%s] Paso 4 — dias_estadia > p99 (%.1f días): {:,} filas (eliminadas: {:,})".format(
            n4, n3 - n4
        ) % (lbl, p99)
    )

    # PASO 5: Convertir columnas numéricas y eliminar valores no numéricos
    # 'DESCONOCIDO' en severidad se convierte a NaN y se descarta
    df[COL_SEVERIDAD] = pd.to_numeric(df[COL_SEVERIDAD], errors='coerce')
    df[COL_PESO]      = pd.to_numeric(df[COL_PESO],      errors='coerce')
    df['edad']        = pd.to_numeric(df['edad'],         errors='coerce')
    df = df[df[COL_SEVERIDAD].notna()]
    n5 = len(df)
    logger.info("[%s] Paso 5 — Severidad no numérica: {:,} filas (eliminadas: {:,})".format(n5, n4 - n5) % lbl)

    # PASO 6: Mantener solo hospitales con suficientes casos
    # Razón: los efectos fijos por hospital requieren varianza intra-hospital
    # adecuada. Hospitales con muy pocos casos producen estimaciones inestables
    # y aumentan el error tipo I en las comparaciones post-hoc.
    conteos = df.groupby(COL_HOSPITAL)['dias_estadia'].count()
    hospitales_validos = conteos[conteos >= MIN_CASOS].index
    n_antes = len(df)
    df = df[df[COL_HOSPITAL].isin(hospitales_validos)]
    logger.info(
        "[%s] Paso 6 — Mínimo %d casos: {:,} filas (eliminadas: {:,}), hospitales válidos: %d".format(
            len(df), n_antes - len(df)
        ) % (lbl, MIN_CASOS, len(hospitales_validos))
    )

    # Optimización de memoria: convertir variables categóricas repetidas
    df[COL_HOSPITAL]  = df[COL_HOSPITAL].astype('category')
    df[COL_SEVERIDAD] = df[COL_SEVERIDAD].astype('category')

    return df.reset_index(drop=True)


# Alias en inglés para compatibilidad con el notebook
clean_data = limpiar_datos


# =============================================================================
# TABLA DE COMPLETITUD
# =============================================================================

def tabla_completitud(df: pd.DataFrame, etiqueta: str = '') -> pd.DataFrame:
    """Calcula el porcentaje de completitud para cada columna del DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
    etiqueta : str
        Nombre del grupo diagnóstico para identificar la tabla.

    Retorna
    -------
    pd.DataFrame
        Tabla con columnas: variable, grupo, n_total, n_faltantes, pct_completo.
    """
    n = len(df)
    filas = []
    for col in df.columns:
        # Contar tanto NaN como strings vacíos como "faltantes"
        faltantes = df[col].isna().sum() + (df[col].astype(str).str.strip() == '').sum()
        faltantes = min(int(faltantes), n)  # No puede superar el total
        filas.append({
            'variable':    col,
            'grupo':       etiqueta,
            'n_total':     n,
            'n_faltantes': faltantes,
            'pct_completo': round(100 * (n - faltantes) / n, 1),
        })
    return pd.DataFrame(filas)


# Alias en inglés para compatibilidad con el notebook
completeness_table = tabla_completitud


# =============================================================================
# GESTIÓN DE MEMORIA
# =============================================================================

def liberar_memoria(*dfs) -> None:
    """Elimina DataFrames y fuerza la recolección de basura.

    Útil después de operaciones que generan grandes DataFrames intermedios.

    Parámetros
    ----------
    *dfs : DataFrames a eliminar.
    """
    for df in dfs:
        del df
    gc.collect()


# Alias en inglés para compatibilidad con el notebook
free_memory = liberar_memoria
