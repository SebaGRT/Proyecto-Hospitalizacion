# =============================================================================
# config.py — Configuración centralizada para el Avance 2
# =============================================================================
# Todos los parámetros estadísticos, rutas y semillas se definen aquí.
# Importar este módulo en cualquier script garantiza consistencia total.
# =============================================================================

import os

# ── Parámetros estadísticos ───────────────────────────────────────────────────

ALPHA = 0.05      # Nivel de significancia para todas las pruebas (estándar biomédico)
P99_CUTOFF = 99   # Percentil para eliminar outliers en días_estadia
MIN_CASOS = 30    # Mínimo de pacientes únicos por hospital para incluirlo en modelos
SEMILLA = 42      # Semilla fija para reproducibilidad (muestras, RNG, etc.)
SHAPIRO_N = 5000  # Tamaño de submuestra para Shapiro-Wilk (máximo válido para el test)

# ── Grupos diagnósticos (códigos CIE-10) ─────────────────────────────────────
# Neoplasias: cánceres de mama, colon, recto, cuello de útero y bronquios/pulmón
# Sepsis: septicemia estreptocócica y otras septicemias

CODIGOS_NEOPLASIA = ['C50', 'C18', 'C19', 'C20', 'C53', 'C34']
CODIGOS_SEPSIS    = ['A40', 'A41']

# ── Nombres de columnas en los datos crudos ───────────────────────────────────
# Se centralizan para evitar errores de tipeo en múltiples scripts

COL_HOSPITAL    = 'COD_HOSPITAL'          # Código del establecimiento
COL_NACIMIENTO  = 'FECHA_NACIMIENTO'      # Para calcular edad
COL_INGRESO     = 'FECHA_INGRESO'         # Fecha de inicio de hospitalización
COL_ALTA        = 'FECHAALTA'             # Fecha de egreso
COL_TIPOALTA    = 'TIPOALTA'              # Incluye 'FALLECIDO' para mortalidad
COL_DIAGNOSTICO = 'DIAGNOSTICO1'          # Diagnóstico principal CIE-10
COL_SEVERIDAD   = 'IR_29301_SEVERIDAD'    # Severidad GRD (1=leve, 4=extremo)
COL_PESO        = 'IR_29301_PESO'         # Peso relativo GRD (proxy de complejidad)

# Lista de las 30 columnas de procedimientos CIE-9 disponibles en el dataset
COLS_PROCEDIMIENTO = [f'PROCEDIMIENTO{i}' for i in range(1, 31)]

# Valor exacto en TIPOALTA que indica fallecimiento del paciente
VALOR_FALLECIDO = 'FALLECIDO'

# ── Rutas de archivos ─────────────────────────────────────────────────────────
# BASE_DIR apunta a advance_2/; los datos están un nivel arriba en DATASET-PROBLEMA8/

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))  # .../advance_2/
DIR_DATOS     = os.path.join(BASE_DIR, '..', 'DATASET-PROBLEMA8')
DIR_SALIDAS   = os.path.join(BASE_DIR, 'outputs')
DIR_TABLAS    = os.path.join(DIR_SALIDAS, 'tablas')
DIR_GRAFICOS  = os.path.join(DIR_SALIDAS, 'graficos')
DIR_MODELOS   = os.path.join(DIR_SALIDAS, 'modelos')

# Archivos CSV fuente (un archivo por año)
ARCHIVOS_GRD = [
    'GRD_PUBLICO_2019.csv',
    'GRD_PUBLICO_2020.csv',
    'GRD_PUBLICO_2021.csv',
    'GRD_PUBLICO_EXTERNO_2022.csv',
    'GRD_PUBLICO_2023.csv',
    'GRD_PUBLICO_2024.csv',
]

# ── Configuración de gráficos ─────────────────────────────────────────────────

DPI_FIGURA   = 300                        # Alta resolución para entrega
ESTILO_FIGURA = 'seaborn-v0_8-whitegrid'  # Fondo limpio, adecuado para publicación


# ── Función auxiliar de significancia estadística ─────────────────────────────

def sig_etiqueta(p: float) -> str:
    """Convierte un p-valor en estrellas de significancia.

    Parámetros
    ----------
    p : float
        P-valor obtenido de una prueba estadística.

    Retorna
    -------
    str
        '***' si p < 0.001, '**' si p < 0.01, '*' si p < 0.05,
        '.' si p < 0.10, 'ns' si no es significativo.
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.10:
        return '.'
    return 'ns'


# Alias en inglés para compatibilidad con el notebook
sig_label = sig_etiqueta
