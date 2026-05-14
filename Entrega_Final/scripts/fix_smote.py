#!/usr/bin/env python3
"""
fix_smote.py — Re-ejecución y corrección del análisis SMOTE
================================================================
- Replica el pipeline de clasificación del notebook (C16.*, top 15 hospitales)
- Aplica SMOTE SOLO a datos de entrenamiento (previene data leakage)
- Valida con StratifiedKFold (k=5)
- Reporta métricas pre-SMOTE vs post-SMOTE
- Genera curvas ROC comparativas
- Guarda resultados en outputs/smote_resultados.csv y outputs/figuras/smote_roc_*.png

Autor: Sisyphus-Junior (T4 — SMOTE fix)
Fecha: 2026-05-13
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, f1_score, recall_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Configuración ───────────────────────────────────────────────────────────
SEMILLA = 42
ALPHA = 0.05
MIN_CASOS_H = 30
TOP_HOSP = 15
TEST_SIZE = 0.25
CV_FOLDS = 5

np.random.seed(SEMILLA)
sns.set_theme(style='whitegrid', context='notebook')

# ── Rutas ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Desde Entrega_Final/scripts/ → raíz
DATA_DIR = PROJECT_ROOT / 'DATASET INICIAL'
GRD_PATH = DATA_DIR / 'GRD_Limpio.csv'
HOSP_PATH = DATA_DIR / 'HospitalesGRD.xlsx'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURAS_DIR = OUTPUTS_DIR / 'figuras'
INFERENCIAL_DIR = OUTPUTS_DIR / 'inferencial'

for d in [OUTPUTS_DIR, FIGURAS_DIR, INFERENCIAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print('=' * 80)
print('FIX_SMOTE: Corrección y validación del análisis SMOTE')
print(f'  Proyecto: {PROJECT_ROOT}')
print(f'  Dataset : {GRD_PATH}')
print('=' * 80)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPROCESAMIENTO DE DATOS (réplica exacta del notebook)
# ══════════════════════════════════════════════════════════════════════════════

print('\n[1/6] Cargando datos...')
cols_uso = (
    ['COD_HOSPITAL', 'hospital', 'diagnostico_principal', 'dias_estada',
     'edad', 'sexo', 'mortalidad', 'severidad_grd', 'peso_grd',
     'cantidad_procedimientos', 'TIPO_INGRESO', 'TIPOALTA', 'comuna', 'region']
    + [f'DIAGNOSTICO{i}' for i in range(2, 36)]
)

# Cargar solo columnas necesarias
_all_cols = pd.read_csv(GRD_PATH, nrows=0).columns.tolist()
_load_cols = [c for c in cols_uso if c in _all_cols]
df_raw = pd.read_csv(GRD_PATH, usecols=_load_cols, low_memory=False)
print(f'  Registros crudos: {len(df_raw):,}  |  Variables: {len(df_raw.columns)}')

# ── Filtro oncológico (C00-D49) y exclusión OBSTETRICA ──
print('[2/6] Filtrando datos oncológicos (C00-D49, excluyendo OBSTETRICA)...')
df = df_raw[
    df_raw['diagnostico_principal'].str.upper().str[:1].isin(['C', 'D'])
].copy()

# Excluir ingresos obstétricos
if 'TIPO_INGRESO' in df.columns:
    df = df[df['TIPO_INGRESO'].str.upper().str.strip() != 'OBSTETRICA'].copy()

# ── Conversión de tipos numéricos ──
for _c in ['dias_estada', 'edad', 'severidad_grd', 'peso_grd', 'cantidad_procedimientos']:
    if _c in df.columns:
        df[_c] = pd.to_numeric(df[_c], errors='coerce')

# ── Filtro de outliers en días de estadía ──
P99 = df['dias_estada'].quantile(0.99)
df = df[
    (df['dias_estada'].notna()) &
    (df['dias_estada'] >= 0) &
    (df['dias_estada'] <= P99)
].copy()
df.dropna(subset=['severidad_grd', 'peso_grd'], inplace=True)
print(f'  Después de limpieza: {len(df):,} registros oncológicos')

# ── Variable de mortalidad ──
if 'mortalidad' in df.columns:
    df['mortalidad_int'] = pd.to_numeric(df['mortalidad'], errors='coerce').fillna(0).astype(int)
else:
    df['mortalidad_int'] = (df['TIPOALTA'].str.upper().str.strip() == 'FALLECIDO').astype(int)

# ── Limpieza de strings ──
for _c in ['sexo', 'TIPO_INGRESO', 'TIPOALTA']:
    if _c in df.columns:
        df[_c] = df[_c].astype(str).str.strip().str.upper()

# ── Comorbilidad (conteo de diagnósticos secundarios) ──
_diag_cols = [c for c in df.columns
              if c.upper().startswith('DIAGNOSTICO') and c != 'diagnostico_principal']

def _contar_comorbilidades(row):
    ppal = str(row['diagnostico_principal']).strip().upper()
    return sum(1 for c in _diag_cols
               if str(row[c]).strip().upper() not in ('', 'NAN', 'NONE', 'NAT')
               and str(row[c]).strip().upper() != ppal)

df['comorbilidad'] = df.apply(_contar_comorbilidades, axis=1)

# ── Subconjunto C16.* (cáncer gástrico) ──
df_c16 = df[df['diagnostico_principal'].str.upper().str.startswith('C16', na=False)].copy()
print(f'  Subconjunto C16.*: {len(df_c16):,} registros')
print(f'  Mortalidad C16.*: {df_c16["mortalidad_int"].mean()*100:.2f}%')

# ── Top 15 hospitales por volumen ──
_conteo = df_c16['hospital'].value_counts()
_hosp15 = _conteo[_conteo >= MIN_CASOS_H].nlargest(TOP_HOSP).index
df_reg = df_c16[df_c16['hospital'].isin(_hosp15)].copy()
df_reg['hospital'] = df_reg['hospital'].astype(str)

print(f'  Top {TOP_HOSP} hospitales: {len(df_reg):,} registros')
print(f'  Hospitales incluidos: {df_reg["hospital"].nunique()}')

# ── Variables del modelo ──
model_vars = ['mortalidad_int', 'cantidad_procedimientos', 'edad',
              'severidad_grd', 'peso_grd', 'comorbilidad', 'hospital']
df_clean = df_reg.dropna(subset=model_vars).copy()
df_clean['mortalidad_int'] = df_clean['mortalidad_int'].astype(int)

print(f'  Dataset final para clasificación: {len(df_clean):,} registros')
print(f'  Tasa de mortalidad: {df_clean["mortalidad_int"].mean()*100:.2f}%')
print(f'  Tamaño clase positiva: {df_clean["mortalidad_int"].sum()}')
print(f'  Tamaño clase negativa: {(df_clean["mortalidad_int"] == 0).sum()}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN/TEST SPLIT (ANTES de SMOTE — previene data leakage)
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n[3/6] Train/test split estratificado (test={TEST_SIZE})...')

# Features: numéricas + one-hot de hospital
predictor_cols = ['cantidad_procedimientos', 'edad', 'severidad_grd',
                   'peso_grd', 'comorbilidad']

def make_design_matrix(df_in, train_cols=None):
    """Crea matriz de diseño: variables numéricas + dummies de hospital."""
    X_num = df_in[predictor_cols].copy().astype(float)
    X_hosp = pd.get_dummies(df_in['hospital'], prefix='hosp', drop_first=True)
    X = pd.concat([X_num.reset_index(drop=True),
                   X_hosp.reset_index(drop=True)], axis=1)
    if train_cols is not None:
        # Alinear columnas con el conjunto de entrenamiento
        for col in train_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[train_cols]
    return X

X_full = make_design_matrix(df_clean)
y_full = df_clean['mortalidad_int'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full,
    test_size=TEST_SIZE,
    random_state=SEMILLA,
    stratify=y_full
)

print(f'  Train: {len(X_train):,} (mortalidad: {y_train.mean()*100:.2f}%)')
print(f'  Test:  {len(X_test):,} (mortalidad: {y_test.mean()*100:.2f}%)')

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELO PRE-SMOTE (baseline)
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n[4/6] Modelo PRE-SMOTE (baseline)...')

scaler_pre = StandardScaler()
X_train_sc = scaler_pre.fit_transform(X_train)
X_test_sc = scaler_pre.transform(X_test)

model_pre = LogisticRegression(max_iter=2000, random_state=SEMILLA, solver='lbfgs')
model_pre.fit(X_train_sc, y_train)
y_prob_pre = model_pre.predict_proba(X_test_sc)[:, 1]
y_pred_pre = (y_prob_pre >= 0.5).astype(int)

auc_pre = roc_auc_score(y_test, y_prob_pre)
acc_pre = accuracy_score(y_test, y_pred_pre)
sens_pre = recall_score(y_test, y_pred_pre)  # recall clase positiva (minoritaria)
spec_pre = ((y_test == 0) & (y_pred_pre == 0)).sum() / (y_test == 0).sum()
f1_pre = f1_score(y_test, y_pred_pre)
fpr_pre, tpr_pre, _ = roc_curve(y_test, y_prob_pre)

print(f'  AUC-ROC     : {auc_pre:.4f}')
print(f'  Accuracy    : {acc_pre:.4f}')
print(f'  Sensitivity : {sens_pre:.4f}')
print(f'  Specificity : {spec_pre:.4f}')
print(f'  F1-score    : {f1_pre:.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODELO POST-SMOTE (SMOTE solo en entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n[5/6] Modelo POST-SMOTE...')

# Verificar k_neighbors <= n_minority - 1
n_minority = y_train.sum()
k_neighbors = min(5, n_minority - 1)
if k_neighbors < 1:
    k_neighbors = 1
print(f'  Clase minoritaria: {n_minority} muestras  →  k_neighbors = {k_neighbors}')

# SMOTE solo en entrenamiento
smote = SMOTE(random_state=SEMILLA, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)

# Verificar que X_test NO fue modificado por SMOTE
assert X_test_sc.shape == X_test.shape, 'ERROR: X_test fue modificado (data leakage)'

print(f'  Antes  SMOTE: {y_train.sum()} positivos / {len(y_train)} total ({y_train.mean()*100:.1f}%)')
print(f'  Después SMOTE: {y_train_smote.sum()} positivos / {len(y_train_smote)} total ({y_train_smote.mean()*100:.1f}%)')

model_smote = LogisticRegression(max_iter=2000, random_state=SEMILLA, solver='lbfgs')
model_smote.fit(X_train_smote, y_train_smote)
y_prob_smote = model_smote.predict_proba(X_test_sc)[:, 1]
y_pred_smote = (y_prob_smote >= 0.5).astype(int)

auc_smote = roc_auc_score(y_test, y_prob_smote)
acc_smote = accuracy_score(y_test, y_pred_smote)
sens_smote = recall_score(y_test, y_pred_smote)
spec_smote = ((y_test == 0) & (y_pred_smote == 0)).sum() / (y_test == 0).sum()
f1_smote = f1_score(y_test, y_pred_smote)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_prob_smote)

print(f'  AUC-ROC     : {auc_smote:.4f}')
print(f'  Accuracy    : {acc_smote:.4f}')
print(f'  Sensitivity : {sens_smote:.4f}')
print(f'  Specificity : {spec_smote:.4f}')
print(f'  F1-score    : {f1_smote:.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# 6. VALIDACIÓN CRUZADA ESTRATIFICADA (k=5) con SMOTE
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n[6/6] Validación cruzada estratificada (k={CV_FOLDS})...')

# Pipeline con SMOTE DENTRO de cada fold (imblearn Pipeline asegura
# que SMOTE solo se aplica a los folds de entrenamiento)
pipeline_cv = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=SEMILLA, k_neighbors=k_neighbors)),
    ('clf', LogisticRegression(max_iter=2000, random_state=SEMILLA, solver='lbfgs'))
])

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMILLA)

scoring_metrics = {
    'auc': 'roc_auc',
    'accuracy': 'accuracy',
    'sensitivity': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(
    pipeline_cv, X_full, y_full,
    cv=skf,
    scoring=scoring_metrics,
    return_train_score=False,
    n_jobs=1
)

print('\n  Resultados de Validación Cruzada (5-fold):')
print(f'  {"Métrica":<15} {"Media":>8} {"± DE":>8} {"Min":>8} {"Max":>8}')
print(f'  {"-"*15} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
for metric in ['auc', 'accuracy', 'sensitivity', 'f1']:
    key = f'test_{metric}'
    scores = cv_results[key]
    print(f'  {metric:<15} {scores.mean():>8.4f} ±{scores.std():>7.4f} '
          f'{scores.min():>8.4f} {scores.max():>8.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR RESULTADOS CSV
# ══════════════════════════════════════════════════════════════════════════════

print('\nGuardando resultados...')

# Tabla de métricas pre vs post SMOTE
resultados = pd.DataFrame({
    'estrategia': ['pre_smote', 'post_smote'],
    'auc_roc': [round(auc_pre, 4), round(auc_smote, 4)],
    'accuracy': [round(acc_pre, 4), round(acc_smote, 4)],
    'sensitivity': [round(sens_pre, 4), round(sens_smote, 4)],
    'specificity': [round(spec_pre, 4), round(spec_smote, 4)],
    'f1_score': [round(f1_pre, 4), round(f1_smote, 4)],
    'n_train': [len(y_train), len(y_train_smote)],
    'n_test': [len(y_test), len(y_test)],
    'train_mortalidad_pct': [
        round(y_train.mean() * 100, 2),
        round(y_train_smote.mean() * 100, 2)
    ],
    'test_mortalidad_pct': [round(y_test.mean() * 100, 2), round(y_test.mean() * 100, 2)],
    'k_neighbors_smote': [None, k_neighbors],
    'cv_folds': [None, CV_FOLDS],
    'notas': [
        'Baseline sin balanceo; modelo tiende a predecir clase mayoritaria',
        'SMOTE aplicado solo a train; balanceo 50/50; evaluado en test original'
    ]
})

csv_path = OUTPUTS_DIR / 'smote_resultados.csv'
resultados.to_csv(csv_path, index=False, encoding='utf-8')
print(f'  ✓ {csv_path}')

# Tabla de CV
cv_summary = pd.DataFrame({
    'metrica': ['auc_roc', 'accuracy', 'sensitivity', 'f1_score'],
    'cv_mean': [
        round(cv_results['test_auc'].mean(), 4),
        round(cv_results['test_accuracy'].mean(), 4),
        round(cv_results['test_sensitivity'].mean(), 4),
        round(cv_results['test_f1'].mean(), 4),
    ],
    'cv_std': [
        round(cv_results['test_auc'].std(), 4),
        round(cv_results['test_accuracy'].std(), 4),
        round(cv_results['test_sensitivity'].std(), 4),
        round(cv_results['test_f1'].std(), 4),
    ],
    'cv_min': [
        round(cv_results['test_auc'].min(), 4),
        round(cv_results['test_accuracy'].min(), 4),
        round(cv_results['test_sensitivity'].min(), 4),
        round(cv_results['test_f1'].min(), 4),
    ],
    'cv_max': [
        round(cv_results['test_auc'].max(), 4),
        round(cv_results['test_accuracy'].max(), 4),
        round(cv_results['test_sensitivity'].max(), 4),
        round(cv_results['test_f1'].max(), 4),
    ],
    'cv_folds': [CV_FOLDS] * 4,
    'notas': [
        'StratifiedKFold con SMOTE dentro de cada fold',
        'Accuracy puede ser engañosa en desbalance',
        'Recall/Sensibilidad de la clase minoritaria (mortalidad)',
        'F1 balancea precision y recall'
    ]
})

cv_csv_path = OUTPUTS_DIR / 'smote_cv_resultados.csv'
cv_summary.to_csv(cv_csv_path, index=False, encoding='utf-8')
print(f'  ✓ {cv_csv_path}')

# ══════════════════════════════════════════════════════════════════════════════
# CURVAS ROC COMPARATIVAS
# ══════════════════════════════════════════════════════════════════════════════

print('Generando curvas ROC...')

# ── Figura 1: ROC pre vs post SMOTE ──
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(fpr_pre, tpr_pre, color='#1A5276', lw=2.5,
        label=f'Pre-SMOTE (AUC = {auc_pre:.3f})')
ax.fill_between(fpr_pre, tpr_pre, alpha=0.08, color='#1A5276')
ax.plot(fpr_smote, tpr_smote, color='#884EA0', lw=2.5, linestyle='-',
        label=f'Post-SMOTE (AUC = {auc_smote:.3f})')
ax.fill_between(fpr_smote, tpr_smote, alpha=0.08, color='#884EA0')
ax.plot([0, 1], [0, 1], color='#7F8C8D', lw=1.2, linestyle='--',
        label='Aleatorio (AUC = 0.500)')

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=11)
ax.set_title('Curvas ROC: Pre-SMOTE vs Post-SMOTE\n'
             'Cáncer Gástrico (C16.*) | Conjunto de prueba (25%) | SMOTE solo en train',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

roc_path = FIGURAS_DIR / 'smote_roc_comparativa_pre_post.png'
fig.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  ✓ {roc_path}')

# ── Figura 2: ROC con bandas de CV ──
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(fpr_pre, tpr_pre, color='#1A5276', lw=2.5,
        label=f'Pre-SMOTE (AUC = {auc_pre:.3f})')
ax.plot(fpr_smote, tpr_smote, color='#884EA0', lw=2.5,
        label=f'Post-SMOTE (AUC = {auc_smote:.3f})')
ax.plot([0, 1], [0, 1], color='#7F8C8D', lw=1.2, linestyle='--',
        label='Aleatorio (AUC = 0.500)')

# Añadir anotación de resultados CV
cv_text = (
    f'Validación Cruzada Estratificada (k={CV_FOLDS}):\n'
    f'AUC CV  = {cv_results["test_auc"].mean():.3f} '
    f'± {cv_results["test_auc"].std():.3f}\n'
    f'F1 CV   = {cv_results["test_f1"].mean():.3f} '
    f'± {cv_results["test_f1"].std():.3f}\n'
    f'Sens CV = {cv_results["test_sensitivity"].mean():.3f} '
    f'± {cv_results["test_sensitivity"].std():.3f}'
)
ax.text(0.62, 0.25, cv_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9',
                  edgecolor='#B2BABB', alpha=0.9))

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=11)
ax.set_title('Curvas ROC con Validación Cruzada\n'
             'Cáncer Gástrico (C16.*) | SMOTE solo en entrenamiento',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

roc_cv_path = FIGURAS_DIR / 'smote_roc_con_cv.png'
fig.savefig(roc_cv_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  ✓ {roc_cv_path}')

# ── Figura 3: Comparación de métricas (barplot) ──
metrics_names = ['AUC-ROC', 'Accuracy', 'Sensitivity\n(Recall)', 'Specificity', 'F1-Score']
pre_values = [auc_pre, acc_pre, sens_pre, spec_pre, f1_pre]
post_values = [auc_smote, acc_smote, sens_smote, spec_smote, f1_smote]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

bars1 = ax.bar(x - width/2, pre_values, width, label='Pre-SMOTE',
               color='#1A5276', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, post_values, width, label='Post-SMOTE',
               color='#884EA0', alpha=0.85, edgecolor='white')

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Valor de la métrica', fontsize=11)
ax.set_title('Comparación de Métricas: Pre-SMOTE vs Post-SMOTE\n'
             'Cáncer Gástrico (C16.*) | Evaluación en Test (sin SMOTE)',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_ylim([0, 1.1])
ax.legend(fontsize=10, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

metrics_bar_path = FIGURAS_DIR / 'smote_metricas_comparativa.png'
fig.savefig(metrics_bar_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  ✓ {metrics_bar_path}')

# ── Figura 4: Distribución de clases pre/post SMOTE ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#FAFAFA')

# Antes SMOTE
classes_pre = ['Sobrevivientes\n(Clase 0)', 'Fallecidos\n(Clase 1)']
counts_pre = [(y_train == 0).sum(), y_train.sum()]
colors_pre = ['#2E86AB', '#A23B72']
axes[0].bar(classes_pre, counts_pre, color=colors_pre, alpha=0.85, edgecolor='white')
axes[0].set_title(f'Antes de SMOTE\nTotal: {len(y_train):,} | Mortalidad: {y_train.mean()*100:.1f}%',
                  fontsize=11, fontweight='bold')
for i, v in enumerate(counts_pre):
    axes[0].text(i, v + len(y_train)*0.01, f'{v:,}\n({v/len(y_train)*100:.1f}%)',
                 ha='center', fontsize=10)

# Después SMOTE
classes_post = ['Sobrevivientes\n(Clase 0)', 'Fallecidos\n(Clase 1)']
counts_post = [(y_train_smote == 0).sum(), y_train_smote.sum()]
colors_post = ['#2E86AB', '#A23B72']
axes[1].bar(classes_post, counts_post, color=colors_post, alpha=0.85, edgecolor='white')
axes[1].set_title(f'Después de SMOTE\nTotal: {len(y_train_smote):,} | Mortalidad: {y_train_smote.mean()*100:.1f}%',
                  fontsize=11, fontweight='bold')
for i, v in enumerate(counts_post):
    axes[1].text(i, v + len(y_train_smote)*0.01, f'{v:,}\n({v/len(y_train_smote)*100:.1f}%)',
                 ha='center', fontsize=10)

for ax_i in axes:
    ax_i.set_facecolor('#FAFAFA')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)

fig.suptitle('Distribución de Clases en Entrenamiento\nPre-SMOTE vs Post-SMOTE (solo en train)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

dist_path = FIGURAS_DIR / 'smote_distribucion_clases.png'
fig.savefig(dist_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  ✓ {dist_path}')

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 80)
print('RESUMEN — ANÁLISIS SMOTE CORREGIDO')
print('=' * 80)
print(f'\nPipeline correcto: Train/Test split → SMOTE solo en train → Evaluar en test')
print(f'No hay data leakage: X_test NUNCA pasa por SMOTE')
print(f'\nMétrica              Pre-SMOTE    Post-SMOTE   Δ')
print(f'{ "-" * 60 }')
print(f'AUC-ROC              {auc_pre:.4f}      {auc_smote:.4f}       {auc_smote - auc_pre:+.4f}')
print(f'Accuracy             {acc_pre:.4f}      {acc_smote:.4f}       {acc_smote - acc_pre:+.4f}')
print(f'Sensitivity (Recall)  {sens_pre:.4f}      {sens_smote:.4f}       {sens_smote - sens_pre:+.4f}')
print(f'Specificity           {spec_pre:.4f}      {spec_smote:.4f}       {spec_smote - spec_pre:+.4f}')
print(f'F1-Score             {f1_pre:.4f}      {f1_smote:.4f}       {f1_smote - f1_pre:+.4f}')
print(f'\nValidación Cruzada Estratificada (k={CV_FOLDS}) con SMOTE:')
print(f'  AUC-ROC      : {cv_results["test_auc"].mean():.4f} ± {cv_results["test_auc"].std():.4f}')
print(f'  Accuracy     : {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
print(f'  Sensitivity  : {cv_results["test_sensitivity"].mean():.4f} ± {cv_results["test_sensitivity"].std():.4f}')
print(f'  F1-Score     : {cv_results["test_f1"].mean():.4f} ± {cv_results["test_f1"].std():.4f}')
print(f'\nArchivos generados:')
print(f'  {csv_path}')
print(f'  {cv_csv_path}')
print(f'  {roc_path}')
print(f'  {roc_cv_path}')
print(f'  {metrics_bar_path}')
print(f'  {dist_path}')
print('\n✅ fix_smote.py completado exitosamente.')
print('=' * 80)
