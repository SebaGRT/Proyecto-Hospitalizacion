import json, sys, random
sys.stdout.reconfigure(encoding='utf-8')

def rand_id():
    return ''.join(random.choices('0123456789abcdef', k=8))

def make_lines(src):
    raw = src.split('\n')
    if len(raw) == 1:
        return [raw[0]]
    result = [l + '\n' for l in raw[:-1]]
    if raw[-1]:
        result.append(raw[-1])
    return result

def find_cell(cells, marker, start=0):
    for i in range(start, len(cells)):
        if marker in ''.join(cells[i]['source']):
            return i
    return -1

with open('Avance2_Proyecto_Final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── CELL 47: Normalidad — histogram + QQ themed & numbered ─────────────────────
idx = find_cell(cells, "5.3.1 Verificación de normalidad")
src = ''.join(cells[idx]['source'])
OLD47 = (
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "axes[0].hist(df_focus['cantidad_procedimientos'].dropna(), bins=40, color='steelblue', edgecolor='white')\n"
    "axes[0].set_title('Distribución de cantidad_procedimientos (C16.*)')\n"
    "axes[0].set_xlabel('N° Procedimientos'); axes[0].set_ylabel('Frecuencia')\n"
    "stats.probplot(df_focus['cantidad_procedimientos'].dropna().values, plot=axes[1])\n"
    "axes[1].set_title('Q-Q Plot — cantidad_procedimientos (C16.*)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h1_normalidad_proc_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
NEW47 = (
    "PALETTE_MAIN = '#1A5276'\n"
    "PALETTE_ACC  = '#C0392B'\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "fig.suptitle('Figura 5. Verificación de Normalidad — Procedimientos por Egreso (C16.*)',\n"
    "             fontsize=13, fontweight='bold', y=1.01)\n"
    "\n"
    "from scipy.stats import gaussian_kde\n"
    "data_norm = df_focus['cantidad_procedimientos'].dropna().values\n"
    "axes[0].hist(data_norm, bins=40, color=PALETTE_MAIN, edgecolor='white', alpha=0.85, density=True)\n"
    "xs = np.linspace(data_norm.min(), data_norm.max(), 300)\n"
    "kde_fn = gaussian_kde(data_norm)\n"
    "axes[0].plot(xs, kde_fn(xs), color=PALETTE_ACC, lw=2, label='KDE')\n"
    "axes[0].set_title('Distribución de Procedimientos', fontsize=11)\n"
    "axes[0].set_xlabel('N° Procedimientos', fontsize=10)\n"
    "axes[0].set_ylabel('Densidad', fontsize=10)\n"
    "axes[0].legend(fontsize=9)\n"
    "axes[0].spines['top'].set_visible(False)\n"
    "axes[0].spines['right'].set_visible(False)\n"
    "\n"
    "(osm, osr), (slope_qq, intercept_qq, _) = stats.probplot(data_norm, dist='norm')\n"
    "axes[1].scatter(osm, osr, color=PALETTE_MAIN, s=8, alpha=0.5, label='Datos')\n"
    "line_x = np.array([osm.min(), osm.max()])\n"
    "axes[1].plot(line_x, slope_qq * line_x + intercept_qq, color=PALETTE_ACC, lw=2, label='Normal teórica')\n"
    "axes[1].set_title('Q-Q Plot Normal', fontsize=11)\n"
    "axes[1].set_xlabel('Cuantiles teóricos', fontsize=10)\n"
    "axes[1].set_ylabel('Cuantiles observados', fontsize=10)\n"
    "axes[1].legend(fontsize=9)\n"
    "axes[1].spines['top'].set_visible(False)\n"
    "axes[1].spines['right'].set_visible(False)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h1_normalidad_proc_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
assert OLD47 in src, f"Cell {idx}: OLD47 pattern not found"
cells[idx]['source'] = make_lines(src.replace(OLD47, NEW47))
print(f"✓ Cell {idx}: Normalidad plot upgraded")

# ── CELL 49: Dunn heatmap — diverging palette + α reference line ───────────────
idx = find_cell(cells, "5.3.3 Post-hoc Dunn-Bonferroni")
src = ''.join(cells[idx]['source'])
OLD49 = (
    "    fig, ax = plt.subplots(figsize=(12, 10))\n"
    "    mask_diag = np.eye(len(top15), dtype=bool)\n"
    "    sns.heatmap(p_matrix.astype(float), mask=mask_diag, cmap='RdYlGn_r', vmin=0, vmax=0.1,\n"
    "                annot=True, fmt='.3f', linewidths=0.5, ax=ax)\n"
    "    ax.set_title('p-valores ajustados (Dunn-Bonferroni) — C16.*\\nVerde = no sig. · Rojo = sig. (p_adj < 0.05)', fontsize=11)\n"
    "    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)\n"
    "    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)\n"
    "    plt.tight_layout()\n"
    "    plt.savefig('outputs/inferencial/h1_dunn_heatmap_C16.png', dpi=150, bbox_inches='tight')\n"
    "    plt.show()"
)
NEW49 = (
    "    fig, ax = plt.subplots(figsize=(13, 11))\n"
    "    mask_diag = np.eye(len(top15), dtype=bool)\n"
    "    cmap_dunn = sns.diverging_palette(10, 145, s=80, l=55, as_cmap=True)\n"
    "    sns.heatmap(p_matrix.astype(float), mask=mask_diag,\n"
    "                cmap=cmap_dunn, vmin=0, vmax=0.10,\n"
    "                annot=True, fmt='.3f', linewidths=0.6,\n"
    "                linecolor='white', annot_kws={'size': 8}, ax=ax)\n"
    "    ax.set_title(\n"
    "        'Figura 6. Heatmap Dunn-Bonferroni — p-valores Ajustados entre Hospitales\\n'\n"
    "        'Cáncer Gástrico (C16.*) | Rojo: p < 0.05 · Azul: p ≥ 0.05',\n"
    "        fontsize=11, fontweight='bold', pad=12)\n"
    "    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)\n"
    "    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)\n"
    "    cbar = ax.collections[0].colorbar\n"
    "    cbar.ax.axhline(y=0.05, color='black', lw=1.5, linestyle='--')\n"
    "    cbar.ax.text(2.5, 0.05, 'alpha=0.05', va='center', fontsize=8, color='black')\n"
    "    plt.tight_layout()\n"
    "    plt.savefig('outputs/inferencial/h1_dunn_heatmap_C16.png', dpi=150, bbox_inches='tight')\n"
    "    plt.show()"
)
assert OLD49 in src, f"Cell {idx}: OLD49 pattern not found"
cells[idx]['source'] = make_lines(src.replace(OLD49, NEW49))
print(f"✓ Cell {idx}: Dunn heatmap upgraded")

# ── CELL 54: ROC — fill AUC, threshold marker, themed ─────────────────────────
idx = find_cell(cells, "5.4.2 Evaluación Predictiva")
src = ''.join(cells[idx]['source'])
OLD54 = (
    "# Curva ROC\n"
    "fpr, tpr, _ = roc_curve(test_df['mortalidad_int'], y_prob)\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')\n"
    "ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')\n"
    "ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])\n"
    "ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')\n"
    "ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')\n"
    "ax.set_title('Curva ROC — Modelo Logístico de Mortalidad (C16.*)')\n"
    "ax.legend(loc='lower right')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h2_roc_curve_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
NEW54 = (
    "# Curva ROC — visualización mejorada\n"
    "fpr, tpr, thresholds_roc = roc_curve(test_df['mortalidad_int'], y_prob)\n"
    "fig, ax = plt.subplots(figsize=(8, 7))\n"
    "\n"
    "ax.fill_between(fpr, tpr, alpha=0.15, color='#1A5276')\n"
    "ax.plot(fpr, tpr, color='#1A5276', lw=2.5,\n"
    "        label=f'Modelo logístico (AUC = {auc:.3f})')\n"
    "ax.plot([0, 1], [0, 1], color='#95A5A6', lw=1.2,\n"
    "        linestyle='--', label='Clasificador aleatorio')\n"
    "\n"
    "idx_thresh50 = np.argmin(np.abs(thresholds_roc - 0.5))\n"
    "ax.plot(fpr[idx_thresh50], tpr[idx_thresh50],\n"
    "        'o', color='#C0392B', markersize=10, zorder=5,\n"
    "        label=f'Umbral 0.50 (Sens={tpr[idx_thresh50]:.2f}, Esp={1-fpr[idx_thresh50]:.2f})')\n"
    "\n"
    "ax.set_xlim([-0.01, 1.01]); ax.set_ylim([0.0, 1.05])\n"
    "ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)\n"
    "ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=11)\n"
    "ax.set_title(\n"
    "    'Figura 7. Curva ROC — Mortalidad Intrahospitalaria\\n'\n"
    "    'Cáncer Gástrico (C16.*) | Regresión Logística con Efectos Fijos de Hospital',\n"
    "    fontsize=12, fontweight='bold')\n"
    "ax.legend(loc='lower right', fontsize=10)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h2_roc_curve_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
assert OLD54 in src, f"Cell {idx}: OLD54 pattern not found"
cells[idx]['source'] = make_lines(src.replace(OLD54, NEW54))
print(f"✓ Cell {idx}: ROC curve upgraded")

# ── CELL 62: Transformation plot — add median/mean lines ──────────────────────
idx = find_cell(cells, "5.5.1 Verificación de asimetría")
src = ''.join(cells[idx]['source'])
OLD62 = (
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "sns.histplot(df_reg['dias_estada'], bins=40, kde=True, color='coral', ax=axes[0])\n"
    "axes[0].set_title('dias_estada original (C16.*)')\n"
    "sns.histplot(df_reg['log_dias_estada'], bins=40, kde=True, color='steelblue', ax=axes[1])\n"
    "axes[1].set_title('log(1 + dias_estada) transformado (C16.*)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h3_transformacion_dias_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
NEW62 = (
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "fig.suptitle('Figura 8. Transformación Logarítmica de Días de Estadía — C16.*',\n"
    "             fontsize=13, fontweight='bold', y=1.01)\n"
    "\n"
    "sns.histplot(df_reg['dias_estada'], bins=40, kde=True,\n"
    "             color='#C0392B', alpha=0.75, ax=axes[0])\n"
    "med0  = df_reg['dias_estada'].median()\n"
    "mean0 = df_reg['dias_estada'].mean()\n"
    "axes[0].axvline(med0,  color='#2C3E50', lw=1.8, ls='--', label=f'Mediana={med0:.1f}')\n"
    "axes[0].axvline(mean0, color='#7D3C98', lw=1.8, ls=':',  label=f'Media={mean0:.1f}')\n"
    "axes[0].set_title(f'Escala original (asimetría={df_reg[\"dias_estada\"].skew():.2f})', fontsize=10)\n"
    "axes[0].set_xlabel('Días de estadía', fontsize=10)\n"
    "axes[0].set_ylabel('Densidad', fontsize=10)\n"
    "axes[0].legend(fontsize=9)\n"
    "axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)\n"
    "\n"
    "sns.histplot(df_reg['log_dias_estada'], bins=40, kde=True,\n"
    "             color='#1A5276', alpha=0.75, ax=axes[1])\n"
    "med1  = df_reg['log_dias_estada'].median()\n"
    "mean1 = df_reg['log_dias_estada'].mean()\n"
    "axes[1].axvline(med1,  color='#2C3E50', lw=1.8, ls='--', label=f'Mediana={med1:.2f}')\n"
    "axes[1].axvline(mean1, color='#7D3C98', lw=1.8, ls=':',  label=f'Media={mean1:.2f}')\n"
    "axes[1].set_title(f'Escala log(1+x) (asimetría={df_reg[\"log_dias_estada\"].skew():.2f})', fontsize=10)\n"
    "axes[1].set_xlabel('log(1 + días de estadía)', fontsize=10)\n"
    "axes[1].set_ylabel('Densidad', fontsize=10)\n"
    "axes[1].legend(fontsize=9)\n"
    "axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h3_transformacion_dias_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
assert OLD62 in src, f"Cell {idx}: OLD62 pattern not found"
cells[idx]['source'] = make_lines(src.replace(OLD62, NEW62))
print(f"✓ Cell {idx}: Transformation plot upgraded")

# ── CELL 65: Pred vs Obs — RMSE band, themed, numbered ────────────────────────
idx = find_cell(cells, "5.5.3 Evaluación Predictiva")
src = ''.join(cells[idx]['source'])
OLD65 = (
    "# Gráfico de dispersión: predicción vs observado (escala original)\n"
    "fig, ax = plt.subplots(figsize=(8, 8))\n"
    "ax.scatter(test_ols['dias_estada'], pred_orig, alpha=0.3, edgecolors='none', color='steelblue')\n"
    "lim = [0, max(test_ols['dias_estada'].max(), pred_orig.max())]\n"
    "ax.plot(lim, lim, 'k--', lw=1)\n"
    "ax.set_xlim(lim); ax.set_ylim(lim)\n"
    "ax.set_xlabel('Días de estadía observados')\n"
    "ax.set_ylabel('Días de estadía predichos')\n"
    "ax.set_title(f'Predicción vs Observado — OLS (C16.*)\\nR² = {r2_orig:.3f} | RMSE = {rmse_orig:.1f} días')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h3_pred_vs_obs_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
NEW65 = (
    "# Gráfico predicción vs observado — mejorado\n"
    "fig, ax = plt.subplots(figsize=(8, 7))\n"
    "\n"
    "ax.scatter(test_ols['dias_estada'], pred_orig,\n"
    "           alpha=0.25, edgecolors='none', color='#1A5276', s=18, label='Observaciones')\n"
    "\n"
    "lim_max = max(test_ols['dias_estada'].max(), pred_orig.max()) * 1.02\n"
    "lim = [0, lim_max]\n"
    "ax.plot(lim, lim, color='#C0392B', lw=2, linestyle='--', label='Prediccion perfecta')\n"
    "\n"
    "xs_band = np.linspace(0, lim_max, 200)\n"
    "ax.fill_between(xs_band, xs_band - rmse_orig, xs_band + rmse_orig,\n"
    "                alpha=0.12, color='#C0392B', label=f'+/- RMSE ({rmse_orig:.1f} dias)')\n"
    "\n"
    "ax.set_xlim(lim); ax.set_ylim(lim)\n"
    "ax.set_xlabel('Dias de estadia observados', fontsize=11)\n"
    "ax.set_ylabel('Dias de estadia predichos (back-transform)', fontsize=11)\n"
    "ax.set_title(\n"
    "    f'Figura 9. Prediccion vs Observado — Regresion OLS\\n'\n"
    "    f'Cancer Gastrico (C16.*) | R2 = {r2_orig:.3f} | RMSE = {rmse_orig:.1f} dias | MAE = {mae_orig:.1f} dias',\n"
    "    fontsize=12, fontweight='bold')\n"
    "ax.legend(fontsize=10)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h3_pred_vs_obs_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
assert OLD65 in src, f"Cell {idx}: OLD65 pattern not found"
cells[idx]['source'] = make_lines(src.replace(OLD65, NEW65))
print(f"✓ Cell {idx}: Pred vs Obs upgraded")

# ── CELL 67: OLS coef — full multi-covariate version ──────────────────────────
idx = find_cell(cells, "Gráfico comparativo del coeficiente de procedimientos")
cells[idx]['source'] = make_lines(
    "# -- Coeficientes OLS — todos los predictores principales (sin dummies hospital) --\n"
    "import matplotlib.patches as mpatches\n"
    "\n"
    "excluir_ols = ['Intercept'] + [c for c in tabla_ols.index if 'C(hospital)' in str(c)]\n"
    "vars_ols_plot = [v for v in tabla_ols.index if v not in excluir_ols]\n"
    "\n"
    "coefs_v  = tabla_ols.loc[vars_ols_plot, 'coef']\n"
    "ci_low_v = tabla_ols.loc[vars_ols_plot, 'IC_inf']\n"
    "ci_hi_v  = tabla_ols.loc[vars_ols_plot, 'IC_sup']\n"
    "pvals_v  = tabla_ols.loc[vars_ols_plot, 'p_valor']\n"
    "\n"
    "etiquetas_ols = {\n"
    "    'cantidad_procedimientos': 'Procedimientos adicionales',\n"
    "    'edad':                    'Edad (anios)',\n"
    "    'severidad_grd':           'Severidad GRD',\n"
    "    'peso_grd':                'Peso relativo GRD',\n"
    "    'comorbilidad':            'Comorbilidades (conteo)',\n"
    "}\n"
    "labels_ols = [etiquetas_ols.get(v, v) for v in vars_ols_plot]\n"
    "colors_ols = ['#C0392B' if c > 0 else '#1A5276' for c in coefs_v]\n"
    "sig_ols    = ['*' if p < 0.05 else '' for p in pvals_v]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, max(4, len(vars_ols_plot) * 0.9)))\n"
    "y_pos = np.arange(len(vars_ols_plot))\n"
    "\n"
    "for i, (y, c, cl, ch, col, s) in enumerate(\n"
    "        zip(y_pos, coefs_v, ci_low_v, ci_hi_v, colors_ols, sig_ols)):\n"
    "    ax.plot([cl, ch], [y, y], color=col, linewidth=2.2, solid_capstyle='round')\n"
    "    ax.plot(c, y, 's', color=col, markersize=9, zorder=5)\n"
    "    ax.text(ch + 0.001, y, f'  b={c:.4f} {s}', va='center', fontsize=9.5, color=col)\n"
    "\n"
    "ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)\n"
    "ax.set_yticks(y_pos)\n"
    "ax.set_yticklabels(labels_ols, fontsize=10)\n"
    "ax.set_xlabel('Coeficiente OLS sobre log(1 + dias de estadia)', fontsize=11)\n"
    "ax.set_title(\n"
    "    f'Figura 10. Coeficientes OLS Ajustados — Estadia Hospitalaria\\n'\n"
    "    f'Cancer Gastrico (C16.*) | R2adj = {modelo_ols.rsquared_adj:.3f} | Errores robustos HC3',\n"
    "    fontsize=12, fontweight='bold')\n"
    "\n"
    "red_p  = mpatches.Patch(color='#C0392B', label='Coef > 0 (aumenta estadia)')\n"
    "blue_p = mpatches.Patch(color='#1A5276', label='Coef < 0 (reduce estadia)')\n"
    "ax.legend(handles=[red_p, blue_p], fontsize=9, loc='lower right')\n"
    "ax.text(0.01, -0.09,\n"
    "        '* p < 0.05  |  Barras = IC 95%  |  Efectos fijos de hospital incluidos pero omitidos.',\n"
    "        transform=ax.transAxes, fontsize=8, color='gray')\n"
    "\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/h3_coef_ols_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx}: OLS coefficient plot replaced with full multi-covariate version")

# Clear outputs & save
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
nb['cells'] = cells
with open('Avance2_Proyecto_Final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"\nSaved. Total cells: {len(cells)}")
