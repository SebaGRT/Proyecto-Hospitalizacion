import json, sys
sys.stdout.reconfigure(encoding='utf-8')

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

def replace_in_cell(cells, idx, old, new):
    src = ''.join(cells[idx]['source'])
    assert old in src, f"Pattern not found in cell {idx}:\n---\n{src[:300]}\n---"
    cells[idx]['source'] = make_lines(src.replace(old, new))

with open('Avance2_Proyecto_Final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ─── CELL 15: Univariate distributions 2x2 ─────────────────────────────────
idx15 = find_cell(cells, "4.1.2 Distribuciones univariadas")
replace_in_cell(cells, idx15,
    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n\n"
    "sns.histplot(df['dias_estada'], bins=50, kde=True, color='steelblue', ax=axes[0,0])\n"
    "axes[0,0].set_title('Distribución de Días de Estadía (C00–D49)')\n"
    "axes[0,0].set_xlabel('Días'); axes[0,0].set_ylabel('Frecuencia')\n"
    "\n"
    "sns.histplot(df['edad'].dropna(), bins=50, kde=True, color='coral', ax=axes[0,1])\n"
    "axes[0,1].set_title('Distribución de Edad (C00–D49)')\n"
    "axes[0,1].set_xlabel('Años'); axes[0,1].set_ylabel('Frecuencia')\n"
    "\n"
    "sns.histplot(df['cantidad_procedimientos'], bins=40, kde=False, color='seagreen', ax=axes[1,0])\n"
    "axes[1,0].set_title('Distribución de Cantidad de Procedimientos (C00–D49)')\n"
    "axes[1,0].set_xlabel('N° Procedimientos'); axes[1,0].set_ylabel('Frecuencia')\n"
    "\n"
    "sns.countplot(x=df['severidad_grd'].dropna().astype(int), palette='viridis', ax=axes[1,1])\n"
    "axes[1,1].set_title('Distribución de Severidad GRD (C00–D49)')\n"
    "axes[1,1].set_xlabel('Severidad'); axes[1,1].set_ylabel('Frecuencia')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/eda_univariado_global.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    # ─── NEW ───
    "PALETA = ['#2E75B6','#C0392B','#148F77','#7D3C98']\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "fig.suptitle('Figura 1. Distribuciones Univariadas — Universo Oncológico (CIE-10 C00–D49)',\n"
    "             fontsize=14, fontweight='bold', y=1.01)\n"
    "\n"
    "for ax in axes.flat:\n"
    "    ax.set_facecolor('#FAFAFA')\n"
    "    ax.spines['top'].set_visible(False)\n"
    "    ax.spines['right'].set_visible(False)\n"
    "\n"
    "sns.histplot(df['dias_estada'], bins=50, kde=True, color=PALETA[0], ax=axes[0,0], alpha=0.8)\n"
    "axes[0,0].set_title('A. Días de Estadía', fontweight='bold')\n"
    "axes[0,0].set_xlabel('Días'); axes[0,0].set_ylabel('Frecuencia')\n"
    "axes[0,0].axvline(df['dias_estada'].median(), color=PALETA[1], lw=1.8, ls='--',\n"
    "                  label=f'Mediana={df[\"dias_estada\"].median():.0f}')\n"
    "axes[0,0].legend(fontsize=9)\n"
    "\n"
    "sns.histplot(df['edad'].dropna(), bins=50, kde=True, color=PALETA[1], ax=axes[0,1], alpha=0.8)\n"
    "axes[0,1].set_title('B. Edad del Paciente', fontweight='bold')\n"
    "axes[0,1].set_xlabel('Años'); axes[0,1].set_ylabel('Frecuencia')\n"
    "axes[0,1].axvline(df['edad'].median(), color=PALETA[0], lw=1.8, ls='--',\n"
    "                  label=f'Mediana={df[\"edad\"].median():.1f}')\n"
    "axes[0,1].legend(fontsize=9)\n"
    "\n"
    "sns.histplot(df['cantidad_procedimientos'], bins=40, kde=True, color=PALETA[2],\n"
    "             ax=axes[1,0], alpha=0.8)\n"
    "axes[1,0].set_title('C. Procedimientos por Egreso', fontweight='bold')\n"
    "axes[1,0].set_xlabel('N° Procedimientos'); axes[1,0].set_ylabel('Frecuencia')\n"
    "\n"
    "sev_counts = df['severidad_grd'].dropna().astype(int).value_counts().sort_index()\n"
    "axes[1,1].bar(sev_counts.index, sev_counts.values,\n"
    "              color=[PALETA[3], PALETA[0], PALETA[1], PALETA[2]][:len(sev_counts)],\n"
    "              edgecolor='white', width=0.6)\n"
    "axes[1,1].set_title('D. Severidad GRD', fontweight='bold')\n"
    "axes[1,1].set_xlabel('Nivel de Severidad'); axes[1,1].set_ylabel('Frecuencia')\n"
    "for bar in axes[1,1].patches:\n"
    "    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,\n"
    "                   f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/eda_univariado_global.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx15}: Univariate distributions themed + Figura 1")

# ─── CELL 16: Top-15 diagnoses ──────────────────────────────────────────────
idx16 = find_cell(cells, "# Top 15 diagnósticos oncológicos")
replace_in_cell(cells, idx16,
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "top_diag = df['diagnostico_principal'].value_counts().head(15)\n"
    "sns.barplot(x=top_diag.values, y=top_diag.index, palette='magma', ax=ax)\n"
    "ax.set_title('Top 15 Diagnósticos Oncológicos (CIE-10) — C00-D49')\n"
    "ax.set_xlabel('Frecuencia')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/top15_diagnosticos_onco.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "fig, ax = plt.subplots(figsize=(11, 6))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "ax.set_facecolor('#FAFAFA')\n"
    "top_diag = df['diagnostico_principal'].value_counts().head(15)\n"
    "bars = ax.barh(range(len(top_diag)), top_diag.values,\n"
    "               color='#2E75B6', alpha=0.85, edgecolor='white')\n"
    "# Resaltar C16.* en rojo\n"
    "for i, (diag, bar) in enumerate(zip(top_diag.index, bars)):\n"
    "    if str(diag).startswith('C16'):\n"
    "        bar.set_color('#C0392B')\n"
    "        bar.set_alpha(1.0)\n"
    "    ax.text(bar.get_width() + top_diag.values.max()*0.01, i,\n"
    "            f'{int(bar.get_width()):,}', va='center', fontsize=8.5)\n"
    "ax.set_yticks(range(len(top_diag)))\n"
    "ax.set_yticklabels(top_diag.index, fontsize=9)\n"
    "ax.set_title('Figura 2. Top 15 Diagnósticos Oncológicos (CIE-10 C00–D49)\\n'\n"
    "             'Rojo = Cáncer Gástrico C16.* (caso focal del proyecto)',\n"
    "             fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Número de egresos', fontsize=10)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/top15_diagnosticos_onco.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx16}: Top-15 diagnoses themed + Figura 2")

# ─── CELL 20: Boxplot days × hospital ───────────────────────────────────────
idx20 = find_cell(cells, "# ── BOXPLOT: Días de estadía × Hospital")
replace_in_cell(cells, idx20,
    "fig, ax = plt.subplots(figsize=(16, 6))\n"
    "sns.boxplot(x='hospital', y='dias_estada', data=df_eda, order=orden,\n"
    "            palette='Spectral', showfliers=False, ax=ax)\n"
    "ax.set_title('Distribución de Días de Estadía por Hospital\\n(Diagnóstico controlado: C16.* — Cáncer Gástrico)')\n"
    "ax.set_xlabel('Código Hospital'); ax.set_ylabel('Días de Estadía')\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "for label in ax.get_xticklabels():\n"
    "    label.set_ha('right')\n"
    "    label.set_fontsize(8)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/boxplot_dias_hospital_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "n_hosp = len(orden)\n"
    "paleta_box = sns.color_palette('Blues_d', n_colors=n_hosp)[::-1]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(16, 6))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "ax.set_facecolor('#FAFAFA')\n"
    "sns.boxplot(x='hospital', y='dias_estada', data=df_eda, order=orden,\n"
    "            palette=paleta_box, showfliers=False, linewidth=1.2, ax=ax)\n"
    "\n"
    "# Línea de mediana global\n"
    "mediana_global = df_eda['dias_estada'].median()\n"
    "ax.axhline(mediana_global, color='#C0392B', lw=1.5, ls='--', alpha=0.7,\n"
    "           label=f'Mediana global = {mediana_global:.1f} días')\n"
    "\n"
    "ax.set_title('Figura 3. Distribución de Días de Estadía por Hospital\\n'\n"
    "             'Cáncer Gástrico (C16.*) | Ordenado de mayor a menor mediana | Sin valores extremos',\n"
    "             fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Hospital (código)', fontsize=10)\n"
    "ax.set_ylabel('Días de Estadía', fontsize=10)\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "for label in ax.get_xticklabels():\n"
    "    label.set_ha('right'); label.set_fontsize(8)\n"
    "ax.legend(fontsize=9)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/boxplot_dias_hospital_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx20}: Boxplot themed + Figura 3")

# ─── CELL 22: Mortality bar ──────────────────────────────────────────────────
idx22 = find_cell(cells, "# ── BARPLOT: Tasa de mortalidad")
replace_in_cell(cells, idx22,
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "sns.barplot(x=mort_hosp.index.astype(str), y=mort_hosp.values*100, palette='Reds_r', ax=ax)\n"
    "ax.set_title('Tasa de Mortalidad Intrahospitalaria por Hospital (C16.*)')\n"
    "ax.set_xlabel('Código Hospital'); ax.set_ylabel('Mortalidad (%)')\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "for label in ax.get_xticklabels():\n"
    "    label.set_ha('right')\n"
    "    label.set_fontsize(8)\n"
    "for p in ax.patches:\n"
    "    ax.annotate(f'{p.get_height():.1f}%',\n"
    "                (p.get_x() + p.get_width()/2., p.get_height()),\n"
    "                ha='center', va='bottom', fontsize=7)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/barplot_mortalidad_hospital_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "# Gradiente de color según tasa de mortalidad\n"
    "n_m = len(mort_hosp)\n"
    "paleta_mort = sns.color_palette('Reds', n_colors=n_m + 4)[4:]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "ax.set_facecolor('#FAFAFA')\n"
    "bars_m = ax.bar(range(n_m), mort_hosp.values * 100,\n"
    "                color=paleta_mort, edgecolor='white', linewidth=0.5)\n"
    "\n"
    "# Línea de promedio nacional\n"
    "avg_mort = mort_hosp.values.mean() * 100\n"
    "ax.axhline(avg_mort, color='#2E75B6', lw=1.8, ls='--',\n"
    "           label=f'Promedio del grupo = {avg_mort:.1f}%')\n"
    "\n"
    "for i, (bar, val) in enumerate(zip(bars_m, mort_hosp.values * 100)):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,\n"
    "            f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#333333')\n"
    "\n"
    "ax.set_xticks(range(n_m))\n"
    "ax.set_xticklabels([str(h) for h in mort_hosp.index], rotation=45, ha='right', fontsize=8)\n"
    "ax.set_title('Figura 4. Tasa de Mortalidad Intrahospitalaria por Hospital\\n'\n"
    "             'Cáncer Gástrico (C16.*) | Ordenado de mayor a menor tasa',\n"
    "             fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Hospital (código)', fontsize=10)\n"
    "ax.set_ylabel('Mortalidad intrahospitalaria (%)', fontsize=10)\n"
    "ax.legend(fontsize=9)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/barplot_mortalidad_hospital_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx22}: Mortality barplot themed + Figura 4")

# ─── CELL 24: Procedures bar ────────────────────────────────────────────────
idx24 = find_cell(cells, "# ── BARPLOT: Promedio de procedimientos")
replace_in_cell(cells, idx24,
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "sns.barplot(x=proc_hosp.index.astype(str), y=proc_hosp.values, palette='Blues_r', ax=ax)\n"
    "ax.set_title('Promedio de Procedimientos por Hospital (C16.*)')\n"
    "ax.set_xlabel('Código Hospital'); ax.set_ylabel('Procedimientos promedio')\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "for label in ax.get_xticklabels():\n"
    "    label.set_ha('right')\n"
    "    label.set_fontsize(8)\n"
    "for p in ax.patches:\n"
    "    ax.annotate(f'{p.get_height():.1f}',\n"
    "                (p.get_x() + p.get_width()/2., p.get_height()),\n"
    "                ha='center', va='bottom', fontsize=7)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/barplot_procedimientos_hospital_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "n_p = len(proc_hosp)\n"
    "paleta_proc = sns.color_palette('Blues_d', n_colors=n_p + 3)[3:]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "ax.set_facecolor('#FAFAFA')\n"
    "bars_p = ax.bar(range(n_p), proc_hosp.values,\n"
    "                color=paleta_proc, edgecolor='white', linewidth=0.5)\n"
    "\n"
    "avg_proc = proc_hosp.values.mean()\n"
    "ax.axhline(avg_proc, color='#C0392B', lw=1.8, ls='--',\n"
    "           label=f'Promedio del grupo = {avg_proc:.1f} proc.')\n"
    "\n"
    "for bar, val in zip(bars_p, proc_hosp.values):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,\n"
    "            f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')\n"
    "\n"
    "ax.set_xticks(range(n_p))\n"
    "ax.set_xticklabels([str(h) for h in proc_hosp.index], rotation=45, ha='right', fontsize=8)\n"
    "ax.set_title('Figura 5. Promedio de Procedimientos por Hospital\\n'\n"
    "             'Cáncer Gástrico (C16.*) | Ordenado de mayor a menor intensidad procedimental',\n"
    "             fontsize=12, fontweight='bold')\n"
    "ax.set_xlabel('Hospital (código)', fontsize=10)\n"
    "ax.set_ylabel('Procedimientos promedio por egreso', fontsize=10)\n"
    "ax.legend(fontsize=9)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/graficos/barplot_procedimientos_hospital_C16.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx24}: Procedures barplot themed + Figura 5")

# ─── CELL 40: Chi-sq mortality × sex ────────────────────────────────────────
idx40 = find_cell(cells, "5.2.1 Escenario A")
replace_in_cell(cells, idx40,
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "(tab_A_prop['Fallecido']*100).plot(kind='bar', ax=ax, color=['steelblue','coral'], edgecolor='white')\n"
    "ax.set_title(f'Mortalidad intrahospitalaria por sexo\\nχ²({dof_A}) = {chi2_A:.2f}, p {fmt_p(p_A)}, V = {v_A:.3f}')\n"
    "ax.set_ylabel('Fallecidos (%)'); ax.set_xlabel('')\n"
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=0)\n"
    "for p_bar in ax.patches:\n"
    "    ax.annotate(f'{p_bar.get_height():.2f}%',\n"
    "                (p_bar.get_x()+p_bar.get_width()/2, p_bar.get_height()),\n"
    "                ha='center', va='bottom', fontsize=11)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/cat_A_mortalidad_sexo.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "ax.set_facecolor('#FAFAFA')\n"
    "sexos = tab_A_prop.index.tolist()\n"
    "vals  = (tab_A_prop['Fallecido'] * 100).values\n"
    "colores_sex = ['#2E75B6' if s.upper() in ('M','MASCULINO','HOMBRE') else '#C0392B'\n"
    "               for s in sexos]\n"
    "bars_A = ax.bar(sexos, vals, color=colores_sex, edgecolor='white', width=0.5, alpha=0.88)\n"
    "for bar, val in zip(bars_A, vals):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,\n"
    "            f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')\n"
    "ax.set_title(\n"
    "    f'Figura 7. Mortalidad Intrahospitalaria por Sexo (C16.*)\\n'\n"
    "    f'χ²({dof_A}) = {chi2_A:.2f},  p {fmt_p(p_A)},  V de Cramér = {v_A:.3f}',\n"
    "    fontsize=11, fontweight='bold')\n"
    "ax.set_ylabel('Porcentaje de fallecidos (%)', fontsize=10)\n"
    "ax.set_xlabel('Sexo', fontsize=10)\n"
    "ax.set_xticklabels(sexos, rotation=0, fontsize=11)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/cat_A_mortalidad_sexo.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx40}: Mortality×Sex themed + Figura 7")

# ─── CELL 42: Chi-sq mortality × tipo_ingreso ───────────────────────────────
idx42 = find_cell(cells, "5.2.2 Escenario B")
replace_in_cell(cells, idx42,
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "colores_ti = ['salmon','steelblue','lightgreen']\n"
    "(tab_B_prop['Fallecido']*100).plot(kind='bar', ax=ax, color=colores_ti, edgecolor='white')\n"
    "ax.set_title(f'Mortalidad intrahospitalaria por tipo de ingreso\\nχ²({dof_B}) = {chi2_B:.2f}, p {fmt_p(p_B)}, V = {v_B:.3f}')\n"
    "ax.set_ylabel('Fallecidos (%)'); ax.set_xlabel('')\n"
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=0)\n"
    "for p_bar in ax.patches:\n"
    "    ax.annotate(f'{p_bar.get_height():.2f}%',\n"
    "                (p_bar.get_x()+p_bar.get_width()/2, p_bar.get_height()),\n"
    "                ha='center', va='bottom', fontsize=10)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/cat_B_mortalidad_tipoing.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "paleta_ti = {'URGENCIA': '#C0392B', 'PROGRAMADA': '#2E75B6', 'OBSTETRICA': '#148F77'}\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "ax.set_facecolor('#FAFAFA')\n"
    "tipos  = tab_B_prop.index.tolist()\n"
    "vals_B = (tab_B_prop['Fallecido'] * 100).values\n"
    "colores_B = [paleta_ti.get(str(t).upper(), '#7F8C8D') for t in tipos]\n"
    "bars_B = ax.bar(tipos, vals_B, color=colores_B, edgecolor='white', width=0.5, alpha=0.88)\n"
    "for bar, val in zip(bars_B, vals_B):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,\n"
    "            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
    "ax.set_title(\n"
    "    f'Figura 8. Mortalidad Intrahospitalaria por Tipo de Ingreso (C16.*)\\n'\n"
    "    f'χ²({dof_B}) = {chi2_B:.2f},  p {fmt_p(p_B)},  V de Cramér = {v_B:.3f}\\n'\n"
    "    'Rojo = Urgencia · Azul = Programada · Verde = Obstétrica',\n"
    "    fontsize=10, fontweight='bold')\n"
    "ax.set_ylabel('Porcentaje de fallecidos (%)', fontsize=10)\n"
    "ax.set_xlabel('Tipo de ingreso', fontsize=10)\n"
    "ax.set_xticklabels(tipos, rotation=0, fontsize=10)\n"
    "ax.spines['top'].set_visible(False)\n"
    "ax.spines['right'].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/cat_B_mortalidad_tipoing.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx42}: Mortality×TipoIngreso themed + Figura 8")

# ─── CELL 44: Residuals heatmap tipo_alta × hospital ─────────────────────────
idx44 = find_cell(cells, "5.2.3 Escenario C")
replace_in_cell(cells, idx44,
    "fig, ax = plt.subplots(figsize=(10, 7))\n"
    "sns.heatmap(resid_C, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-8, vmax=8, linewidths=0.5, ax=ax)\n"
    "ax.set_title(f'Residuos estandarizados: tipo de alta × hospital (top 10)\\nχ²({dof_C}) = {chi2_C:.2f}, p {fmt_p(p_C)}, V = {v_C:.3f}')\n"
    "ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)\n"
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/cat_C_residuos_tipoalta_hospital.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()",
    "fig, ax = plt.subplots(figsize=(11, 8))\n"
    "fig.patch.set_facecolor('#FAFAFA')\n"
    "sns.heatmap(resid_C, annot=True, fmt='.2f',\n"
    "            cmap='RdBu_r', center=0, vmin=-8, vmax=8,\n"
    "            linewidths=0.5, linecolor='white',\n"
    "            annot_kws={'size': 9},\n"
    "            cbar_kws={'label': 'Residuo estandarizado', 'shrink': 0.75},\n"
    "            ax=ax)\n"
    "ax.set_title(\n"
    "    f'Figura 9. Residuos Estandarizados: Tipo de Alta × Hospital (C16.*)\\n'\n"
    "    f'χ²({dof_C}) = {chi2_C:.2f},  p {fmt_p(p_C)},  V de Cramér = {v_C:.3f}\\n'\n"
    "    'Rojo = sobrerepresentación · Azul = subrepresentación respecto a independencia',\n"
    "    fontsize=11, fontweight='bold', pad=12)\n"
    "ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)\n"
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)\n"
    "# Umbral de significancia visual\n"
    "ax.text(1.18, -0.04,\n"
    "        '|residuo| > 2:\\ncombinación significativa\\n(aprox. p < 0.05)',\n"
    "        transform=ax.transAxes, fontsize=8, color='#555555',\n"
    "        bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2E75B6', alpha=0.8))\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/inferencial/cat_C_residuos_tipoalta_hospital.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)
print(f"✓ Cell {idx44}: Residuals heatmap themed + Figura 9")

# ─── Update figure numbers in inferential cells (now offset) ─────────────────
# Cell 47 has "Figura 6" → should be Figura 10
idx47 = find_cell(cells, "Figura 6 — Verificación de Normalidad")
if idx47 >= 0:
    src = ''.join(cells[idx47]['source'])
    cells[idx47]['source'] = make_lines(
        src.replace('Figura 6 — Verificación de Normalidad', 'Figura 10 — Verificación de Normalidad'))
    print(f"✓ Cell {idx47}: Renumbered Figura 6→10")

# Cell 49 "Figura 7" → Figura 11
idx49 = find_cell(cells, "Figura 7 — Heatmap Post-hoc Dunn-Bonferroni")
if idx49 >= 0:
    src = ''.join(cells[idx49]['source'])
    cells[idx49]['source'] = make_lines(
        src.replace('Figura 7 — Heatmap Post-hoc Dunn-Bonferroni', 'Figura 11 — Heatmap Post-hoc Dunn-Bonferroni'))
    print(f"✓ Cell {idx49}: Renumbered Figura 7→11")

# Cell 54 "Figura 8" → Figura 12
idx54 = find_cell(cells, "Figura 8 — Curva ROC")
if idx54 >= 0:
    src = ''.join(cells[idx54]['source'])
    cells[idx54]['source'] = make_lines(
        src.replace('Figura 8 — Curva ROC', 'Figura 12 — Curva ROC'))
    print(f"✓ Cell {idx54}: Renumbered Figura 8→12")

# Cell 62 "Figura 9" → Figura 13
idx62 = find_cell(cells, "Figura 9 — Transformación logarítmica")
if idx62 >= 0:
    src = ''.join(cells[idx62]['source'])
    cells[idx62]['source'] = make_lines(
        src.replace('Figura 9 — Transformación logarítmica', 'Figura 13 — Transformación logarítmica'))
    print(f"✓ Cell {idx62}: Renumbered Figura 9→13")

# Cell 65 "Figura 10" → Figura 14
idx65 = find_cell(cells, "Figura 10 — Evaluación Predictiva del Modelo OLS")
if idx65 >= 0:
    src = ''.join(cells[idx65]['source'])
    cells[idx65]['source'] = make_lines(
        src.replace('Figura 10 — Evaluación Predictiva del Modelo OLS', 'Figura 14 — Evaluación Predictiva del Modelo OLS'))
    print(f"✓ Cell {idx65}: Renumbered Figura 10→14")

# Cell 67 "Figura 11" → Figura 15
idx67 = find_cell(cells, "Figura 11 — Forest Plot OLS")
if idx67 >= 0:
    src = ''.join(cells[idx67]['source'])
    cells[idx67]['source'] = make_lines(
        src.replace('Figura 11 — Forest Plot OLS', 'Figura 15 — Forest Plot OLS'))
    print(f"✓ Cell {idx67}: Renumbered Figura 11→15")

# Clear outputs & save
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
nb['cells'] = cells
with open('Avance2_Proyecto_Final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"\n✓ All early-graph cells themed. Notebook saved. Total cells: {len(cells)}")
