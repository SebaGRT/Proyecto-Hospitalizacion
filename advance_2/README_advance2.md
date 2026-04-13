# Avance 2 — Análisis Inferencial Completo

## Objetivo
Implementación del análisis inferencial completo con modelado estadístico,
visualizaciones interpretadas y conclusiones coherentes con la literatura.

**Pregunta de investigación:** ¿Varía significativamente la intensidad de
procedimientos médicos entre hospitales públicos para pacientes con diagnóstico
de neoplasia o sepsis, y está esa variación asociada a diferencias en mortalidad
intrahospitalaria y días de estadía?

---

## Estructura del proyecto

```
advance_2/
├── config.py                    # Parámetros centralizados (α, semillas, rutas)
├── utils.py                     # Funciones reutilizables (carga, limpieza, derivación)
├── 01_eda_profundo.py           # EDA profundo + 6 visualizaciones
├── 02_preparacion_datos.py      # Documentación de decisiones de limpieza
├── 03_tests_normalidad.py       # Shapiro-Wilk → justifica Kruskal-Wallis
├── 04_kruskal_wallis.py         # H1: Kruskal-Wallis + Dunn-Bonferroni
├── 05_regresion_logistica.py    # H2: Regresión logística (mortalidad)
├── 06_regresion_ols.py          # H3: OLS + VIF (días de estadía)
├── 07_sintesis_resultados.py    # Visualizaciones finales + tabla comparativa
├── Advance2_Analisis_Completo.ipynb  # Notebook ejecutable de punta a punta
├── README_advance2.md           # Este archivo
└── outputs/
    ├── tablas/                  # Tablas CSV de resultados
    ├── graficos/                # Gráficos PNG (300 dpi)
    └── modelos/                 # Summary de modelos + métricas JSON
```

---

## Hipótesis

| N° | Hipótesis | Prueba | Variable dependiente |
|----|-----------|--------|---------------------|
| H1 | Variabilidad inter-hospital significativa en procedimientos | Shapiro-Wilk → Kruskal-Wallis → Dunn-Bonferroni | n_procedimientos, dias_estadia, n_proc_unicos |
| H2 | Intensidad de procedimientos asociada a mortalidad | Regresión logística + efectos fijos por hospital | mortalidad (TIPOALTA == 'FALLECIDO') |
| H3 | Intensidad de procedimientos predice estadía independientemente de severidad | OLS + efectos fijos por hospital + VIF | dias_estadia |

---

## Grupos diagnósticos

| Grupo | Códigos CIE-10 | Tipo de atención |
|-------|---------------|-----------------|
| Neoplasia | C50, C18, C19, C20, C53, C34 | Programada (cirugía electiva) |
| Sepsis | A40, A41 | Urgencia (cuidados intensivos) |

---

## Cómo ejecutar

### Opción 1: Scripts individuales (desde la carpeta `advance_2/`)
```bash
cd advance_2/

python 01_eda_profundo.py           # → outputs/graficos/
python 02_preparacion_datos.py      # Documenta decisiones de limpieza
python 03_tests_normalidad.py       # → outputs/tablas/
python 04_kruskal_wallis.py         # → outputs/tablas/
python 05_regresion_logistica.py    # → outputs/tablas/ + modelos/
python 06_regresion_ols.py          # → outputs/tablas/ + modelos/
python 07_sintesis_resultados.py    # → tabla comparativa final
```

### Opción 2: Notebook completo
```bash
jupyter notebook Advance2_Analisis_Completo.ipynb
```
Ejecutar todas las celdas de arriba hacia abajo. Produce todos los outputs.

---

## Dependencias

```
pandas>=1.5
numpy>=1.23
scipy>=1.9
statsmodels>=0.13
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.1
openpyxl
scikit-posthocs   # opcional para Dunn post-hoc (se usa implementación manual si falta)
```

Instalar:
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn openpyxl scikit-posthocs
```

---

## Archivos de salida

### Tablas (`outputs/tablas/`)
| Archivo | Contenido |
|---------|-----------|
| `tabla_completitud.csv` | % completitud por variable y grupo |
| `resultados_shapiro_wilk.csv` | W-stat, p-valor, conclusión por grupo |
| `resultados_kruskal_wallis.csv` | H-stat, p-valor, N hospitales por variable |
| `dunn_posthoc_neoplasia.csv` | P-valores pareados corregidos por Bonferroni |
| `dunn_posthoc_sepsis.csv` | P-valores pareados corregidos por Bonferroni |
| `coeficientes_regresion_logistica.csv` | OR, IC95%, p-valores (H2) |
| `coeficientes_regresion_ols.csv` | β, IC95%, p-valores (H3) |
| `diagnostico_vif.csv` | VIF por predictor y grupo |
| `tabla_comparativa_neoplasia_vs_sepsis.csv` | Resumen cruzado de las 3 hipótesis |

### Gráficos (`outputs/graficos/`)
| Archivo | Contenido |
|---------|-----------|
| `01_distribuciones.png` | Histogramas + KDE: dias_estadia, n_procedimientos |
| `02_qqplots.png` | Q-Q plots para evaluación de normalidad |
| `03_boxplot_severidad.png` | Boxplots por nivel de severidad GRD |
| `04_violin_hospitales.png` | Violinplots inter-hospital (top 15) |
| `05_barras_procedimientos.png` | Media ± IC95% de procedimientos por hospital |
| `06_scatter_peso_estadia.png` | Peso GRD vs. días de estadía (scatter) |
| `07_comparacion_coeficientes.png` | β de n_procedimientos: neoplasia vs. sepsis |
| `08_diagnostico_residuos.png` | Residuos OLS: histograma + Q-Q |
| `09_predichos_vs_observados.png` | Predichos vs. observados (scatter OLS) |
| `10_heatmap_hospitales.png` | Efectos fijos por hospital (heatmap) |

---

## Parámetros estadísticos

| Parámetro | Valor | Justificación |
|-----------|-------|--------------|
| α (nivel de significancia) | 0.05 | Estándar en investigación biomédica |
| Corrección de Bonferroni | Sí (Dunn post-hoc) | Múltiples comparaciones pareadas |
| Mínimo casos por hospital | 30 | Varianza intra-hospital suficiente para efectos fijos |
| Corte P99 (dias_estadia) | Percentil 99 por grupo | Elimina extremos sin perder casos severos típicos |
| Muestra Shapiro-Wilk | 5000 (semilla=42) | Válido para n ≤ 5000; reproducible |

---

## Fuente de datos

GRD Público FONASA 2019–2024, en `../DATASET-PROBLEMA8/`:
- `GRD_PUBLICO_2019.csv` a `GRD_PUBLICO_2024.csv`
- Separador: `|` | Encoding: UTF-8
- ~5.8 millones de registros en total
