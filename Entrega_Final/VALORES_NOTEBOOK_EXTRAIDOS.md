# VALORES EXTRAÍDOS DEL NOTEBOOK — Fuente de Verdad para Corrección del Docx

**Notebook:** `Proyecto_Final_AmatHerreraRodriguez.ipynb`
**Nota importante:** El notebook tiene discrepancias INTERNAS entre el texto descriptivo (markdown narrativo) y los outputs reales de código. Los valores de **código ejecutado (output)** son los CORRECTOS; los valores en markdown narrativo a veces no fueron actualizados.

---

## SECCIÓN 3 — DATASET

| Variable | Valor (output código) | Markdown narrativo (⚠ puede no coincidir) |
|---|---|---|
| Universo total inicial cargado | **454,235** (48 variables) | "aproximadamente 14 millones de egresos" |
| Universo oncológico C00-D49 (filtro CIE-10) | **419,510** | 457,717 o 422,894 (tabla) |
| Universo oncológico C00-D49 (post-limpieza) | **414,755** | 418,572 |
| Cáncer gástrico C16.* | **17,096** (4.1% del oncológico) | 17,335 |
| Dataset regresión (top 15 hospitales, ≥30 casos) | **9,740** | 9,855 |
| Mortalidad universo oncológico C00-D49 | **4.95%** | — |
| Mortalidad C16.* | **5.56%** | 7.07% (en texto markdown antiguo) |
| Mortalidad dataset regresión | **3.99%** | ~4% |

---

## SECCIÓN 4 — LIMPIEZA (Pipeline de filtros)

| Paso | Operación | N resultante | Δ |
|---|---|---|---|
| Carga inicial | Lectura del dataset | **454,235** | — |
| Filtro 1 | CIE-10: restringir a C00-D49 | **419,510** | −34,725 |
| Filtro 2 | Exclusión obstétrica (TIPO_INGRESO) | **418,442** | −1,068 |
| Filtro 3 | Truncar dias_estada a P99 (= **23 días**), dropna, fix types | **414,755** | −3,687 |
| Filtro 4 | Subset focal: diagnóstico C16.* | **17,096** | — |
| Filtro 5 | Top 15 hospitales con ≥30 casos | **9,740** | — |

⚠ La **tabla markdown** en el notebook (Sección 3) muestra números DIFERENTES: 422,894 / 421,824 / 418,572 / 17,335 / 9,855. Esos son valores DE PLANIFICACIÓN o de una versión anterior. Los valores del **código ejecutado** arriba son los correctos.

---

## SECCIÓN 5.1 — ESTADÍSTICA DESCRIPTIVA (C16.*, N=17,096)

| Variable | Valor |
|---|---|
| Edad | **66.5 ± 12.0** años (media ± DE) |
| % Masculino | **67.3%** |
| Estadía (mediana) | **5** días |
| Estadía (media ± DE) | **6.24 ± 5.77** días |
| Procedimientos (media ± DE) | **8.82 ± 6.47** |
| Procedimientos (mediana) | **8.00** |
| Severidad GRD (media ± DE) | **1.56 ± 0.99** |
| Peso GRD (media ± DE) | **1.20 ± 0.81** |
| Comorbilidad (media ± DE) | **4.37 ± 3.72** |
| Comorbilidad (mediana) | **4.00** (máx: 34) |
| Mortalidad | **5.56%** |
| % Ingreso urgencia | **35.2%** |

---

## SECCIÓN 5.3 — ANÁLISIS BIVARIADO (Chi-cuadrado)

### Escenario A: Mortalidad × Sexo

| Estadístico | Valor |
|---|---|
| χ² | **2.72** |
| df | **1** |
| p | **0.0990** (NO significativo al 5%) |
| OR | **1.13** |
| IC 95% | **[0.98; 1.30]** |
| N | 17,096 |

### Escenario B: Mortalidad × Tipo de Ingreso (sin obstétrica)

| Estadístico | Valor |
|---|---|
| χ² | **1209.36** |
| df | **1** |
| p | **< 0.001** (~0.000000) |
| N | 17,096 |
| OR | No calculado explícitamente en código (diferencia extrema: PROGRAMADA 1.06% vs URGENCIA 13.82%) |

### Escenario C: Tipo de Alta × Top 10 Hospitales

| Estadístico | Valor |
|---|---|
| χ² | **248.53** |
| df | **81** |
| p | **< 0.001** (~0.000000) |
| N | 7,379 (top 10 hospitales) |

---

## SECCIÓN 6.2 — KRUSKAL-WALLIS (H₁: diferencias entre hospitales)

| Estadístico | Valor |
|---|---|
| Shapiro-Wilk W | **0.916** (n=5000 muestra), p < 0.001 |
| H de Kruskal-Wallis | **1909.40** |
| df | **24** |
| p | **< 0.001** |
| N | **12,925** (25 hospitales, cada uno ≥30 casos) |
| ε² (tamaño del efecto) | **0.146** (efecto GRANDE, ≥0.14) |
| Dunn-Bonferroni | **189** de 300 pares significativos = **63.0%** |

⚠ **EL NOTEBOOK SE CONTRADICE A SÍ MISMO:** La Sección 8 (Conclusiones) en markdown dice "H = 412.3, p < 0.001, ε² = 0.08". El código ejecutado da **H = 1909.40, ε² = 0.146**. El valor del código es el correcto — la conclusión markdown NO fue actualizada.

---

## SECCIÓN 6.3 — REGRESIÓN LOGÍSTICA (H₂: mortalidad)

| Parámetro | Valor |
|---|---|
| N | **9,740** |
| AUC-ROC | **0.849** |
| Pseudo-R² McFadden | **0.259** |

### Predictores:

| Predictor | OR | IC 95% | p |
|---|---|---|---|
| Procedimientos | **1.053** | [1.030; 1.076] | **< 0.001** *** |
| Edad | **0.999** | [0.990; 1.008] | **0.7804** (ns) |
| Severidad GRD | **5.973** | [4.811; 7.415] | **< 0.001** *** |
| Peso GRD | **0.693** | [0.597; 0.805] | **< 0.001** *** |
| Comorbilidad | **1.064** | [1.032; 1.097] | **< 0.001** *** |

**Matriz de confusión (umbral 0.5):** TN=2,334 | FP=4 | FN=97 | TP=0
- Sensibilidad = **0.0%**, Especificidad = **99.8%** (el modelo predice 0 para todos por desbalance de clases)

⚠ La Sección 8 (Conclusiones) dice "AUC-ROC = 0.823" y "OR procedimientos = 1.05 [1.02–1.08]". Son valores REDONDEADOS/ANTIGUOS en markdown. Los valores reales del código son AUC=**0.849** y OR=**1.053 [1.030–1.076]**.

---

## SECCIÓN 6.4 — OLS (H₃: días de estadía)

| Parámetro global | Valor |
|---|---|
| N | **9,740** |
| R² | **0.6332** |
| R² ajustado | **0.6325** |
| F | **948.40** (F(19, 9720)) |
| p (modelo) | **≈ 0** |
| MAE | **3.29** días |
| RMSE | **5.41** días |

### Predictores (con errores HC3):

| Predictor | β | SE | IC 95% | p | Dirección |
|---|---|---|---|---|---|
| Intercepto | **0.5510** | — | [0.461; 0.641] | **< 0.001** *** | + |
| Procedimientos | **0.0924** | 0.0013 | — | **< 0.001** *** | **POSITIVO** (+9.68%) |
| Edad | **0.0013** | 0.0005 | — | **0.0157** * | **POSITIVO** (+0.13%) |
| Severidad GRD | **0.3603** | 0.0100 | — | **< 0.001** *** | **POSITIVO** (+43.37%) |
| Peso GRD | **0.0007** | 0.0121 | — | **0.9550** ns | **NO significativo** |
| Comorbilidad | **−0.0324** | 0.0023 | — | **< 0.001** *** | **NEGATIVO** (−3.18%) |

🔴 **DISCREPANCIAS CRÍTICAS vs docx:**

1. **Peso GRD NO es significativo**: p = **0.9550** (ns). Si el docx dice que es significativo o tiene signo, está MAL.
2. **Comorbilidad tiene signo NEGATIVO**: β = **−0.0324**. Si el docx reporta β positivo, está MAL.
3. **R² = 0.6332** y **R² ajustado = 0.6325**. Si el docx reporta otros valores, están MAL.

---

## SECCIÓN 6.5 — MODELO MULTINIVEL (Intercepto aleatorio por hospital)

| Parámetro | Valor |
|---|---|
| σ²_hospital (varianza intercepto) | **0.000000** |
| σ²_residual | **0.358788** |
| **ICC** | **0.0000 (0.00%)** |
| N hospitales | **15** |
| N observaciones | **9,740** |

### Notas sobre convergencia:
- **Convergencia exitosa** técnicamente, pero la varianza del efecto aleatorio colapsó a cero (estructura de covarianza singular).
- Log-Likelihood reportado como **"inf"** (inestable por varianza cero).
- **No hay efecto hospital detectable** en el modelo multinivel → ICC = 0.
- No se reporta Likelihood Ratio Test vs modelo nulo.

### Efectos fijos (idénticos a OLS):

| Predictor | β | SE | p |
|---|---|---|---|
| Procedimientos | **0.0924** | 0.0013 | < 0.001 *** |
| Edad | **0.0013** | 0.0005 | **0.0129** * |
| Severidad GRD | **0.3603** | 0.0100 | < 0.001 *** |
| Peso GRD | **0.0007** | 0.0121 | **0.9512** ns |
| Comorbilidad | **−0.0324** | 0.0023 | < 0.001 *** |
| Intercepto | NaN | — | — |

---

## SECCIÓN 6.6 — IDH (Análisis geográfico-socioeconómico)

🔴 **ATENCIÓN: Los p-values en el docx están INVENTADOS o son de otra versión.**

| Estadístico | Valor |
|---|---|
| % vinculación exitosa | **99.78%** (17,059 / 17,096) |
| ⚠ Nota | IDH **simulado** por región, no IDH real |

### Kruskal-Wallis: Estadía × Quintil IDH

| Estadístico | Valor |
|---|---|
| H | **6.2338** |
| df | **4** |
| p | **0.1824** ← **NO SIGNIFICATIVO** |
| Conclusión | No hay diferencias en estadía por quintil de IDH |

### Chi-cuadrado: Mortalidad × Quintil IDH

| Estadístico | Valor |
|---|---|
| χ² | **5.2290** |
| df | **4** |
| p | **0.2646** ← **NO SIGNIFICATIVO** |
| Conclusión | No hay asociación entre mortalidad y quintil de IDH |

### Mortalidad y N por quintil:

| Quintil | Mortalidad | N |
|---|---|---|
| Q1 (IDH más bajo) | **5.07%** | 3,412 |
| Q2 | **5.33%** | 3,412 |
| Q3 | **5.34%** | 3,411 |
| Q4 | **5.89%** | 3,412 |
| Q5 (IDH más alto) | **6.15%** | 3,412 |

- **Medianas de estadía:** Todas idénticas = **5.0 días** en todos los quintiles.

🔴 Si el docx reporta:
- p < 0.001 o p = 0.003 para IDH → **FALSO**. Los valores reales son p = **0.1824** y p = **0.2646** (NO significativos).
- "Diferencias significativas por IDH" → **FALSO**. No las hay.

---

## SECCIÓN 6.7 — K-MEANS (Clustering de hospitales)

| Parámetro | Valor |
|---|---|
| k óptimo | **2** |
| Silhouette score | **0.407** |
| Método | Silhouette + elbow (rango [2, 4]) |
| Varianza PCA explicada | PC1 = **58.3%**, PC2 = **19.9%** (total: **78.2%**) |
| Inercia final | **163.94** |

### Centroides:

| Variable | Cluster 0 | Cluster 1 |
|---|---|---|
| Mortalidad | **16.13%** | **5.14%** |
| Días estadía | **7.29** | **6.27** |
| Procedimientos | **11.09** | **8.27** |
| % Urgencia | **82.27%** | **34.40%** |
| Severidad GRD | **2.14** | **1.56** |
| N hospitales | **18** | **39** |
| N casos totales | **2,108** | **14,854** |

⚠ El texto markdown de la sección describe 3 clusters, pero el **código ejecutado seleccionó k = 2**. El docx debe usar k=2.

---

## SECCIÓN 7 — DISCUSIÓN (Valores numéricos mencionados)

| Frase/contexto | Número |
|---|---|
| "AUC > 0.80. La curva ROC muestra que discrimina adecuadamente" | AUC > 0.80 |
| "Cada procedimiento adicional se asocia con ~9.7% más días de estadía" | +9.7% |
| "R² de 0.64. La combinación de factores clínicos e institucionales explica..." | R² = 0.64 |
| Misma sección también menciona: AUC > 0.80, +9.7%, R² = 0.64 (repetido en 7.1) | — |
| Gradiente socioeconómico Q1 vs Q5 (7.2) | Q1, Q5 (sin cifras específicas) |
| ICC del modelo multinivel (7.2) | "fracción no trivial" (pero ICC real = 0.00%) — ⚠ CONTRADICE |
| Limitaciones: TNM, histología, ECOG (7.3) | Solo menciones cualitativas |
| **Conclusiones (Secc 8)**: "H = 412.3, p < 0.001, ε² = 0.08" | ⚠ DISCREPA con código (H=1909.40, ε²=0.146) |
| "más del 60% de pares de hospitales con diferencias significativas" | 63% ✅ (coincide) |
| "OR = 1.05; IC95%: 1.02–1.08; p < 0.001" | ⚠ DISCREPA: código da OR=1.053 [1.030;1.076] |
| "OR = 6.12" para severidad | ⚠ DISCREPA: código da OR=5.973 |
| "AUC-ROC = 0.823" | ⚠ DISCREPA: código da AUC=0.849 |
| "~4% de mortalidad" (desbalance de clase) | 3.99% ✅ |
| "β = 0.093; p < 0.001. Cada procedimiento adicional: +9.7% estadía" | β = 0.0924 ✅ |

---

## RESUMEN DE DISCREPANCIAS CRÍTICAS DOCX vs NOTEBOOK

| # | Sección | Docx (según Oracle) | Notebook (código real) | Gravedad |
|---|---|---|---|---|
| 1 | IDH: KW estadía | p < 0.001 o p = 0.003 | **p = 0.1824** (NO signif.) | 🔴 CRÍTICO |
| 2 | IDH: χ² mortalidad | p < 0.001 o p = 0.003 | **p = 0.2646** (NO signif.) | 🔴 CRÍTICO |
| 3 | OLS: signo comorbilidad | Positivo (según docx) | **NEGATIVO** (β = −0.0324) | 🔴 CRÍTICO |
| 4 | OLS: significancia Peso GRD | Significativo (según docx) | **p = 0.9550** (NO signif.) | 🔴 CRÍTICO |
| 5 | K-means: número de clusters | ¿k=3? | **k=2** (silhouette = 0.407) | 🟡 ALTO |
| 6 | Multinivel: ICC | Posiblemente reportado > 0 | **ICC = 0.0000** | 🔴 CRÍTICO |
| 7 | Logística: AUC | Posiblemente 0.823 (Conclusión) | **0.849** (código) | 🟡 ALTO |
| 8 | KW H₁: H y ε² | Posiblemente H=412.3, ε²=0.08 | **H=1909.40, ε²=0.146** | 🟡 ALTO |
| 9 | Conteos pipeline | Posiblemente 418,572 / 17,335 | **414,755 / 17,096** | 🟡 ALTO |
| 10 | ~25 discrepancias adicionales | Varios chi², descriptivos | Ver valores arriba | Varios |

---

**Los valores presentados en este documento provienen de las celdas de output del código ejecutado en el notebook.** Son los valores que deben aparecer en el docx corregido.
