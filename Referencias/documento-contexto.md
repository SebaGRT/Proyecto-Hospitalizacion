# Contexto: Incorporación de Análisis Inferencial en Advance2

> **Sesión origen:** 2026-04-29
> **Objetivo:** Integrar los tests de hipótesis del notebook `03_Analisis_Inferencial.ipynb` en `Advance2_Analisis_Completo.ipynb`, con enfoque exclusivamente oncológico y considerando el notebook de referencia de regresión.

---

## 1. Descripción de los Notebooks Analizados

### 1.1 `Advance2_Analisis_Completo.ipynb` (Notebook destino)
- **Dataset:** `GRD_Limpio.csv` (~454 k registros, 2019–2024), autogenerado desde CSVs anuales.
- **Foco:** Oncología C00–D49 (CIE-10). EDA bivariado con énfasis en **C16.* (cáncer gástrico)**.
- **Variables clave:**
  - `hospital` (COD_HOSPITAL)
  - `cantidad_procedimientos` (conteo de PROCEDIMIENTO1..30)
  - `mortalidad` (booleano derivado de TIPOALTA)
  - `dias_estada` (FECHAALTA − FECHA_INGRESO)
  - `peso_grd`, `severidad_grd`, `edad`, `sexo`, `tipo_ingreso`
- **Análisis actual:** 100% descriptivo y exploratorio (EDA). **Sin tests de hipótesis ni modelos inferenciales.**
- **Outputs:** `outputs/graficos/` y `outputs/tablas/`

### 1.2 `03_Analisis_Inferencial.ipynb` (Notebook fuente de tests)
- **Dataset:** `df_clean_final_2019_2024.csv` (~441 k registros)
- **Foco:** Dos grupos diagnósticos — **Neoplasias** (C50, C18–C20, C53, C34) y **Sepsis** (A40, A41)
- **Tests implementados:**
  | Sección | Test | Variable dependiente |
  |---|---|---|
  | 2 | Chi-cuadrado (4 escenarios: A, B, C, D) | Variables categóricas (mortalidad, tipo de alta) |
  | 3 | Kruskal-Wallis + Dunn-Bonferroni | `n_procedimientos` por hospital (H1) |
  | 4 | Regresión Logística (OR + IC95%) | `mortalidad_intrahospitalaria` (H2) |
  | 5 | Regresión OLS con log-transform | `dias_estada` (H3) |
- **Tamaño de efecto:** ε² (Kruskal-Wallis), V de Cramér (Chi-cuadrado), Pseudo-R² McFadden (logit)
- **Outputs:** `advance_3/outputs/`

### 1.3 `Notebook_Final_Regresion_Hospitalizaciones.ipynb` (Notebook de referencia)
- **Tipo:** Notebook didáctico del curso
- **Aportes clave:**
  - Interpretación estructurada de modelos (coeficientes, intercepto, F-statistic, R², AIC/BIC)
  - **Evaluación predictiva:** train/test split (`sklearn`), MAE, RMSE, R² en test
  - Uso de `statsmodels` con fórmulas tipo R y variables categóricas (`C(variable)`)

---

## 2. Mapeo de Variables (03 → Advance2)

| Variable en 03 | Variable en Advance2 | Acción requerida |
|---|---|---|
| `n_procedimientos` | `cantidad_procedimientos` | Renombrar o crear alias |
| `mortalidad_intrahospitalaria` (0/1) | `mortalidad` (bool) | Convertir a entero (`astype(int)`) |
| `COD_HOSPITAL` | `hospital` | Renombrar |
| `peso_relativo_grd` | `peso_grd` | Renombrar |
| `severidad_grd` | `severidad_grd` | Mismo nombre; verificar tipo numérico |
| `edad` | `edad` | Mismo nombre; verificar tipo numérico |
| `TIPO_INGRESO` | `tipo_ingreso` / `TIPO_INGRESO` | Verificar nombre real en Advance2 |
| `TIPOALTA` | `tipo_alta` / `TIPOALTA` | Verificar nombre real en Advance2 |

> **Nota:** Advance2 usa el dataset limpio `GRD_Limpio.csv`, por lo que no es necesario replicar la carga desde `df_clean_final_2019_2024.csv` si ambos datasets son equivalentes. Verificar que las columnas necesarias existan.

---

## 3. Rediseño Propuesto: Análisis Inferencial Exclusivamente Oncológico

### 3.1 Principios del rediseño
1. **Eliminar el enfoque dual Neoplasia/Sepsis** y centrar todo en el universo oncológico C00–D49.
2. **Mantener coherencia con el EDA de Advance2**, que ya filtra oncología y profundiza en C16.*
3. **Incorporar la evaluación predictiva** del notebook de referencia (train/test, MAE/RMSE, AUC-ROC) para robustecer la validación de modelos.
4. **Adaptar los escenarios categóricos** a las variables disponibles en el dataset oncológico.

### 3.2 Estructura propuesta para la nueva sección

Insertar al final de `Advance2_Analisis_Completo.ipynb`, después de la sección 4.2 (EDA bivariado C16.*):

```
## 5. Análisis Inferencial

### 5.1 Preparación inferencial
- Mapeo de variables (renombrar para compatibilidad con funciones de 03)
- Filtrado de hospitales con volumen mínimo (MIN_CASOS_HOSPITAL, ej. ≥20 casos)
- Creación de subconjuntos: todo oncología (C00–D49) y opcionalmente C16.*
- Definición de constantes: ALPHA = 0.05, SEMILLA = 42, TOP_HOSP_KW, TOP_HOSP_REG

### 5.2 Variables Categóricas — Tests Chi-Cuadrado
**Escenarios adaptados a oncología:**
- **Escenario A:** Mortalidad × Sexo (2×2) → χ² + Odds Ratio + Fisher exacta
- **Escenario B:** Mortalidad × Tipo de ingreso (2×3) → χ² + post-hoc Bonferroni + V de Cramér
- **Escenario C:** Tipo de alta (5 categorías) × Hospital top N → χ² + residuos estandarizados + heatmap
- (Opcional) **Escenario D:** Tipo de alta × Tipo de ingreso (3×5) → χ² + residuos

> Nota: El Escenario A original (Mortalidad × Grupo diagnóstico 2×2) no aplica porque Advance2 ya filtra solo oncología.

### 5.3 Hipótesis 1 — Variabilidad en intensidad de procedimientos por hospital
**Pregunta:** ¿La distribución del número de procedimientos difiere entre hospitales para pacientes oncológicos?

**Flujo:**
1. Shapiro-Wilk sobre `cantidad_procedimientos` (submuestra si n > 5000)
2. Kruskal-Wallis (H, p, ε²) con top N hospitales por volumen
3. Si p < 0.05: Dunn post-hoc con corrección Bonferroni
4. Heatmap de p-valores ajustados (top 15 hospitales)
5. Exportar tabla CSV y gráfico

**Funciones a migrar de 03:** `test_shapiro`, `kruskal_wallis`, `dunn_bonferroni`, `preparar_grupos_kw`

### 5.4 Hipótesis 2 — Mortalidad intrahospitalaria (Regresión Logística)
**Pregunta:** ¿El número de procedimientos se asocia con la probabilidad de mortalidad, controlando por severidad, edad y hospital?

**Modelo:**
```
logit(mortalidad) = β₀ + β₁·cantidad_procedimientos + β₂·edad + β₃·severidad_grd
                    + β₄·peso_grd + Σ γₖ·C(hospital)ₖ + ε
```

**Flujo:**
1. Ajuste del modelo logístico con `statsmodels` (`method='bfgs'`)
2. Extracción de OR + IC95% + p-valores
3. Reporte de Pseudo-R² (McFadden), AIC, Log-Likelihood
4. **(Nuevo, del referencia)** Train/test split + AUC-ROC + matriz de confusión
5. Forest plot del OR de `cantidad_procedimientos`
6. Exportar tabla CSV

**Funciones a migrar de 03:** `ajustar_logit` (adaptada a nombres de variables de Advance2)

### 5.5 Hipótesis 3 — Días de estadía (Regresión OLS)
**Pregunta:** ¿El número de procedimientos predice la duración de la estadía, controlando por severidad y hospital?

**Modelo:**
```
log(1 + dias_estada) = β₀ + β₁·cantidad_procedimientos + β₂·edad + β₃·severidad_grd
                       + β₄·peso_grd + Σ γₖ·C(hospital)ₖ + ε
```

**Flujo:**
1. Verificar asimetría de `dias_estada`; aplicar `log(1 + x)` si |asimetría| > 1.0
2. Ajuste OLS con errores robustos HC3 (`cov_type='HC3'`)
3. Reporte de coeficientes β + IC95% + p-valores + R² ajustado + F-statistic
4. **(Nuevo, del referencia)** Train/test split + MAE + RMSE + R² en test
5. Gráfico comparativo de coeficientes
6. Exportar tabla CSV

**Funciones a migrar de 03:** `ajustar_ols` (adaptada a nombres de variables de Advance2)

### 5.6 Síntesis Ejecutiva
- Tabla resumen de los 3 modelos (H1, H2, H3)
- Interpretación clínica integrada
- Lista de outputs generados

---

## 4. Funciones a Migrar desde `03_Analisis_Inferencial.ipynb`

Copiar estas funciones a la celda de funciones auxiliares de Advance2 (sección 2.1):

### 4.1 Tests categóricos
- `fmt_p(p)` — formato APA 7 para p-valores
- `cramers_v(table)` — tamaño del efecto V de Cramér
- `interpret_cramers_v(v, r, c)` — interpretación cualitativa del efecto
- `standardized_residuals(table)` — residuos estandarizados para localizar celdas

### 4.2 Tests no paramétricos
- `test_shapiro(arr, semilla, n_max)` — test de normalidad con submuestra
- `kruskal_wallis(grupos_vals, nombres_grupos)` — test H + ε²
- `dunn_bonferroni(grupos_vals, nombres_grupos)` — post-hoc Dunn con corrección
- `preparar_grupos_kw(df, variable, top_n, min_n)` — preparar grupos para KW

### 4.3 Modelos de regresión
- `ajustar_logit(df_grp, nombre_grupo, top_n, min_n)` — logística con OR
- `ajustar_ols(df_grp, var_dep, nombre_grupo, top_n, min_n)` — OLS con HC3

---

## 5. Decisiones Pendientes

Antes de implementar, resolver:

1. **¿Sobre qué universo corren los tests?**
   - [ ] Todo el dataset oncológico C00–D49 (~454 k registros)
   - [ ] Solo el subgrupo C16.* (cáncer gástrico) (~17 k registros), coherente con el EDA bivariado de Advance2
   - [ ] Ambos: primero C00–D49 y luego C16.* como análisis focal

2. **¿Incorporar evaluación predictiva?**
   - [ ] Sí: agregar train/test split, MAE/RMSE (OLS), AUC-ROC (logit) — como en el referencia
   - [ ] No: mantener solo enfoque explicativo/inferencial (estilo 03)

3. **¿Comparar subgrupos oncológicos dentro de los tests?**
   - [ ] Sí: por ejemplo, C16.* vs otros cánceres, o por sitio anatómico
   - [ ] No: analizar el conjunto oncológico sin segmentar

4. **¿Crear un notebook separado o integrar en Advance2?**
   - [ ] Integrar todo en `Advance2_Analisis_Completo.ipynb`
   - [ ] Crear notebook aparte (ej. `Advance2_Analisis_Inferencial.ipynb`) que importe el dataset limpio

---

## 6. Notas Técnicas

- **Colinealidad hospital-severidad:** El notebook 03 menciona esta limitación. En Advance2, `severidad_grd` y `peso_grd` pueden estar correlacionadas con el hospital (complejidad). Considerar VIF si se reporta la colinealidad.
- **Eventos raros:** Si la mortalidad en C16.* es muy baja, el modelo logístico puede tener problemas de convergencia. Verificar tasa de mortalidad antes del ajuste.
- **Outputs:** Se sugiere crear carpeta `outputs/inferencial/` para no mezclar con descriptivos.
- **Reproducibilidad:** Usar `np.random.default_rng(SEMILLA)` en todas las funciones que usen aleatoriedad (submuestras, train/test split).

---

## 7. Archivos Relacionados

| Ruta | Descripción |
|---|---|
| `Avance 2/Advance2_Analisis_Completo.ipynb` | Notebook destino (EDA descriptivo) |
| `Avance 2/03_Analisis_Inferencial.ipynb` | Notebook fuente (tests H1-H3) |
| `Referencias/Notebook_Final_Regresion_Hospitalizaciones.ipynb` | Referencia didáctico (evaluación predictiva) |
| `DATASET INICIAL/GRD_Limpio.csv` | Dataset limpio usado por Advance2 |
| `DATASET INICIAL/df_clean_final_2019_2024.csv` | Dataset limpio usado por 03 |

---

*Documento generado automáticamente para continuar la tarea en una sesión posterior.*
