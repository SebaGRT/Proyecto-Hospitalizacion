# Retroalimentación Experta — Avance 2
## "Variabilidad en el Tratamiento Oncológico y sus Efectos sobre la Mortalidad y la Estadía Hospitalaria en el Sistema Público Chileno"

**Revisado por:** Evaluador externo (perfil: investigador senior, análisis de datos clínicos y políticas de salud)
**Fecha de revisión:** 29 de abril de 2026
**Entrega límite:** 8 de mayo de 2026 — quedan **9 días**
**Integrantes:** Vicente · José Tomás · Sebastián

---

## 1. Evaluación Global

Este proyecto se encuentra en un nivel técnico considerablemente superior al promedio esperado para un segundo avance de pregrado. La sofisticación metodológica es genuina: el uso de Kruskal-Wallis con post-hoc Dunn-Bonferroni, regresión logística con efectos fijos por hospital, errores robustos HC3 en OLS, y la integración del IDH comunal como variable contextual exceden con creces los requerimientos mínimos de la rúbrica.

Sin embargo, el proyecto enfrenta **brechas formales graves** que, de no corregirse antes del 8 de mayo, impactarán directamente la nota. La más crítica: **no existe informe escrito separado** (el requerimiento explícito es PDF + Word de hasta 10 páginas en formato APA). El cuaderno Jupyter, por muy bien documentado que esté, no reemplaza ese entregable.

### Nota estimada en estado actual: **5,2 / 7,0**
> El rigor analítico es notable (habría llegado a 6,5+), pero las brechas formales —informe escrito ausente, README faltante, figuras sin numerar, tablas categóricas ausentes, sección de referencias incompleta— arrastran la calificación. Subsanar los puntos críticos listados a continuación puede llevar el proyecto a **6,4–6,7**.

---

## 2. Lo que está muy bien (fortalezas a conservar)

| Aspecto | Detalle |
|---|---|
| **Diseño inferencial** | Tres hipótesis bien diferenciadas, con tests apropiados para cada tipo de variable y distribución. La justificación metodológica de Kruskal-Wallis sobre ANOVA es explícita y correcta. |
| **Modelos de regresión** | Efectos fijos por hospital en logit y OLS es la decisión correcta para datos de panel de establecimientos. HC3 en OLS indica comprensión real de heteroscedasticidad. |
| **Interpretación clínica** | Las celdas Markdown tienen razonamiento clínico de alto nivel: la distinción entre "efecto confusor por indicación" en procedimientos y la mediación del sexo vía urgencia es precisa. |
| **Análisis IDH (Tabla 3)** | La incorporación del IDH comunal y el análisis de correlación no lineal entre desarrollo socioeconómico y varianza hospitalaria va muy por encima del requerimiento. Es publicable como nota metodológica. |
| **Evaluación predictiva** | ROC + matriz de confusión + classification report + back-transform para el OLS: evaluación predictiva completa en ambos modelos. |
| **Forest plot completo** | El forest plot multi-covariable con escala logarítmica y anotación de significancia estadística es visualización de nivel publicación. |

---

## 3. Brechas Críticas — Lo que FALTA (corregir antes del 8 de mayo)

Cada ítem está marcado con prioridad: 🔴 bloqueante para la nota · 🟠 importante · 🟡 mejora.

### 🔴 BRECHA 1 — Informe escrito separado (PDF + Word)
**Qué falta:** El requerimiento §2.1 exige un informe de máx. 10 páginas en formato APA, entregado en PDF y Word, independiente del cuaderno.
**Qué hacer:** Redactar el informe usando las secciones del rubric como índice: portada → introducción/pregunta → descripción dataset → limpieza → EDA → tests → modelo → discusión → próximos pasos → referencias. El notebook ya contiene todo el contenido; es un ejercicio de síntesis y redacción formal, no de análisis nuevo. Tiempo estimado: 6–8 horas en grupo si se dividen secciones.

### 🔴 BRECHA 2 — README.md en el repositorio
**Qué falta:** El requerimiento §2.2 exige un README con nombre del proyecto, integrantes, fuente de datos e instrucciones de reproducción.
**Qué hacer:** Crear `README.md` en la raíz del repositorio. Debe incluir: (1) nombre y descripción de 2 líneas, (2) integrantes, (3) fuente: "GRD Público MINSAL/FONASA 2019–2024, disponible en [URL]", (4) instrucciones: "Instalar dependencias: `pip install -r requirements.txt`; ejecutar `Avance2_Proyecto_Final.ipynb` desde la raíz del repositorio". Tiempo estimado: 20 minutos.

### 🔴 BRECHA 3 — Sección de Referencias APA al final del notebook
**Qué falta:** Las citas a Wennberg, Munir et al. (2024) y Kamaraju et al. (2022) aparecen en texto corrido sin lista de referencias formal al final.
**Qué hacer:** Agregar una celda Markdown al final del notebook con encabezado `## Referencias` y lista en formato APA 7. Ejemplo:
```
Wennberg, J., & Gittelsohn, A. (1973). Small area variations in health care delivery. *Science*, *182*(4117), 1102–1108. https://doi.org/10.1126/science.182.4117.1102
```
Incluir también pandas, scikit-learn, statsmodels y MINSAL como fuentes de datos.

### 🟠 BRECHA 4 — Tabla de frecuencias de variables categóricas
**Qué falta:** El rubric §4.1 exige estadísticas descriptivas también para variables categóricas (frecuencias absolutas, porcentajes, tablas resumen). El EDA actual solo tiene descriptivas numéricas (Tabla 1).
**Qué hacer:** Agregar en la Sección 4.1 una tabla que muestre distribución de: `sexo`, `TIPO_INGRESO`, `TIPOALTA`, `region` (y `severidad_grd` como ordinal). Bastará un `.value_counts(normalize=True)` formateado como DataFrame con columnas `n` y `%`. Para C16.* específicamente, mostrar también distribución de subcódigos (C16.0–C16.9).

### 🟠 BRECHA 5 — Matriz de correlación entre variables numéricas
**Qué falta:** El rubric §4.2 menciona explícitamente "matriz de correlación si corresponde". Para este dataset corresponde.
**Qué hacer:** Agregar un heatmap de correlación de Spearman (no Pearson, dado que las variables son asimétricas) entre: `dias_estada`, `edad`, `cantidad_procedimientos`, `peso_grd`, `severidad_grd`, `comorbilidad`, `mortalidad_int`. Usar `df_focus` para hacerlo específico a C16.*. Interpretar las correlaciones más fuertes (|r| > 0.3) en relación a la pregunta de investigación.

### 🟠 BRECHA 6 — Figuras y tablas sin numeración
**Qué falta:** El rubric §11 exige "figuras y tablas numeradas". Ningún gráfico tiene número (Figura 1, Figura 2, etc.).
**Qué hacer:** Agregar numeración en los títulos de todos los gráficos y tablas: `ax.set_title('Figura 3. Distribución de Días de Estadía...')`. También numerar las tablas en las celdas Markdown precedentes: `**Tabla 2.** Estadísticas descriptivas por hospital`.

### 🟠 BRECHA 7 — Sección de introducción formal
**Qué falta:** El notebook pasa directamente del título al dataset sin una introducción (§3.2 del rubric exige: contexto del problema, justificación de relevancia, pregunta de investigación actualizada, hipótesis general explícita).
**Qué hacer:** Agregar una celda Markdown entre la portada (celda 0) y la descripción del dataset (celda 1) con los siguientes elementos: (a) párrafo de contexto (~3 oraciones sobre variabilidad oncológica en sistemas públicos), (b) pregunta de investigación como oración interrogativa directa, (c) hipótesis general como oración declarativa ("Se hipotetiza que el hospital de atención es un determinante independiente y significativo de la mortalidad intrahospitalaria y la duración de estadía en pacientes con cáncer gástrico, incluso controlando por severidad clínica del caso"), (d) justificación de por qué los datos GRD permiten responderla.

### 🟡 BRECHA 8 — Verificación de duplicados ausente en limpieza
**Qué falta:** El rubric §3.4 exige documentar la identificación y tratamiento de duplicados.
**Qué hacer:** Agregar una línea en la sección de limpieza: `n_dup = df.duplicated().sum(); print(f'Registros duplicados: {n_dup}')`. Aunque no haya duplicados, documentar el chequeo es obligatorio.

### 🟡 BRECHA 9 — Evaluación crítica incompleta del modelo de regresión
**Qué falta:** El rubric §6.5 exige "posibles problemas de interpretación, limitaciones del modelo, variables importantes que podrían faltar". La celda de interpretación OLS menciona los coeficientes pero no discute multicolinealidad (VIF), ni homocedasticidad de residuos, ni si la distribución de residuos es aproximadamente normal.
**Qué hacer:** Agregar, al final de la sección H₃, una celda Markdown con análisis crítico: (1) reportar VIF para las variables del modelo (hay código en statsmodels), (2) mencionar explícitamente que la ausencia de estadio TNM, tipo histológico e índice de Charlson es una limitación importante del modelo, (3) comentar que los efectos fijos de hospital absorben heterogeneidad no observada pero impiden interpretar el efecto de características del hospital.

### 🟡 BRECHA 10 — Discusión no responde explícitamente las 6 preguntas del rubric
**Qué falta:** §7 del rubric pide responder seis preguntas específicas. La discusión actual integra bien los resultados pero no las responde estructuradamente.
**Qué hacer:** Reestructurar la celda de discusión con 6 sub-apartados explícitos usando negritas o subtítulos: **Hallazgos más importantes**, **¿Los resultados apoyan la hipótesis?**, **Patrones relevantes observados**, **Hallazgos inesperados**, **Limitaciones del análisis**, **¿Qué falta para el avance final?**

---

## 4. Observaciones de mejora por sección

### Sección 3 — Limpieza
- El corte en P99 para estadía es razonable, pero **no se documenta cuántos registros con `mortalidad = NaN` (si los hay) se eliminan**. Agregar conteo.
- La variable `comorbilidad` se calcula a nivel global (`df`) pero su distribución descriptiva específica en `df_focus` (C16.*) no aparece como parte del EDA. Agregar una fila de descriptiva de `comorbilidad` en la Tabla 1 del subconjunto focal.

### Sección 4 — EDA
- El boxplot de días por hospital es potente, pero no incluye la mediana como anotación numérica sobre cada caja. Un lector del informe escrito que vea la figura sin el código no puede leer los valores exactos. Añadir `ax.text()` con la mediana sobre cada caja o incluir una tabla complementaria.
- Falta un análisis de distribución por **año** (2019–2024). El dataset cubre 6 años. La pandemia COVID-19 (2020–2021) probablemente alteró patrones de hospitalización oncológica. Si el análisis temporal no se controla, hay un confusor no reconocido. Como mínimo, agregar una celda que muestre el volumen anual y note si hay disrupción visible.

### Sección 5 — Tests de hipótesis
- Los Escenarios A, B y C son correctos, pero **la justificación del test Chi-cuadrado vs. Fisher exact está implícita** (el código elige automáticamente). En el informe escrito, explicitar el criterio (frecuencia esperada mínima < 5 → Fisher; si no → Chi²).
- El post-hoc Dunn-Bonferroni es la elección correcta, pero en el informe debe reportarse cuántos pares del total son significativos y **cuál es el par con mayor diferencia de rangos** (el más clínicamente relevante), no solo el heatmap de p-valores.

### Sección 6 — Modelos de regresión
- **Logístico:** La tasa de mortalidad ~5% genera un problema de desbalance de clases severo. Aunque se menciona en la interpretación de la matriz de confusión, no se discute ni evalúa el uso de `class_weight='balanced'` o remuestreo (SMOTE). Para el avance final, al menos mencionar esta limitación y su efecto en el recall de la clase positiva.
- **OLS:** La `tabla_ols` que se muestra en el notebook incluye las dummies de hospital pero la celda de interpretación de coeficientes solo comenta las variables continuas. El rubric exige interpretar al menos 3 coeficientes; asegurarse de que esa interpretación quede explícita y en lenguaje aplicado (no solo "coef = 0.034, p < 0.001").

---

## 5. Recomendaciones para el Avance Final (Avance 3)

Estas van más allá del Avance 2 pero deben figurar en la sección "Próximos pasos":

1. **Modelo multinivel (mixed effects):** Tratar hospital como efecto aleatorio en lugar de fijo. Esto permite estimar la varianza entre hospitales (ICC) y hacer inferencia sobre el "efecto hospital" en la población de hospitales, no solo en la muestra. Paquete recomendado: `pymer4` o `statsmodels.MixedLM`.

2. **Control por año:** Incluir `_anio` (disponible en el dataset) como variable de control o efecto fijo temporal en los modelos. Sin esto, los efectos estimados pueden estar confundidos por tendencias temporales o por el impacto de la pandemia.

3. **Análisis de sensibilidad del umbral de corte (P99):** Mostrar que los resultados principales son robustos bajo P95 y P97 como puntos alternativos de corte de outliers.

4. **Análisis de supervivencia (Kaplan-Meier):** Si el dataset incluye fecha de ingreso y fecha de alta, un análisis de tiempo hasta el alta o tiempo hasta la muerte agregaría una dimensión longitudinal que enriquece la narrativa de variabilidad hospitalaria.

5. **Visualización geográfica:** Un mapa de calor a nivel región/servicio de salud de la mortalidad promedio y la varianza de estadía sería el gráfico más comunicativo para la presentación oral.

---

## 6. Lista de verificación antes de entregar

```
[ ] Informe escrito en PDF y Word (máx. 10 páginas, APA)
[ ] README.md creado y pusheado al repositorio
[ ] Sección ## Referencias al final del notebook (APA 7)
[ ] Tabla de frecuencias de variables categóricas (sexo, tipo_ingreso, tipoalta, region)
[ ] Matriz de correlación de Spearman (heatmap) para variables numéricas de C16.*
[ ] Todas las figuras y tablas numeradas (Figura 1, Figura 2..., Tabla 1, Tabla 2...)
[ ] Celda de Introducción con pregunta de investigación + hipótesis general explícita
[ ] Chequeo de duplicados documentado en sección de limpieza
[ ] Discusión reestructurada en 6 sub-apartados
[ ] VIF reportado para el modelo OLS
[ ] Distribución de comorbilidad en df_focus incluida en EDA
[ ] Análisis de volumen por año (2019–2024) incluido
```

---

*Este documento fue preparado con el propósito de maximizar la calidad del entregable final. La profundidad analítica del equipo es real y merece materializarse en una nota que la refleje. Los puntos críticos son exclusivamente de forma y presentación, no de sustancia estadística — esa parte ya está hecha y está bien hecha.*
