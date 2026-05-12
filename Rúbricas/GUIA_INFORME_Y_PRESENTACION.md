# Guía de Estructura: Informe Escrito y Presentación Oral
## Variabilidad en el Tratamiento Oncológico — Proyecto Final

**Equipo:** Vicente Rodríguez · José Tomás Amat · Sebastián Herrera  
**Curso:** Análisis de Datos e Inferencia Estadística — UDD Ingeniería 2026

---

# PARTE I — INFORME ESCRITO

> **Criterios de evaluación (peso 25% de la nota final):**
> Planteamiento del problema (10%) · Hipótesis y objetivos (10%) · Metodología (15%) · Resultados (20%) · Discusión (15%) · Conclusiones (10%) · Calidad de redacción (10%) · Integración Avances 1 y 2 (10%)

---

## Estructura del informe (máximo 12 páginas sin anexos)

---

### PORTADA

```
Título:   Variabilidad en el Tratamiento Oncológico y sus Efectos sobre la
          Mortalidad y la Estadía Hospitalaria en el Sistema Público Chileno
Equipo:   Vicente Rodríguez · José Tomás Amat · Sebastián Herrera
Curso:    Análisis de Datos e Inferencia Estadística — Ingeniería UDD
Fecha:    Mayo 2026
```

---

### 1. RESUMEN / ABSTRACT (½ página)

Redacta 150–200 palabras que cubran:
- **Contexto:** segunda causa de muerte en Chile; cáncer gástrico lidera en el sistema público.
- **Pregunta:** ¿determina el hospital los resultados clínicos más allá del perfil del paciente?
- **Datos:** GRD MINSAL/FONASA 2019–2024; ~9 855 egresos C16.* en 15 hospitales.
- **Métodos:** EDA, Kruskal-Wallis + Dunn, regresión logística, OLS HC3, modelo multinivel (ICC), análisis IDH, K-means.
- **Hallazgos clave:** variabilidad estructural confirmada (H₁); OR procedimientos = 1.05 sobre mortalidad (H₂); R² = 0.636 en días de estadía (H₃); ICC confirma efecto institucional real; gradiente socioeconómico Q1→Q5 en mortalidad.
- **Conclusión:** la variabilidad hospitalaria es estructural, inequitativa y potencialmente evitable.

---

### 2. INTRODUCCIÓN Y PREGUNTA DE INVESTIGACIÓN (1 página)

**Qué incluir:**
- Contexto epidemiológico: incidencia y mortalidad por cáncer gástrico en Chile (citar MINSAL/GBD 2023).
- Motivación del estudio: variabilidad clínica no justificada como problema de equidad en salud.
- Marco conceptual: enfoque Wennberg (2010) de variabilidad legítima vs. institucional.
- **Pregunta de investigación:** *¿En qué medida el hospital de atención determina los días de estadía, la cantidad de procedimientos y la mortalidad intrahospitalaria en pacientes con cáncer gástrico (C16.*) en el sistema público chileno, controlando las características clínicas del paciente?*
- **Hipótesis general:** existe variabilidad inter-hospitalaria significativa que no se explica por las características observables del paciente.
- Integración con Avance 1: mencionar que en el A1 se estableció la pregunta y se exploró la base global; en el A2 se realizó el análisis focal en C16.*; en la entrega final se amplía a análisis multinivel, IDH y clustering.

**Criterio de calidad:** la pregunta debe ser específica, medible y coherente con los análisis realizados.

---

### 3. DESCRIPCIÓN DEL DATASET (¾ página)

**Tabla resumen obligatoria:**

| Atributo | Descripción |
|----------|-------------|
| Fuente | GRD Público MINSAL/FONASA (Ministerio de Salud de Chile) |
| Período | 2019–2024 (6 años) |
| Universo | ~457 000 egresos hospitalarios |
| Universo oncológico (C00–D49) | ~422 000 egresos tras filtro CIE-10 |
| Subconjunto focal (C16.*) | ~17 300 egresos de cáncer gástrico |
| Dataset de regresión (top 15 hospitales) | ~9 855 egresos |
| Unidad de análisis | Egreso hospitalario (no paciente único) |

**Variables clave a describir:**
- `dias_estada` (VD en H₃): distribución asimétrica (P99 = 25 días); transformación log(1+x).
- `cantidad_procedimientos` (VI principal): media ~X; asimetría rechaza normalidad (Shapiro-Wilk p < 0.001).
- `mortalidad_int` (VD en H₂): variable binaria 0/1; tasa ~4.05% → desbalance severo.
- `severidad_grd`, `peso_grd`: proxies de complejidad clínica del episodio.
- `hospital`: factor de agrupación; 15 hospitales públicos con ≥ 30 casos C16.*.
- `comorbilidad`: conteo de diagnósticos secundarios activos (derivada).
- `IDH comunal`: índice de desarrollo humano del municipio de residencia del paciente (cruce externo).

**Integración Avances:** señalar explícitamente los cambios introducidos en el Avance 2 respecto al Avance 1 (filtro CIE-10 refinado, exclusión obstétrica, truncación P99, variable comorbilidad).

---

### 4. LIMPIEZA Y PREPARACIÓN DE DATOS (¾ página)

Describir el pipeline en 5 etapas con tabla de resultados:

| Etapa | Operación | N resultante | Δ |
|-------|-----------|--------------|---|
| 1 | Filtro CIE-10 C00–D49 | ~422 000 | −34 000 |
| 2 | Exclusión egresos obstétricos | ~421 000 | −1 000 |
| 3 | Truncación outliers dias_estada (P99 = 25 días) + fix tipos | ~418 000 | −3 000 |
| 4 | Subconjunto focal C16.* | ~17 300 | — |
| 5 | Top 15 hospitales (≥ 30 casos C16.*) | ~9 855 | — |

**Decisiones metodológicas a justificar:**
- Elección del P99 (no P95 ni P99.9) para el corte de outliers.
- Por qué se excluyen egresos obstétricos (contexto clínico incomparable).
- Derivación de `comorbilidad` como conteo de diagnósticos secundarios (proxy Charlson-Deyo simplificado).
- Transformación `log_dias_estada = log(1 + dias_estada)` justificada por asimetría > 1.63.

---

### 5. ANÁLISIS EXPLORATORIO DE DATOS — EDA (1½ páginas)

#### 5.1 Estadística descriptiva
Presentar **Tabla 1** con descriptivas globales (media ± DE, mediana, P25–P75) para las 5 variables numéricas clave, comparando el universo oncológico C00–D49 vs. el subconjunto C16.*.

Destacar:
- La mortalidad en C16.* (4.05%) duplica la del universo oncológico general.
- La mediana de procedimientos difiere entre hospitales en hasta X unidades.

#### 5.2 Visualizaciones (referenciar Figuras 1–6 del notebook)
Para cada figura incluir: descripción breve + **interpretación en 2–3 oraciones**.

- **Figura 1** (distribuciones univariadas): señalar asimetría de `dias_estada` y `cantidad_procedimientos`.
- **Figura 2** (correlación): destacar correlación positiva procedimientos–dias_estada (r ≈ 0.55), que motiva H₃.
- **Figura 3** (boxplot días por hospital): rango intercuartílico varía de 3 a 9 días entre hospitales → motivación principal del estudio.
- **Figura 4** (mortalidad por hospital): diferencia máxima entre hospitales ≈ 8–12 puntos porcentuales.
- **Figura 5** (procedimientos por hospital): diferencia máxima en media de procedimientos ≈ X–Y.
- **Figura 6** (urgencias por sexo): los hombres representan ~X% de los ingresos; los ingresos de urgencia concentran mayor mortalidad.

#### 5.3 Relaciones bivariadas relevantes
- Mortalidad × Tipo de ingreso: diferencia de ~13 pp entre urgencia y programada → mayor hallazgo bivariado.
- Mortalidad × Sexo: tendencia a mayor mortalidad en hombres (OR ≈ 1.2; no significativo al 5%).
- Tipo de alta × Hospital: residuos estandarizados muestran perfiles institucionales diferenciados (Escenario C).

---

### 6. ANÁLISIS INFERENCIAL Y MODELOS (2½ páginas)

#### 6.1 Tests chi-cuadrado (asociaciones categóricas)
Reportar los 3 escenarios en una tabla resumen con χ², df, p-valor y decisión:
- A: Mortalidad × Sexo
- B: Mortalidad × Tipo de ingreso (**hallazgo más fuerte; excluir OBSTÉTRICA**)
- C: Tipo de alta × Top 10 hospitales

#### 6.2 Hipótesis 1 — Kruskal-Wallis + Dunn-Bonferroni
- Justificación: Shapiro-Wilk rechaza normalidad → KW en lugar de ANOVA.
- Resultados: H = XX; p < 0.001; ε² = XX (tamaño del efecto).
- Post-hoc: N% de los pares de hospitales con diferencias significativas → variabilidad estructural.
- **Interpretación clínica:** el hospital de atención cambia la cantidad de procedimientos que recibe un paciente con el mismo diagnóstico.

#### 6.3 Hipótesis 2 — Regresión Logística
Especificación del modelo:
```
mortalidad_int ~ procedimientos + edad + severidad_grd + peso_grd + comorbilidad + C(hospital)
```
Reportar en **Tabla de OR**:

| Predictor | OR | IC95% | p-valor |
|-----------|-----|-------|---------|
| Procedimientos adicionales | ~1.05 | [1.02–1.08] | < 0.001 |
| Severidad GRD | ~6.1 | [X–Y] | < 0.001 |
| Peso relativo GRD | ~0.X | [X–Y] | < 0.05 |
| Edad (años) | ~1.0X | [X–Y] | n.s. |
| Comorbilidad (conteo) | ~1.0X | [X–Y] | < 0.05 |

Métricas: Pseudo-R² McFadden = XX; AUC-ROC = 0.82; discutir desbalance de clase (~4% de mortalidad).

#### 6.4 Hipótesis 3 — Regresión OLS Múltiple
Especificación:
```
log(1 + dias_estada) ~ procedimientos + edad + severidad_grd + peso_grd + comorbilidad + C(hospital)
```
Métricas: R² = 0.636; R² ajustado = 0.635; F(19; 9835) = 978.1; p < 0.001. Errores robustos HC3.

Interpretaciones **clave** (en lenguaje aplicado):
- *"Manteniendo constantes las demás variables, cada procedimiento adicional se asocia con un aumento del 9.7% en los días de estadía (β = 0.093; p < 0.001)."*
- *"La severidad GRD tiene el mayor impacto clínico: cada nivel adicional incrementa la estadía en ~X%."*
- *"Los efectos fijos por hospital son estadísticamente significativos, confirmando que el establecimiento tiene un efecto propio sobre la estadía más allá del perfil del paciente."*

---

### 7. ANÁLISIS AVANZADOS (1 página)

Resumir los 4 análisis de extensión en prosa, con un resultado central por cada uno:

**Modelo multinivel (Sección 12):** ICC = X.XX%, lo que significa que el X.X% de la varianza en log(días de estadía) es atribuible a diferencias entre hospitales, no al paciente. Esto cuantifica el *efecto institucional* de forma estandarizada y comparable.

**Análisis IDH comunal (Sección 13):** Existe un gradiente socioeconómico significativo (Kruskal-Wallis p < XX; Chi-cuadrado mortalidad p < XX): los pacientes de comunas en el quintil inferior (Q1) presentan mayor mortalidad y estadías más largas que los del quintil superior (Q5). El gradiente es de ~X puntos porcentuales en mortalidad entre Q1 y Q5.

**Clustering K-means (Sección 14):** Se identificaron 3 tipologías hospitalarias: (1) alta complejidad/urgente — mayor mortalidad y procedimientos; (2) quirúrgico/especializado — estadías largas y alta complejidad programada; (3) general/baja complejidad — menores recursos y menor intensidad terapéutica. Estos clusters son accionables para políticas de redistribución de recursos.

**Refinamiento desbalance de clase (Sección 15):** Con umbral 0.5, el baseline tiene sensibilidad casi nula para detectar fallecidos (~4% de la muestra). Class-weight='balanced' y SMOTE mejoran la sensibilidad a costa de especificidad; para uso clínico preventivo se recomienda umbral 0.10 con cualquier estrategia.

---

### 8. DISCUSIÓN (¾ página)

**Estructura sugerida:**
1. Respuesta directa a la pregunta de investigación (1 párrafo).
2. Comparación con literatura internacional — Kamaraju et al. (2022), OECD (2023) (1 párrafo).
3. **Implicancias socioeconómicas** — ciclo de desventaja, costo de la variabilidad (1 párrafo, énfasis aquí).
4. Limitaciones principales — 3 a 4 puntos concisos.

---

### 9. CONCLUSIONES (½ página)

**Estructura sugerida (3 párrafos):**
1. Respuesta sintética a las 3 hipótesis + resultado del ICC.
2. Implicación central de política pública: el hospital importa, la variabilidad es evitable.
3. Próximos pasos: datos de seguimiento post-alta, nivel socioeconómico individual, estadiaje TNM.

---

### 10. REFERENCIAS (APA, ½ página)

Mínimo incluir:
- Hollander & Wolfe (1999). *Nonparametric statistical methods.*
- Kamaraju et al. (2022). Hospital-level variation in gastric cancer. *J Surgical Oncology.*
- Ministerio de Salud de Chile (2024). *Base GRD Público MINSAL/FONASA 2019–2024.*
- OECD (2023). *Health at a Glance.*
- Wennberg, J.E. (2010). *Tracking Medicine.*
- Fuentes de las librerías: pandas, numpy, statsmodels, scikit-learn, matplotlib, seaborn.

---

### ANEXOS (no cuentan en el límite de páginas)

- Tabla completa de OR del modelo logístico (con dummies de hospital).
- Tabla de coeficientes OLS (con dummies de hospital).
- Centroides K-means en escala original.
- Tabla Dunn-Bonferroni completa.

---
---

# PARTE II — PRESENTACIÓN ORAL (13 MINUTOS)

> **Criterios de evaluación (peso 50% de la nota final):**
> Estructura (20%) · Claridad problema e hipótesis (15%) · Metodología y análisis (15%) · Presentación de resultados (20%) · Discusión y conclusiones (15%) · Calidad comunicativa y tiempo (15%)

---

## Guión por segmentos (13 minutos exactos)

---

### DIAPOSITIVA 1 — Portada y presentación del equipo (0:00–0:30)

**Tiempo:** 30 segundos  
**Contenido:**
- Título completo del proyecto
- Nombres del equipo
- UDD · Ingeniería · 2026

**Qué decir:**
> *"Buenas [tardes/días]. Somos Vicente Rodríguez, José Tomás Amat y Sebastián Herrera. Hoy presentamos nuestro proyecto final sobre variabilidad hospitalaria oncológica en el sistema público chileno."*

---

### DIAPOSITIVA 2 — El problema y la pregunta (0:30–2:00)

**Tiempo:** 90 segundos  
**Contenido:**
- 1 dato impactante: "El cáncer gástrico es la principal causa oncológica de muerte en el sistema público chileno."
- 1 imagen: mapa o gráfico de mortalidad por cáncer en Chile.
- La pregunta de investigación en grande: *¿Determina el hospital de atención los resultados clínicos más allá del perfil del paciente?*
- Las 3 hipótesis operativas (H₁, H₂, H₃) en bullets.

**Qué decir:**
> *"El problema central es la variabilidad. Dos pacientes con el mismo diagnóstico, la misma edad y la misma severidad pueden recibir tratamientos completamente distintos —y tener resultados distintos— simplemente por el hospital donde llegaron. Eso no debería ocurrir, y cuando ocurre, es una señal de inequidad institucional. Nuestra pregunta es: ¿cuánta de esa variabilidad se explica realmente por el hospital, una vez que controlamos las características del paciente?"*

---

### DIAPOSITIVA 3 — Dataset y metodología (2:00–3:30)

**Tiempo:** 90 segundos  
**Contenido:**
- Tabla resumen del dataset (GRD, 2019–2024, ~9 855 egresos C16.*, 15 hospitales).
- Diagrama de flujo del pipeline de limpieza (5 etapas en una imagen).
- Listado vertical de métodos: EDA → Kruskal-Wallis → Regresión logística → OLS → Multinivel → IDH → K-means.

**Qué decir:**
> *"Trabajamos con la base GRD del MINSAL, ~9 855 egresos de cáncer gástrico en los 15 hospitales públicos con mayor volumen. El dataset tiene variables clínicas —severidad GRD, procedimientos, comorbilidad, edad— que usamos como controles. Nuestro análisis tiene cinco capas: exploración, tests de hipótesis, regresión, modelo multinivel y análisis socioeconómico."*

---

### DIAPOSITIVA 4 — EDA: la variabilidad existe y es grande (3:30–5:00)

**Tiempo:** 90 segundos  
**Contenido:**
- **Figura 3** (boxplot días de estadía por hospital) — mostrar el rango de medianas entre hospitales.
- **Figura 4** (barplot mortalidad por hospital) — mostrar la diferencia entre el hospital con menor y mayor mortalidad.
- Un solo número destacado: "Diferencia máxima en mortalidad entre hospitales: ~X puntos porcentuales."

**Qué decir:**
> *"El EDA confirma que la variabilidad existe y que es grande. Aquí ven los días de estadía para el mismo diagnóstico —cáncer gástrico— en 15 hospitales. Las medianas varían de 3 a 9 días. La mortalidad intrahospitalaria varía de menos del 1% a más del X% entre establecimientos. Esto no es ruido estadístico; es variabilidad real que necesita explicación."*

---

### DIAPOSITIVA 5 — H₁: Kruskal-Wallis (5:00–6:15)

**Tiempo:** 75 segundos  
**Contenido:**
- Resultado del test: H = XX; p < 0.001; ε² = XX.
- **Figura 7** (heatmap Dunn-Bonferroni) — mostrar que la mayoría de los pares son significativamente distintos.
- Una frase de interpretación.

**Qué decir:**
> *"La Hipótesis 1 pregunta si los procedimientos difieren entre hospitales. El test de Kruskal-Wallis —no paramétrico porque la distribución no es normal— rechaza contundentemente la hipótesis nula: p menor a 0.001, con un tamaño del efecto de X. El heatmap post-hoc de Dunn muestra que el X% de los pares de hospitales tienen distribuciones de procedimientos significativamente distintas. La variabilidad no se concentra en uno o dos establecimientos atípicos; es generalizada y sistémica."*

---

### DIAPOSITIVA 6 — H₂: Regresión Logística (6:15–7:30)

**Tiempo:** 75 segundos  
**Contenido:**
- **Figura Forest Plot** de OR (variables principales, no dummies de hospital).
- AUC-ROC en grande: 0.82.
- Una tabla pequeña con los 3 OR más importantes.

**Qué decir:**
> *"La Hipótesis 2 modela la mortalidad. El forest plot muestra que cada procedimiento adicional aumenta las odds de mortalidad en un 5%, controlando por severidad, peso GRD, edad y comorbilidad. Pero el predictor dominante es la severidad GRD —OR de 6.1— lo que sugiere que parte de la asociación entre procedimientos y mortalidad refleja confusión por indicación: los pacientes más graves reciben más procedimientos. El modelo discrimina bien: AUC de 0.82."*

---

### DIAPOSITIVA 7 — H₃: Regresión OLS (7:30–8:45)

**Tiempo:** 75 segundos  
**Contenido:**
- Ecuación del modelo (simplificada, en texto).
- Tabla con 3 coeficientes principales: procedimientos (β = 0.093), severidad GRD, efectos fijos de hospital.
- R² = 0.636 destacado.
- **Figura 12** (diagnóstico de residuos) — mostrar brevemente.

**Qué decir:**
> *"La Hipótesis 3 cuantifica el efecto sobre los días de estadía. Cada procedimiento adicional se asocia con un aumento del 9.7% en la estadía, controlando por todo lo demás. El R² de 0.636 indica que nuestro modelo explica el 64% de la varianza. Los errores robustos HC3 corrigen la heterocedasticidad. El 36% restante corresponde a variables clínicas que GRD no captura: estadio TNM, ECOG, histología."*

---

### DIAPOSITIVA 8 — Modelo Multinivel e ICC (8:45–10:00)

**Tiempo:** 75 segundos  
**Contenido:**
- ICC = X.X% en grande.
- Interpretación visual: "De cada 100 días de variabilidad en la estadía, X se explican por el hospital."
- **Figura 15** (forest plot de interceptos aleatorios por hospital).

**Qué decir:**
> *"Para cuantificar el 'efecto hospital' de forma directa, ajustamos un modelo multinivel con intercepto aleatorio por hospital. El coeficiente de correlación intraclase —ICC— nos dice que el X% de la varianza en la estadía es atribuible al establecimiento, no al paciente. El forest plot muestra los hospitales ordenados por su intercepto aleatorio: los hospitales a la derecha tienen estadías sistemáticamente más largas para el mismo perfil clínico."*

---

### DIAPOSITIVA 9 — Impacto Socioeconómico: IDH y K-means (10:00–11:30)

**Tiempo:** 90 segundos — **énfasis especial aquí**  
**Contenido:**
- **Figura 17** (mortalidad por quintil IDH): mostrar el gradiente Q1→Q5.
- **Figura 19** (heatmap K-means clusters): 3 tipologías hospitalarias.
- 2 números impactantes: diferencia de mortalidad Q1 vs Q5 + nombre de los 3 clusters.

**Qué decir:**
> *"Aquí está el hallazgo más importante desde el punto de vista de política pública. Cuando cruzamos los datos con el IDH comunal, encontramos un gradiente socioeconómico claro: los pacientes de comunas con menor desarrollo —quintil 1— tienen mayor mortalidad y más días de estadía que los del quintil 5, incluso controlando por severidad. El lugar de nacimiento y residencia determina en parte las probabilidades de sobrevivir un episodio oncológico en el sistema público. El K-means refuerza esto: identificamos tres tipologías hospitalarias. Los hospitales del cluster de alta complejidad/urgente —donde llegan más diagnósticos tardíos— sirven desproporcionadamente a comunas de menor IDH. Esto cierra un ciclo: menor acceso preventivo, diagnóstico tardío, peores resultados."*

---

### DIAPOSITIVA 10 — Discusión, Conclusiones y Recomendaciones (11:30–12:30)

**Tiempo:** 60 segundos  
**Contenido:**
- 3 bullets de conclusión (H₁ ✓ H₂ ✓ H₃ ✓).
- 1 bullet sobre el ICC.
- 1 bullet sobre el gradiente IDH.
- 3 recomendaciones de política (en bullets concisos).

**Qué decir:**
> *"En síntesis: la variabilidad hospitalaria en cáncer gástrico en Chile es real, estadísticamente robusta y clínicamente relevante. El hospital importa más allá del paciente. Y esa importancia se distribuye de forma inequitativa: los pacientes más vulnerables son atendidos en los establecimientos con mayores desafíos. Las recomendaciones son tres: estandarizar protocolos en hospitales de alta variabilidad, invertir diferencialmente en comunas de bajo IDH, e implementar sistemas de derivación temprana de casos complejos."*

---

### DIAPOSITIVA 11 — Limitaciones y Próximos Pasos (12:30–13:00)

**Tiempo:** 30 segundos  
**Contenido:**
- 3 limitaciones (bullets muy concisos).
- 2 próximos pasos (bullets concisos).
- Agradecimiento.

**Qué decir:**
> *"Las principales limitaciones son la ausencia de estadiaje TNM, el análisis del IDH a nivel comunal y no individual, y la naturaleza transversal de los datos que limita la causalidad. Como próximos pasos, conectar los GRD con registros de mortalidad del Registro Civil para seguimiento a 30–90 días, e incorporar datos de nivel socioeconómico individual. Muchas gracias."*

---

## Resumen de timing

| Diapositiva | Contenido | Tiempo | Acumulado |
|-------------|-----------|--------|-----------|
| 1 | Portada | 0:30 | 0:30 |
| 2 | Problema y pregunta | 1:30 | 2:00 |
| 3 | Dataset y metodología | 1:30 | 3:30 |
| 4 | EDA: variabilidad existe | 1:30 | 5:00 |
| 5 | H₁ Kruskal-Wallis | 1:15 | 6:15 |
| 6 | H₂ Regresión logística | 1:15 | 7:30 |
| 7 | H₃ Regresión OLS | 1:15 | 8:45 |
| 8 | Modelo multinivel ICC | 1:15 | 10:00 |
| 9 | IDH + K-means (ÉNFASIS AQUÍ) | 1:30 | 11:30 |
| 10 | Conclusiones y recomendaciones | 1:00 | 12:30 |
| 11 | Limitaciones y cierre | 0:30 | 13:00 |

---

## Tips de presentación

- **No leer las diapositivas.** Las slides son apoyo visual; el mensaje va en tu voz.
- **Figura por slide.** Una figura bien explicada vale más que tres sin interpretación.
- **Usa números concretos.** "El hospital explica el X% de la varianza" es más poderoso que "hay variabilidad".
- **Conecta siempre con la pregunta.** Cada resultado debe responder: "¿y esto qué nos dice sobre si el hospital determina los resultados?"
- **El segmento 9 (IDH/K-means) es el diferenciador.** Aquí está la parte más original del trabajo; dénle los 90 segundos completos.
- **Practica las transiciones.** El tiempo justo (13 min) se logra ensayando, no improvisando.
- **Preguntas frecuentes del profesor:** "¿por qué Kruskal-Wallis y no ANOVA?" (normalidad rechazada por Shapiro-Wilk) · "¿qué significa el ICC?" (% de varianza atribuible al hospital) · "¿cómo interpretan el OR de 1.05?" (confusión por indicación; los más graves reciben más procedimientos).

---

*Guía preparada para el Proyecto Final — Análisis de Datos e Inferencia Estadística, UDD Ingeniería 2026.*
