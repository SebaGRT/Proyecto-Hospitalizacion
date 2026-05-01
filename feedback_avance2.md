# Retroalimentación — Segundo Avance de Proyecto
## Variabilidad en el Tratamiento Oncológico y sus Efectos sobre la Mortalidad y la Estadía Hospitalaria en el Sistema Público Chileno

**Equipo:** Vicente · José Tomás · Sebastián  
**Evaluador:** Revisión experta basada en los Requerimientos del Segundo Avance (UDD — Análisis de Datos e Inferencia Estadística)  
**Fecha de revisión:** 30 de abril de 2026  
**Fecha de entrega:** 8 de mayo de 2026

---

## 1. Evaluación Global

Este trabajo es técnicamente uno de los más ambiciosos y rigurosos que se puede ver a este nivel formativo. La elección del tema —variabilidad hospitalaria oncológica en el sistema público chileno— es relevante, oportuna y perfectamente conectada con la agenda de política sanitaria nacional. El uso de datos GRD MINSAL/FONASA con ~454.000 egresos oncológicos, la implementación de modelos de regresión con efectos fijos hospitalarios, errores robustos HC3 y la interpretación clínica profunda a lo largo de todo el notebook son méritos genuinamente destacables.

Sin embargo, existen **brechas formales críticas** entre lo que la rúbrica exige y lo que actualmente se entrega. Si estas brechas no se cierran antes del 8 de mayo, el puntaje final quedará muy por debajo del potencial real del proyecto.

**Calificación estimada en estado actual: 5.5 – 6.0 / 7.0**  
**Calificación potencial tras correcciones indicadas: 6.8 – 7.0 / 7.0**

---

## 2. Lo que está muy bien ✓

| Dimensión | Fortaleza destacada |
|---|---|
| **Pregunta de investigación** | Clara, específica, medible y alineada con el análisis realizado |
| **Pipeline end-to-end** | Carga → limpieza → EDA → inferencia → modelos → evaluación, completamente reproducible |
| **Tests de hipótesis** | Tres hipótesis bien formuladas; Kruskal-Wallis justificado por violación de normalidad; post-hoc Dunn-Bonferroni riguroso |
| **Modelamiento** | Logit + OLS con efectos fijos hospitalarios, HC3, train/test split estratificado — excede los mínimos del rubric |
| **Interpretación clínica** | Las celdas Markdown tienen razonamiento clínico genuino, no solo lectura de p-valores |
| **Análisis socioeconómico (IDH)** | La Tabla 3 con IDH comunal es un aporte metodológico singular que va más allá de lo requerido |
| **Evaluación de modelos** | ROC + AUC + confusion matrix + punto de Youden para logit; MAE/RMSE/R² + hexbin + diagnóstico residual para OLS |
| **Variable comorbilidad** | Derivada de DIAGNOSTICO2–35, mejora el control clínico de ambos modelos |
| **Discusión** | Responde las 6 preguntas requeridas por el rubric con profundidad analítica |

---

## 3. Brechas Críticas — Lo que FALTA ✗

> Estas son las brechas que pueden costar puntos directos en la rúbrica. Prioridad máxima antes del 8 de mayo.

### 3.1 ❌ Informe Escrito (CRÍTICO — entregable obligatorio)

**Problema:** La rúbrica exige un informe escrito en PDF y Word de máximo 10 páginas con formato APA. El notebook NO es el informe. Actualmente no existe ningún documento separado.

**Qué hacer:**
- Redactar el informe en Word/Google Docs con las secciones: portada, introducción y pregunta de investigación, descripción del dataset, limpieza y preparación de datos, EDA, hipótesis, modelo de regresión, discusión, próximos pasos, referencias.
- Cada sección puede basarse en los markdown cells del notebook, pero debe estar escrita en **prosa continua**, con figuras numeradas, tablas en formato APA y citas correctas.
- Las figuras y tablas van como anexo si no caben en las 10 páginas.
- **Sin este documento, la entrega está incompleta por definición.**

**Portada obligatoria:**
- Nombre del proyecto
- Integrantes (Vicente, José Tomás, Sebastián)
- Fecha de entrega
- Nombre del profesor/a y ayudante
- Curso: Análisis de Datos e Inferencia Estadística

### 3.2 ❌ README.md en el Repositorio (CRÍTICO)

**Problema:** El repositorio no tiene README. La rúbrica lo exige explícitamente.

**Qué hacer:** Crear `README.md` en la raíz con al menos:
```markdown
# Variabilidad en el Tratamiento Oncológico — Sistema Público Chileno
**Integrantes:** Vicente Rodríguez · José Tomás Herrera · Sebastián Amat
**Curso:** Análisis de Datos e Inferencia Estadística — UDD 2026
**Dataset:** GRD Público MINSAL/FONASA 2019–2024 (solicitar acceso al equipo)
## Reproducción
1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Abrir `Avance2_Proyecto_Final.ipynb` con Jupyter
4. Ejecutar: Kernel → Restart & Run All
```

### 3.3 ❌ Referencias en Formato APA (IMPORTANTE)

**Problema:** Se citan autores (Wennberg, Gittelsohn, Munir et al., Kamaraju et al.) en el texto de los markdown, pero no existe bibliografía formal al final del notebook ni del informe.

**Qué hacer:** Agregar una celda Markdown al final del notebook con referencias APA completas. Ejemplo:
> Wennberg, J., & Gittelsohn, A. (1973). Small area variations in health care delivery. *Science*, 182(4117), 1102–1108. https://doi.org/10.1126/science.182.4117.1102

---

## 4. Observaciones de Mejora por Sección

### 4.1 Limpieza de Datos

**✓ Fortaleza:** Se documentan el filtro oncológico CIE-10, la exclusión obstétrica, el corte de outliers (P99), la normalización de tipos y la derivación de comorbilidad.

**✗ Debilidad — duplicados:** No hay verificación de duplicados. La rúbrica lo exige explícitamente. Agregar en la sección 3:
```python
n_dup = df.duplicated(subset=['CIP_ENCRIPTADO', 'FECHA_INGRESO', 'FECHAALTA']).sum()
print(f'Registros potencialmente duplicados (mismo paciente+fechas): {n_dup}')
```

**✗ Debilidad — missing values:** Los nulos existen en el dataset pero no hay una decisión documentada por variable (¿se imputan? ¿se eliminan? ¿se ignoran?). Agregar una tabla que muestre `% de nulos por variable` y la decisión tomada para cada una.

### 4.2 EDA — Estadística Descriptiva

**✓ Fortaleza:** La Tabla 1 es completa para variables numéricas. Las tablas de frecuencias categóricas y la matriz de correlación (agregadas en esta sesión) cierran los principales gaps del EDA original.

**✗ Debilidad pendiente:** Existe Tabla 1 para el universo C00–D49, pero no hay tabla equivalente para el subconjunto focal C16.*. La rúbrica pide describir las variables del estudio, y el estudio se enfoca en C16.*. Agregar **Tabla 1b** con descriptivas de `df_focus`:
```python
cols_num = ['dias_estada', 'edad', 'cantidad_procedimientos',
            'severidad_grd', 'peso_grd', 'comorbilidad']
display(df_focus[cols_num].describe().T.round(2))
```

### 4.3 EDA — Visualizaciones

**✓ Fortaleza:** Boxplot por hospital, barplot de mortalidad, barplot de procedimientos, análisis de urgencias por sexo — todos con interpretaciones clínicas profundas.

**✗ Debilidad:** No hay gráfico de dispersión entre variables numéricas clave. La rúbrica menciona explícitamente "gráficos de dispersión" como tipo esperado. Agregar:
```python
# Scatter: procedimientos vs. estadía, coloreado por mortalidad
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df_focus['cantidad_procedimientos'],
                     df_focus['dias_estada'],
                     c=df_focus['mortalidad_int'],
                     cmap='RdBu_r', alpha=0.3, s=10)
plt.colorbar(scatter, label='Mortalidad (0=No, 1=Sí)')
ax.set_xlabel('Cantidad de procedimientos'); ax.set_ylabel('Días de estadía')
ax.set_title('Figura 5 — Dispersión: Procedimientos vs. Estadía (C16.*)')
```

### 4.4 Hipótesis 1 — Kruskal-Wallis

**✓ Fortaleza:** Formulación correcta de H0/H1, justificación del test no paramétrico, reporte del estadístico H, p-valor y tamaño de efecto ε².

**✗ Debilidad menor:** No se reporta el N por hospital ni la mediana de procedimientos por establecimiento como parte del reporte formal. Una tabla resumen de `{hospital: N, mediana_proc, rango}` fortalece la narrativa del test.

### 4.5 Hipótesis 2 — Regresión Logística

**✓ Fortaleza:** Especificación correcta, OR con IC95%, forest plot completo con todos los covariables, ROC con punto de Youden, confusion matrix con análisis clínico de TN/TP/FN/FP.

**✗ Debilidad:** No se reporta la bondad de ajuste del modelo logístico (Pseudo-R²). Agregar en el informe:
```python
print(f"Pseudo-R² McFadden: {1 - modelo_logit.llf / modelo_logit.llnull:.4f}")
```

**✗ Debilidad:** La justificación de la selección de variables es implícita. El informe debe decir explícitamente: *"Se incluyó `severidad_grd` porque captura la complejidad del episodio; `peso_grd` porque es proxy de recursos requeridos; `comorbilidad` porque..."*

### 4.6 Hipótesis 3 — Regresión OLS

**✓ Fortaleza (tras mejoras):** Especificación con tabla de justificación de variables, coeficientes con Δ% interpretable, interpretación de 5 covariables en lenguaje aplicado, diagnóstico residual de 4 paneles, errores robustos HC3.

**✗ Debilidad crítica — placeholders en la interpretación:** Las celdas de interpretación usan `β_proc × 100`, `β_sev × 100` etc. en lugar de los valores numéricos reales del modelo. **Esta es la brecha más importante para el informe**: la interpretación debe decir, por ejemplo:

> *"Manteniendo constantes edad, severidad, comorbilidad y hospital, cada procedimiento adicional se asocia con un aumento del 3.2% en los días de estadía (β = 0.032, p < 0.001, IC 95%: [0.028, 0.036]). Para un paciente con estadía base de 7 días, esto representa 0.22 días adicionales por procedimiento, o equivalente a 2.2 días adicionales para alguien sometido a 10 procedimientos más que la mediana."*

Esta especificidad es lo que diferencia un análisis riguroso de una descripción procedimental.

### 4.7 Discusión

**✓ Fortaleza (tras reescritura):** Ahora responde explícitamente las 6 preguntas requeridas por la rúbrica.

**✗ Debilidad pendiente:** La discusión debe conectar los resultados con la literatura revisada en el Avance 1. Citar al menos 2–3 estudios internacionales y explicar si los hallazgos chilenos convergen o divergen. Por ejemplo: *"Mientras Munir et al. (2024) encontraron que el volumen hospitalario explicaba el X% de la variación en mortalidad por cáncer gástrico en EE.UU., nuestros resultados sugieren que en el sistema público chileno el efecto institucional persiste incluso en hospitales de alto volumen, lo que..."*

---

## 5. Plan de Trabajo para los 8 Días Restantes

| Día | Tarea | Responsable sugerido |
|---|---|---|
| 1–2 | Redactar informe escrito (estructura + introducción + dataset + limpieza) | Todos |
| 2–3 | Completar informe: EDA + hipótesis + modelo (texto narrativo con resultados reales) | Vicente / José Tomás |
| 3 | Reemplazar placeholders de coeficientes con valores reales del modelo | Sebastián |
| 4 | Revisión APA: citas, referencias bibliográficas, numeración de figuras | José Tomás |
| 4–5 | Crear README.md + verificar reproducibilidad del notebook desde cero | Sebastián |
| 5–6 | Agregar Tabla 1b (C16.*), scatter procedimientos vs. estadía, verificación duplicados | Vicente |
| 6–7 | Revisión cruzada del informe y del notebook | Todos |
| 7–8 | Preparación de presentación oral (10–15 min) | Todos |

---

## 6. Recomendaciones Metodológicas para el Avance Final

Estas son mejoras que elevarían el proyecto de "muy bueno" a "publicable":

1. **Modelo multinivel (mixed effects):** Reemplazar efectos fijos por efectos aleatorios hospitalarios con `statsmodels.MixedLM`. Permite cuantificar la varianza entre hospitales como parámetro (ICC) y es el estándar en la literatura de variabilidad hospitalaria.

2. **Análisis longitudinal 2019–2024:** Comparar la variabilidad pre-pandemia (2019) vs. durante pandemia (2020–2021) vs. post-pandemia (2022–2024). COVID-19 alteró dramáticamente los patrones de hospitalización oncológica.

3. **Mapa coroplético de Chile:** Visualizar tasas de mortalidad y varianza de estadía por región/hospital sobre un mapa geográfico. Impacto visual excepcional para la presentación oral.

4. **Análisis de supervivencia:** Si es posible reconstruir el seguimiento desde el egreso con datos de años consecutivos, un análisis Kaplan-Meier + Cox añadiría una dimensión temporal al proyecto.

5. **Umbrales alternativos en el modelo logístico:** Explorar umbral 0.30 en lugar de 0.50 para aumentar la sensibilidad (recall de Fallecidos), que es clínicamente más relevante que la especificidad.

---

## 7. Observación Final del Evaluador

> *"El análisis técnico de este trabajo es sólido y en varios aspectos supera lo que se espera a este nivel. El equipo demuestra comprensión genuina de los datos y capacidad para conectar los resultados estadísticos con la realidad clínica del sistema de salud chileno. Lo que le falta a esta entrega no es rigor analítico —que ya existe— sino el documento que lo comunique de forma estructurada y la bibliografía que lo contextualice dentro del campo."*

Con una semana disponible, el equipo tiene tiempo suficiente para cerrar las brechas críticas (informe escrito + README + referencias) y entregar un trabajo de nivel de excelencia. El análisis está listo; la presentación formal es lo que está en deuda con la rúbrica.

---

*Este documento es una herramienta de preparación interna del equipo. No reemplaza la evaluación oficial del docente.*
