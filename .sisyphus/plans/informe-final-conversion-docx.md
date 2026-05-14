# Plan: Conversión y Profundización del Informe Final — LaTeX a DOCX

## TL;DR

> **Quick Summary**: Convertir el paper LaTeX estilo Lancet (inglés) a un informe DOCX profesional en español (formato APA/UCC) usando python-docx, profundizando todos los contenidos con los análisis completos del notebook de Jupyter. Simultáneamente corregir discrepancias numéricas en el LaTeX inglés y generar su versión DOCX.
>
> **Deliverables**:
> - `Entrega_Final/Informe_Final_AmatHerreraRodriguez_v2.docx` — Informe español, ≤12 páginas, 13+ figuras, APA 7ª
> - `Entrega_Final/main_paper_v2.tex` — LaTeX inglés corregido con valores reales del notebook
> - `Entrega_Final/main_paper_v2.docx` — DOCX inglés estilo Lancet desde LaTeX corregido
> - `Entrega_Final/referencia_numerica.csv` — Tabla de referencia con todos los valores estadísticos del notebook
> - Archivos originales preservados como `.bak` (no se modifican)
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T2 → T10 → T13 (reference table → assembly → polish)

---

## Context

### Original Request
"Utilizando todo lo de Entrega_Final, y las Rúbricas en este, traspasa el informe en LaTeX a .docx, manteniendo el formato, y profundiza en los contenidos de estos mismos con todo lo presente en el notebook de Jupyter. Guíate con todo el contenido de Rúbricas. Genera copias para modificar, dejando los archivos originales como backup. Trabaja como un Data Scientist, con experiencia en el campo de la salud. El notebook de Jupyter no se modifica."

### Interview Summary

**Key Discussions**:
- **Idiomas**: Ambos — español (principal, para entrega UDD) + inglés (paper actualizado)
- **Formato DOCX español**: Columna única APA/UCC con paleta de colores Lancet (verde #006046)
- **Formato DOCX inglés**: Adaptar estructura Lancet de dos columnas desde LaTeX
- **Nivel de profundización**: Profunda — incluir TODOS los análisis del notebook
- **Secciones a fortalecer**: Todas por igual
- **Valores numéricos**: Notebook es fuente de verdad — corregir TODAS las discrepancias en LaTeX
- **Análisis incompletos**: Re-ejecutar y corregir ICC multinivel y SMOTE antes de reportar
- **Nombres de autores**: Vicente Amat, José Tomás Herrera, Sebastián Rodríguez (orden README/DOCX existente)
- **DOCX existente**: Versión separada en español (Avance 2) — no modificar, crear nueva versión

**Research Findings**:
- **LaTeX**: Paper inglés 836 líneas, Lancet 2-columnas, 4 secciones, valores numéricos desactualizados
- **Notebook**: ~15 secciones en español, análisis completos con valores correctos
- **Discrepancias numéricas críticas**: KW H=412.3→1909.40, ε²=0.08→0.146, AUC=0.823→0.849, mortalidad=4.05%→5.56%, R²=0.636→0.633
- **ICC multinivel**: Notebook reporta ICC=0.000 (probable fallo de convergencia) — requiere debugging
- **SMOTE**: Implementación notebook requiere validación (train/test split correcto)
- **Figuras**: 21+ figuras en `outputs/` con nombres `figuraN_*`
- **Tablas CSV**: En `outputs/tablas/`
- **python-docx**: Sin soporte nativo para ecuaciones, referencias cruzadas, columnas decimales, o diseño a dos columnas sin manipulación XML frágil

### Metis Review

**Identified Gaps** (addressed):
- **Layout dos columnas en DOCX**: Técnicamente posible pero frágil con python-docx. Para DOCX español se usa columna única (alineado con rúbrica UDD). Versión inglés se genera vía LaTeX→pandoc→DOCX.
- **Ecuaciones**: Usar Unicode (β̂, ε², χ², R²) o imágenes para fórmulas complejas
- **Referencias cruzadas**: Texto estático ("Tabla 1", "Figura 2") — sin campos dinámicos
- **Cajas estilo mdframed**: Simular con tabla de una celda con borde verde y fondo tintado
- **Nombres de autores**: Usar orden "Vicente Amat, José Tomás Herrera, Sebastián Rodríguez"
- **Scope creep**: Rúbricas + notebook definen el límite; no añadir análisis nuevos

---

## Work Objectives

### Core Objective
Generar dos versiones DOCX profesionales (español UDD + inglés Lancet) que integren y profundicen TODOS los análisis del notebook de Jupyter, con valores numéricos corregidos, formato académico riguroso, y archivos originales preservados.

### Concrete Deliverables
1. `Entrega_Final/backup/` — Copia de seguridad de todos los archivos originales
2. `Entrega_Final/referencia_numerica.csv` — Tabla maestra de valores estadísticos extraídos del notebook
3. `Entrega_Final/Informe_Final_AmatHerreraRodriguez_v2.docx` — Informe español completo
4. `Entrega_Final/main_paper_v2.tex` — LaTeX inglés con valores corregidos
5. `Entrega_Final/main_paper_v2.docx` — DOCX inglés desde LaTeX corregido

### Definition of Done
- [ ] Todos los archivos originales tienen copia `.bak` o están en `backup/`
- [ ] `referencia_numerica.csv` contiene todos los valores estadísticos con celdas de notebook de origen
- [ ] Informe español: 12 secciones GUIA_INFORME, ≤12 páginas, 13+ figuras, 6+ referencias APA 7ª
- [ ] Informe español: todos los valores numéricos coinciden con notebook (fuente de verdad)
- [ ] Informe español: cada hipótesis (H1, H2, H3) respondida explícitamente en conclusiones
- [ ] Paper inglés LaTeX: compila sin errores, valores numéricos corregidos
- [ ] Paper inglés DOCX: preserva formato Lancet, incluye análisis profundizados
- [ ] ICC multinivel re-ejecutado con convergencia verificada
- [ ] SMOTE re-ejecutado con validación cruzada correcta
- [ ] Notebook Jupyter NO modificado

### Must Have
- Valores numéricos del notebook como única fuente de verdad
- 12 secciones requeridas por rúbrica GUIA_INFORME
- Paleta de colores Lancet (verde #006046, grises) en ambos DOCX
- Todas las figuras del notebook incluidas y referenciadas
- APA 7ª edición para referencias y citas
- Análisis avanzados: multinivel (ICC), oncología (C16/C33-34/C61), IDH, K-means, SMOTE

### Must NOT Have (Guardrails)
- NO modificar el notebook de Jupyter
- NO modificar archivos originales (LaTeX, DOCX existente)
- NO añadir análisis nuevos no presentes en el notebook
- NO exceder 12 páginas en informe español
- NO usar referencias cruzadas dinámicas de Word (frágiles)
- NO usar ecuaciones OMML (incompatibles con LibreOffice); usar Unicode o imágenes
- NO diseño a dos columnas en DOCX español (usar columna única)
- NO usar inteligencia artificial generativa para redactar conclusiones clínicas no respaldadas por datos

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — Toda la verificación es ejecutada por agentes. Sin excepciones.

### Test Decision
- **Infrastructure exists**: NO (no hay framework de testing en este proyecto)
- **Automated tests**: None (proyecto de análisis de datos, no de software)
- **Framework**: N/A
- **Agent-Executed QA**: MANDATORY para todas las tareas

### QA Policy
Cada tarea incluye escenarios QA ejecutables por agente. Evidencia en `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Contenido/Markdown**: grep/búsqueda de texto para verificar presencia de secciones, valores, referencias
- **DOCX**: python-docx para verificar estructura, estilos, conteo de figuras/tablas
- **LaTeX**: compilación + grep en PDF para verificar valores
- **Datos numéricos**: comparación programática CSV vs notebook

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Inicio inmediato — fundación + scaffolding, 4 tareas paralelas):
├── T1: Backup de archivos originales [quick]
├── T2: Extraer tabla de referencia numérica del notebook [deep]
├── T3: Re-ejecutar ICC multinivel (debug convergencia) [deep]
└── T4: Re-ejecutar SMOTE (corregir validación) [deep]

Wave 2 (Después de Wave 1 — redacción de secciones, 5 tareas paralelas):
├── T5: DOCX español — Portada, Resumen, Palabras clave, Introducción [writing]
├── T6: DOCX español — Problema, Hipótesis, Objetivos [writing]
├── T7: DOCX español — Metodología completa [writing]
├── T8: DOCX español — Resultados: EDA + Descriptivos + H1 Kruskal-Wallis [writing]
└── T9: DOCX español — Resultados: H2 Logit + H3 OLS + Avanzados [writing]

Wave 3 (Después de Wave 2 — discusión + ensamblaje + inglés, 4 tareas):
├── T10: DOCX español — Discusión + Conclusiones + Referencias + Anexos [writing]
├── T11: Ensamblar DOCX español (python-docx) [deep]
├── T12: Corregir valores numéricos en LaTeX inglés [quick]
└── T13: Generar DOCX inglés desde LaTeX corregido + profundizar [deep]

Wave 4 (Después de Wave 3 — pulido final, 2 tareas paralelas):
├── T14: Pulido final DOCX español (ToC, formato, APA compliance) [quick]
└── T15: Pulido final DOCX inglés [quick]

Wave FINAL (Después de TODAS las tareas — 4 revisiones paralelas):
├── F1: Plan Compliance Audit (oracle)
├── F2: Code Quality Review (unspecified-high)
├── F3: Real Manual QA (unspecified-high)
└── F4: Scope Fidelity Check (deep)
→ Presentar resultados → Esperar OK explícito del usuario

Critical Path: T2 → T8/T9 → T11 → T14 → F1-F4 → user okay
Parallel Speedup: ~65% más rápido que secuencial
Max Concurrent: 5 (Waves 1 & 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1   | —         | —      | 1    |
| T2   | —         | T5-T10, T12 | 1 |
| T3   | —         | T8, T9  | 1    |
| T4   | —         | T9      | 1    |
| T5   | T2        | T11     | 2    |
| T6   | T2        | T11     | 2    |
| T7   | T2        | T11     | 2    |
| T8   | T2, T3    | T11     | 2    |
| T9   | T2, T3, T4 | T11   | 2    |
| T10  | T2        | T11     | 3    |
| T11  | T5-T10    | T14     | 3    |
| T12  | T2        | T13     | 3    |
| T13  | T12       | T15     | 3    |
| T14  | T11       | —       | 4    |
| T15  | T13       | —       | 4    |
| F1   | ALL       | —       | FINAL |
| F2   | ALL       | —       | FINAL |
| F3   | ALL       | —       | FINAL |
| F4   | ALL       | —       | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 4 — T1 → `quick`, T2 → `deep`, T3 → `deep`, T4 → `deep`
- **Wave 2**: 5 — T5-T10 → `writing`
- **Wave 3**: 4 — T10 → `writing`, T11 → `deep`, T12 → `quick`, T13 → `deep`
- **Wave 4**: 2 — T14 → `quick`, T15 → `quick`
- **FINAL**: 4 — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Verification = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

- [ ] 1. **Crear backups de todos los archivos originales**

  **What to do**:
  - Crear directorio `Entrega_Final/backup/`
  - Copiar `main_paper.tex` → `backup/main_paper.tex` y `main_paper.tex.bak` (en mismo dir)
  - Copiar `Informe_Final_AmatHerreraRodriguez.docx` → `backup/Informe_Final_AmatHerreraRodriguez.docx` y `.bak`
  - Copiar `Proyecto_Final_AmatHerreraRodriguez.ipynb` → `backup/Proyecto_Final_AmatHerreraRodriguez.ipynb` y `.bak`
  - Copiar directorio `Rúbricas/` completo → `backup/Rúbricas/`
  - Copiar `outputs/` completo → `backup/outputs/`
  - Generar y guardar checksum MD5 del notebook original: `md5sum Proyecto_Final_AmatHerreraRodriguez.ipynb > backup/notebook_checksum.md5`
  - Crear `backup/README.md` documentando qué es cada archivo y fecha del backup

  **Must NOT do**:
  - NO modificar, mover ni renombrar ningún archivo original
  - NO hacer commit de los backups si los originales ya están en git

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Operaciones de sistema de archivos simples, sin lógica compleja
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2, T3, T4)
  - **Blocks**: None (foundation task)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `Entrega_Final/` — Listar todos los archivos a respaldar

  **Acceptance Criteria**:
  - [ ] Directorio `Entrega_Final/backup/` existe con todos los archivos
  - [ ] Archivos `.bak` existen junto a cada original en `Entrega_Final/`
  - [ ] `backup/notebook_checksum.md5` contiene hash MD5 del notebook original
  - [ ] `diff original backup/original` no muestra diferencias para cada archivo

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — todos los backups creados correctamente
    Tool: Bash
    Preconditions: Archivos originales existen en Entrega_Final/
    Steps:
      1. ls Entrega_Final/backup/ → verificar que contiene: main_paper.tex, Informe_Final_AmatHerreraRodriguez.docx, Proyecto_Final_AmatHerreraRodriguez.ipynb, Rúbricas/, outputs/, notebook_checksum.md5, README.md
      2. ls Entrega_Final/*.bak → verificar que existen .bak para .tex, .docx, .ipynb
      3. diff Entrega_Final/main_paper.tex Entrega_Final/backup/main_paper.tex → sin diferencias
      4. md5sum -c backup/notebook_checksum.md5 → "OK"
    Expected Result: 8+ archivos en backup/, 3 archivos .bak junto a originales, diffs vacíos, checksum OK
    Failure Indicators: Falta algún archivo en backup/, diff muestra diferencias, checksum falla
    Evidence: .sisyphus/evidence/task-1-backup-verify.txt (output de ls + diff + md5sum)

  Scenario: Edge case — backups no sobrescriben backups existentes
    Tool: Bash
    Preconditions: backup/ ya existe (segunda ejecución simulada)
    Steps:
      1. Ejecutar script de backup con flag --no-clobber
      2. Verificar que archivos existentes NO fueron modificados (mtime original)
    Expected Result: Segunda ejecución no modifica archivos ya respaldados
    Evidence: .sisyphus/evidence/task-1-backup-noclobber.txt
  ```

  **Commit**: YES (grupo T1)
  - Message: `chore(backup): create backup copies of all original files`
  - Files: `Entrega_Final/backup/`, `Entrega_Final/*.bak`

- [ ] 2. **Extraer tabla de referencia numérica del notebook**

  **What to do**:
  - Leer el notebook `Proyecto_Final_AmatHerreraRodriguez.ipynb` sistemáticamente
  - Extraer TODOS los valores estadísticos clave en una tabla CSV estructurada con columnas:
    `analisis, metrica, valor, notebook_celda, notebook_linea, notas`
  - Valores mínimos a extraer:
    - **H1 Kruskal-Wallis**: H, ε², p-valor, medianas por grupo (12+ grupos), resultados Dunn post-hoc (todas las comparaciones significativas)
    - **H2 Regresión Logística**: AUC, accuracy, sensitivity, specificity, OR por variable (procedures, age, sex, urgency, HDI, etc.), IC 95%, matriz de confusión
    - **H3 Regresión Lineal**: R², R² ajustado, β por variable, p-valores, VIF, F-statistic
    - **Descriptivos**: n total, n por hospital, % mortalidad, % urgencia, media/mediana días estancia, distribución por sexo, edad
    - **Chi-cuadrado exploratorio**: χ² y p-valor para: mortalidad×sexo, mortalidad×urgencia, alta×hospital
    - **Multinivel ICC**: ICC valor, grupo (hospital)
    - **Oncología**: medias/medianas C16 vs C33-C34 vs C61, Kruskal-Wallis entre grupos
    - **IDH**: tasas por quintil Q1-Q5
    - **K-means**: silhouette score, tamaños de clusters, centros
    - **SMOTE**: métricas pre/post SMOTE (AUC, accuracy)
  - Incluir columna `notebook_celda` con número de celda (In[N]) para trazabilidad
  - Incluir columna `discrepancia_latex` marcando valores que difieren del LaTeX original

  **Must NOT do**:
  - NO modificar el notebook
  - NO redondear valores — extraer con precisión completa
  - NO omitir valores "no significativos" — extraer todo

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requiere lectura exhaustiva de notebook con ~15 secciones, extracción precisa de docenas de valores estadísticos, trazabilidad a celdas
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3, T4)
  - **Blocks**: T5, T6, T7, T8, T9, T10, T12 (todas las tareas de contenido dependen de estos números)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Notebook con todos los análisis (NO modificar)
  - `Entrega_Final/main_paper.tex` — Paper LaTeX con valores a comparar para columna `discrepancia_latex`

  **Acceptance Criteria**:
  - [ ] `referencia_numerica.csv` existe con ≥40 filas de valores estadísticos
  - [ ] Cada fila tiene valor numérico exacto, celda de notebook de origen, y notas
  - [ ] Columna `discrepancia_latex` marcada como SÍ/NO para cada valor
  - [ ] Valores verificados contra notebook: H=1909.40, ε²=0.146, AUC=0.849, R²=0.633, mortalidad=5.56%

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — tabla completa y precisa
    Tool: Bash (python3)
    Preconditions: Notebook accesible en Entrega_Final/
    Steps:
      1. python3 -c "import csv; rows = list(csv.DictReader(open('Entrega_Final/referencia_numerica.csv'))); print(f'Total rows: {len(rows)}')" → ≥40
      2. Verificar valores clave: grep "H1.*Kruskal" referencia_numerica.csv → H=1909.40, ε²=0.146
      3. Verificar H2: grep "H2.*AUC" referencia_numerica.csv → 0.849
      4. Verificar % mortalidad: grep "mortalidad" referencia_numerica.csv → 5.56
      5. Verificar todas las filas tienen notebook_celda no vacío
    Expected Result: ≥40 filas, valores clave correctos, trazabilidad completa
    Failure Indicators: Valores no coinciden con notebook, filas sin celda de origen, <40 filas
    Evidence: .sisyphus/evidence/task-2-reference-table.txt

  Scenario: Negative — verificar que valores del LaTeX NO se usaron como fuente
    Tool: Bash (python3)
    Preconditions: referencia_numerica.csv existe
    Steps:
      1. Filtrar filas con discrepancia_latex=SÍ
      2. Verificar que valor en CSV no coincide con valor en LaTeX (comparar contra extracted_latex_values.csv)
    Expected Result: Todas las filas con discrepancia_latex=SÍ usan el valor del notebook, no del LaTeX
    Evidence: .sisyphus/evidence/task-2-discrepancies-verified.txt
  ```

  **Commit**: YES (grupo T2)
  - Message: `feat(data): extract numerical reference table from notebook`
  - Files: `Entrega_Final/referencia_numerica.csv`

- [ ] 3. **Re-ejecutar y corregir ICC multinivel**

  **What to do**:
  - Localizar la celda del notebook donde se ejecuta el modelo multinivel (statsmodels MixedLM)
  - El notebook reporta ICC=0.000 — esto es casi seguro un fallo de convergencia o modelo mal especificado
  - Crear script `Entrega_Final/scripts/fix_icc_multinivel.py` que:
    1. Replique exactamente la preparación de datos del notebook
    2. Ejecute el modelo multinivel con variables del notebook
    3. Verifique convergencia (`model.converged` debe ser True)
    4. Si no converge: probar con diferentes optimizadores (`bfgs`, `lbfgs`, `powell`), escalar variables, o usar fórmula alternativa
    5. Extraer componentes de varianza: σ²_between (hospital) y σ²_within (residual)
    6. Calcular ICC = σ²_between / (σ²_between + σ²_within)
    7. Reportar ICC con interpretación (≥0.05 = efecto hospital relevante)
  - Guardar resultados en `outputs/icc_multinivel_resultados.csv`
  - Actualizar `referencia_numerica.csv` con el ICC corregido (añadir fila o actualizar existente)

  **Must NOT do**:
  - NO modificar el notebook
  - NO cambiar la estructura del modelo (mismas variables que el notebook)
  - NO reportar ICC sin verificar convergencia

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requiere debugging estadístico de modelo multinivel, comprensión de componentes de varianza, posible ajuste de optimizadores
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T4)
  - **Blocks**: T8, T9 (resultados avanzados dependen del ICC corregido)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas del modelo multinivel (buscar "MixedLM", "ICC", "multinivel")
  - `statsmodels` docs: `MixedLM` API, parámetros de convergencia
  - `Entrega_Final/referencia_numerica.csv` — Actualizar con ICC corregido

  **Acceptance Criteria**:
  - [ ] Script `fix_icc_multinivel.py` ejecuta sin errores
  - [ ] Modelo converge (`converged=True`)
  - [ ] ICC > 0.000 (valor razonable, típicamente 0.02-0.25 en datos hospitalarios)
  - [ ] `outputs/icc_multinivel_resultados.csv` contiene: σ²_between, σ²_within, ICC, converged, método
  - [ ] `referencia_numerica.csv` actualizado con ICC corregido

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — modelo converge y produce ICC válido
    Tool: Bash
    Preconditions: Script fix_icc_multinivel.py existe
    Steps:
      1. python3 scripts/fix_icc_multinivel.py
      2. Verificar output contiene "Converged: True"
      3. Leer outputs/icc_multinivel_resultados.csv → ICC > 0.000
      4. Verificar ICC = σ²_between / (σ²_between + σ²_within) (consistencia interna)
    Expected Result: Script ejecuta sin errores, convergencia=True, ICC > 0, cálculo internamente consistente
    Failure Indicators: Error de convergencia, ICC=0, división por cero, excepción no manejada
    Evidence: .sisyphus/evidence/task-3-icc-results.txt (output completo del script)

  Scenario: Negative — script maneja gracefulmente falta de datos
    Tool: Bash
    Preconditions: Eliminar temporalmente columna necesaria (simulado)
    Steps:
      1. Ejecutar script con datos incompletos
      2. Verificar que el error es informativo (no traceback genérico)
    Expected Result: Mensaje de error claro: "Falta la columna X requerida para el modelo multinivel"
    Evidence: .sisyphus/evidence/task-3-icc-error-handling.txt
  ```

  **Commit**: YES (grupo T3)
  - Message: `fix(analysis): re-run multilevel ICC with convergence verification`
  - Files: `scripts/fix_icc_multinivel.py`, `outputs/icc_multinivel_resultados.csv`

- [ ] 4. **Re-ejecutar y corregir análisis SMOTE**

  **What to do**:
  - Localizar la celda del notebook donde se aplica SMOTE
  - Verificar que SMOTE se aplica SOLO a datos de entrenamiento (no a test)
  - Crear script `Entrega_Final/scripts/fix_smote.py` que:
    1. Replique el pipeline de clasificación del notebook
    2. Asegure train/test split ANTES de SMOTE (si no está así, corregir)
    3. Aplique SMOTE solo a X_train, y_train
    4. Entrene modelo balanceado y evalúe en X_test (sin SMOTE)
    5. Reporte métricas pre-SMOTE vs post-SMOTE: AUC, accuracy, sensitivity (recall clase minoritaria), specificity, F1
    6. Use validación cruzada estratificada (StratifiedKFold, k=5) para métricas robustas
  - Guardar resultados en `outputs/smote_resultados.csv`
  - Guardar gráficos comparativos (ROC pre vs post) en `outputs/figuras/`
  - Actualizar `referencia_numerica.csv` con métricas SMOTE corregidas

  **Must NOT do**:
  - NO aplicar SMOTE antes del train/test split (data leakage)
  - NO modificar el notebook
  - NO usar accuracy como única métrica (problema desbalanceado)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requiere conocimiento de validación de modelos desbalanceados, data leakage prevention, validación cruzada estratificada
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T3)
  - **Blocks**: T9 (resultados avanzados dependen de métricas SMOTE corregidas)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas SMOTE (buscar "SMOTE", "imblearn", "balance")
  - `imbalanced-learn` docs: `SMOTE` API, `Pipeline` con SMOTE
  - `Entrega_Final/referencia_numerica.csv` — Actualizar con métricas SMOTE corregidas

  **Acceptance Criteria**:
  - [ ] Script `fix_smote.py` ejecuta sin errores ni warnings de data leakage
  - [ ] SMOTE aplicado solo a training data (verificable en código)
  - [ ] `outputs/smote_resultados.csv` contiene métricas pre y post SMOTE
  - [ ] Validación cruzada k=5 implementada
  - [ ] ROC curves comparativas guardadas en `outputs/figuras/`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — SMOTE correctamente aplicado con CV
    Tool: Bash
    Preconditions: Script fix_smote.py existe, datos disponibles
    Steps:
      1. python3 scripts/fix_smote.py
      2. Verificar output contiene "Train size" y "Test size" como números diferentes
      3. Verificar SMOTE aplicado solo a train: buscar "SMOTE" en el script, confirmar que X_test NO se pasa a SMOTE
      4. Leer outputs/smote_resultados.csv → métricas pre y post SMOTE presentes
      5. Verificar métrica recall para clase minoritaria ≥ mejora sobre baseline
    Expected Result: No data leakage, métricas de validación cruzada, ROC curves generadas
    Failure Indicators: X_test pasado a SMOTE, métricas sospechosamente altas (posible leakage), sin CV
    Evidence: .sisyphus/evidence/task-4-smote-results.txt

  Scenario: Edge case — verificar que la clase minoritaria tiene suficientes muestras
    Tool: Bash
    Preconditions: Script ejecutado
    Steps:
      1. python3 -c "
import pandas as pd
y = pd.read_csv('data.csv')['mortalidad']
print(f'Clase minoritaria: {y.value_counts().min()} muestras')
print(f'k_neighbors en SMOTE debe ser ≤ {y.value_counts().min() - 1}')
"
    Expected Result: k_neighbors ≤ muestras_minoritarias - 1 (no error "Expected n_neighbors <= n_samples")
    Evidence: .sisyphus/evidence/task-4-smote-neighbors-check.txt
  ```

  **Commit**: YES (grupo T4)
  - Message: `fix(analysis): re-run SMOTE with correct train/test split and CV`
  - Files: `scripts/fix_smote.py`, `outputs/smote_resultados.csv`, `outputs/figuras/smote_roc_*.png`

- [ ] 5. **Redactar DOCX español — Portada, Resumen, Palabras Clave, Introducción**

  **What to do**:
  - Escribir archivo `Entrega_Final/secciones/seccion_1_portada_intro.md` con:
    - **Portada**: Título del informe, nombres de autores (Vicente Amat, José Tomás Herrera, Sebastián Rodríguez), asignatura, profesor, fecha
    - **Resumen/Abstract** (español, ≤250 palabras): Objetivo, métodos, resultados principales (H1 ε²=0.146, H2 AUC=0.849, H3 R²=0.633), conclusiones
    - **Palabras clave**: 4-6 keywords (ej: hospitalización, días de estancia, mortalidad, GRD, Kruskal-Wallis, regresión logística)
    - **Sección 1 — Introducción**: Contexto del sistema de salud chileno, relevancia de predecir días de estancia y mortalidad, estudios previos (citar 2-3 referencias APA), justificación del estudio, organización del informe
  - Usar valores del notebook y `referencia_numerica.csv` como fuente de verdad
  - Insertar referencias a figuras existentes usando ruta `outputs/figuras/figuraN_*.png`
  - Estilo: Académico-científico, español formal, tono de Data Science en salud

  **Must NOT do**:
  - NO usar valores del LaTeX original (usar solo notebook/referencia_numerica.csv)
  - NO exceder 250 palabras en resumen
  - NO inventar referencias bibliográficas no presentes en el notebook

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Redacción académica en español, estructura de paper científico, tono formal Data Science
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T7, T8, T9)
  - **Blocks**: T11 (ensamblaje DOCX)
  - **Blocked By**: T2 (referencia numérica)

  **References**:
  - `Entrega_Final/referencia_numerica.csv` — Valores exactos para resumen
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas introductorias (contexto, justificación)
  - `Entrega_Final/Rúbricas/` — Secciones 1-2 de GUIA_INFORME
  - `Entrega_Final/main_paper.tex` — Estructura del abstract original (referencia de estilo)
  - `Entrega_Final/Informe_Final_AmatHerreraRodriguez.docx` — DOCX español existente (referencia de contenido base, NO copiar)

  **Acceptance Criteria**:
  - [ ] Archivo `secciones/seccion_1_portada_intro.md` existe
  - [ ] Contiene las 4 subsecciones: portada, resumen, palabras clave, introducción
  - [ ] Resumen ≤250 palabras (verificable con `wc -w`)
  - [ ] Nombres de autores correctos: Vicente Amat, José Tomás Herrera, Sebastián Rodríguez
  - [ ] 4-6 palabras clave en español
  - [ ] Introducción referencia al menos 2 fuentes APA

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — sección completa y valores correctos
    Tool: Bash
    Preconditions: secciones/seccion_1_portada_intro.md existe
    Steps:
      1. wc -w secciones/seccion_1_portada_intro.md → verificar resumen ≤250 palabras
      2. grep "Vicente Amat" secciones/seccion_1_portada_intro.md → encontrado
      3. grep "ε²" secciones/seccion_1_portada_intro.md → contiene 0.146
      4. grep "AUC" secciones/seccion_1_portada_intro.md → contiene 0.849
      5. grep "Palabras clave" secciones/seccion_1_portada_intro.md -A 3 → 4-6 términos
    Expected Result: Todas las verificaciones positivas, valores coinciden con referencia_numerica.csv
    Failure Indicators: Valores del LaTeX original (ε²=0.08, AUC=0.823), resumen >250 palabras, nombres incorrectos
    Evidence: .sisyphus/evidence/task-5-section1-verify.txt

  Scenario: Negative — verificar que NO usa valores del LaTeX original
    Tool: Bash (grep)
    Steps:
      1. grep "0.08" secciones/seccion_1_portada_intro.md → NO debe aparecer (valor antiguo ε²)
      2. grep "0.823" secciones/seccion_1_portada_intro.md → NO debe aparecer (valor antiguo AUC)
      3. grep "4.05" secciones/seccion_1_portada_intro.md → NO debe aparecer (valor antiguo mortalidad)
    Expected Result: Ningún valor numérico antiguo del LaTeX presente
    Evidence: .sisyphus/evidence/task-5-section1-no-old-values.txt
  ```

  **Commit**: YES (grupo T5-T10)
  - Message: `docs(informe): write front matter, abstract, and introduction for Spanish DOCX`
  - Files: `secciones/seccion_1_portada_intro.md`

- [ ] 6. **Redactar DOCX español — Problema, Hipótesis, Objetivos**

  **What to do**:
  - Escribir archivo `Entrega_Final/secciones/seccion_2_prob_hip_obj.md` con:
    - **Sección 2 — Problema de Investigación**: ¿Qué factores clínicos y sociodemográficos predicen los días de estancia y la mortalidad? Contextualizar en sistema GRD chileno. Describir el problema de la variabilidad en hospitalizaciones.
    - **Sección 3a — Hipótesis**: 
      - H1: Existen diferencias significativas en días de estancia entre hospitales (KW)
      - H2: Variables clínicas (n° procedimientos, edad, urgencia, sexo, HDI) predicen significativamente la probabilidad de mortalidad (Logit)
      - H3: Variables clínicas y sociodemográficas predicen significativamente los días de estancia (OLS)
    - **Sección 3b — Objetivos**:
      - General: Caracterizar y modelar los factores asociados a hospitalización usando datos GRD
      - Específicos: (1) Comparar días de estancia entre hospitales, (2) Modelar predictores de mortalidad, (3) Modelar predictores de días de estancia, (4) Explorar patrones avanzados (multinivel, oncología, IDH, clustering)
  - Cada hipótesis: justificación basada en EDA del notebook
  - Objetivos específicos numerados y vinculados a cada hipótesis

  **Must NOT do**:
  - NO incluir resultados en esta sección (es solo planteamiento)
  - NO usar valores numéricos que pertenecen a resultados
  - NO formular hipótesis adicionales no analizadas en el notebook

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Redacción académica, formulación de hipótesis, estructura científica
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T5, T7, T8, T9)
  - **Blocks**: T11
  - **Blocked By**: T2

  **References**:
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas de planteamiento de hipótesis y EDA exploratorio
  - `Entrega_Final/Rúbricas/` — GUIA_INFORME sección 3
  - `Entrega_Final/referencia_numerica.csv` — Confirmar que hipótesis cubren todos los análisis

  **Acceptance Criteria**:
  - [ ] Archivo `secciones/seccion_2_prob_hip_obj.md` existe
  - [ ] Contiene problema de investigación (contexto chileno, GRD)
  - [ ] Contiene 3 hipótesis (H1, H2, H3) claramente formuladas
  - [ ] Contiene objetivo general + 4 objetivos específicos
  - [ ] Cada objetivo vinculado a una hipótesis

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — estructura completa y coherente
    Tool: Bash (grep)
    Steps:
      1. grep "H1" secciones/seccion_2_prob_hip_obj.md → menciona Kruskal-Wallis o diferencias entre hospitales
      2. grep "H2" secciones/seccion_2_prob_hip_obj.md → menciona regresión logística o mortalidad
      3. grep "H3" secciones/seccion_2_prob_hip_obj.md → menciona regresión lineal o días de estancia
      4. grep "Objetivo general" secciones/seccion_2_prob_hip_obj.md → encontrado
      5. grep "Objetivo específico" secciones/seccion_2_prob_hip_obj.md → al menos 4 ocurrencias
    Expected Result: 3 hipótesis, 1 objetivo general, 4+ objetivos específicos, todos coherentes
    Failure Indicators: Hipótesis sin método asociado, objetivos sin hipótesis, sección vacía
    Evidence: .sisyphus/evidence/task-6-section2-verify.txt
  ```

  **Commit**: YES (grupo T5-T10)
  - Message: `docs(informe): write problem statement, hypotheses, and objectives`
  - Files: `secciones/seccion_2_prob_hip_obj.md`

- [ ] 7. **Redactar DOCX español — Metodología completa**

  **What to do**:
  - Escribir archivo `Entrega_Final/secciones/seccion_3_metodologia.md` con todas las subsecciones requeridas:
    - **Diseño de estudio**: Observacional, retrospectivo, transversal, análisis de datos secundarios (GRD)
    - **Población y muestra**: Descripción de la base GRD, criterios de inclusión/exclusión del notebook, n final, justificación del tamaño muestral
    - **Variables**: 
      - Dependientes: días_estancia (continua), mortalidad (binaria)
      - Independientes principales: n_procedimientos, edad, sexo, urgencia, HDI_comuna
      - Control/agrupación: hospital_id (para multinivel), diagnostico_grupo (C16, C33-C34, C61 para oncología)
      - Operacionalización de cada variable con tipo y codificación
    - **Instrumentos**: Python 3, pandas, numpy, scipy, statsmodels, sklearn, scikit-posthocs, matplotlib, seaborn
    - **Procedimientos**: Pipeline de limpieza de datos del notebook (missing values, outliers, encoding)
    - **Aspectos éticos**: Datos anonimizados, uso secundario, no requiere consentimiento informado
    - **Plan de análisis**: 
      - Análisis descriptivo (medianas, frecuencias, visualizaciones)
      - Análisis bivariado (chi-cuadrado exploratorio)
      - H1: Kruskal-Wallis + Dunn post-hoc (ε² para tamaño de efecto)
      - H2: Regresión logística binaria (OR, IC 95%, AUC, matriz de confusión)
      - H3: Regresión lineal múltiple (β, R², VIF, diagnóstico de supuestos)
      - Análisis avanzados: multinivel (ICC), ANOVA/KW oncológico, IDH por quintiles, K-means clustering, SMOTE

  **Must NOT do**:
  - NO incluir resultados en metodología
  - NO omitir ningún método usado en el notebook (todos deben describirse)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Redacción técnica de metodología estadística, requiere precisión en nomenclatura de pruebas
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T5, T6, T8, T9)
  - **Blocks**: T11
  - **Blocked By**: T2

  **References**:
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Pipeline de análisis completo
  - `Entrega_Final/Rúbricas/` — GUIA_INFORME sección 4 (metodología)
  - `Entrega_Final/referencia_numerica.csv` — Confirmar que todos los métodos tienen valores asociados

  **Acceptance Criteria**:
  - [ ] Archivo `secciones/seccion_3_metodologia.md` existe
  - [ ] Contiene TODAS las subsecciones: diseño, población, variables (con tabla de operacionalización), instrumentos, procedimientos, aspectos éticos, plan de análisis
  - [ ] Cada variable definida con tipo, codificación y rol (dependiente/independiente/control)
  - [ ] Plan de análisis cubre TODOS los métodos del notebook (KW, Logit, OLS, multinivel, KW oncológico, IDH, K-means, SMOTE)
  - [ ] Sección de aspectos éticos presente

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — metodología exhaustiva
    Tool: Bash (grep)
    Steps:
      1. grep -ci "diseño" secciones/seccion_3_metodologia.md → ≥1
      2. grep -ci "población" secciones/seccion_3_metodologia.md → ≥1
      3. grep -ci "variable" secciones/seccion_3_metodologia.md → ≥3
      4. grep -ci "kruskal" secciones/seccion_3_metodologia.md → ≥1
      5. grep -ci "regresión logística" secciones/seccion_3_metodologia.md → ≥1
      6. grep -ci "regresión lineal" secciones/seccion_3_metodologia.md → ≥1
      7. grep -ci "multinivel" secciones/seccion_3_metodologia.md → ≥1
      8. grep -ci "SMOTE" secciones/seccion_3_metodologia.md → ≥1
      9. grep -ci "éticos\|ética\|consentimiento" secciones/seccion_3_metodologia.md → ≥1
    Expected Result: Los 9 grep retornan ≥1, confirmando cobertura completa
    Failure Indicators: Algún método del notebook no mencionado, ausencia de aspectos éticos
    Evidence: .sisyphus/evidence/task-7-metodologia-verify.txt
  ```

  **Commit**: YES (grupo T5-T10)
  - Message: `docs(informe): write complete methodology section`
  - Files: `secciones/seccion_3_metodologia.md`

- [ ] 8. **Redactar DOCX español — Resultados: EDA + Descriptivos + H1 Kruskal-Wallis**

  **What to do**:
  - Escribir archivo `Entrega_Final/secciones/seccion_4_resultados_parte1.md` con:
    - **Análisis Exploratorio de Datos (EDA)**:
      - Describir e insertar figuras 1-6 del notebook (tendencias temporales, distribuciones, boxplots)
      - Tabla 1: Estadísticos descriptivos de variables demográficas y clínicas (n, %, media, mediana, SD, min-max)
      - Tabla 2: Características por hospital (n pacientes, % mortalidad, mediana días estancia, % urgencia)
      - Chi-cuadrado exploratorio: mortalidad×sexo (χ², p=0.099), mortalidad×urgencia (χ², p<0.001), alta×hospital (χ², p<0.001)
    - **Resultados H1 — Kruskal-Wallis**:
      - H=1909.40, ε²=0.146, p<0.001
      - Tabla 3: Medianas y rangos intercuartílicos de días de estancia por hospital (top 15)
      - Figura 7: Boxplot días de estancia por hospital
      - Figura 8: Violin plot con medianas
      - Dunn post-hoc: resultados de comparaciones múltiples (hospitales con diferencias significativas)
      - Interpretación: tamaño de efecto ε²=0.146 indica diferencia moderada entre hospitales
  - Cada tabla: diseñada para insertar en DOCX (formato markdown)
  - Cada figura: ruta exacta a `outputs/figuras/figuraN_*.png`
  - Todas las referencias a figuras usan "Figura N" (no "Figure N")

  **Must NOT do**:
  - NO usar valores antiguos del LaTeX (H=412.3, ε²=0.08)
  - NO reportar significancia sin tamaño de efecto (ε²)
  - NO omitir resultados no significativos (chi-cuadrado sexo p=0.099 debe incluirse)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Redacción de resultados estadísticos con precisión numérica, descripción de figuras
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T5, T6, T7, T9)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:
  - `Entrega_Final/referencia_numerica.csv` — Valores exactos de H, ε², chi-cuadrado
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas EDA y H1
  - `outputs/figuras/figura1_*` a `figura8_*` — Figuras EDA y H1
  - `outputs/tablas/` — CSVs con datos para tablas descriptivas

  **Acceptance Criteria**:
  - [ ] Archivo `secciones/seccion_4_resultados_parte1.md` existe
  - [ ] Contiene EDA con 6+ figuras referenciadas y descritas
  - [ ] Tabla 1 y Tabla 2 con estadísticos descriptivos completos
  - [ ] Chi-cuadrado con 3 escenarios (mortalidad×sexo, mortalidad×urgencia, alta×hospital)
  - [ ] H1: H=1909.40, ε²=0.146, p<0.001 correctos
  - [ ] Dunn post-hoc con resultados de comparaciones significativas
  - [ ] 8+ figuras referenciadas con rutas verificables

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — valores H1 correctos y figuras existen
    Tool: Bash
    Steps:
      1. grep "1909.40" secciones/seccion_4_resultados_parte1.md → encontrado
      2. grep "0.146" secciones/seccion_4_resultados_parte1.md → encontrado (ε²)
      3. grep "p.*<.*0.001" secciones/seccion_4_resultados_parte1.md → encontrado
      4. grep "Dunn" secciones/seccion_4_resultados_parte1.md → encontrado
      5. Extraer todas las rutas de figuras: grep -oP 'outputs/figuras/[^)\s]+' secciones/seccion_4_resultados_parte1.md | while read f; do [ -f "$f" ] && echo "OK: $f" || echo "MISSING: $f"; done → todas OK
    Expected Result: Valores correctos, Dunn presente, figuras existen
    Failure Indicators: H=412.3 o ε²=0.08 (valores antiguos), figuras inexistentes
    Evidence: .sisyphus/evidence/task-8-results-h1-verify.txt

  Scenario: Edge case — verificar chi-cuadrado no significativo incluido
    Tool: Bash (grep)
    Steps:
      1. grep "sexo.*0.099\|0.099.*sexo" secciones/seccion_4_resultados_parte1.md → encontrado
      2. Verificar que se menciona como "no significativo" o "p > 0.05"
    Expected Result: χ² sexo p=0.099 reportado y correctamente interpretado
    Evidence: .sisyphus/evidence/task-8-chisq-nonsig.txt
  ```

  **Commit**: YES (grupo T5-T10)
  - Message: `docs(informe): write results: EDA, descriptive statistics, and H1 Kruskal-Wallis`
  - Files: `secciones/seccion_4_resultados_parte1.md`

- [ ] 9. **Redactar DOCX español — Resultados: H2 Logit + H3 OLS + Análisis Avanzados**

  **What to do**:
  - Escribir archivo `Entrega_Final/secciones/seccion_5_resultados_parte2.md` con:
    - **Resultados H2 — Regresión Logística (Mortalidad)**:
      - AUC=0.849, accuracy, sensitivity, specificity
      - Tabla 4: OR, IC 95%, p-valor por variable (procedures OR=1.053, edad, sexo, urgencia, HDI)
      - Figura 11: Matriz de confusión
      - Figura 12: Curva ROC
      - Interpretación: n° procedimientos es predictor significativo (OR=1.053, p<0.001)
    - **Resultados H3 — Regresión Lineal (Días de Estancia)**:
      - R²=0.633, R² ajustado, F-statistic
      - Tabla 5: Coeficientes β, error estándar, t, p-valor, VIF por variable
      - β procedimientos = 0.0924 (cada procedimiento adicional → +9.7% días estancia)
      - Diagnóstico de supuestos: normalidad (Q-Q plot), homocedasticidad, multicolinealidad (VIF<5)
      - Figuras 14-16: Q-Q, residuos vs ajustados, leverage
    - **Resultados Análisis Avanzados**:
      - **Multinivel ICC**: Valor corregido de T3, interpretación (efecto hospital)
      - **Oncología**: Comparación C16 (estómago) vs C33-C34 (pulmón) vs C61 (próstata) — KW entre grupos, medias/medianas días estancia y mortalidad por grupo
      - **IDH por quintiles**: Gradiente Q1→Q5 en mortalidad y días estancia
      - **K-means clustering**: k=3, silhouette score, caracterización de clusters (perfiles de pacientes)
      - **SMOTE**: Métricas pre/post SMOTE corregidas de T4, ROC comparativa
  - Formato consistente con T8 (mismas convenciones de tablas y figuras)
  - Usar `referencia_numerica.csv` para TODOS los valores

  **Must NOT do**:
  - NO reportar AUC=0.823 o R²=0.636 (valores LaTeX antiguos)
  - NO reportar ICC=0.000 (usar valor corregido de T3)
  - NO omitir métricas de diagnóstico de modelos (VIF, Q-Q, residuos)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Redacción técnica de resultados de regresión y análisis multivariados
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T5, T6, T7, T8)
  - **Blocks**: T11
  - **Blocked By**: T2, T3, T4

  **References**:
  - `Entrega_Final/referencia_numerica.csv` — Todos los valores de H2, H3, avanzados
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas H2, H3, avanzados
  - `outputs/figuras/figura11_*` a `figura21_*` — Figuras de regresión y avanzados
  - `outputs/tablas/` — CSVs de resultados de modelos
  - `outputs/icc_multinivel_resultados.csv` — ICC corregido de T3
  - `outputs/smote_resultados.csv` — SMOTE corregido de T4

  **Acceptance Criteria**:
  - [ ] Archivo `secciones/seccion_5_resultados_parte2.md` existe
  - [ ] H2: AUC=0.849, OR procedures=1.053, matriz de confusión, ROC
  - [ ] H3: R²=0.633, β procedures=0.0924, VIF<5, diagnóstico de supuestos
  - [ ] Multinivel ICC con valor corregido (≠0.000)
  - [ ] Oncología: 3 subgrupos comparados (C16, C33-C34, C61)
  - [ ] IDH: gradiente Q1→Q5 descrito
  - [ ] K-means: k=3, silhouette score, caracterización de clusters
  - [ ] SMOTE: métricas pre/post con validación correcta
  - [ ] 8+ figuras referenciadas con rutas verificables

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — valores H2, H3, avanzados correctos
    Tool: Bash
    Steps:
      1. grep "0.849" secciones/seccion_5_resultados_parte2.md → encontrado (AUC)
      2. grep "1.053" secciones/seccion_5_resultados_parte2.md → encontrado (OR)
      3. grep "0.633" secciones/seccion_5_resultados_parte2.md → encontrado (R²)
      4. grep "0.0924" secciones/seccion_5_resultados_parte2.md → encontrado (β)
      5. grep "C16.*C33.*C61\|estómago.*pulmón.*próstata" secciones/seccion_5_resultados_parte2.md → encontrado (oncología)
      6. grep "silhouette\|silueta" secciones/seccion_5_resultados_parte2.md → encontrado (K-means)
      7. grep "ICC" secciones/seccion_5_resultados_parte2.md → valor ≠ 0.000
      8. grep "SMOTE" secciones/seccion_5_resultados_parte2.md → presente
      9. Extraer rutas de figuras y verificar existencia (≥8 figuras)
    Expected Result: Los 9 checks pasan, todas las figuras existen
    Failure Indicators: Valor AUC=0.823 o R²=0.636 (antiguos), ICC=0.000, figuras faltantes
    Evidence: .sisyphus/evidence/task-9-results-h2h3-verify.txt

  Scenario: Edge case — verificar que SMOTE describe data leakage prevention
    Tool: Bash (grep)
    Steps:
      1. grep -i "train.*test\|entrenamiento.*prueba" secciones/seccion_5_resultados_parte2.md → encontrado
      2. grep -i "SMOTE.*solo.*train\|SMOTE.*entrenamiento" secciones/seccion_5_resultados_parte2.md → encontrado
    Expected Result: Texto indica explícitamente que SMOTE se aplicó solo a datos de entrenamiento
    Evidence: .sisyphus/evidence/task-9-smote-data-leakage-check.txt
  ```

  **Commit**: YES (grupo T5-T10)
  - Message: `docs(informe): write results: H2 Logit, H3 OLS, and advanced analyses`
  - Files: `secciones/seccion_5_resultados_parte2.md`

- [ ] 10. **Redactar DOCX español — Discusión, Conclusiones, Referencias, Anexos**

  **What to do**:
  - Escribir archivo `Entrega_Final/secciones/seccion_6_disc_conc_ref.md` con:
    - **Sección 7 — Discusión**:
      - Interpretación de resultados principales en contexto de literatura
      - H1: Variabilidad entre hospitales (ε²=0.146) — ¿es clínicamente relevante? Comparar con benchmarks
      - H2: n° procedimientos como predictor de mortalidad (OR=1.053) — interpretación clínica
      - H3: Predictores de días de estancia (R²=0.633) — ¿explica suficiente varianza?
      - Hallazgos avanzados: ICC (efecto hospital), oncología (diferencias entre tipos de cáncer), IDH (gradiente socioeconómico), clusters (perfiles de pacientes)
      - **Limitaciones**: Datos secundarios (GRD), no hay variables clínicas detalladas (comorbilidades, severidad), diseño transversal, SMOTE con limitaciones en clase muy minoritaria
      - Comparación con literatura: citar 2-3 estudios previos relevantes (APA)
    - **Sección 8 — Conclusiones**:
      - Responder cada hipótesis explícitamente:
        - H1: SÍ, diferencias significativas entre hospitales (ε²=0.146, p<0.001)
        - H2: SÍ, variables clínicas predicen mortalidad (AUC=0.849)
        - H3: SÍ, modelo explica 63.3% de varianza en días de estancia
      - Implicaciones prácticas para gestión hospitalaria
      - Recomendaciones para investigación futura
    - **Sección 9 — Referencias**:
      - 6+ referencias en formato APA 7ª edición
      - Incluir: fuentes de datos (MINSAL/GRD), paquetes de Python (statsmodels, sklearn, scipy), literatura de hospitalización
    - **Anexos** (si espacio lo permite):
      - Tablas completas de resultados (Dunn post-hoc, VIF, estadísticos por hospital)

  **Must NOT do**:
  - NO hacer afirmaciones causales (diseño observacional)
  - NO extrapolar conclusiones más allá de los datos
  - NO omitir limitaciones del estudio
  - NO usar menos de 6 referencias APA

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Discusión científica con interpretación crítica, conclusiones basadas en evidencia, referencias APA
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11, T12, T13)
  - **Blocks**: T11
  - **Blocked By**: T2

  **References**:
  - `Entrega_Final/referencia_numerica.csv` — Valores para discusión y conclusiones
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Celdas de discusión (si existen)
  - `Entrega_Final/Rúbricas/` — GUIA_INFORME secciones 7-9 + criterios de "Excelente"
  - `Entrega_Final/Informe_Final_AmatHerreraRodriguez.docx` — Referencias existentes (verificar si hay APA)
  - `Entrega_Final/main_paper.tex` — Referencias en LaTeX para adaptar

  **Acceptance Criteria**:
  - [ ] Archivo `secciones/seccion_6_disc_conc_ref.md` existe
  - [ ] Discusión: interpreta H1, H2, H3 + avanzados con contexto clínico
  - [ ] Discusión: incluye sección de limitaciones (mínimo 4 limitaciones)
  - [ ] Discusión: compara con al menos 2 estudios previos
  - [ ] Conclusiones: responde H1, H2, H3 explícitamente (SÍ/NO + evidencia)
  - [ ] Conclusiones: implicaciones prácticas y recomendaciones
  - [ ] Referencias: 6+ en formato APA 7ª
  - [ ] Anexos: tablas complementarias (si ≤12 páginas lo permite)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — discusión y conclusiones completas
    Tool: Bash (grep)
    Steps:
      1. grep -ci "limitación\|limitante" secciones/seccion_6_disc_conc_ref.md → ≥3
      2. grep "H1.*sí\|H1.*SÍ\|H1.*significativa" secciones/seccion_6_disc_conc_ref.md → encontrado
      3. grep "H2.*sí\|H2.*SÍ\|H2.*significativa" secciones/seccion_6_disc_conc_ref.md → encontrado
      4. grep "H3.*sí\|H3.*SÍ\|H3.*significativa" secciones/seccion_6_disc_conc_ref.md → encontrado
      5. grep -cP '\(\d{4}\)' secciones/seccion_6_disc_conc_ref.md → ≥6 (referencias APA con año)
    Expected Result: Limitaciones identificadas, 3 hipótesis respondidas, ≥6 refs APA
    Failure Indicators: Hipótesis sin respuesta explícita, sin limitaciones, <6 referencias
    Evidence: .sisyphus/evidence/task-10-disc-conc-verify.txt

  Scenario: Edge case — verificar que NO hay afirmaciones causales
    Tool: Bash (grep)
    Steps:
      1. grep -i "causa\|provoca\|determina\|efecto directo" secciones/seccion_6_disc_conc_ref.md → contar
      2. Si >0, verificar contexto (¿se matiza como asociación o se afirma causalidad?)
    Expected Result: Lenguaje correlacional/asociativo, no causal (ej: "se asocia con", no "causa")
    Evidence: .sisyphus/evidence/task-10-no-causal-claims.txt
  ```

  **Commit**: YES (grupo T5-T10)
  - Message: `docs(informe): write discussion, conclusions, references, and appendices`
  - Files: `secciones/seccion_6_disc_conc_ref.md`

- [ ] 11. **Ensamblar DOCX español con python-docx**

  **What to do**:
  - Crear script `Entrega_Final/scripts/ensamblar_docx.py` que:
    1. Importe `python-docx` y cree documento en blanco
    2. Configure estilos personalizados:
       - Tema de color: verde Lancet `#006046` para headings, `#2D7D64` para subheadings, gris `#666666` para texto secundario
       - Fuente: Calibri 12pt cuerpo, 11pt tablas, negrita para headings
       - Espaciado: 1.5 interlineado, márgenes 2.54cm (APA)
       - Heading 1: 14pt bold verde oscuro
       - Heading 2: 13pt bold verde medio
       - Heading 3: 12pt bold verde claro
    3. Procese cada archivo de sección (`secciones/seccion_*_*.md`) secuencialmente:
       - Parsear markdown (headings → Word headings, párrafos → párrafos, listas → bullets)
       - Insertar figuras: detectar `outputs/figuras/figuraN_*` → `doc.add_picture()` con caption "Figura N: descripción"
       - Insertar tablas: detectar tablas markdown → crear `doc.add_table()` con borde verde sutil
       - Ecuaciones: detectar `$...$` o `$$...$$` → inline Unicode (χ², ε², β̂, R²) o imagen renderizada
    4. Añadir numeración automática de páginas en footer
    5. Añadir encabezado con título abreviado
    6. Insertar saltos de página entre secciones principales
    7. Generar Tabla de Contenidos (ToC) al inicio (después de portada)
    8. Guardar como `Informe_Final_AmatHerreraRodriguez_v2.docx`
  - Verificar que el documento resultante:
    - Tiene ≤12 páginas
    - Contiene ≥13 figuras insertadas
    - Contiene todas las tablas (mínimo 5)
    - Las secciones siguen el orden GUIA_INFORME

  **Must NOT do**:
  - NO usar diseño a dos columnas (columna única APA)
  - NO usar ecuaciones OMML (incompatibles con LibreOffice)
  - NO insertar figuras como links (incrustar en documento)
  - NO usar macros VBA

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: python-docx avanzado con estilos personalizados, inserción de figuras y tablas, parseo markdown, formato APA
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T12, T13)
  - **Blocks**: T14 (pulido español)
  - **Blocked By**: T5, T6, T7, T8, T9, T10

  **References**:
  - `secciones/seccion_1_portada_intro.md` a `secciones/seccion_6_disc_conc_ref.md` — Contenido a ensamblar
  - `outputs/figuras/` — Todas las figuras a insertar
  - `Entrega_Final/referencia_numerica.csv` — Validación de valores
  - `python-docx` documentation — API para estilos, tablas, imágenes
  - `Entrega_Final/main_paper.tex` — Referencia de colores Lancet (#006046)

  **Acceptance Criteria**:
  - [ ] `Informe_Final_AmatHerreraRodriguez_v2.docx` generado exitosamente
  - [ ] Documento abre sin errores en LibreOffice y Word
  - [ ] ≤12 páginas (incluyendo portada, figuras, tablas, referencias)
  - [ ] ≥13 figuras incrustadas (no vinculadas)
  - [ ] ≥5 tablas con formato consistente
  - [ ] 12 secciones GUIA_INFORME en orden correcto
  - [ ] Colores Lancet aplicados en headings y bordes de tabla
  - [ ] ToC funcional (actualizable en Word)
  - [ ] Números de página en footer

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — DOCX generado con todas las secciones
    Tool: Bash (python3)
    Preconditions: DOCX generado
    Steps:
      1. python3 -c "
from docx import Document
doc = Document('Entrega_Final/Informe_Final_AmatHerreraRodriguez_v2.docx')
headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
print(f'Headings: {len(headings)}')
for h in headings: print(f'  - {h}')
"
      2. Verificar ≥12 headings
      3. Conteo de figuras: python3 -c "
from docx import Document
doc = Document('...')
images = sum(1 for rel in doc.part.rels.values() if 'image' in rel.reltype)
print(f'Images: {images}')
" → ≥13
      4. Conteo de tablas: python3 -c "
from docx import Document
doc = Document('...')
print(f'Tables: {len(doc.tables)}')
" → ≥5
    Expected Result: ≥12 headings, ≥13 imágenes, ≥5 tablas
    Failure Indicators: <12 headings, <13 imágenes, <5 tablas, error al abrir DOCX
    Evidence: .sisyphus/evidence/task-11-docx-structure.txt

  Scenario: Edge case — verificar formato APA y colores Lancet
    Tool: Bash (python3)
    Steps:
      1. Verificar color headings: python3 -c "
from docx import Document
from docx.shared import RGBColor
doc = Document('...')
for p in doc.paragraphs:
    if p.style.name.startswith('Heading') and p.runs:
        color = p.runs[0].font.color.rgb if p.runs[0].font.color and p.runs[0].font.color.rgb else 'default'
        print(f'{p.style.name}: {color}')
"
      2. Verificar que algún heading usa RGBColor(0, 96, 70) o similar (#006046)
      3. Verificar márgenes APA (2.54cm): python3 -c "
from docx import Document
doc = Document('...')
for section in doc.sections:
    print(f'Margins: L={section.left_margin}, R={section.right_margin}, T={section.top_margin}, B={section.bottom_margin}')
"
    Expected Result: Color verde en headings, márgenes ~2.54cm (aprox 914400 EMUs)
    Evidence: .sisyphus/evidence/task-11-docx-format.txt

  Scenario: Negative — verificar que NO usa dos columnas
    Tool: Bash (python3)
    Steps:
      1. python3 -c "
from docx import Document
from docx.oxml.ns import qn
doc = Document('...')
for section in doc.sections:
    cols = section._sectPr.find(qn('w:cols'))
    if cols is not None:
        num = cols.get(qn('w:num'), '1')
        print(f'Columns: {num}')
    else:
        print('Single column (default)')
"
    Expected Result: Single column (num=1 o sin elemento cols)
    Evidence: .sisyphus/evidence/task-11-single-column.txt
  ```

  **Commit**: YES (grupo T11)
  - Message: `feat(docx): assemble Spanish DOCX with python-docx and Lancet styling`
  - Files: `scripts/ensamblar_docx.py`, `Informe_Final_AmatHerreraRodriguez_v2.docx`

- [ ] 12. **Corregir valores numéricos en LaTeX inglés**

  **What to do**:
  - Crear copia: `cp main_paper.tex main_paper_v2.tex` (respetando .bak original)
  - Actualizar TODOS los valores numéricos discrepantes en `main_paper_v2.tex` usando `referencia_numerica.csv`:
    - Resumen/Abstract: KW H=412.3→1909.40, ε²=0.08→0.146, AUC=0.823→0.849, R²=0.636→0.633, mortalidad 4.05%→5.56%
    - Cuerpo del paper: todos los valores de tablas y texto
    - Tablas LaTeX: actualizar valores en entornos `tabular`
    - Si el LaTeX usa top-15 hospitales para regresión (y notebook usa todos), añadir nota explicativa
  - Añadir contenido nuevo no presente en LaTeX original:
    - Subsección de oncología (C16, C33-C34, C61)
    - Subsección de IDH por quintiles
    - Subsección de SMOTE (con resultados corregidos de T4)
    - Actualizar ICC con valor corregido de T3
  - Compilar LaTeX para verificar que no hay errores:
    ```bash
    cd Entrega_Final && pdflatex main_paper_v2.tex
    ```
  - Verificar que todas las figuras referenciadas existen

  **Must NOT do**:
  - NO modificar `main_paper.tex` original
  - NO cambiar la estructura del documento (secciones, formato)
  - NO eliminar contenido existente (solo actualizar números y añadir)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Edición de valores en archivo de texto, búsqueda y reemplazo sistemático
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T11, T13)
  - **Blocks**: T13 (generación DOCX inglés)
  - **Blocked By**: T2 (referencia numérica)

  **References**:
  - `Entrega_Final/main_paper.tex` — Original (NO modificar)
  - `Entrega_Final/referencia_numerica.csv` — Valores corregidos (fuente de verdad)
  - `Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb` — Contenido para nuevas subsecciones
  - `outputs/figuras/` — Verificar que figuras referenciadas existen

  **Acceptance Criteria**:
  - [ ] `main_paper_v2.tex` existe (copia de main_paper.tex con cambios)
  - [ ] `main_paper.tex` NO modificado (verificable con diff)
  - [ ] `pdflatex main_paper_v2.tex` compila sin errores
  - [ ] Valores actualizados: H=1909.40, ε²=0.146, AUC=0.849, R²=0.633, mortalidad=5.56%
  - [ ] Nuevas subsecciones añadidas: oncología, IDH, SMOTE
  - [ ] ICC actualizado al valor corregido de T3

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — LaTeX compila y valores correctos
    Tool: Bash
    Steps:
      1. cd Entrega_Final && pdflatex -interaction=nonstopmode main_paper_v2.tex
      2. Verificar exit code = 0
      3. Verificar no hay "undefined reference" warnings
      4. grep "1909.40" main_paper_v2.tex → encontrado
      5. grep "0.146" main_paper_v2.tex → encontrado (ε²)
      6. grep "0.849" main_paper_v2.tex → encontrado (AUC)
      7. grep "5.56" main_paper_v2.tex → encontrado (mortalidad)
      8. grep -c "SMOTE\|oncolog\|IDH\|Human Development" main_paper_v2.tex → ≥3 (nuevas subsecciones)
    Expected Result: Compilación exitosa, valores corregidos, contenido nuevo presente
    Failure Indicators: Error de compilación, valores antiguos (412.3, 0.08, 0.823), falta contenido nuevo
    Evidence: .sisyphus/evidence/task-12-latex-verify.txt (output de compilación + greps)

  Scenario: Edge case — original intacto
    Tool: Bash
    Steps:
      1. diff main_paper.tex main_paper_v2.tex → muestra diferencias (esperado)
      2. diff main_paper.tex main_paper.tex.bak → sin diferencias (backup de T1)
    Expected Result: main_paper.tex no tiene cambios, main_paper_v2.tex sí tiene cambios
    Evidence: .sisyphus/evidence/task-12-original-intact.txt
  ```

  **Commit**: YES (grupo T12)
  - Message: `fix(latex): correct all numerical values and add missing analyses`
  - Files: `main_paper_v2.tex`

- [ ] 13. **Generar DOCX inglés desde LaTeX corregido + profundizar**

  **What to do**:
  - Convertir `main_paper_v2.tex` a DOCX:
    - Opción A (preferida): `pandoc main_paper_v2.tex -o main_paper_v2.docx --reference-doc=template_lancet.docx` (si existe template)
    - Opción B: python-docx manual replicando estructura Lancet dos columnas
  - Si es python-docx:
    - Crear script `scripts/ensamblar_docx_english.py`
    - Configurar página tamaño A4 o Letter (según Lancet), dos columnas (vía XML en python-docx)
    - Parsear secciones del LaTeX corregido
    - Aplicar mismos estilos Lancet que versión española pero en inglés
  - Profundizar contenido con análisis del notebook:
    - Añadir oncología subgrupos
    - Añadir IDH quintiles
    - Añadir SMOTE
    - Añadir tabla completa de Dunn post-hoc
  - Guardar como `main_paper_v2.docx`

  **Must NOT do**:
  - NO perder el formato Lancet de dos columnas
  - NO usar valores antiguos (debe usar main_paper_v2.tex corregido)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Conversión LaTeX a DOCX con preservación de formato dos columnas, posible manipulación XML
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T11, T12)
  - **Blocks**: T15 (pulido inglés)
  - **Blocked By**: T12 (LaTeX corregido)

  **References**:
  - `Entrega_Final/main_paper_v2.tex` — LaTeX corregido de T12
  - `Entrega_Final/main_paper.tex` — Estructura original de referencia
  - `Entrega_Final/referencia_numerica.csv` — Valores para profundización
  - `outputs/figuras/` — Figuras a insertar

  **Acceptance Criteria**:
  - [ ] `main_paper_v2.docx` generado exitosamente
  - [ ] Documento abre sin errores
  - [ ] Formato dos columnas preservado (o justificado si no es viable)
  - [ ] Valores numéricos coinciden con main_paper_v2.tex (corregidos)
  - [ ] Contenido profundizado: oncología, IDH, SMOTE presentes

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — DOCX inglés generado con formato Lancet
    Tool: Bash (python3)
    Steps:
      1. python3 -c "
from docx import Document
doc = Document('Entrega_Final/main_paper_v2.docx')
print(f'Paragraphs: {len(doc.paragraphs)}')
print(f'Tables: {len(doc.tables)}')
headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
print(f'Headings: {len(headings)}')
"
      2. Verificar ≥4 secciones (Abstract, Introduction, Results, Discussion/Conclusion)
      3. grep -c "SMOTE\|oncology\|IDH\|Human Development" (en texto extraído) → ≥2
      4. Verificar valores corregidos en texto extraído
    Expected Result: DOCX válido, secciones presentes, contenido profundizado
    Failure Indicators: Error al abrir, <3 secciones, sin contenido nuevo
    Evidence: .sisyphus/evidence/task-13-english-docx-verify.txt

  Scenario: Edge case — verificar formato dos columnas (si se usa python-docx)
    Tool: Bash (python3)
    Steps:
      1. python3 -c "
from docx import Document
from docx.oxml.ns import qn
doc = Document('...')
for section in doc.sections:
    cols = section._sectPr.find(qn('w:cols'))
    if cols is not None:
        num = cols.get(qn('w:num'), '1')
        print(f'Columns: {num}')
"
    Expected Result: 2 columnas o justificado (si 2 columnas no fue viable, documentar razón)
    Evidence: .sisyphus/evidence/task-13-columns-check.txt
  ```

  **Commit**: YES (grupo T13)
  - Message: `feat(docx): generate English DOCX from corrected LaTeX with deepened content`
  - Files: `main_paper_v2.docx`, `scripts/ensamblar_docx_english.py` (si aplica)

- [ ] 14. **Pulido final DOCX español (ToC, APA compliance, QA final)**

  **What to do**:
  - Abrir `Informe_Final_AmatHerreraRodriguez_v2.docx`
  - Verificar y corregir:
    - Tabla de Contenidos actualizada y correcta
    - Numeración de páginas en footer (números arábigos, centrados)
    - Encabezado con título abreviado
    - Numeración secuencial de figuras (Figura 1, Figura 2, ...)
    - Numeración secuencial de tablas (Tabla 1, Tabla 2, ...)
    - Referencias cruzadas en texto: "Figura X" y "Tabla Y" corresponden a las correctas
    - Formato APA 7ª:
      - Márgenes 2.54cm (1 pulgada) en los 4 lados
      - Interlineado 2.0 en todo el documento (excepto tablas/figuras)
      - Sangría en primera línea de cada párrafo (1.27cm)
      - Referencias con sangría francesa (hanging indent)
      - Citas en texto: (Autor, Año)
    - Consistencia de fuentes: Calibri 12pt cuerpo, 11pt tablas
    - Colores Lancet consistentes en todos los headings
    - Sin párrafos huérfanos o viudos al final de página
    - Todas las figuras tienen caption descriptivo
    - Todas las tablas tienen título y nota al pie si aplica
    - ≤12 páginas (si excede, compactar sin perder contenido: reducir espaciado de figuras, combinar anexos)
  - Guardar versión final pulida

  **Must NOT do**:
  - NO añadir contenido nuevo (solo correcciones de formato)
  - NO cambiar valores numéricos
  - NO modificar el orden de secciones

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verificación y corrección de formato, sin cambios de contenido
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T15)
  - **Blocks**: None (última tarea de español)
  - **Blocked By**: T11 (ensamblaje)

  **References**:
  - `Entrega_Final/Informe_Final_AmatHerreraRodriguez_v2.docx` — Documento a pulir
  - `Entrega_Final/Rúbricas/` — GUIA_INFORME + criterios de formato
  - APA 7th Edition guidelines — Márgenes, interlineado, sangrías, referencias
  - `Entrega_Final/main_paper.tex` — Referencia de colores Lancet

  **Acceptance Criteria**:
  - [ ] DOCX abre sin errores
  - [ ] ≤12 páginas
  - [ ] ≥13 figuras con numeración secuencial y captions
  - [ ] ≥5 tablas con numeración secuencial y títulos
  - [ ] ToC funcional con números de página correctos
  - [ ] Encabezado y pie de página presentes
  - [ ] Formato APA verificado (márgenes, interlineado, sangrías, referencias)
  - [ ] Colores Lancet consistentes
  - [ ] Sin párrafos huérfanos/viudos

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — documento cumple todas las especificaciones
    Tool: Bash (python3)
    Steps:
      1. python3 -c "
from docx import Document
from docx.shared import Inches, Cm
doc = Document('Entrega_Final/Informe_Final_AmatHerreraRodriguez_v2.docx')

# Page count (approximate by paragraph count / density)
total_chars = sum(len(p.text) for p in doc.paragraphs)
print(f'Total chars: {total_chars}')

# Count images
images = sum(1 for rel in doc.part.rels.values() if 'image' in rel.reltype)
print(f'Images: {images}')

# Count tables
print(f'Tables: {len(doc.tables)}')

# Check margins
for s in doc.sections:
    left_inches = s.left_margin / 914400
    right_inches = s.right_margin / 914400
    print(f'Margins: L={left_inches:.2f}in R={right_inches:.2f}in')
    assert abs(left_inches - 1.0) < 0.1, 'Left margin not 1 inch'

# Check headings
headings = [(p.style.name, p.text[:60]) for p in doc.paragraphs if p.style.name.startswith('Heading')]
print(f'Headings: {len(headings)}')
assert len(headings) >= 12, f'Only {len(headings)} headings'
for style, text in headings:
    print(f'  [{style}] {text}')
"
    Expected Result: ≥12 headings, ≥13 imágenes, ≥5 tablas, márgenes APA, sin assertion errors
    Failure Indicators: <12 headings, <13 imágenes, márgenes incorrectos
    Evidence: .sisyphus/evidence/task-14-final-qa.txt

  Scenario: Edge case — verificar referencias APA con sangría francesa
    Tool: Bash (python3)
    Steps:
      1. python3 -c "
from docx import Document
doc = Document('...')
# Find references section
in_refs = False
ref_paras = []
for p in doc.paragraphs:
    if 'referencia' in p.text.lower() or 'bibliografía' in p.text.lower():
        in_refs = True
        continue
    if in_refs and p.text.strip():
        ref_paras.append(p)
        if len(ref_paras) >= 6:
            break
print(f'Reference paragraphs found: {len(ref_paras)}')
for i, p in enumerate(ref_paras):
    fmt = p.paragraph_format
    indent = fmt.first_line_indent
    left = fmt.left_indent
    print(f'  Ref {i+1}: first_line={indent}, left={left}')
"
      2. Verificar hanging indent (first_line negativo o left_indent > 0)
    Expected Result: ≥6 referencias con formato APA (sangría francesa detectable)
    Evidence: .sisyphus/evidence/task-14-apa-refs.txt
  ```

  **Commit**: YES (grupo T14)
  - Message: `style(docx): final formatting polish for Spanish DOCX`
  - Files: `Informe_Final_AmatHerreraRodriguez_v2.docx`

- [ ] 15. **Pulido final DOCX inglés**

  **What to do**:
  - Abrir `main_paper_v2.docx`
  - Verificar y corregir:
    - Formato Lancet preservado (dos columnas, tipografía, colores)
    - Research in Context box (si existía en original) correctamente formateado
    - Structured abstract con headings correctos (Background, Methods, Findings, Interpretation)
    - Numeración de figuras y tablas secuencial
    - Referencias en formato Vancouver (estilo Lancet) o APA (según original)
    - Valores numéricos coinciden con `referencia_numerica.csv`
    - Sin errores de conversión LaTeX→DOCX (caracteres escapados, fórmulas rotas)
    - Contenido profundizado (oncología, IDH, SMOTE) está presente
  - Guardar versión final

  **Must NOT do**:
  - NO añadir contenido nuevo (solo correcciones de formato)
  - NO cambiar el estilo de referencias

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verificación y corrección de formato
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T14)
  - **Blocks**: None (última tarea de inglés)
  - **Blocked By**: T13 (generación DOCX inglés)

  **References**:
  - `Entrega_Final/main_paper_v2.docx` — Documento a pulir
  - `Entrega_Final/main_paper.tex` — Referencia de formato original
  - `Entrega_Final/referencia_numerica.csv` — Validación de valores

  **Acceptance Criteria**:
  - [ ] DOCX abre sin errores
  - [ ] Formato dos columnas funcional
  - [ ] Research in Context box presente (si existía)
  - [ ] Structured abstract con 4 secciones
  - [ ] ≥4 figuras insertadas
  - [ ] ≥4 tablas
  - [ ] Valores corregidos (AUC=0.849, ε²=0.146, R²=0.633)
  - [ ] Contenido profundizado presente (oncología, IDH, SMOTE)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — DOCX inglés cumple especificaciones Lancet
    Tool: Bash (python3)
    Steps:
      1. python3 -c "
from docx import Document
doc = Document('Entrega_Final/main_paper_v2.docx')
images = sum(1 for rel in doc.part.rels.values() if 'image' in rel.reltype)
print(f'Images: {images}')
print(f'Tables: {len(doc.tables)}')
headings = [p.text[:80] for p in doc.paragraphs if p.style.name.startswith('Heading')]
print(f'Headings: {len(headings)}')
for h in headings: print(f'  - {h}')
"
      2. Verificar ≥4 headings (Abstract, Introduction, Results, Discussion)
      3. Verificar valores corregidos en texto
    Expected Result: ≥4 headings, ≥4 imágenes, ≥4 tablas, valores corregidos
    Failure Indicators: Headings insuficientes, sin imágenes, valores antiguos
    Evidence: .sisyphus/evidence/task-15-english-qa.txt

  Scenario: Edge case — sin artefactos de conversión LaTeX
    Tool: Bash (python3)
    Steps:
      1. Extraer todo el texto del DOCX
      2. Buscar artefactos: grep -E '\\\\[a-zA-Z]+{|\\$\\$|\\{|\\}'
    Expected Result: Sin artefactos de LaTeX (sin \textbf{}, $$, {, } escapados)
    Evidence: .sisyphus/evidence/task-15-no-latex-artifacts.txt
  ```

  **Commit**: YES (grupo T15)
  - Message: `style(docx): final formatting polish for English DOCX`
  - Files: `main_paper_v2.docx`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.**

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, grep content, check DOCX structure). For each "Must NOT Have": search for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Content Quality Review** — `unspecified-high`
  Abrir DOCX español, verificar: las 12 secciones GUIA_INFORME presentes, ≤12 páginas, ≥13 figuras, ≥6 referencias APA, valores numéricos coinciden con `referencia_numerica.csv`, errores ortográficos/gramaticales, calidad de redacción académica, consistencia terminológica. Abrir DOCX inglés, verificar: formato Lancet preservado, valores corregidos.
  Output: `Secciones [12/12] | Páginas [N≤12] | Figuras [N≥13] | Refs [N≥6] | Coherencia numérica [N/N] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Ejecutar TODOS los escenarios QA de TODAS las tareas. Verificar integración cruzada (las figuras referenciadas en Resultados coinciden con las insertadas, los valores en texto coinciden con tablas). Probar casos extremos: ¿archivos originales intactos? ¿Notebook sin modificar? Guardar en `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read deliverables. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Flag unaccounted changes. Verify notebook unmodified (md5 checksum).
  Output: `Tasks [N/N compliant] | Creep [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **T1**: `chore(backup): create backup copies of all original files` — backup/
- **T2**: `feat(data): extract numerical reference table from notebook` — referencia_numerica.csv
- **T3-T4**: `fix(analysis): re-run ICC and SMOTE with corrected methodology` — outputs/
- **T5-T10**: `docs(informe): write section content for Spanish DOCX` — sections/*.md
- **T11**: `feat(docx): assemble Spanish DOCX with python-docx` — Informe_Final_v2.docx
- **T12**: `fix(latex): correct numerical discrepancies in English paper` — main_paper_v2.tex
- **T13**: `feat(docx): generate English DOCX from corrected LaTeX` — main_paper_v2.docx
- **T14-T15**: `style(docx): final formatting polish for both DOCX versions` — *_v2.docx

---

## Success Criteria

### Verification Commands
```bash
# Verificar backups intactos
diff Entrega_Final/main_paper.tex Entrega_Final/backup/main_paper.tex  # Expected: no output (identical)

# Verificar notebook no modificado
md5sum Entrega_Final/Proyecto_Final_AmatHerreraRodriguez.ipynb  # Expected: hash constante

# Verificar DOCX español tiene todas las secciones
python3 -c "
from docx import Document
doc = Document('Entrega_Final/Informe_Final_AmatHerreraRodriguez_v2.docx')
headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
print(f'Secciones: {len(headings)}')  # Expected: ≥12
print(f'Párrafos: {len(doc.paragraphs)}')
"

# Verificar conteo de figuras
ls outputs/figuras/figura*.png | wc -l  # Expected: ≥13
```

### Final Checklist
- [ ] Todos los archivos originales preservados (backup + .bak)
- [ ] Notebook Jupyter sin modificar (md5 checksum)
- [ ] `referencia_numerica.csv` contiene todos los valores con trazabilidad a celdas del notebook
- [ ] Informe español: 12 secciones GUIA_INFORME, ≤12 páginas, ≥13 figuras, ≥6 refs APA
- [ ] Informe español: ICC y SMOTE con valores corregidos (no 0.000 ni warnings)
- [ ] Paper inglés LaTeX: compila sin errores, valores actualizados
- [ ] Paper inglés DOCX: formato Lancet preservado
- [ ] Todas las hipótesis respondidas en conclusiones
- [ ] "Must NOT Have" respetados
