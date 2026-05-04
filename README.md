# Variabilidad en el Tratamiento Oncológico y sus Efectos sobre la Mortalidad y la Estadía Hospitalaria en el Sistema Público Chileno

**Curso:** Análisis de Datos e Inferencia Estadística — UDD 2026  
**Profesora:** Karen Oróstica  
**Ayudante:** David Hernández

---

## Integrantes

- Vicente Rodríguez
- José Tomás Amat
- Sebastián Herrera

---

## Fuente de Datos

Los datos utilizados en este proyecto corresponden a los **GRD Públicos del Ministerio de Salud de Chile (MINSAL)**, gestionados por FONASA, correspondientes a los años **2019–2024**.
El archivo notebook principal utiliza principalmente una versión depurada de los datasets GRD correspondiente a aquel periodo.

- **Dataset principal:** `DATASET INICIAL/GRD_Limpio.csv`
- **Dimensiones:** ~454.000 registros de egresos hospitalarios y ~145 variables (seleccionadas las clínicamente relevantes).
- **Enfoque:** Oncología CIE-10 C00–D49, con énfasis en cáncer gástrico (C16.*).
- **Acceso:** Los datos GRD son de carácter público y pueden solicitarse a través del portal de datos abiertos del MINSAL o FONASA. El repositorio no incluye el dataset por restricciones de tamaño.

---

## Requisitos

- Python >= 3.10
- Jupyter Notebook / JupyterLab

### Dependencias principales

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

> Nota: Si existe un entorno virtual configurado en el proyecto (`.venv`), actívalo antes de instalar las dependencias.

---

## Instrucciones para Reproducir el Análisis

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd Proyecto-Hospitalizacion
   ```

2. **Obtener el dataset:**
   - Descargar los archivos GRD Públicos 2019–2024 desde la fuente oficial (MINSAL/FONASA).
   - Colocar el archivo `GRD_Limpio.csv` dentro de la carpeta `DATASET INICIAL/`.

3. **(Opcional) Activar entorno virtual:**
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # o
   .venv\Scripts\activate     # Windows
   ```

4. **Instalar dependencias:**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
   ```

5. **Abrir el notebook principal:**
   ```bash
   jupyter notebook "Avance 2/Avance2_Proyecto_AmatHerreraRodríguez.ipynb"
   ```

6. **Ejecutar el análisis:**
   - En Jupyter, seleccionar: `Kernel` → `Restart & Run All`
   - O ejecutar celda por celda según se requiera.

---

## Estructura del Repositorio

```
Proyecto-Hospitalizacion/
├── Avance 1/                                 # Notebooks y archivos del primer avance
│   ├── EDA_Egresos_Hospitalarios_Chile.ipynb
│   ├── EDA_Egresos_Hospitalarios_Chile (Cargado).ipynb
│   ├── Estimacion_Estadistica (1).ipynb
│   ├── Proyecto-Hospitales.ipynb
│   ├── unir_grd_fonasa.ipynb
│   ├── nuevas_celdas_analisis.ipynb
│   ├── test_sample_utf8_load.py
│   ├── instrucciones.txt
│   └── codigos_C00_D49.txt
├── Avance 2/                                 # Notebook y outputs del segundo avance
│   ├── Avance2_Proyecto_AmatHerreraRodríguez.ipynb  # Notebook principal con análisis completo
│   └── outputs/
│       ├── graficos/                         # Visualizaciones EDA e inferenciales
│       ├── tablas/                           # Tablas descriptivas exportadas
│       └── inferencial/                      # Resultados de tests de hipótesis y modelos
├── DATASET INICIAL/                          # Dataset limpio y archivos auxiliares
│   ├── GRD_Limpio.csv                        # Dataset principal (~454k registros)
│   ├── idh_comunas_2024.csv                  # Datos socioeconómicos (IDH comunal)
│   ├── HospitalesGRD.xlsx                    # Mapeo de códigos a nombres de hospitales
│   ├── comunas-regiones.json                 # Mapeo comunas-regiones
│   ├── regiones-provincias.json              # Mapeo regiones-provincias
│   ├── CIE-10-filtrado.ipynb                 # Notebook de filtrado de códigos CIE-10
│   └── codigos_C00_D49.txt                   # Lista de códigos CIE-10 oncológicos
├── DATASET-PROBLEMA8/                        # Datasets originales GRD por año (2019–2024)
│   ├── df_clean_final_2019_2024.csv          # Dataset alternativo limpio
│   ├── GRD_PUBLICO_2019.csv
│   ├── GRD_PUBLICO_2020.csv
│   ├── GRD_PUBLICO_2021.csv
│   ├── GRD_PUBLICO_EXTERNO_2022.csv
│   ├── GRD_PUBLICO_2023.csv
│   ├── GRD_PUBLICO_2024.csv
│   ├── CIE-10.xlsx
│   ├── Lista-Tabular-CIE-10-1-1.xls
│   └── utf8/                                 # Versiones en UTF-8 de los CSV anuales
├── Documentos/                               # Documentos del proyecto (presentaciones e informes)
│   ├── Avance1_Presentacion.pdf
│   ├── Avance1_AmatHerreraRodríguez.pdf
│   ├── Reporte_Avance2_Final.docx
│   ├── Reporte_Avance2_Final.pdf
│   └── Variabilidad_Hospitalaria_Presentation.pptx
├── outputs/                                  # Gráficos y tablas generados (raíz)
│   ├── graficos/                             # Visualizaciones EDA
│   │   ├── barplot_mortalidad_hospital_C16.png
│   │   ├── barplot_procedimientos_hospital_C16.png
│   │   ├── boxplot_dias_hospital_C16.png
│   │   ├── categoricas_globales.png
│   │   ├── correlacion_variables_numericas.png
│   │   ├── eda_univariado_global.png
│   │   ├── eda_urgencias_por_sexo_C16.png
│   │   └── top15_diagnosticos_onco.png
│   ├── tablas/                               # Tablas descriptivas exportadas
│   │   ├── 03_varianza_por_hospital.csv
│   │   ├── h3_ols_coefs_C16.csv
│   │   ├── tabla_descriptiva_hospital_C16.csv
│   │   └── tabla1_descriptivas_global.csv
│   └── inferencial/                          # Resultados de tests de hipótesis y modelos
│       ├── cat_A_mortalidad_sexo.png
│       ├── cat_B_mortalidad_tipoing.png
│       ├── cat_B_posthoc_bonferroni.csv
│       ├── cat_C_residuos_tipoalta_hospital.png
│       ├── h1_dunn_heatmap_C16.png
│       ├── h1_dunn_resultados_C16.csv
│       ├── h1_normalidad_proc_C16.png
│       ├── h2_confusion_matrix_C16.png
│       ├── h2_forest_plot_OR_C16.png
│       ├── h2_forest_plot_completo_C16.png
│       ├── h2_logit_OR_C16.csv
│       ├── h2_roc_curve_C16.png
│       ├── h3_coef_ols_C16.png
│       ├── h3_ols_coefs_C16.csv
│       ├── h3_pred_vs_obs_C16.png
│       ├── h3_residual_diagnostics_C16.png
│       └── h3_transformacion_dias_C16.png
├── Referencias/                              # Documentos de referencia y notebooks guía
│   ├── Requerimientos_Avance2_Analisis_de_Datos_fecha_entrega.pdf
│   ├── documento-contexto.md
│   ├── Notebook_Final_Regresion_Hospitalizaciones.ipynb
│   ├── notebook_tests_categoricos_GRD (1).ipynb
│   ├── Clase 19 - Metodos Numericos.ipynb
│   ├── Clase 20 - Metodos numéricos.ipynb
│   └── Clase 21 - Metodos numéricos.ipynb
└── .gitignore
```

> **Nota:** Los archivos de datos (`.csv`, `.xlsx`, `.xls`) están excluidos del repositorio mediante `.gitignore` por su tamaño. Los archivos listados en `DATASET INICIAL/` y `DATASET-PROBLEMA8/` deben obtenerse desde la fuente oficial MINSAL/FONASA.

---

## Resumen del Análisis

Este proyecto investiga en qué medida el **hospital de atención** determina los días de estadía, la cantidad de procedimientos y la mortalidad intrahospitalaria en pacientes oncológicos clínicamente comparables (CIE-10 C16.* — Cáncer Gástrico) del sistema público chileno.

### Hipótesis planteadas

| Hipótesis | Enunciado | Test estadístico |
|---|---|---|
| **H₁** | La distribución de la cantidad de procedimientos difiere significativamente entre hospitales para pacientes con cáncer gástrico. | Kruskal-Wallis + Dunn-Bonferroni |
| **H₂** | La cantidad de procedimientos se asocia con la probabilidad de mortalidad intrahospitalaria, controlando por edad, severidad GRD y hospital. | Regresión Logística (OR, IC95%) |
| **H₃** | La cantidad de procedimientos predice la duración de la estadía hospitalaria, controlando por edad, severidad GRD y hospital. | Regresión OLS Múltiple (β, IC95%) |

Nivel de significancia: **α = 0.05**.

---

## Notas

- El análisis completo está contenido en el notebook `Avance 2/Avance2_Proyecto_AmatHerreraRodríguez.ipynb`.
- Para dudas o solicitudes de acceso a los datos, contactar a los integrantes del equipo.

---

*Proyecto académico — Universidad del Desarrollo (UDD), 2026.*