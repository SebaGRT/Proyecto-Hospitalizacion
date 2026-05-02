# Variabilidad en el Tratamiento Oncológico y sus Efectos sobre la Mortalidad y la Estadía Hospitalaria en el Sistema Público Chileno

**Curso:** Análisis de Datos e Inferencia Estadística — UDD 2026

---

## Integrantes

- Vicente Rodríguez
- José Tomás Amat 
- Sebastián Herrera 

---

## Fuente de Datos

Los datos utilizados en este proyecto corresponden a los **GRD Públicos del Ministerio de Salud de Chile (MINSAL)**, gestionados por FONASA, correspondientes a los años **2019–2024**.
El archivo notebook principal utiliza principalmente una version depurada de los datasets GRD correspondientes a aquel periodo.

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
   jupyter notebook Avance2_Proyecto_Final.ipynb
   ```

6. **Ejecutar el análisis:**
   - En Jupyter, seleccionar: `Kernel` → `Restart & Run All`
   - O ejecutar celda por celda según se requiera.

---

## Estructura del Repositorio

```
Proyecto-Hospitalizacion/
├── Avance2_Proyecto_Final.ipynb      # Notebook principal con análisis completo (EDA + inferencia)
├── Avance2_Proyecto_Final.py         # Exportación Python del notebook principal
├── DATASET INICIAL/                  # Dataset limpio y archivos auxiliares
│   ├── GRD_Limpio.csv                # Dataset principal (~454k registros)
│   ├── df_clean_final_2019_2024.csv  # Dataset alternativo limpio
│   ├── idh_comunas_2024.csv          # Datos socioeconómicos (IDH comunal)
│   ├── CIE-10-filtrado.ipynb         # Notebook de filtrado de códigos CIE-10
│   └── ...
├── DATASET-PROBLEMA8/                # Datasets originales GRD por año (2019–2024)
├── Avance 1/                         # Notebooks y archivos del primer avance
├── Avance 2/                         # Notebooks y outputs del segundo avance
├── outputs/                          # Gráficos y tablas generados
│   ├── graficos/                     # Visualizaciones EDA
│   └── inferencial/                  # Resultados de tests de hipótesis
├── Referencias/                      # Documentos de referencia y notebooks guía
├── docs/                             # Documentos del proyecto (presentaciones, informes)
├── theme_early_graphs.py             # Script de tematización de gráficos
├── upgrade_graphs.py                 # Script de actualización de figuras del notebook
└── feedback_avance2.md               # Retroalimentación del segundo avance
```

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

- El análisis completo está contenido en el notebook `Avance2_Proyecto_Final.ipynb`.
- Los scripts `theme_early_graphs.py` y `upgrade_graphs.py` son herramientas auxiliares para la actualización masiva de estilos y figuras dentro del notebook.
- Para dudas o solicitudes de acceso a los datos, contactar a los integrantes del equipo.

---

*Proyecto académico — Universidad del Desarrollo (UDD), 2026.*
