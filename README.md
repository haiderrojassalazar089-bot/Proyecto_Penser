<div align="center">

# Proyecto PENSER — Del Índice a los Arquetipos

**Consultorio de Estadística y Ciencia de Datos**  
Universidad Santo Tomás · Facultad de Estadística  
Curso: Consultoría e Investigación · Octavo Semestre (2026-1)

**Equipo:** Haider Rojas · Sergio Prieto

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Tests](https://img.shields.io/badge/Tests-37%20passed-brightgreen)
![Status](https://img.shields.io/badge/Estado-Completado-success)

</div>

---

## ¿De qué trata este proyecto?

Los graduados de la Universidad Santo Tomás no son todos iguales. Unos dominan el liderazgo pero batallan con el inglés. Otros tienen alta satisfacción con su vida pero menor inserción laboral. Otros lo tienen todo.

Este proyecto analiza **dos bases de datos** del Estudio PENSER 2026 y construye **arquetipos de graduados** mediante métodos estadísticos avanzados: Análisis Factorial Exploratorio (AFE), Análisis de Correspondencia Múltiple (ACM) y Clustering Jerárquico Ward.

El punto de partida fue revisar el Índice de Impacto de Egresados que ya existía, encontrar sus limitaciones, y construir algo más robusto, reproducible y metodológicamente riguroso.

---

## Bases de Datos Analizadas

| Base | Registros | Columnas útiles | Instrumento |
|---|---|---|---|
| **Estudio de Percepción Egresados** | 2.530 | 109 | Competencias Likert 1–5, bienestar, trayectoria |
| **Base Depurada PENSER 2025** | 1.129 | 64 | Logro competencias (Muy bajo→Muy alto), incidencia formación |

---

## Resultados — Base Percepción Egresados

### Arquetipos identificados

| # | Arquetipo | n | % | Perfil |
|---|---|---|---|---|
| 0 | **El Subjetivamente Satisfecho** | 594 | 23.5% | F1+F2+F3 bajos. Solo 29.5% recomendaría la USTA. |
| 1 | **El Profesional Consolidado** | 602 | 23.8% | F1+F2+F3 medios. 90.5% recomendaría la USTA. |
| 2 | **El Líder de Alto Desempeño** | 1.334 | 52.7% | F1+F2+F3 altos. Referente institucional. |

### Factores AFE

| Factor | Nombre | Varianza |
|---|---|---|
| F1 | Competencias Cognitivas y Comunicativas | 21.6% |
| F2 | Satisfacción y Correspondencia Laboral | 11.2% |
| F3 | Competencias Tecnológicas e Inserción Laboral | 17.7% |

### Hallazgos clave

- **Segunda lengua**: media 2.76/5 — brecha transversal en los 3 arquetipos (43% calificó con 1 o 2)
- **Arquetipo 0**: 29.5% recomendaría la USTA y 18.9% estudiaría otra vez — señal de alerta institucional
- **Movilidad social**: 20.7% ascendió de estrato socioeconómico tras graduarse

---

## Resultados — Base Depurada PENSER 2025

### Arquetipos identificados

| # | Arquetipo | n | % | Perfil |
|---|---|---|---|---|
| 0 | **El Graduado en Desarrollo** | 587 | 52.0% | F1+F2+F3 bajos. Score incidencia 14.6/30. |
| 1 | **El Profesional Impactado** | 242 | 21.4% | F1+F2+F3 medios. Score incidencia 19.3/30. |
| 2 | **El Líder con Alta Incidencia** | 300 | 26.6% | F1+F2+F3 altos. Score incidencia 20.3/30. |

### Factores AFE

| Factor | Nombre | Varianza |
|---|---|---|
| F1 | Competencias Transversales | 34.3% |
| F2 | Incidencia en Bienestar | 19.5% |
| F3 | Competencias Profesionales del Programa | 14.5% |

### Limitación documentada

Los arquetipos de esta base están fuertemente influenciados por la sede de graduación (Bucaramanga → Arquetipo 2, Villavicencio → Arquetipo 1, Bogotá → Arquetipo 0). Esto sugiere diferencias entre sedes más que perfiles individuales distintos. Se documenta como limitación metodológica.

---

## Calidad de Datos

### Base Percepción Egresados

| Problema | Eliminados | Justificación |
|---|---|---|
| Filas vacías | 2 | Sin ninguna respuesta |
| Fila fantasma | 1 | Números de pregunta en lugar de respuestas |
| Duplicados exactos | 48 | Formularios enviados más de una vez |
| Formularios incompletos | 15 | < 10% de respuestas |
| **Total** | **66 (2.5%)** | |

### Base Depurada PENSER 2025

| Problema | Eliminados | Justificación |
|---|---|---|
| Duplicados exactos | 0 | Base ya depurada |
| Columnas >90% nulos | 79 | Preguntas de nivel educativo de hermanos |
| **Filas eliminadas** | **0** | Base limpia |

---

## Metodología (ambas bases)

### Etapa 1 — Ingesta y Validación
Pipeline de carga con reporte automático de calidad. Detecta PII, duplicados, outliers, valores inválidos. Guarda en formato Parquet.

### Etapa 2 — Construcción de Variables
Codificación de escalas ordinales, binarias, categóricas y construcción de scores compuestos.

### Etapa 3 — Modelamiento (AFE + ACM + Ward)

**AFE** sobre variables de competencias y satisfacción:
- Variables Likert NO son normales → correlación de **Spearman** (no Pearson)
- KMO > 0.93 en ambas bases (Excelente) y Bartlett p≈0
- Rotación **Oblimin** (factores correlacionados entre sí)
- 3 factores por criterio Kaiser

**ACM** sobre variables categóricas nominales:
- 6 variables en base percepción, 4 en base depurada
- 3 dimensiones retenidas en ambas bases

**Clustering Jerárquico Ward**:
- Input: scores AFE (3) + coordenadas ACM (3) = 6 dimensiones
- k=3 seleccionado por parsimonia e interpretabilidad
- Validación: silueta, Davies-Bouldin, Calinski-Harabasz

### Etapa 4 — Evaluación e Interpretación
Perfiles de competencias, bienestar, satisfacción, trayectoria laboral y recomendaciones institucionales.

---

## Estructura del Proyecto

```
Proyecto_Penser/
│
├── data/
│   ├── raw/          ← Bases originales (no versionadas)
│   ├── interim/      ← Bases validadas (no versionadas)
│   └── processed/    ← Bases procesadas (no versionadas)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── percepcion/          ← Pipeline base percepción egresados
│   │   ├── ingest.py
│   │   ├── features.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── depurada/            ← Pipeline base depurada PENSER 2025
│       ├── ingest_depurada.py
│       ├── features_depurada.py
│       ├── train_depurada.py
│       └── evaluate_depurada.py
│
├── models/           ← Modelos serializados (no versionados)
├── artifacts/        ← Artefactos generados (no versionados)
│
├── tests/
│   ├── test_ingest.py    ← 14 pruebas
│   ├── test_features.py  ← 12 pruebas
│   └── test_train.py     ← 11 pruebas
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cómo Reproducir el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/haiderrojassalazar089-bot/Proyecto_Penser.git
cd Proyecto_Penser
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Parche requerido (compatibilidad factor_analyzer con sklearn 1.8)

```bash
FA_PATH=$(python3 -c "import factor_analyzer, os; print(os.path.dirname(factor_analyzer.__file__))")/factor_analyzer.py
sed -i 's/force_all_finite="allow-nan"/ensure_all_finite="allow-nan"/g' $FA_PATH
sed -i 's/force_all_finite=True/ensure_all_finite=True/g' $FA_PATH
```

### 4. Agregar las bases de datos

```
data/raw/ESTUDIO_DE_PERCEPCION_EGRESADOS.xlsx
data/raw/DATA_DEPURADA_PENSER_2025.xlsx
```

### 5. Ejecutar pipeline — Base Percepción

```bash
python src/percepcion/ingest.py
python src/percepcion/features.py
python src/percepcion/train.py
python src/percepcion/evaluate.py
```

### 6. Ejecutar pipeline — Base Depurada

```bash
python src/depurada/ingest_depurada.py
python src/depurada/features_depurada.py
python src/depurada/train_depurada.py
python src/depurada/evaluate_depurada.py
```

### 7. Pruebas unitarias

```bash
pytest tests/ -v
# 37 passed
```

---

## Stack Tecnológico

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python 3.12 |
| Manipulación de datos | pandas, numpy |
| Machine Learning | scikit-learn |
| Análisis Factorial | factor_analyzer (AFE, KMO, Bartlett) |
| Análisis Correspondencia | prince (MCA) |
| Estadística | scipy |
| Calidad de código | pytest (37 pruebas unitarias) |
| Entorno | GitHub Codespaces |
| Formato de datos | Parquet |

---

<div align="center">

Universidad Santo Tomás · Facultad de Estadística · 2026-1  
Consultorio de Estadística y Ciencia de Datos

</div>