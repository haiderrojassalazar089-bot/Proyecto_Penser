"""
features.py
===========
Depuración, codificación y construcción de variables analíticas — Estudio PENSER.

Autores : Yeimy Alarcón · Karen Suarez · Maria José Galindo
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2025-2

Dimensiones construidas:
-------------------------
1. Competencias (23 vars Likert 1-5)     → renombradas a nombres cortos
2. Bienestar y movilidad social          → binarias codificadas 0/1 + índice de movilidad
3. Satisfacción con la formación         → escala textual → numérica
4. Trayectoria laboral                   → variables categóricas ordinales codificadas
5. Variables sociodemográficas           → género, sede, nivel de formación
6. Índice de movilidad social            → estrato_actual - estrato_al_graduar
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
INTERIM_PATH   = Path("data/interim")
PROCESSED_PATH = Path("data/processed")

# ---------------------------------------------------------------------------
# Mapeo de nombres largos → cortos para competencias
# ---------------------------------------------------------------------------
RENAME_COMPETENCIAS = {
    "          Exponer las ideas de forma clara y efectiva por medios   escritos\n        ": "com_escrita",
    "Exponer oralmente las ideas de manera clara y efectiva":                                "com_oral",
    "Pensamiento crítico y analítico":                                                       "pensamiento_critico",
    "Uso y comprensión de razonamiento y métodos cuantitativos":                             "metodos_cuantitativos",
    "Uso y comprensión de métodos cualitativos":                                             "metodos_cualitativos",
    "Leer y comprender material académico (artículos, libros, etc)":                        "lectura_academica",
    "Capacidad argumentativa":                                                               "argumentacion",
    "Escribir o hablar en una segunda lengua":                                               "segunda_lengua",
    "Crear ideas originales y soluciones":                                                   "creatividad",
    "Capacidad de resolver conflictos interpersonales":                                      "resolucion_conflictos",
    "Habilidades de liderazgo":                                                              "liderazgo",
    "Asumir responsabilidades y tomar decisiones":                                           "toma_decisiones",
    "Identificar, plantear y resolver problemas":                                            "resolucion_problemas",
    "Habilidad para formular, ejecutar y evaluar una investigación o proyecto":              "investigacion",
    "Utilizar herramientas informáticas básicas":                                            "herramientas_informaticas",
    "Capacidad para comprender y desenvolverse en contextos multiculturales":                "contextos_multiculturales",
    "Conocimientos y habilidades relacionados con la inserción en el mercado laboral":       "insercion_laboral",
    "Capacidad de usar las técnicas, habilidades y herramientas modernas necesarias para la inserción en el mercado laboral": "herramientas_modernas",
    "Buscar, analizar, administrar y compartir información":                                 "gestion_informacion",
    "Habilidades para trabajar en equipo":                                                   "trabajo_equipo",
    "Capacidad de aprender y mantenerse actualizado por su cuenta":                         "aprendizaje_autonomo",
    "Adquirir conocimientos de distintas áreas":                                             "conocimientos_multidisciplinares",
    "Capacidad de identificar problemas éticos y morales":                                   "etica",
}

COLS_COMPETENCIAS = list(RENAME_COMPETENCIAS.values())

# ---------------------------------------------------------------------------
# Mapeo de columnas de bienestar → nombres cortos
# ---------------------------------------------------------------------------
RENAME_BIENESTAR = {
    "¿Adquirió bienes materiales después de obtener su título de pregrado y posgrado?\xa0(casa, carro, inversión a largo plazo).": "adquirio_bienes",
    "¿La calidad de su vivienda mejoró después de obtener su título de pregrado y posgrado?\xa0(compra o adecuaciones).":          "mejoro_vivienda",
    "¿Sus condiciones de salud han mejorado desde que obtuvo su título de pregrado o posgrado en la Universidad Santo Tomás?":     "mejoro_salud",
    "¿Pudo acceder a un esquema de seguridad social con mayores privilegios después de obtener su título de pregrado y posgrado?": "acceso_seguridad_social",
    "¿Su asistencia a eventos culturales, deportivos o artísticos se incrementó después de obtener su título de pregrado y posgrado?": "incremento_cultural",
    "¿Está satisfecho con el tiempo de ocio que tiene disponible después de obtener su título de pregrado y posgrado?":            "satisfecho_ocio",
    "¿Continúa en contacto con su red de amigos universitarios?":                                                                  "red_amigos",
}

COLS_BIENESTAR = list(RENAME_BIENESTAR.values())

# ---------------------------------------------------------------------------
# Escalas de satisfacción (texto → número)
# ---------------------------------------------------------------------------
ESCALA_SATISFACCION = {
    "Nulo (0)": 0, "Regular (1)": 1, "Aceptable (2)": 2,
    "Medio (3)": 3, "Adecuado (4)": 4, "Pleno (5)": 5,
}

# ---------------------------------------------------------------------------
# Variables ordinales de trayectoria laboral
# ---------------------------------------------------------------------------
ESCALA_TIEMPO_EMPLEO = {
    "Menos de 1 mes": 1,
    "Entre 1 mes y 3 meses": 2,
    "Entre 3 meses y 6 meses": 3,
    "Entre 6 meses y 1 año": 4,
    "Entre 1 año y 2 años": 5,
    "Más de 2 años": 6,
}

ESCALA_NIVEL_CARGO = {
    "Auxiliar": 1,
    "Técnico": 2,
    "Profesional": 3,
    "Coordinador": 4,
    "Jefe": 5,
    "Directivo": 6,
    "Gerente": 7,
}

ESCALA_RELACION_SECTOR = {
    "Nada relacionado": 1,
    "Relación Indirecta": 2,
    "Relación Directa": 3,
}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra competencias y bienestar a nombres cortos manejables."""
    df = df.rename(columns={**RENAME_COMPETENCIAS, **RENAME_BIENESTAR})
    log.info(f"Columnas renombradas: {len(RENAME_COMPETENCIAS)} competencias + {len(RENAME_BIENESTAR)} bienestar.")
    return df


def limpiar_competencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que las competencias estén en rango [1-5].
    Valores fuera de rango → NaN (ya corregidos en ingest, esta es validación final).
    """
    cols = [c for c in COLS_COMPETENCIAS if c in df.columns]
    corregidas = 0
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = df[col].notna() & ~df[col].between(1, 5)
        if mask.any():
            df.loc[mask, col] = np.nan
            corregidas += 1
    log.info(f"Competencias validadas: {len(cols)} columnas. Corregidas: {corregidas}.")
    return df


def codificar_binarias(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas Si/No a 1/0."""
    cols = [c for c in COLS_BIENESTAR if c in df.columns]
    for col in cols:
        df[col] = df[col].map({"Si": 1, "No": 0, "SI": 1, "NO": 0, "si": 1, "no": 0})
    log.info(f"Variables binarias codificadas: {len(cols)} columnas.")
    return df


def codificar_satisfaccion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las escalas textuales de satisfacción a numéricas.
    Ejemplo: 'Adecuado (4)' → 4
    """
    cols_satisfaccion = {
        "Indique el grado de cumplimiento de sus expectativas de formación con la oferta del programa.": "satisfaccion_formacion",
        "Indique el efecto que tuvo su título de pregrado o posgrado en el mejoramiento de su calidad de vida": "efecto_calidad_vida",
        "Indique el grado de satisfacción general que tiene de su vida después de obtener su título de pregrado y posgrado": "satisfaccion_vida",
        "Indique el grado de correspondencia entre sus funciones en su primer empleo y las competencias desarrolladas durante el programa de pregrado o posgrado:": "correspondencia_primer_empleo",
        "Indique el grado de correspondencia entre sus funciones en su empleo actual y las competencias desarrolladas durante el programa de pregrado o posgrado": "correspondencia_empleo_actual",
        "Indique la relación entre el sector de su empleo actual y su título de pregrado o posgrado": "relacion_sector_actual",
    }
    creadas = 0
    for col_orig, col_nuevo in cols_satisfaccion.items():
        if col_orig in df.columns:
            df[col_nuevo] = df[col_orig].map(ESCALA_SATISFACCION)
            creadas += 1
    log.info(f"Variables de satisfacción codificadas: {creadas} columnas nuevas.")
    return df


def calcular_movilidad_social(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el índice de movilidad social como la diferencia entre
    el estrato actual y el estrato al momento de graduarse.
    Positivo = ascenso social · Negativo = descenso · 0 = sin cambio
    """
    col_grad   = "Estrato socioeconómico en el momento de obtener el título de pregrado o posgrado"
    col_actual = "Estrato socioeconómico actual"
    if col_grad in df.columns and col_actual in df.columns:
        estrato_grad   = pd.to_numeric(df[col_grad],   errors="coerce")
        estrato_actual = pd.to_numeric(df[col_actual], errors="coerce")
        df["movilidad_social"] = estrato_actual - estrato_grad
        dist = df["movilidad_social"].value_counts().sort_index()
        log.info(f"Índice de movilidad social calculado. Distribución: {dist.to_dict()}")
    else:
        log.warning("No se encontraron columnas de estrato para calcular movilidad social.")
    return df


def codificar_nivel_cargo(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica el nivel del cargo (primer empleo y empleo actual) de forma ordinal."""
    cols = {
        "Especifique el nivel del cargo desempeñado en su primer empleo:": "nivel_cargo_primer_empleo",
        "Especifique el nivel del cargo desempeñado en su empleo actual":  "nivel_cargo_actual",
    }
    for col_orig, col_nuevo in cols.items():
        if col_orig in df.columns:
            df[col_nuevo] = df[col_orig].map(ESCALA_NIVEL_CARGO)
    log.info("Nivel de cargo codificado (ordinal).")
    return df


def codificar_tiempo_primer_empleo(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica el tiempo en conseguir el primer empleo de forma ordinal."""
    col = "Indique el tiempo que tardó en lograr su primer empleo tras obtener su título de pregrado o posgrado"
    if col in df.columns:
        df["tiempo_primer_empleo"] = df[col].map(ESCALA_TIEMPO_EMPLEO)
        log.info("Tiempo primer empleo codificado (ordinal).")
    return df


def codificar_relacion_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica la relación entre el empleo y el título de forma ordinal."""
    cols = {
        "Indique la relación entre el sector de su primer empleo y el título de pregrado o posgrado que obtuvo.": "relacion_sector_primer_empleo",
    }
    for col_orig, col_nuevo in cols.items():
        if col_orig in df.columns:
            df[col_nuevo] = df[col_orig].map(ESCALA_RELACION_SECTOR)
    log.info("Relación sector-título codificada (ordinal).")
    return df


def codificar_sociodemograficas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variables sociodemográficas clave:
    - Género → dummies
    - Sede → dummies
    - Nivel de formación → ordinal
    """
    # Género
    col_genero = "Genero:"
    if col_genero in df.columns:
        dummies_genero = pd.get_dummies(df[col_genero], prefix="genero", drop_first=False)
        df = pd.concat([df, dummies_genero], axis=1)
        log.info(f"Género codificado: {list(dummies_genero.columns)}")

    # Sede
    col_sede = "En su relación como estudiante en la Universidad Santo Tomás. ¿De qué sede o seccional es graduado?:"
    if col_sede in df.columns:
        dummies_sede = pd.get_dummies(df[col_sede], prefix="sede", drop_first=False)
        df = pd.concat([df, dummies_sede], axis=1)
        log.info(f"Sede codificada: {list(dummies_sede.columns)}")

    # Nivel de formación
    col_nivel = "Cuál es el mayor nivel de estudios o título universitario obtenido (responda su último nivel de estudios aprobado):"
    nivel_map = {
        "Pregrado": 1,
        "Posgrado Especialización": 2,
        "Posgrado Maestría": 3,
        "Posgrado Doctorado": 4,
    }
    if col_nivel in df.columns:
        df["nivel_formacion"] = df[col_nivel].map(nivel_map)
        log.info("Nivel de formación codificado (ordinal).")

    return df


def calcular_score_bienestar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un score compuesto de bienestar sumando las variables binarias.
    Rango: 0 (ningún indicador de bienestar) a 7 (todos los indicadores).
    """
    cols = [c for c in COLS_BIENESTAR if c in df.columns]
    if cols:
        df["score_bienestar"] = df[cols].sum(axis=1, skipna=True)
        log.info(f"Score de bienestar calculado (0-{len(cols)}). Media: {df['score_bienestar'].mean():.2f}")
    return df


def resumen_variables_finales(df: pd.DataFrame) -> None:
    """Imprime un resumen de las variables construidas."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  RESUMEN DE VARIABLES CONSTRUIDAS — FEATURES.PY")
    print(sep)

    comp = [c for c in COLS_COMPETENCIAS if c in df.columns]
    bien = [c for c in COLS_BIENESTAR if c in df.columns]
    nuevas = ["movilidad_social", "score_bienestar", "satisfaccion_formacion",
              "efecto_calidad_vida", "satisfaccion_vida", "nivel_formacion",
              "tiempo_primer_empleo", "nivel_cargo_primer_empleo", "nivel_cargo_actual",
              "correspondencia_primer_empleo", "correspondencia_empleo_actual"]
    nuevas_ok = [c for c in nuevas if c in df.columns]

    print(f"\n📊 VARIABLES ANALÍTICAS")
    print(f"   Competencias (Likert 1-5) : {len(comp)} variables")
    print(f"   Bienestar (binarias 0/1)  : {len(bien)} variables")
    print(f"   Variables construidas     : {len(nuevas_ok)} variables nuevas")
    print(f"   Total columnas en base    : {df.shape[1]}")
    print(f"   Total registros           : {df.shape[0]:,}")

    print(f"\n📈 ESTADÍSTICAS COMPETENCIAS")
    if comp:
        stats = df[comp].describe().loc[["mean", "std", "min", "max"]].round(2)
        print(stats.T.to_string())

    print(f"\n🏠 MOVILIDAD SOCIAL")
    if "movilidad_social" in df.columns:
        mov = df["movilidad_social"].value_counts().sort_index()
        pct = (mov / mov.sum() * 100).round(1)
        for v, n in mov.items():
            label = "sin cambio" if v == 0 else ("ascenso" if v > 0 else "descenso")
            print(f"   {int(v):+d} estrato ({label}): {n} graduados ({pct[v]}%)")

    print(f"\n✅ BIENESTAR")
    if "score_bienestar" in df.columns:
        print(f"   Score promedio: {df['score_bienestar'].mean():.2f} / 7")
        print(f"   Score máximo  : {df['score_bienestar'].max():.0f} / 7")
        print(f"   Score mínimo  : {df['score_bienestar'].min():.0f} / 7")

    print(f"\n{sep}\n")


def guardar_procesada(df: pd.DataFrame, filename: str = "base_procesada.parquet") -> None:
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    # Convertir object mixtos a string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), other=None)
    ruta = PROCESSED_PATH / filename
    df.to_parquet(ruta, index=False)
    log.info(f"Base procesada guardada en: {ruta}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Pipeline completo de construcción de variables."""
    log.info("Iniciando features.py...")
    df = pd.read_parquet(INTERIM_PATH / "base_cargada.parquet")
    log.info(f"Base cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # Renombrar
    df = renombrar_columnas(df)

    # Limpiar y validar
    df = limpiar_competencias(df)
    df = codificar_binarias(df)

    # Construir variables nuevas
    df = codificar_satisfaccion(df)
    df = calcular_movilidad_social(df)
    df = codificar_nivel_cargo(df)
    df = codificar_tiempo_primer_empleo(df)
    df = codificar_relacion_sector(df)
    df = codificar_sociodemograficas(df)
    df = calcular_score_bienestar(df)

    # Resumen y guardado
    resumen_variables_finales(df)
    guardar_procesada(df)

    return df


if __name__ == "__main__":
    run()