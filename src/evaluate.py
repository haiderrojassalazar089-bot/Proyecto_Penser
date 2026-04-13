"""
evaluate.py
===========
Descripción, interpretación y reporte de los arquetipos de graduados USTA.

Autores : Yeimy Alarcón · Karen Suarez · Maria José Galindo
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2025-2

Arquetipos identificados (3):
------------------------------
0 — El Graduado en Desarrollo  : competencias bajas-medias, mayor área de mejora
1 — El Profesional Consolidado : competencias medias-altas, perfil más común (49%)
2 — El Líder de Alto Desempeño : competencias muy altas en todas las dimensiones
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
ARTIFACTS_PATH = Path("artifacts")

# ---------------------------------------------------------------------------
# Definición de arquetipos
# ---------------------------------------------------------------------------
ARQUETIPOS = {
    0: {
        "nombre": "El Subjetivamente Satisfecho",
        "descripcion": (
            "Graduado con competencias en desarrollo. Sus puntuaciones se ubican "
            "en el rango bajo-medio (2.5–3.2). Sus mayores fortalezas son la "
            "lectura académica y el trabajo en equipo, mientras que presenta "
            "brechas importantes en segunda lengua, inserción laboral y herramientas "
            "modernas. Representa el grupo con mayor potencial de mejora."
        ),
        "fortalezas": ["lectura_academica", "trabajo_equipo", "com_escrita"],
        "debilidades": ["segunda_lengua", "insercion_laboral", "herramientas_modernas"],
        "recomendacion": (
            "Fortalecer programas de inglés, empleabilidad y herramientas digitales "
            "para este segmento de graduados."
        ),
    },
    1: {
        "nombre": "El Profesional Consolidado",
        "descripcion": (
            "El perfil más común (≈49% de los graduados). Competencias medias-altas "
            "(3.2–4.2). Destaca en toma de decisiones, ética y trabajo en equipo. "
            "Mantiene la brecha en segunda lengua y herramientas de inserción laboral, "
            "pero en menor grado que el arquetipo anterior. Perfil profesional sólido "
            "y equilibrado."
        ),
        "fortalezas": ["toma_decisiones", "etica", "trabajo_equipo"],
        "debilidades": ["segunda_lengua", "insercion_laboral", "herramientas_modernas"],
        "recomendacion": (
            "Potenciar habilidades de segunda lengua y actualización tecnológica "
            "para elevar la competitividad en el mercado laboral."
        ),
    },
    2: {
        "nombre": "El Líder de Alto Desempeño",
        "descripcion": (
            "Graduado con las competencias más altas en todas las dimensiones (4.2–4.9). "
            "Sobresale en toma de decisiones, trabajo en equipo y ética. Incluso en "
            "segunda lengua —la debilidad transversal de todos los arquetipos— obtiene "
            "la puntuación más alta del grupo (3.31). Representa el modelo de egresado "
            "que la formación USTA aspira a consolidar."
        ),
        "fortalezas": ["toma_decisiones", "trabajo_equipo", "etica"],
        "debilidades": ["segunda_lengua", "insercion_laboral", "herramientas_modernas"],
        "recomendacion": (
            "Capitalizar este perfil como referente institucional y explorar "
            "programas de mentoría entre graduados."
        ),
    },
}

# Variables de competencias para el análisis
COLS_COMPETENCIAS = [
    "com_escrita", "com_oral", "pensamiento_critico", "metodos_cuantitativos",
    "metodos_cualitativos", "lectura_academica", "argumentacion", "segunda_lengua",
    "creatividad", "resolucion_conflictos", "liderazgo", "toma_decisiones",
    "resolucion_problemas", "investigacion", "herramientas_informaticas",
    "contextos_multiculturales", "insercion_laboral", "herramientas_modernas",
    "gestion_informacion", "trabajo_equipo", "aprendizaje_autonomo",
    "conocimientos_multidisciplinares", "etica",
]

COLS_BIENESTAR = [
    "adquirio_bienes", "mejoro_vivienda", "mejoro_salud",
    "acceso_seguridad_social", "incremento_cultural",
    "satisfecho_ocio", "red_amigos",
]

COLS_SATISFACCION = [
    "satisfaccion_vida", "satisfaccion_formacion",
    "efecto_calidad_vida", "score_bienestar",
]


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_base(filename: str = "base_con_arquetipos.parquet") -> pd.DataFrame:
    ruta = ARTIFACTS_PATH / filename
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró: {ruta}\n"
            f"Ejecuta primero: python src/train.py"
        )
    df = pd.read_parquet(ruta)
    log.info(f"Base con arquetipos cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    log.info(f"Arquetipos encontrados: {sorted(df['arquetipo'].unique())}")
    return df


def perfil_competencias(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de medias de competencias por arquetipo."""
    cols = [c for c in COLS_COMPETENCIAS if c in df.columns]
    perfil = df.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def perfil_bienestar(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de proporciones de bienestar por arquetipo (variables 0/1)."""
    cols = [c for c in COLS_BIENESTAR if c in df.columns]
    if not cols:
        return pd.DataFrame()

    # Convertir a numérico si es string
    df_bien = df[cols + ["arquetipo"]].copy()
    for col in cols:
        df_bien[col] = pd.to_numeric(df_bien[col], errors="coerce")

    perfil = df_bien.groupby("arquetipo")[cols].mean().round(3) * 100
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def perfil_satisfaccion(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de satisfacción y bienestar general por arquetipo."""
    cols = [c for c in COLS_SATISFACCION if c in df.columns]
    if not cols:
        return pd.DataFrame()
    df_sat = df[cols + ["arquetipo"]].copy()
    for col in cols:
        df_sat[col] = pd.to_numeric(df_sat[col], errors="coerce")
    perfil = df_sat.groupby("arquetipo")[cols].mean().round(3)
    perfil.index = perfil.index.map(
        lambda x: f"{x} — {ARQUETIPOS.get(x, {}).get('nombre', f'Arquetipo {x}')}"
    )
    return perfil


def distribucion_arquetipos(df: pd.DataFrame) -> pd.Series:
    """Distribución de registros por arquetipo."""
    counts = df["arquetipo"].value_counts().sort_index()
    total = counts.sum()
    return counts, total


def brecha_segunda_lengua(df: pd.DataFrame) -> None:
    """
    Análisis específico de segunda lengua — debilidad transversal identificada
    en todos los arquetipos. Hallazgo clave para recomendaciones institucionales.
    """
    if "segunda_lengua" not in df.columns:
        return
    col = pd.to_numeric(df["segunda_lengua"], errors="coerce")
    print(f"\n🌐 HALLAZGO CLAVE — SEGUNDA LENGUA")
    print(f"   Media global          : {col.mean():.2f} / 5.0")
    print(f"   Es la competencia más baja en los 3 arquetipos")
    print(f"   Distribución de respuestas:")
    for v in [1, 2, 3, 4, 5]:
        n = (col == v).sum()
        pct = n / col.notna().sum() * 100
        barra = "█" * int(pct / 2)
        print(f"   {v} — {n:4d} graduados ({pct:.1f}%) {barra}")


def movilidad_social_por_arquetipo(df: pd.DataFrame) -> None:
    """Analiza la movilidad social dentro de cada arquetipo."""
    if "movilidad_social" not in df.columns:
        return
    df_mov = df.copy()
    df_mov["movilidad_social"] = pd.to_numeric(df_mov["movilidad_social"], errors="coerce")
    print(f"\n📈 MOVILIDAD SOCIAL POR ARQUETIPO")
    for arq in sorted(df["arquetipo"].unique()):
        sub = df_mov[df_mov["arquetipo"] == arq]["movilidad_social"].dropna()
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        ascenso = (sub > 0).sum()
        sin_cambio = (sub == 0).sum()
        descenso = (sub < 0).sum()
        print(f"   {arq} — {nombre}")
        print(f"      Ascenso   : {ascenso:4d} ({ascenso/len(sub)*100:.1f}%)")
        print(f"      Sin cambio: {sin_cambio:4d} ({sin_cambio/len(sub)*100:.1f}%)")
        print(f"      Descenso  : {descenso:4d} ({descenso/len(sub)*100:.1f}%)")


def imprimir_reporte_completo(df: pd.DataFrame) -> None:
    """Imprime el reporte completo de arquetipos."""
    sep = "=" * 65
    counts, total = distribucion_arquetipos(df)

    print(f"\n{sep}")
    print("  REPORTE DE ARQUETIPOS — ESTUDIO PENSER EGRESADOS USTA")
    print(sep)

    # Distribución
    print(f"\n👥 DISTRIBUCIÓN DE ARQUETIPOS (n={total:,})")
    for arq, n in counts.items():
        nombre = ARQUETIPOS.get(arq, {}).get("nombre", f"Arquetipo {arq}")
        pct = n / total * 100
        barra = "█" * int(pct / 2)
        print(f"   {arq} — {nombre:<38}: {n:4d} ({pct:.1f}%) {barra}")

    # Descripción de cada arquetipo
    print(f"\n📋 DESCRIPCIÓN DE ARQUETIPOS")
    for arq in sorted(df["arquetipo"].unique()):
        info = ARQUETIPOS.get(arq, {})
        nombre = info.get("nombre", f"Arquetipo {arq}")
        n = counts.get(arq, 0)
        print(f"\n  ── Arquetipo {arq}: {nombre} (n={n}) ──")
        print(f"  {info.get('descripcion', '')}")
        print(f"  ✅ Fortalezas : {', '.join(info.get('fortalezas', []))}")
        print(f"  ⚠️  Brechas    : {', '.join(info.get('debilidades', []))}")
        print(f"  💡 Recomenda  : {info.get('recomendacion', '')}")

    # Perfil de competencias
    print(f"\n📊 MEDIAS DE COMPETENCIAS POR ARQUETIPO (escala 1–5)")
    perfil = perfil_competencias(df)
    if not perfil.empty:
        print(perfil.T.to_string())

    # Bienestar
    print(f"\n🏠 INDICADORES DE BIENESTAR POR ARQUETIPO (% que respondió Sí)")
    bien = perfil_bienestar(df)
    if not bien.empty:
        print(bien.T.to_string())

    # Satisfacción
    print(f"\n😊 SATISFACCIÓN Y CALIDAD DE VIDA POR ARQUETIPO")
    sat = perfil_satisfaccion(df)
    if not sat.empty:
        print(sat.T.to_string())

    # Hallazgos especiales
    brecha_segunda_lengua(df)
    movilidad_social_por_arquetipo(df)

    print(f"\n{sep}")
    print("  RECOMENDACIONES INSTITUCIONALES")
    print(sep)
    print("""
  1. Segunda lengua es la competencia más débil en los 3 arquetipos.
     Reforzar programas de inglés es prioritario para toda la institución.

  2. Inserción laboral y herramientas modernas son las segundas brechas.
     Los programas de empleabilidad y actualización tecnológica deben
     fortalecerse, especialmente para el Arquetipo 0.

  3. El 49% de graduados son Profesionales Consolidados — un resultado
     positivo que muestra el impacto general de la formación USTA.

  4. El 13% clasificado como Subjetivamente Satisfecho representa el
     grupo prioritario para intervención y seguimiento institucional.
    """)
    print(f"{sep}\n")


def guardar_reportes(df: pd.DataFrame) -> None:
    """Guarda los perfiles en CSV para uso en informes."""
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    # Perfil de competencias
    perfil = perfil_competencias(df)
    if not perfil.empty:
        perfil.T.to_csv(ARTIFACTS_PATH / "perfil_competencias_arquetipos.csv")
        log.info("Perfil competencias guardado.")

    # Perfil bienestar
    bien = perfil_bienestar(df)
    if not bien.empty:
        bien.T.to_csv(ARTIFACTS_PATH / "perfil_bienestar_arquetipos.csv")
        log.info("Perfil bienestar guardado.")

    # Base con nombre de arquetipo
    df_out = df.copy()
    df_out["nombre_arquetipo"] = df_out["arquetipo"].map(
        lambda x: ARQUETIPOS.get(x, {}).get("nombre", f"Arquetipo {x}")
    )
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)
    df_out.to_parquet(ARTIFACTS_PATH / "base_con_arquetipos.parquet", index=False)
    log.info("Base final con nombres de arquetipos actualizada.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    """Pipeline completo de evaluación e interpretación de arquetipos."""
    log.info("Iniciando evaluate.py...")
    df = cargar_base()
    imprimir_reporte_completo(df)
    guardar_reportes(df)
    log.info("Evaluación completada.")


if __name__ == "__main__":
    run()