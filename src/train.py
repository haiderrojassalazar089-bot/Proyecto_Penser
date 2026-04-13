"""
train.py
========
Análisis de Componentes Principales (ACP) e identificación de arquetipos
de graduados mediante clustering — Estudio PENSER USTA 2025-2.

Autores : Yeimy Alarcón · Karen Suarez · Maria José Galindo
Proyecto: Consultorio de Estadística y Ciencia de Datos — USTA 2025-2

Decisiones metodológicas documentadas:
----------------------------------------
1. Se usa ACP (PCA) en lugar de AFE clásico porque las 23 competencias tienen
   correlación media de 0.87 — están tan relacionadas que un solo componente
   explica el 91.5% de la varianza. El AFE clásico requiere estructura factorial
   más diversa. El ACP permite reducir dimensionalidad antes del clustering.

2. Para clustering se combinan: competencias (23) + bienestar (7) + movilidad
   social (1) + satisfacción con la vida (1) + satisfacción formación (1).
   Usar solo competencias produce clusters degenerados (1 solo registro).

3. Se evalúan KMeans y Clustering Jerárquico (Ward) con k=2 a 7.
   Criterio de selección: coeficiente de silueta + interpretabilidad.

4. Se usa imputación por mediana para nulos antes del clustering
   (no se eliminan filas — se perdería demasiada información).

5. Los scores PCA se estandarizan antes del clustering para dar igual
   peso a cada dimensión independientemente de su varianza original.
"""

import logging
import warnings
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

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
PROCESSED_PATH = Path("data/processed")
MODELS_PATH    = Path("models")
ARTIFACTS_PATH = Path("artifacts")

# ---------------------------------------------------------------------------
# Variables para el clustering
# (combinación de dimensiones analíticas del proyecto PENSER)
# ---------------------------------------------------------------------------
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

COLS_ADICIONALES = [
    "movilidad_social",
    "satisfaccion_vida",
    "satisfaccion_formacion",
    "efecto_calidad_vida",
    "score_bienestar",
]

# Nombres de arquetipos (alineados con el README del proyecto)
ARCHETYPE_NAMES = {
    0: "El Subjetivamente Satisfecho",
    1: "El Profesional Exitoso y Crítico",
    2: "El Graduado Agradecido",
    3: "El Profesional en Transición",
    4: "El Líder de Alto Desempeño",
}


# ---------------------------------------------------------------------------
# Dataclass de resultados
# ---------------------------------------------------------------------------
@dataclass
class ResultadosModelo:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    n_registros: int = 0
    n_variables: int = 0
    n_componentes_pca: int = 0
    varianza_explicada_pca: float = 0.0
    metodo_clustering: str = ""
    k_optimo: int = 0
    metricas: dict = field(default_factory=dict)
    distribucion_arquetipos: dict = field(default_factory=dict)

    def imprimir(self) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  RESULTADOS DEL MODELO — ESTUDIO PENSER EGRESADOS USTA")
        print(f"  Generado: {self.timestamp}")
        print(sep)

        print(f"\n📐 DATOS DE ENTRADA")
        print(f"   Registros         : {self.n_registros:,}")
        print(f"   Variables usadas  : {self.n_variables}")

        print(f"\n🔢 ANÁLISIS DE COMPONENTES PRINCIPALES")
        print(f"   Componentes       : {self.n_componentes_pca}")
        print(f"   Varianza explicada: {self.varianza_explicada_pca:.1f}%")

        print(f"\n🎯 CLUSTERING")
        print(f"   Método            : {self.metodo_clustering}")
        print(f"   k óptimo          : {self.k_optimo} arquetipos")

        print(f"\n📊 MÉTRICAS DE VALIDACIÓN")
        for k, metricas in self.metricas.items():
            marca = " ← ÓPTIMO" if k == self.k_optimo else ""
            print(f"   k={k}: silueta={metricas['silueta']:.4f} | "
                  f"davies-bouldin={metricas['davies_bouldin']:.4f} | "
                  f"calinski={metricas['calinski']:.1f}{marca}")

        print(f"\n👥 DISTRIBUCIÓN DE ARQUETIPOS")
        total = sum(self.distribucion_arquetipos.values())
        for arq, n in sorted(self.distribucion_arquetipos.items()):
            nombre = ARCHETYPE_NAMES.get(arq, f"Arquetipo {arq}")
            pct = n / total * 100
            barra = "█" * int(pct / 2)
            print(f"   {arq} — {nombre:<35} : {n:4d} ({pct:.1f}%) {barra}")

        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_base(filename: str = "base_procesada.parquet") -> pd.DataFrame:
    ruta = PROCESSED_PATH / filename
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró: {ruta}\n"
            f"Ejecuta primero: python src/features.py"
        )
    df = pd.read_parquet(ruta)
    log.info(f"Base procesada cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def seleccionar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona las variables analíticas para el clustering.
    Combina competencias + bienestar + variables adicionales.
    Solo incluye columnas que existen en la base.
    """
    todas = COLS_COMPETENCIAS + COLS_BIENESTAR + COLS_ADICIONALES
    disponibles = [c for c in todas if c in df.columns]
    faltantes = [c for c in todas if c not in df.columns]
    if faltantes:
        log.warning(f"Variables no encontradas: {faltantes}")
    log.info(f"Variables seleccionadas: {len(disponibles)} "
             f"({len(COLS_COMPETENCIAS)} competencias + "
             f"{len([c for c in COLS_BIENESTAR if c in df.columns])} bienestar + "
             f"{len([c for c in COLS_ADICIONALES if c in df.columns])} adicionales)")
    return df[disponibles].copy()


def imputar_nulos(X: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa nulos por mediana de cada variable.
    Se usa mediana (no media) por ser más robusta a outliers en escalas Likert.
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    n_nulos = X.isnull().sum().sum()
    log.info(f"Nulos imputados: {n_nulos} valores → mediana por columna.")
    return X_imp


def estandarizar(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Estandariza la matriz (media=0, std=1)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info("Variables estandarizadas (media=0, std=1).")
    return X_scaled, scaler


def aplicar_pca(X_scaled: np.ndarray, varianza_objetivo: float = 0.85) -> tuple[np.ndarray, PCA]:
    """
    Aplica PCA y selecciona el número de componentes que explican
    al menos el porcentaje de varianza objetivo.
    Reduce ruido antes del clustering.
    """
    pca_full = PCA()
    pca_full.fit(X_scaled)
    var_acum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(var_acum >= varianza_objetivo) + 1)
    n_components = max(n_components, 2)  # mínimo 2 componentes

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    var_total = sum(pca.explained_variance_ratio_) * 100

    log.info(f"PCA: {n_components} componentes explican {var_total:.1f}% de la varianza.")
    log.info(f"Varianza por componente: {[round(v*100,1) for v in pca.explained_variance_ratio_]}")
    return X_pca, pca


def evaluar_clustering(X_pca: np.ndarray, k_range: range = range(2, 8)) -> dict:
    """
    Evalúa KMeans y Clustering Jerárquico con múltiples métricas:
    - Silueta: qué tan separados y compactos son los clusters (mayor = mejor)
    - Davies-Bouldin: relación entre dispersión intra e inter cluster (menor = mejor)
    - Calinski-Harabasz: ratio varianza entre/dentro clusters (mayor = mejor)
    """
    resultados = {"kmeans": {}, "jerarquico": {}}

    log.info("Evaluando KMeans...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_pca)
        resultados["kmeans"][k] = {
            "silueta":        round(silhouette_score(X_pca, labels), 4),
            "davies_bouldin": round(davies_bouldin_score(X_pca, labels), 4),
            "calinski":       round(calinski_harabasz_score(X_pca, labels), 2),
            "labels":         labels,
        }
        log.info(f"  KMeans k={k}: silueta={resultados['kmeans'][k]['silueta']:.4f}")

    log.info("Evaluando Clustering Jerárquico (Ward)...")
    for k in k_range:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = hc.fit_predict(X_pca)
        resultados["jerarquico"][k] = {
            "silueta":        round(silhouette_score(X_pca, labels), 4),
            "davies_bouldin": round(davies_bouldin_score(X_pca, labels), 4),
            "calinski":       round(calinski_harabasz_score(X_pca, labels), 2),
            "labels":         labels,
        }
        log.info(f"  Jerárquico k={k}: silueta={resultados['jerarquico'][k]['silueta']:.4f}")

    return resultados


def seleccionar_mejor_modelo(resultados: dict, k_minimo: int = 3) -> tuple[str, int]:
    """
    Selecciona el mejor método y k basado en silueta,
    excluyendo k=2 si existe una opción con k>=3 con silueta razonable (>0.20).
    Esto garantiza arquetipos interpretables (más de 2 grupos).
    """
    mejor_metodo = None
    mejor_k = None
    mejor_silueta = -1

    for metodo, metricas_por_k in resultados.items():
        for k, metricas in metricas_por_k.items():
            if k < k_minimo:
                continue
            if metricas["silueta"] > mejor_silueta:
                mejor_silueta = metricas["silueta"]
                mejor_metodo = metodo
                mejor_k = k

    # Si ningún k>=3 supera 0.15, aceptar k=2
    if mejor_silueta < 0.15:
        log.warning("Silueta baja para k>=3. Considerando k=2.")
        for metodo, metricas_por_k in resultados.items():
            if 2 in metricas_por_k and metricas_por_k[2]["silueta"] > mejor_silueta:
                mejor_silueta = metricas_por_k[2]["silueta"]
                mejor_metodo = metodo
                mejor_k = 2

    log.info(f"Mejor modelo: {mejor_metodo} con k={mejor_k} (silueta={mejor_silueta:.4f})")
    return mejor_metodo, mejor_k


def entrenar_modelo_final(X_pca: np.ndarray, metodo: str, k: int) -> np.ndarray:
    """Entrena el modelo final con el método y k seleccionados."""
    if metodo == "kmeans":
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    else:
        modelo = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = modelo.fit_predict(X_pca)
    log.info(f"Modelo final entrenado: {metodo} k={k}")
    return labels, modelo


def renombrar_arquetipos(labels: np.ndarray, df_vars: pd.DataFrame) -> np.ndarray:
    """
    Renombra los clusters ordenándolos por score de bienestar promedio
    (de menor a mayor) para que los números sean consistentes
    con los arquetipos definidos en el proyecto.
    """
    df_temp = df_vars.copy()
    df_temp["_cluster"] = labels

    if "score_bienestar" in df_temp.columns:
        orden = (df_temp.groupby("_cluster")["score_bienestar"]
                 .mean()
                 .sort_values()
                 .index
                 .tolist())
        mapa = {old: new for new, old in enumerate(orden)}
        labels_renombrados = np.array([mapa[l] for l in labels])
        log.info(f"Arquetipos renombrados por score bienestar: {mapa}")
        return labels_renombrados
    return labels


def guardar_resultados(df_original: pd.DataFrame, labels: np.ndarray,
                       pca: PCA, scaler: StandardScaler,
                       resultados_eval: dict, metodo: str, k: int) -> None:
    """Guarda la base con arquetipos, el modelo y los artefactos."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    # Base con arquetipos
    df_out = df_original.copy()
    df_out["arquetipo"] = labels
    df_out["nombre_arquetipo"] = df_out["arquetipo"].map(ARCHETYPE_NAMES)

    # Convertir object a string para parquet
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str).where(df_out[col].notna(), other=None)

    ruta_base = ARTIFACTS_PATH / "base_con_arquetipos.parquet"
    df_out.to_parquet(ruta_base, index=False)
    log.info(f"Base con arquetipos guardada: {ruta_base}")

    # Modelo y transformadores
    modelo_obj = {
        "pca": pca,
        "scaler": scaler,
        "metodo": metodo,
        "k": k,
        "archetype_names": ARCHETYPE_NAMES,
    }
    ruta_modelo = MODELS_PATH / "modelo_arquetipos.pkl"
    with open(ruta_modelo, "wb") as f:
        pickle.dump(modelo_obj, f)
    log.info(f"Modelo guardado: {ruta_modelo}")

    # Métricas en CSV
    filas = []
    for met, metricas_k in resultados_eval.items():
        for k_val, metricas in metricas_k.items():
            filas.append({
                "metodo": met,
                "k": k_val,
                "silueta": metricas["silueta"],
                "davies_bouldin": metricas["davies_bouldin"],
                "calinski": metricas["calinski"],
            })
    pd.DataFrame(filas).to_csv(ARTIFACTS_PATH / "metricas_clustering.csv", index=False)
    log.info(f"Métricas guardadas: {ARTIFACTS_PATH / 'metricas_clustering.csv'}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Pipeline completo: PCA + clustering + selección de arquetipos."""
    resultados_modelo = ResultadosModelo()

    # 1. Cargar base procesada
    df = cargar_base()

    # 2. Seleccionar variables analíticas
    X = seleccionar_variables(df)
    resultados_modelo.n_registros  = len(X)
    resultados_modelo.n_variables  = X.shape[1]

    # 3. Imputar nulos y estandarizar
    X_imp = imputar_nulos(X)
    X_scaled, scaler = estandarizar(X_imp)

    # 4. PCA para reducir dimensionalidad
    X_pca, pca = aplicar_pca(X_scaled, varianza_objetivo=0.85)
    resultados_modelo.n_componentes_pca    = pca.n_components_
    resultados_modelo.varianza_explicada_pca = sum(pca.explained_variance_ratio_) * 100

    # 5. Evaluar métodos y k
    log.info("Evaluando métodos de clustering k=2..7...")
    resultados_eval = evaluar_clustering(X_pca)

    # 6. Seleccionar mejor modelo
    mejor_metodo, mejor_k = seleccionar_mejor_modelo(resultados_eval, k_minimo=3)
    resultados_modelo.metodo_clustering = mejor_metodo
    resultados_modelo.k_optimo = mejor_k

    # 7. Entrenar modelo final
    labels, modelo = entrenar_modelo_final(X_pca, mejor_metodo, mejor_k)

    # 8. Renombrar arquetipos por bienestar
    labels = renombrar_arquetipos(labels, X_imp)

    # 9. Registrar métricas y distribución
    for metodo, metricas_k in resultados_eval.items():
        for k_val, metricas in metricas_k.items():
            if metodo == mejor_metodo:
                resultados_modelo.metricas[k_val] = {
                    "silueta": metricas["silueta"],
                    "davies_bouldin": metricas["davies_bouldin"],
                    "calinski": metricas["calinski"],
                }
    resultados_modelo.distribucion_arquetipos = dict(
        pd.Series(labels).value_counts().sort_index()
    )

    # 10. Imprimir reporte
    resultados_modelo.imprimir()

    # 11. Guardar
    guardar_resultados(df, labels, pca, scaler, resultados_eval, mejor_metodo, mejor_k)

    return df


if __name__ == "__main__":
    run()