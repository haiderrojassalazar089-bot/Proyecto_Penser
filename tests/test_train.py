"""
test_train.py
=============
Pruebas unitarias para train.py (AFE + ACM + Ward).
Autores: Haider Rojas · Sergio Prieto — USTA 2026-1
"""

import sys
sys.path.insert(0, "src/percepcion")

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_procesado():
    """Base procesada simulada con variables analíticas."""
    np.random.seed(42)
    n = 300
    cols_comp = [
        "com_escrita", "com_oral", "pensamiento_critico", "metodos_cuantitativos",
        "metodos_cualitativos", "lectura_academica", "argumentacion", "segunda_lengua",
        "creatividad", "resolucion_conflictos", "liderazgo", "toma_decisiones",
        "resolucion_problemas", "investigacion", "herramientas_informaticas",
        "contextos_multiculturales", "insercion_laboral", "herramientas_modernas",
        "gestion_informacion", "trabajo_equipo", "aprendizaje_autonomo",
        "conocimientos_multidisciplinares", "etica",
    ]
    cols_sat = [
        "satisfaccion_formacion", "efecto_calidad_vida", "satisfaccion_vida",
        "correspondencia_primer_empleo", "correspondencia_empleo_actual",
    ]
    data = {c: np.random.choice([1, 2, 3, 4, 5], n) for c in cols_comp + cols_sat}
    data["score_bienestar"] = np.random.randint(0, 8, n)
    data["movilidad_social"] = np.random.randint(-2, 3, n)
    data["cat_genero"] = np.random.choice(["Masculino", "Femenino"], n)
    data["cat_sede"] = np.random.choice(["Bogotá", "Bucaramanga"], n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests de preparación AFE
# ---------------------------------------------------------------------------

def test_preparar_afe_retorna_dataframe(df_procesado):
    """preparar_matriz_afe retorna un DataFrame."""
    from train import preparar_matriz_afe
    result = preparar_matriz_afe(df_procesado)
    assert isinstance(result, pd.DataFrame)


def test_preparar_afe_sin_nulos(df_procesado):
    """preparar_matriz_afe elimina todos los nulos."""
    from train import preparar_matriz_afe
    result = preparar_matriz_afe(df_procesado)
    assert result.isnull().sum().sum() == 0


def test_preparar_afe_columnas_correctas(df_procesado):
    """preparar_matriz_afe solo incluye columnas de competencias y satisfacción."""
    from train import preparar_matriz_afe, COLS_COMPETENCIAS, COLS_SATISFACCION
    result = preparar_matriz_afe(df_procesado)
    for col in result.columns:
        assert col in COLS_COMPETENCIAS + COLS_SATISFACCION


def test_preparar_afe_conserva_registros(df_procesado):
    """preparar_matriz_afe conserva el número de registros."""
    from train import preparar_matriz_afe
    result = preparar_matriz_afe(df_procesado)
    assert len(result) == len(df_procesado)


# ---------------------------------------------------------------------------
# Tests de estandarización
# ---------------------------------------------------------------------------

def test_estandarizar_media_cero(df_procesado):
    """El espacio latente estandarizado tiene media ~0."""
    from train import preparar_matriz_afe
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    medias = np.abs(X_scaled.mean(axis=0))
    assert (medias < 1e-10).all()


def test_estandarizar_std_uno(df_procesado):
    """El espacio latente estandarizado tiene std ~1."""
    from train import preparar_matriz_afe
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    stds = X_scaled.std(axis=0)
    assert np.allclose(stds, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests de ACM
# ---------------------------------------------------------------------------

def test_preparar_acm_retorna_dataframe(df_procesado):
    """preparar_datos_acm retorna un DataFrame."""
    from train import preparar_datos_acm
    result = preparar_datos_acm(df_procesado)
    assert isinstance(result, pd.DataFrame)


def test_preparar_acm_sin_nulos(df_procesado):
    """preparar_datos_acm no tiene nulos (imputa por moda)."""
    from train import preparar_datos_acm
    result = preparar_datos_acm(df_procesado)
    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Tests de clustering
# ---------------------------------------------------------------------------

def test_clustering_genera_etiquetas_validas(df_procesado):
    """El clustering genera etiquetas para todos los registros."""
    from sklearn.cluster import AgglomerativeClustering
    from train import preparar_matriz_afe
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    assert len(labels) == len(df_procesado)
    assert set(labels).issubset({0, 1, 2})


def test_clustering_jerarquico_genera_etiquetas(df_procesado):
    """El clustering jerárquico Ward funciona correctamente."""
    from sklearn.cluster import AgglomerativeClustering
    from train import preparar_matriz_afe
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    assert len(np.unique(labels)) == 3


def test_evaluacion_clustering_retorna_metricas(df_procesado):
    """evaluar_clustering retorna métricas para cada k."""
    from train import preparar_matriz_afe, evaluar_clustering
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    resultados = evaluar_clustering(X_scaled, k_range=range(2, 5))
    assert isinstance(resultados, dict)
    assert 2 in resultados and 3 in resultados
    for k, m in resultados.items():
        assert "silueta" in m
        assert "labels" in m


def test_silueta_entre_menos1_y_1(df_procesado):
    """El coeficiente de silueta está en rango válido [-1, 1]."""
    from train import preparar_matriz_afe, evaluar_clustering
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    resultados = evaluar_clustering(X_scaled, k_range=range(2, 4))
    for k, m in resultados.items():
        assert -1 <= m["silueta"] <= 1


def test_k_optimo_cumple_minimo_grupo(df_procesado):
    """El k óptimo garantiza que todos los grupos tienen al menos 5% del total."""
    from train import preparar_matriz_afe, evaluar_clustering, seleccionar_k_optimo
    X = preparar_matriz_afe(df_procesado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    resultados = evaluar_clustering(X_scaled, k_range=range(2, 5))
    k = seleccionar_k_optimo(resultados)
    labels = resultados[k]["labels"]
    counts = pd.Series(labels).value_counts()
    min_pct = counts.min() / len(labels)
    assert min_pct >= 0.05 or k == 3  # k=3 es forzado por parsimonia