import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

@pytest.fixture
def dummy_pca_data():
    """Genera un DataFrame simulado con datos de PCA"""
    np.random.seed(42)
    normal_points = np.random.normal(loc=0, scale=1, size=(50, 2))  # Grupo de puntos normales
    outlier_points = np.random.normal(loc=5, scale=1, size=(5, 2))  # Grupo de outliers

    data = np.vstack([normal_points, outlier_points])
    df = pd.DataFrame(data, columns=["PC1", "PC2"])
    df['Star'] = ['Star_' + str(i) for i in range(len(df))]
    return df

@pytest.mark.parametrize("contamination, expected_min_anomalies", [
    (0.01, 1),
    (0.05, 2),
    (0.1, 3),
])
def test_isolation_forest_thresholds(dummy_pca_data, contamination, expected_min_anomalies):
    """Testea cómo Isolation Forest responde a diferentes niveles de contamination"""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(dummy_pca_data[["PC1", "PC2"]])

    # Contamos cuántos puntos fueron clasificados como anómalos (-1)
    num_anomalies = (anomaly_labels == -1).sum()

    print(f"Contamination: {contamination} - Anomalías detectadas: {num_anomalies}")

    # Verificamos que detecte al menos el mínimo esperado de anomalías
    assert num_anomalies >= expected_min_anomalies, (
        f"Esperaba al menos {expected_min_anomalies} anomalías, pero encontré {num_anomalies}"
    )
