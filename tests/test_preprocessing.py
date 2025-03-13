import pytest
import pandas as pd
import numpy as np
from src.preprocessing import remove_isolated_outliers  # Importa la función


@pytest.fixture
def dummy_lightcurve():
    """Crea una curva de luz de prueba sin outliers."""
    np.random.seed(42)
    timecorr = np.linspace(0, 10, 1000)  # Simulación de tiempo
    brillo = 1.0 + 0.01 * np.sin(timecorr)  # Señal limpia con pequeñas variaciones
    return pd.DataFrame({'timecorr': timecorr, 'brillo': brillo})


@pytest.fixture
def lightcurve_with_outliers(dummy_lightcurve):
    """Crea una curva de luz con algunos outliers artificiales."""
    df = dummy_lightcurve.copy()  # Asegurar que no se modifica la original
    df.loc[100, 'brillo'] = 10  # Outlier alto
    df.loc[200, 'brillo'] = -5  # Outlier bajo
    df.loc[300, 'brillo'] = 3  # Otro outlier alto
    return df


def test_no_outliers(dummy_lightcurve):
    """Prueba que la función no elimine nada si no hay outliers."""
    df_filtered = remove_isolated_outliers(dummy_lightcurve, window=10, sigma=5)
    assert len(df_filtered) == len(dummy_lightcurve), "Error: Se eliminaron puntos en una curva limpia"


def test_remove_outliers(lightcurve_with_outliers):
    """Prueba que la función elimine los outliers correctamente."""
    df_filtered = remove_isolated_outliers(lightcurve_with_outliers, window=10, sigma=2)

    # Comprobar que los outliers fueron eliminados
    assert 100 not in df_filtered.index, "Outlier en índice 100 no fue eliminado"
    assert 200 not in df_filtered.index, "Outlier en índice 200 no fue eliminado"
    assert 300 not in df_filtered.index, "Outlier en índice 300 no fue eliminado"


def test_keep_structure(lightcurve_with_outliers):
    """Prueba que la estructura de la curva de luz se mantenga tras eliminar outliers."""
    df_filtered = remove_isolated_outliers(lightcurve_with_outliers, window=10, sigma=5)

    # Verificar que la mayoría de los datos se mantuvieron
    assert len(df_filtered) > 900, "Error: Se eliminaron demasiados puntos"
