# auto.py
from src.pca_analysis import load_processed_data, standardize_data
from src.autoenconder_model import (
    build_autoencoder,
    train_autoencoder,
    compute_reconstruction_errors,
    plot_reconstruction_error,
    detect_anomalies
)

import os
import pandas as pd
import numpy as np

# Ruta a los datos preprocesados
DATA_DIR = 'notebooks/data/processed/'

# 1. Cargar datos
data_matrix = load_processed_data(DATA_DIR)
standardized_data = standardize_data(data_matrix)

# 2. Entrenar autoencoder
autoencoder = build_autoencoder(input_dim=standardized_data.shape[1])
train_autoencoder(autoencoder, standardized_data, epochs=20, batch_size=8)

# 3. Obtener errores
errors = compute_reconstruction_errors(autoencoder, standardized_data)

# 4. Visualizar
plot_reconstruction_error(errors)

# 5. Detectar anomalías
threshold = np.percentile(errors, 90)
labels = detect_anomalies(errors, threshold)

# 6. Guardar
df = pd.DataFrame({'ErrorReconstruccion': errors, 'Anomalia': labels})
df.to_csv(os.path.join(DATA_DIR, 'autoencoder_anomaly_results.csv'), index=False)
print("✅ Resultados guardados.")

import pandas as pd
import os

# Rutas a los archivos
DATA_DIR = 'notebooks/data/processed/'
path_auto = os.path.join(DATA_DIR, 'autoencoder_anomaly_results.csv')
path_iforest = os.path.join(DATA_DIR, 'pca_isolation_forest_results.csv')

# Cargar resultados
df_auto = pd.read_csv(path_auto)
df_iforest = pd.read_csv(path_iforest)

# Validación: ¿cuántas estrellas hay? Deben ser 8 en ambos
assert len(df_auto) == len(df_iforest), "❌ Número de estrellas no coincide entre ambos resultados"

# Añadir nombres de estrellas al df_auto (en el mismo orden que df_iforest)
df_auto['Star'] = df_iforest['Star']

# Renombrar columnas para claridad
df_auto.rename(columns={'Anomalia': 'Anom_Autoencoder'}, inplace=True)
df_iforest.rename(columns={'Anomaly': 'Anom_IsolationForest'}, inplace=True)

# Unir ambos DataFrames
df_comparacion = pd.merge(df_auto[['Star', 'Anom_Autoencoder']],
                          df_iforest[['Star', 'Anom_IsolationForest']],
                          on='Star')

# Mostrar tabla comparativa
print("📊 Comparación de métodos:")
print(df_comparacion)

# Guardar la comparación
df_comparacion.to_csv(os.path.join(DATA_DIR, 'comparacion_modelos.csv'), index=False)
print("✅ Comparación guardada como 'comparacion_modelos.csv'")
