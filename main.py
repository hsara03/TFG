from src.preprocessing import load_lightcurve, plot_lightcurve, remove_isolated_outliers, interpolate_missing_values, \
    save_processed_data, plot_removed_outliers

# 1. Cargar datos
file_path = "notebooks/data/raw/curva_luz_kepler.csv"
df = load_lightcurve(file_path)

# 2. Obtener límites del brillo para mantener la misma escala en ambas gráficas
brillo_min, brillo_max = df['brillo'].min(), df['brillo'].max()

# 3. Graficar datos originales con título claro y misma escala
plot_lightcurve(df, title="Curva de Luz - Antes del Procesamiento", ylim=(brillo_min, brillo_max))

# 4. Filtrar outliers
df_filtered = remove_isolated_outliers(df, window=10, sigma=5)


# 2. Comparar los DataFrames antes y después del filtrado
print(f"Total de puntos antes del filtrado: {len(df)}")
print(f"Total de puntos después del filtrado: {len(df_filtered)}")

# 3. Verificar si hay alguna diferencia real
if df.equals(df_filtered):
    print("✅ Los DataFrames son exactamente iguales. No se eliminó ningún punto.")
else:
    print("⚠️ Los DataFrames son diferentes. Se han eliminado algunos puntos.")
plot_removed_outliers(df, df_filtered)

# 5. Interpolar datos faltantes
df_processed = interpolate_missing_values(df_filtered)

# 6. Guardar datos limpios
save_processed_data(df_processed, "notebooks/data/processed/curva_luz_kepler.csv")

# 7. Graficar datos procesados con título claro y misma escala
plot_lightcurve(df_processed, title="Curva de Luz - Después del Procesamiento", ylim=(brillo_min, brillo_max))

from src.pca_analysis import load_processed_data, standardize_data
from src.autoenconder_model import (
    build_autoencoder,
    train_autoencoder,
    compute_reconstruction_errors,
    plot_reconstruction_error,
    detect_anomalies
)

import os
import numpy as np
import pandas as pd

# 1. Cargar y normalizar los datos
DATA_DIR = 'notebooks/data/processed/'
data_matrix = load_processed_data(DATA_DIR)
standardized_data = standardize_data(data_matrix)

# 2. Crear y entrenar el autoencoder
autoencoder = build_autoencoder(input_dim=standardized_data.shape[1])
history = train_autoencoder(autoencoder, standardized_data, epochs=100, batch_size=8)

# 3. Calcular errores de reconstrucción
errors = compute_reconstruction_errors(autoencoder, standardized_data)

# 4. Visualizar los errores
plot_reconstruction_error(errors)

# 5. Elegir un umbral y detectar anomalías
threshold = np.percentile(errors, 90)  # por ejemplo: top 10% como anomalías
anomaly_labels = detect_anomalies(errors, threshold)

# 6. Guardar resultados
df_results = pd.DataFrame({
    'ErrorReconstruccion': errors,
    'Anomalia': anomaly_labels
})
df_results.to_csv(os.path.join(DATA_DIR, 'autoencoder_anomaly_results.csv'), index=False)
print("✅ Resultados guardados como 'autoencoder_anomaly_results.csv'")
