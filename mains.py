import os
import pandas as pd
from src.preprocessings import load_lightcurve, remove_outliers, plot_all_stages

# Definir rutas
raw_data_path = os.path.join(os.getcwd(), "notebooks", "data", "raw")
processed_data_path = os.path.join(os.getcwd(), "notebooks", "data", "processed")
os.makedirs(processed_data_path, exist_ok=True)

if not os.path.exists(raw_data_path):
    raise FileNotFoundError(f"❌ La carpeta de datos no existe: {raw_data_path}")

files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]

if not files:
    raise FileNotFoundError(f"❌ No se encontraron archivos CSV en {raw_data_path}. Asegúrate de que los datos están en la carpeta correcta.")

print(f"📂 Se encontraron {len(files)} archivos CSV en {raw_data_path}.")

# 🚀 PROCESAR CADA ARCHIVO ENCONTRADO
for file in files:
    file_path = os.path.join(raw_data_path, file)
    processed_file_path = os.path.join(processed_data_path, file)
    print(f"🔍 Procesando {file}...")

    # 1️⃣ Cargar datos
    df = load_lightcurve(file_path)

    # 2️⃣ Verificar si la columna 'brillo' tiene valores válidos
    if df['brillo'].isna().all():
        print(f"⚠️ Todos los valores de 'brillo' son NaN en {file}. Saltando archivo...")
        continue

    # 3️⃣ Filtrar outliers
    df_filtered = remove_outliers(df, sigma=1.5)

    # 4️⃣ Comparar antes y después del filtrado
    print(f"Total de puntos antes del filtrado: {len(df)}")
    print(f"Total de puntos después del filtrado: {len(df_filtered)}")

    if df.equals(df_filtered):
        print("✅ No se eliminaron outliers.")
    else:
        print("⚠️ Se eliminaron algunos puntos atípicos.")

    # 5️⃣ Guardar datos limpios
    df_filtered.to_csv(processed_file_path, index=False)
    print(f"✅ Datos procesados guardados en: {processed_file_path}")

    # 6️⃣ Obtener límites comunes de brillo para las gráficas
    brillo_min = min(df['brillo'].min(), df_filtered['brillo'].min())
    brillo_max = max(df['brillo'].max(), df_filtered['brillo'].max())
    ylim = (brillo_min, brillo_max)

    # 7️⃣ Graficar las tres etapas con mismo rango
    plot_all_stages(df, df_filtered, ylim=ylim)

print("✅ Todas las curvas de luz han sido procesadas correctamente.")

# 🔎 PASO EXTRA: Análisis PCA
from src.pca_analysis import load_processed_data, standardize_data, apply_pca, plot_pca_results, save_pca_results

print("🚀 Iniciando análisis PCA sobre los datos procesados...")

data_matrix = load_processed_data(processed_data_path)
standardized_data = standardize_data(data_matrix)
principal_components, explained_variance = apply_pca(standardized_data)
plot_pca_results(principal_components, explained_variance)
save_pca_results(principal_components, output_path=os.path.join(processed_data_path, 'pca_results.csv'))
print("✅ Análisis PCA completado y resultados guardados.")

import pandas as pd
import plotly.express as px
import os

# Crear DataFrame con los resultados y nombres de estrellas
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
df_pca['Star'] = [
    "Kepler-10",
    "Kepler-22",
    "Kepler-62",
    "Kepler-186",
    "Kepler-452",
    "KIC 12557548",
    "KIC 3542116",
    "KIC 8462852"
]

# Crear el gráfico interactivo
fig = px.scatter(
    df_pca,
    x='PC1',
    y='PC2',
    text='Star',
    title='PCA Interactivo de Curvas de Luz de Estrellas',
    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'},
    template='plotly_white'
)

# Añadir los nombres de las estrellas como texto en el hover (tooltip)
fig.update_traces(textposition='top center', marker=dict(size=10, color='blue'))

# Mostrar el gráfico
fig.show()

# Guardar el gráfico como archivo HTML interactivo
fig.write_html(os.path.join(processed_data_path, 'pca_star_plot_interactive.html'))

print("✅ Gráfico PCA interactivo generado correctamente.")
from sklearn.ensemble import IsolationForest
import numpy as np
import plotly.express as px

print("🚀 Iniciando análisis de anomalías con Isolation Forest...")

# Creamos el modelo Isolation Forest
iso_forest = IsolationForest(contamination='auto', random_state=42)
anomaly_labels = iso_forest.fit_predict(df_pca[['PC1', 'PC2']])
anomaly_scores = iso_forest.decision_function(df_pca[['PC1', 'PC2']])

# Añadimos los resultados al DataFrame
df_pca['Anomaly'] = anomaly_labels
df_pca['Score'] = anomaly_scores

# Visualizamos resultados con plotly
fig = px.scatter(
    df_pca,
    x='PC1',
    y='PC2',
    color='Anomaly',
    hover_data=['Star', 'Score'],
    title='Detección de Anomalías con Isolation Forest',
    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'Anomaly': 'Anomalía'},
    template='plotly_white'
)

fig.update_traces(marker=dict(size=12))
fig.show()

# Guardamos resultados
df_pca.to_csv(os.path.join(processed_data_path, 'pca_isolation_forest_results.csv'), index=False)
fig.write_html(os.path.join(processed_data_path, 'isolation_forest_interactive.html'))

print("✅ Análisis de anomalías completado y resultados guardados.")

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

