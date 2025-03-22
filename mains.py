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