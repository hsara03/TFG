import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_lightcurve(file_path):
    df = pd.read_csv(file_path)
    print(f"\n📂 Columnas disponibles en {file_path}: {df.columns.tolist()}")

    if 'quality' in df.columns:
        df = df[df['quality'] == 0].reset_index(drop=True)

    if 'timecorr' in df.columns:
        time_col = 'timecorr'
        print(f"✅ Usando 'timecorr' como tiempo en {file_path}.")
    elif 'cadenceno' in df.columns:
        df['cadenceno'] -= df['cadenceno'].min()
        time_col = 'cadenceno'
        print(f"⚠️ No se encontró 'timecorr', usando 'cadenceno' como tiempo relativo.")
    else:
        raise KeyError(f"❌ No se encontró ninguna columna de tiempo en {file_path}.")

    if 'pdcsap_flux' in df.columns:
        df['pdcsap_flux'] = df['pdcsap_flux'].fillna(df['pdcsap_flux'].median())
        if (df['pdcsap_flux'] <= 0).any():
            print(f"⚠️ Algunos valores de 'pdcsap_flux' son <= 0 en {file_path}, esto podría afectar la normalización.")
        df['brillo'] = df['pdcsap_flux'] / np.median(df['pdcsap_flux'])
    elif 'flux' in df.columns:
        df['brillo'] = df['flux']
    else:
        raise KeyError(f"❌ No se encontró ninguna columna de brillo en {file_path}.")

    df['brillo'] = df['brillo'].fillna(df['brillo'].mean())
    print(df[[time_col, 'brillo']].head())
    return df[[time_col, 'brillo']].rename(columns={time_col: 'tiempo'})


def plot_lightcurve(df, title="Curva de Luz", ylim=None):
    plt.figure(figsize=(10, 5))
    plt.plot(df['tiempo'], df['brillo'], 'k.', markersize=1, alpha=0.5)
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Brillo Normalizado')
    plt.title(title)
    if ylim is None or np.isnan(ylim[0]) or np.isnan(ylim[1]):
        ylim = (df['brillo'].min(), df['brillo'].max())
    plt.ylim(ylim)
    plt.show()


def plot_difference(original_df, filtered_df, ylim):
    merged = original_df.merge(filtered_df, on=['tiempo', 'brillo'], how='left', indicator=True)
    eliminados = merged[merged['_merge'] == 'left_only']

    plt.figure(figsize=(10, 5))
    plt.plot(original_df['tiempo'], original_df['brillo'], 'k.', markersize=1, alpha=0.5, label='Original')
    plt.plot(eliminados['tiempo'], eliminados['brillo'], 'ro', markersize=2, alpha=0.8, label='Outliers eliminados')
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Brillo Normalizado')
    plt.title('Outliers eliminados sobre la Curva Original')
    plt.ylim(ylim)
    plt.legend()
    plt.show()


def plot_all_stages(original_df, filtered_df, ylim):
    plot_lightcurve(original_df, title="Curva de Luz - Original", ylim=ylim)
    plot_difference(original_df, filtered_df, ylim=ylim)
    plot_lightcurve(filtered_df, title="Curva de Luz - Filtrada", ylim=ylim)


def remove_outliers(df, sigma=1.5):
    q1, q3 = np.percentile(df['brillo'], [10, 90])
    iqr = q3 - q1
    lower_bound = q1 - sigma * iqr
    upper_bound = q3 + sigma * iqr
    df_filtered = df[(df['brillo'] >= lower_bound) & (df['brillo'] <= upper_bound)]

    if len(df_filtered) < 0.05 * len(df):
        print("⚠️ Se evitaron eliminaciones masivas, conservando datos originales.")
        return df.reset_index(drop=True)

    print(f"Se eliminaron {len(df) - len(df_filtered)} outliers.")
    return df_filtered.reset_index(drop=True)


def process_lightcurve(file_path, output_path):
    df = load_lightcurve(file_path)
    df_filtered = remove_outliers(df)
    ylim = (min(df['brillo'].min(), df_filtered['brillo'].min()), max(df['brillo'].max(), df_filtered['brillo'].max()))
    plot_all_stages(df, df_filtered, ylim=ylim)
    df_filtered.to_csv(output_path, index=False)
    print(f"✅ Datos procesados guardados en: {output_path}\n")


def process_all_lightcurves(input_folder, output_folder):
    import os
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"\n📂 Se encontraron {len(files)} archivos CSV en {input_folder}.")

    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        print(f"🔍 Procesando {file}...")
        process_lightcurve(input_path, output_path)

    print("✅ Todas las curvas de luz han sido procesadas correctamente.")

# Ejemplo de uso:
# process_all_lightcurves('ruta/de/entrada', 'ruta/de/salida')
