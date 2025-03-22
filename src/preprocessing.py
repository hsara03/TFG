import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_lightcurve(file_path):
    """Carga la curva de luz desde un archivo CSV y usa la columna correcta."""
    df = pd.read_csv(file_path)

    # Filtrar solo datos de buena calidad
    df = df[df['quality'] == 0]

    # Usar la columna 'flux' en lugar de 'pdcsap_flux'
    if 'flux' in df.columns:
        df['brillo'] = df['flux']  # Crear una columna con un nombre más claro
    else:
        print("⚠️ Advertencia: No se encontró la columna 'flux'. Usando 'pdcsap_flux'.")
        df['brillo'] = df['pdcsap_flux'] / np.median(df['pdcsap_flux'])  # Normalizar si es necesario

    return df

def plot_lightcurve(df, title="Curva de Luz", ylim=None):
    """Grafica la curva de luz con un título personalizado y la misma escala de ejes."""
    plt.figure(figsize=(10,5))
    plt.plot(df['timecorr'], df['brillo'], 'k.', markersize=1, alpha=0.5)
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Brillo Normalizado')
    plt.title(title)

    # Mantener la misma escala de brillo si se proporciona un límite
    if ylim:
        plt.ylim(ylim)

    plt.show()

def plot_removed_outliers(original_df, filtered_df):
    """Grafica los puntos eliminados por el filtrado de outliers."""
    datos_eliminados = original_df[~original_df.index.isin(filtered_df.index)]

    plt.figure(figsize=(10,5))
    plt.plot(original_df['timecorr'], original_df['brillo'], 'k.', markersize=1, alpha=0.5, label="Datos originales")
    plt.plot(datos_eliminados['timecorr'], datos_eliminados['brillo'], 'ro', markersize=3, label="Puntos eliminados")
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Brillo Normalizado')
    plt.title('Puntos Eliminados en el Filtrado de Outliers')
    plt.legend()
    plt.show()




def remove_isolated_outliers(df, window=10, sigma=2):
    """Elimina solo los puntos aislados que no siguen un patrón continuo."""

    df_filtered = df.copy()

    # Calcular la media y desviación estándar en una ventana deslizante
    df_filtered['rolling_mean'] = df_filtered['brillo'].rolling(window=window, center=True).mean()
    df_filtered['rolling_std'] = df_filtered['brillo'].rolling(window=window, center=True).std()

    # Determinar los límites superior e inferior para considerar un punto como outlier
    limite_superior = df_filtered['rolling_mean'] + sigma * df_filtered['rolling_std']
    limite_inferior = df_filtered['rolling_mean'] - sigma * df_filtered['rolling_std']

    # Identificar los outliers
    outliers = (df_filtered['brillo'] > limite_superior) | (df_filtered['brillo'] < limite_inferior)

    # ✅ CORRECCIÓN: Filtrar directamente los valores fuera de los límites
    df_filtered = df_filtered[~outliers]

    print(f"Se eliminaron {len(df) - len(df_filtered)} puntos atípicos aislados.")

    # Eliminar columnas auxiliares antes de devolver
    df_filtered = df_filtered.drop(columns=['rolling_mean', 'rolling_std'])

    return df_filtered


def interpolate_missing_values(df):
    """Interpola valores faltantes en la columna 'brillo' solo si hay NaNs."""
    df_interpolated = df.copy()

    # Verificar si hay valores NaN antes de interpolar
    if df_interpolated['brillo'].isna().sum() > 0:
        df_interpolated['brillo'] = df_interpolated['brillo'].interpolate(method='linear')

        # Rellenar NaN restantes (inicio y fin de la serie)
        df_interpolated['brillo'] = df_interpolated['brillo'].fillna(df_interpolated['brillo'].median())

    return df_interpolated

def save_processed_data(df, output_path):
    """Guarda los datos procesados en un archivo CSV."""
    df.to_csv(output_path, index=False)
    print(f"✅ Datos procesados guardados en: {output_path}")
