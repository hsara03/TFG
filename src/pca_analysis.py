import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ruta a tus datos preprocesados
DATA_DIR = 'notebooks/data/processed/'

# Paso 1: Cargar todos los CSV preprocesados
def load_processed_data(data_dir, num_points=300):
    all_curves = []

    for filename in os.listdir(data_dir):
        # Solo archivos .csv que empiecen por "curva_luz_"
        if filename.endswith('.csv') and filename.startswith('curva_luz_'):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)

            # Comprobamos si tiene la columna "brillo"
            if 'brillo' not in df.columns:
                print(f"⚠️ {filename} no tiene columna 'brillo'. Saltando...")
                continue

            if df.shape[0] < 2:
                print(f"⚠️ {filename} está vacío o tiene muy pocos datos. Saltando...")
                continue

            # Interpolación para igualar longitudes
            original_x = np.linspace(0, 1, num=len(df))
            target_x = np.linspace(0, 1, num=num_points)
            interpolated_flux = np.interp(target_x, original_x, df['brillo'].values)

            all_curves.append(interpolated_flux)
            print(f"✅ Cargada curva: {filename}")

        else:
            print(f"⏭️ Ignorada: {filename} (no es curva de luz válida)")

    return np.array(all_curves)


# Paso 2: Estandarizar datos (normalización por característica)
def standardize_data(data_matrix):
    scaler = StandardScaler()
    return scaler.fit_transform(data_matrix)

# Paso 3: Aplicar PCA
def apply_pca(data_matrix, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_matrix)
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance

# Paso 4: Visualización
def plot_pca_results(principal_components, explained_variance):
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)
    plt.title(f'PCA de curvas de luz\nVarianza explicada: PC1 {explained_variance[0]:.2f}, PC2 {explained_variance[1]:.2f}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Paso 5 (opcional): Guardar los resultados
def save_pca_results(principal_components, output_path='pca_results.csv'):
    df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df.to_csv(output_path, index=False)
    print(f'Resultados de PCA guardados en: {output_path}')

if __name__ == '__main__':
    print("Cargando datos preprocesados...")
    data_matrix = load_processed_data(DATA_DIR)

    print("Estandarizando datos...")
    standardized_data = standardize_data(data_matrix)

    print("Aplicando PCA...")
    principal_components, explained_variance = apply_pca(standardized_data)

    print("Visualizando resultados...")
    plot_pca_results(principal_components, explained_variance)

    print("Guardando resultados...")
    save_pca_results(principal_components)
