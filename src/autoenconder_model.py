import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def build_autoencoder(input_dim, encoding_dim=64):
    """
    Crea un autoencoder simple con codificador y decodificador simétricos.
    - input_dim: número de puntos de la curva (ej. 300)
    - encoding_dim: tamaño del cuello de botella
    """
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = Dense(128, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    # Modelo completo
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return autoencoder


def train_autoencoder(model, data, epochs=100, batch_size=16, verbose=1):
    """
    Entrena el autoencoder con los datos.
    - model: el autoencoder
    - data: matriz N x 300 (curvas normalizadas)
    """
    history = model.fit(data, data,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=verbose)
    return history


def compute_reconstruction_errors(model, data):
    """
    Calcula el error de reconstrucción MSE para cada muestra.
    """
    reconstructed = model.predict(data)
    errors = np.mean(np.square(data - reconstructed), axis=1)
    return errors


def plot_reconstruction_error(errors, threshold=None):
    """
    Dibuja un histograma de errores de reconstrucción y marca un umbral si se da.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel("Error de reconstrucción (MSE)")
    plt.ylabel("Número de curvas")
    plt.title("Distribución del error de reconstrucción")

    if threshold:
        plt.axvline(threshold, color='r', linestyle='--', label=f"Umbral = {threshold:.4f}")
        plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def detect_anomalies(errors, threshold):
    """
    Clasifica las curvas como normales o anómalas según el umbral.
    Devuelve una lista de etiquetas: 1 = normal, -1 = anomalía
    """
    return np.where(errors > threshold, -1, 1)
