import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np

def resolver_rutas_dataset():
    """
    Resuelve de forma dinamica y segura las rutas del dataset EMNIST.
    Prioridad de busqueda:
      1. Variables de entorno (EMNIST_TRAIN_PATH, EMNIST_TEST_PATH)
      2. Carpeta local 'data/' en la raiz del proyecto
      3. Carpeta 'Downloads/DB' en el directorio de usuario (fallback de desarrollo)
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"

    env_train = os.getenv("EMNIST_TRAIN_PATH")
    env_test = os.getenv("EMNIST_TEST_PATH")

    def _resolver(env_val, filename):
        if env_val and os.path.exists(env_val):
            return Path(env_val)
        local_data = data_dir / filename
        if local_data.exists():
            return local_data
        user_downloads = Path.home() / "Downloads" / "DB" / filename
        if user_downloads.exists():
            return user_downloads
        return local_data

    ruta_train = _resolver(env_train, "emnist-digits-train.csv")
    ruta_test  = _resolver(env_test, "emnist-digits-test.csv")
    return ruta_train, ruta_test

RUTA_TRAIN, RUTA_TEST = resolver_rutas_dataset()

def cargar_datos(ruta_train=None, ruta_test=None):
    train_path = Path(ruta_train) if ruta_train else RUTA_TRAIN
    test_path  = Path(ruta_test) if ruta_test else RUTA_TEST

    print("=" * 50)
    print("  CARGA DEL DATASET EMNIST DIGITS")
    print("=" * 50)

    if not train_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] No se encontro el archivo de entrenamiento en:\n  -> {train_path}\n"
            f"Por favor, descarga 'emnist-digits-train.csv' y colocalo en la carpeta 'data/' "
            f"o configura la variable de entorno EMNIST_TRAIN_PATH."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] No se encontro el archivo de prueba en:\n  -> {test_path}\n"
            f"Por favor, descarga 'emnist-digits-test.csv' y colocalo en la carpeta 'data/' "
            f"o configura la variable de entorno EMNIST_TEST_PATH."
        )

    print(f"\nRuta Train : {train_path}")
    print(f"Ruta Test  : {test_path}")

    print("\nCargando datos de entrenamiento...")
    datos_train = np.loadtxt(train_path, delimiter=',', dtype='float32')

    print("Cargando datos de testeo...")
    datos_test = np.loadtxt(test_path, delimiter=',', dtype='float32')

    y_train = datos_train[:, 0].astype(int)
    X_train = datos_train[:, 1:].astype('float32')

    y_test = datos_test[:, 0].astype(int)
    X_test = datos_test[:, 1:].astype('float32')

    n_features = X_train.shape[1]

    print(f"\nX_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print(f"n_features (columnas): {n_features}")
    print(f"Clases (digitos): {np.unique(y_train)}")

    return X_train, y_train, X_test, y_test, n_features

def normalizar(X_train, X_test):
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    print("\nNormalizacion aplicada: pixeles / 255.0")
    print(f"Rango X_train -> min: {X_train.min()}  max: {X_train.max()}")

    return X_train, X_test
