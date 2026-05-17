# pyrefly: ignore [missing-import]
import numpy as np

RUTA_TRAIN = r'C:\Users\ayuqu\Downloads\DB\emnist-digits-train.csv'
RUTA_TEST  = r'C:\Users\ayuqu\Downloads\DB\emnist-digits-test.csv'

def cargar_datos():
    print("=" * 50)
    print("  CARGA DEL DATASET EMNIST DIGITS")
    print("=" * 50)

    print("\nCargando datos de entrenamiento...")
    datos_train = np.loadtxt(RUTA_TRAIN, delimiter=',', dtype='float32')

    print("Cargando datos de testeo...")
    datos_test = np.loadtxt(RUTA_TEST, delimiter=',', dtype='float32')

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
