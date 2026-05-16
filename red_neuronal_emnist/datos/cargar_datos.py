"""
cargar_datos.py
---------------
Responsabilidad: Leer los archivos CSV del dataset EMNIST Digits
y devolver los datos listos para entrenar y testear el modelo.

Dataset EMNIST Digits:
  - Imagenes de digitos escritos a mano (0 al 9)
  - Cada imagen es 28x28 pixeles = 784 valores (variables/columnas)
  - Formato CSV: columna 0 = etiqueta, columnas 1-784 = pixeles
  
Librerias usadas: numpy
"""

# pyrefly: ignore [missing-import]
import numpy as np

RUTA_TRAIN = r'C:\Users\ayuqu\Downloads\DB\emnist-digits-train.csv'
RUTA_TEST  = r'C:\Users\ayuqu\Downloads\DB\emnist-digits-test.csv'


def cargar_datos():
    """
    Carga el dataset EMNIST Digits desde los archivos CSV.
    
    Retorna:
        X_train : array con pixeles de entrenamiento  (n_muestras, 784)
        y_train : array con etiquetas de entrenamiento (n_muestras,)
        X_test  : array con pixeles de testeo          (n_muestras, 784)
        y_test  : array con etiquetas de testeo         (n_muestras,)
        n_features : numero de columnas/variables = 784
    """
    print("=" * 50)
    print("  CARGA DEL DATASET EMNIST DIGITS")
    print("=" * 50)

    print("\nCargando datos de entrenamiento...")
    datos_train = np.loadtxt(RUTA_TRAIN, delimiter=',', dtype='float32')

    print("Cargando datos de testeo...")
    datos_test = np.loadtxt(RUTA_TEST, delimiter=',', dtype='float32')

    # Separar etiquetas (columna 0) y pixeles (columnas 1 a 784)
    y_train = datos_train[:, 0].astype(int)
    X_train = datos_train[:, 1:].astype('float32')

    # Las y_test se guardan por separado (el profesor dijo: guardarlas aparte)
    y_test = datos_test[:, 0].astype(int)
    X_test = datos_test[:, 1:].astype('float32')

    n_features = X_train.shape[1]  # = 784 columnas de pixeles

    print(f"\nX_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print(f"n_features (columnas): {n_features}")
    print(f"Clases (digitos): {np.unique(y_train)}")

    return X_train, y_train, X_test, y_test, n_features


def normalizar(X_train, X_test):
    """
    Normaliza los valores de pixeles de [0, 255] a [0.0, 1.0].
    Esto mejora la convergencia del descenso del gradiente.
    
    Retorna:
        X_train normalizado
        X_test  normalizado
    """
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    print("\nNormalizacion aplicada: pixeles / 255.0")
    print(f"Rango X_train -> min: {X_train.min()}  max: {X_train.max()}")

    return X_train, X_test
