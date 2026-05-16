"""
arquitectura.py
---------------
Responsabilidad: Definir y construir la arquitectura de la red neuronal
usando keras.Sequential() tal como explico el profesor.

Arquitectura del profesor (imagen en clase):
  Input(784) -> Dense(25, relu) -> Dense(15, relu) -> Dense(5, relu) -> Dense(10, softmax)

Conceptos del profesor aplicados aqui:
  - keras.Sequential() : modelo secuencial instanciado en variable
  - model.add()        : agrega capas a la arquitectura
  - input_shape        : SOLO en la primera capa (n_features = 784)
  - relu               : evita vanishing gradient (pesos muy pequeños)
  - Cada capa PUEDE tener distinta funcion de activacion (lo dijo el profe)
  - softmax en salida  : 10 neuronas, una por digito (0-9)

Librerias usadas: tensorflow, keras
"""

from tensorflow import keras
from tensorflow.keras import layers


def construir_modelo(n_features):
    """
    Construye la arquitectura de la red neuronal.

    Parametros:
        n_features (int): numero de columnas/variables del dataset (784)

    Retorna:
        model: modelo keras.Sequential con la arquitectura lista
    """
    print("\n" + "=" * 50)
    print("  ARQUITECTURA DE LA RED NEURONAL")
    print("=" * 50)

    # Instanciar el modelo secuencial - como el profesor
    model = keras.Sequential()

    # --- CAPAS OCULTAS ---
    # Solo la primera capa lleva input_shape (n_features=784)
    # relu: funcion de activacion que evita el vanishing gradient
    # El profesor dijo: cada capa puede tener distinta funcion de activacion
    # pyrefly: ignore [unexpected-keyword]
    model.add(layers.Dense(25, activation='relu', input_shape=(n_features,)))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(5,  activation='relu'))

    # --- CAPA DE SALIDA ---
    # 10 neuronas = 10 digitos posibles (0 al 9)
    # softmax: convierte las salidas en probabilidades (suma = 1)
    model.add(layers.Dense(10, activation='softmax'))

    print(f"\nEntrada  : {n_features} neuronas (pixeles 28x28)")
    print("Oculta 1 : 25 neuronas | activacion: relu")
    print("Oculta 2 : 15 neuronas | activacion: relu")
    print("Oculta 3 :  5 neuronas | activacion: relu")
    print("Salida   : 10 neuronas | activacion: softmax (digitos 0-9)")

    return model
