"""
arquitectura.py
---------------
Responsabilidad: Definir y construir la arquitectura de la red neuronal
usando keras.Sequential() tal como explico el profesor.

Arquitectura mejorada (distintas activaciones por capa - como dijo el profesor):
  Input(784)
    -> Dense(128, relu)    : capa ancha para extraer caracteristicas
    -> Dense(64,  relu)    : reduccion con relu (evita vanishing gradient)
    -> Dense(32,  tanh)    : el profesor dijo: cada capa PUEDE tener distinta activacion
    -> Dense(16,  sigmoid) : demostracion de activacion diferente
    -> Dense(10,  softmax) : salida: 10 probabilidades (digitos 0-9)

Conceptos del profesor aplicados aqui:
  - keras.Sequential() : modelo secuencial instanciado en variable
  - model.add()        : agrega capas a la arquitectura
  - input_shape        : SOLO en la primera capa (n_features = 784)
  - relu               : evita vanishing gradient (pesos que se desvanecen)
  - Cada capa PUEDE tener distinta funcion de activacion (lo dijo el profe)
  - softmax en salida  : 10 neuronas, una por digito (0-9)

Librerias usadas: tensorflow, keras
"""

from tensorflow import keras
from tensorflow.keras import layers
from config import CAPAS_OCULTAS, N_SALIDA, ACTIVACION_SALIDA


def construir_modelo(n_features):
    """
    Construye la arquitectura de la red neuronal.

    Parametros:
        n_features (int): numero de columnas/variables del dataset (784)

    Retorna:
        model: modelo keras.Sequential con la arquitectura lista
    """
    print("\n" + "=" * 60)
    print("  ARQUITECTURA DE LA RED NEURONAL")
    print("=" * 60)

    # Instanciar el modelo secuencial - como el profesor
    model = keras.Sequential()

    # --- CAPAS OCULTAS (definidas en config.py) ---
    # La primera capa lleva input_shape, las demas no
    # Bucle: agrega/quita capas desde config.py sin tocar este archivo
    for i, (neuronas, activacion) in enumerate(CAPAS_OCULTAS):
        if i == 0:
            # pyrefly: ignore [unexpected-keyword]
            model.add(layers.Dense(neuronas, activation=activacion,
                                   # pyrefly: ignore [unexpected-keyword]
                                   input_shape=(n_features,)))
            print(f"Entrada   : {n_features} variables (pixeles 28x28)")
        else:
            model.add(layers.Dense(neuronas, activation=activacion))

        # Nota sobre vanishing gradient:
        # tanh/sigmoid con pocas capas (<10) no causan problema
        nota = "" if activacion == 'relu' else " (ok con pocas capas)"
        print(f"Oculta {i+1:2d}  : {neuronas:3d} neuronas | act: {activacion}{nota}")

    # --- CAPA DE SALIDA (siempre softmax para 10 clases) ---
    model.add(layers.Dense(N_SALIDA, activation=ACTIVACION_SALIDA))
    print(f"Salida    : {N_SALIDA:3d} neuronas | act: {ACTIVACION_SALIDA} (probabilidades 0-9)")

    return model
