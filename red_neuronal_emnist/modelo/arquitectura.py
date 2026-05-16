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

    # --- CAPA 1: entrada + primera capa oculta ---
    # Solo la primera capa lleva input_shape (n_features=784)
    # relu: funcion mas usada, evita el vanishing gradient
    # pyrefly: ignore [unexpected-keyword]
    model.add(layers.Dense(128, activation='relu', input_shape=(n_features,)))

    # --- CAPA 2: segunda capa oculta ---
    # relu: sigue extrayendo caracteristicas sin perder gradiente
    model.add(layers.Dense(64, activation='relu'))

    # --- CAPA 3: tercera capa oculta ---
    # tanh: el profesor dijo que CADA CAPA PUEDE TENER DISTINTA ACTIVACION
    # tanh da salidas entre -1 y 1, util para capas intermedias
    model.add(layers.Dense(32, activation='tanh'))

    # --- CAPA 4: cuarta capa oculta ---
    # sigmoid: otra activacion distinta, salida entre 0 y 1
    model.add(layers.Dense(16, activation='sigmoid'))

    # --- CAPA DE SALIDA ---
    # 10 neuronas = 10 digitos posibles (0 al 9)
    # softmax: convierte las 10 salidas en probabilidades (suman = 1)
    # SOLO se usa softmax en la salida, NO en capas ocultas
    model.add(layers.Dense(10, activation='softmax'))

    print(f"\nEntrada   : {n_features} variables (pixeles 28x28)")
    print("Oculta 1  : 128 neuronas | activacion: relu    (evita vanishing gradient)")
    print("Oculta 2  :  64 neuronas | activacion: relu    (extraccion de caracteristicas)")
    print("Oculta 3  :  32 neuronas | activacion: tanh    (activacion distinta - prof.)")
    print("Oculta 4  :  16 neuronas | activacion: sigmoid (activacion distinta - prof.)")
    print("Salida    :  10 neuronas | activacion: softmax (digitos 0-9, suma prob.= 1)")

    return model
