"""
main.py
-------
Punto de entrada principal del proyecto.
Orquesta el flujo completo de la red neuronal tal como explico el profesor:

  1. Cargar datos de EMNIST Digits
  2. Normalizar pixeles
  3. Construir arquitectura  (keras.Sequential + model.add)
  4. Compilar modelo         (entropia cruzada + adam + accuracy)
  5. Entrenar modelo         (model.fit - 30 epocas)
  6. Predecir y evaluar      (model.predict solo con testeo)

Librerias permitidas: numpy, tensorflow, keras
"""

# ============================================================
# IMPORTS - Solo las librerias permitidas por el docente
# ============================================================

# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {keras.__version__}")
print(f"NumPy      : {np.__version__}")

# ============================================================
# IMPORTAR MODULOS DEL PROYECTO
# ============================================================
from datos.cargar_datos            import cargar_datos, normalizar
from modelo.arquitectura           import construir_modelo
from modelo.compilar               import compilar_modelo
from modelo.entrenar               import entrenar_modelo
from evaluacion.testeo             import evaluar_modelo
from visualizacion.tensorboard_viz import obtener_callback_tensorboard
from visualizacion.ver_red         import imprimir_arquitectura_ascii, guardar_imagen_red


# ============================================================
# FLUJO PRINCIPAL
# ============================================================
if __name__ == '__main__':

    # --- PASO 1: Cargar datos ---
    X_train, y_train, X_test, y_test, n_features = cargar_datos()

    # --- PASO 2: Normalizar ---
    X_train, X_test = normalizar(X_train, X_test)

    # --- PASO 3: Construir arquitectura ---
    model = construir_modelo(n_features)

    # --- PASO 4: Compilar ---
    model = compilar_modelo(model)

    # --- PASO 5: Configurar visualizacion con TensorBoard ---
    # TensorBoard es parte de TensorFlow -> libreria permitida
    # Guarda graficas de Loss y Accuracy por epoca
    tb_callback = obtener_callback_tensorboard()

    # --- PASO 6: Entrenar (30 epocas como el profesor) ---
    model, historial = entrenar_modelo(
        model, X_train, y_train,
        epocas=30,
        callbacks=[tb_callback]
    )

    # --- PASO 7: Evaluar con datos de testeo ---
    # Las y_test se guardaron aparte (como dijo el profesor)
    y_predichas, exactitud, error = evaluar_modelo(model, X_test, y_test)

    # --- PASO 8: Visualizar arquitectura de la red ---
    imprimir_arquitectura_ascii(model)       # diagrama en consola (siempre funciona)
    guardar_imagen_red(model)                # imagen PNG en C:\logs_emnist\ (requiere graphviz)
