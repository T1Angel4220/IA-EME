# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf

from tensorflow import keras

print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {keras.__version__}")
print(f"NumPy      : {np.__version__}")

from datos.cargar_datos            import cargar_datos, normalizar
from modelo.arquitectura           import construir_modelo
from modelo.compilar               import compilar_modelo
from modelo.entrenar               import entrenar_modelo
from evaluacion.testeo             import evaluar_modelo
from visualizacion.tensorboard_viz import obtener_callback_tensorboard
from visualizacion.ver_red         import imprimir_arquitectura_ascii, guardar_imagen_red

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, n_features = cargar_datos()
    X_train, X_test = normalizar(X_train, X_test)
    
    model = construir_modelo(n_features)
    model = compilar_modelo(model)
    
    tb_callback = obtener_callback_tensorboard()
    
    model, historial = entrenar_modelo(
        model, X_train, y_train,
        epocas=30,
        callbacks=[tb_callback]
    )
    
    y_predichas, exactitud, error = evaluar_modelo(model, X_test, y_test)
    
    imprimir_arquitectura_ascii(model)
    guardar_imagen_red(model)
