"""
entrenar.py
-----------
Responsabilidad: Ejecutar el entrenamiento del modelo con model.fit().

Conceptos del profesor aplicados aqui:
  - model.fit() : ejecuta el modelo sobre los datos de entrenamiento
  - epochs      : numero de repeticiones del entrenamiento (30 epocas)
  - batch_size  : tamano del lote por paso de gradiente
  - Adam ajusta la tasa de aprendizaje: saltos grandes al inicio, pequenos al final

Librerias usadas: numpy, tensorflow (keras dentro de model.fit)
"""

# pyrefly: ignore [missing-import]
import numpy as np
from config import EPOCAS, BATCH_SIZE


def entrenar_modelo(model, X_train, y_train, epocas=None, batch_size=None, callbacks=None):
    """
    Entrena el modelo con los datos de entrenamiento.
    Los valores de epocas y batch_size se toman de config.py si no se especifican.

    Parametros:
        model      : modelo keras compilado
        X_train    : pixeles de entrenamiento normalizados
        y_train    : etiquetas de entrenamiento
        epocas     : numero de repeticiones (default desde config.py)
        batch_size : tamano de lote para el gradiente
        callbacks  : lista de callbacks (ej. TensorBoard)

    Retorna:
        model    : modelo entrenado
        historial: objeto con accuracy y loss por epoca
    """
    epocas     = epocas     or EPOCAS
    batch_size = batch_size or BATCH_SIZE

    print("\n" + "=" * 50)
    print(f"  ENTRENAMIENTO DEL MODELO ({epocas} EPOCAS)")
    print("=" * 50)
    print(f"\nmodel.fit() -> {epocas} epocas | batch_size={batch_size}\n")

    # model.fit - ejecuta el modelo compilado
    # callbacks: permite conectar TensorBoard para visualizar graficamente
    historial = model.fit(
        X_train,
        y_train,
        epochs=epocas,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Mostrar resumen de exactitud por epoca
    print("\n--- Resumen de exactitud por epoca ---")
    accuracies = historial.history['accuracy']
    for i, acc in enumerate(accuracies):
        barra = int(acc * 30)
        print(f"  Epoca {i+1:2d}: [{'#' * barra}{' ' * (30 - barra)}] {acc * 100:.2f}%")

    exactitud_final = accuracies[-1]
    print(f"\nExactitud final de entrenamiento: {exactitud_final:.4f} ({exactitud_final * 100:.2f}%)")

    return model, historial
