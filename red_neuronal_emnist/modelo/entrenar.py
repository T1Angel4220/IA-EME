"""
entrenar.py
-----------
Responsabilidad: Ejecutar el entrenamiento del modelo compilado.

Conceptos del profesor aplicados aqui:
  - model.fit()  : ejecuta el modelo compilado
  - epochs=30    : 30 repeticiones/epocas (como dijo el profesor)
  - En cada epoca se muestra la exactitud (accuracy)
  - Al inicio el modelo demora mas, luego converge

Librerias usadas: numpy, tensorflow, keras
"""

# pyrefly: ignore [missing-import]
import numpy as np


def entrenar_modelo(model, X_train, y_train, epocas=30, batch_size=64, callbacks=None):
    """
    Entrena el modelo con los datos de entrenamiento.

    Parametros:
        model      : modelo keras compilado
        X_train    : pixeles de entrenamiento normalizados
        y_train    : etiquetas de entrenamiento
        epocas     : numero de repeticiones (default 30, como el profesor)
        batch_size : tamano de lote para el gradiente
        callbacks  : lista de callbacks (ej. TensorBoard para visualizacion)

    Retorna:
        model    : modelo entrenado (ajustado)
        historial: objeto con accuracy y loss por epoca
    """
    print("\n" + "=" * 50)
    print(f"  ENTRENAMIENTO DEL MODELO ({epocas} EPOCAS)")
    print("=" * 50)
    print(f"\nmodel.fit() -> {epocas} epocas | batch_size={batch_size}\n")

    # model.fit - ejecuta el modelo compilado, 30 epocas como el profesor
    # callbacks: permite conectar TensorBoard para visualizar graficamente
    historial = model.fit(
        X_train,
        y_train,
        epochs=epocas,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("\n--- Resumen de exactitud por epoca ---")
    for epoca, acc in enumerate(historial.history['accuracy'], 1):
        barra = '#' * int(acc * 30)
        print(f"  Epoca {epoca:2d}: [{barra:<30}] {acc * 100:.2f}%")

    acc_final = historial.history['accuracy'][-1]
    print(f"\nExactitud final de entrenamiento: {acc_final:.4f} ({acc_final * 100:.2f}%)")

    return model, historial
