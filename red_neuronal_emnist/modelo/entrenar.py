# pyrefly: ignore [missing-import]
import numpy as np
from config import EPOCAS, BATCH_SIZE

def entrenar_modelo(model, X_train, y_train, epocas=None, batch_size=None, callbacks=None):
    epocas     = epocas     or EPOCAS
    batch_size = batch_size or BATCH_SIZE

    print("\n" + "=" * 50)
    print(f"  ENTRENAMIENTO DEL MODELO ({epocas} EPOCAS)")
    print("=" * 50)
    print(f"\nmodel.fit() -> {epocas} epocas | batch_size={batch_size}\n")

    historial = model.fit(
        X_train,
        y_train,
        epochs=epocas,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("\n--- Resumen de exactitud por epoca ---")
    accuracies = historial.history['accuracy']
    for i, acc in enumerate(accuracies):
        barra = int(acc * 30)
        print(f"  Epoca {i+1:2d}: [{'#' * barra}{' ' * (30 - barra)}] {acc * 100:.2f}%")

    exactitud_final = accuracies[-1]
    print(f"\nExactitud final de entrenamiento: {exactitud_final:.4f} ({exactitud_final * 100:.2f}%)")

    return model, historial
