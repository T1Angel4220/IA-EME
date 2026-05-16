"""
testeo.py
---------
Responsabilidad: Evaluar el modelo entrenado SOLO con datos de testeo.

Conceptos del profesor aplicados aqui:
  - model.predict()  : predice SOLO con X_test (datos de testeo guardados aparte)
  - y_predichas      : clases predichas por el modelo
  - y_verdaderas     : etiquetas reales del conjunto de testeo
  - error_testeo     : 1 - exactitud (no debe ser mayor a 0.10)
  - El profesor dijo: si el error es bajo, la arquitectura es adecuada

Librerias usadas: numpy
"""

# pyrefly: ignore [missing-import]
import numpy as np


def evaluar_modelo(model, X_test, y_test):
    """
    Realiza predicciones con el modelo entrenado y calcula el error de testeo.
    
    Como dijo el profesor:
      - Las y_test se guardan aparte desde el inicio
      - model.predict() solo recibe X_test
      - Se contrastan y_predichas vs y_verdaderas
      - El error = 1 - exactitud no debe ser muy grande

    Parametros:
        model  : modelo keras entrenado
        X_test : pixeles de testeo normalizados
        y_test : etiquetas verdaderas de testeo (guardadas aparte)

    Retorna:
        y_predichas     : clases predichas
        exactitud_testeo: exactitud en datos de testeo
        error_testeo    : error = 1 - exactitud
    """
    print("\n" + "=" * 50)
    print("  EVALUACION CON DATOS DE TESTEO")
    print("=" * 50)

    # model.predict() SOLO con las X de testeo
    # (las y_test las guardamos aparte, como dijo el profesor)
    print("\nEjecutando model.predict(X_test)...")
    y_pred_probabilidades = model.predict(X_test)

    # La neurona con mayor probabilidad = clase predicha
    y_predichas = np.argmax(y_pred_probabilidades, axis=1)

    # y verdaderas = las que guardamos aparte (y_test)
    y_verdaderas = y_test

    # -----------------------------------------------
    # Contrastar predicciones vs etiquetas verdaderas
    # como explico el profesor
    # -----------------------------------------------
    correctas        = np.sum(y_predichas == y_verdaderas)
    total            = len(y_verdaderas)
    exactitud_testeo = correctas / total
    error_testeo     = 1 - exactitud_testeo  # error = 1 - exactitud

    print(f"\nPrimeras 10 predicciones : {y_predichas[:10]}")
    print(f"Primeras 10 reales       : {y_verdaderas[:10]}")

    print("\n" + "=" * 50)
    print("  RESULTADO FINAL EN DATOS DE TESTEO")
    print("=" * 50)
    print(f"  Total muestras testeo  : {total}")
    print(f"  Predicciones correctas : {correctas}")
    print(f"  Exactitud de testeo    : {exactitud_testeo:.4f}  ({exactitud_testeo * 100:.2f}%)")
    print(f"  Error de testeo        : {error_testeo:.4f}")

    # El profesor dijo: el error no puede ser muy grande (>= 0.90 de exactitud)
    print("\n  Verificacion del modelo:")
    if error_testeo <= 0.10:
        print("  [OK] Arquitectura ADECUADA  (error <= 0.10, exactitud >= 90%)")
    else:
        print("  [--] Arquitectura necesita ajustes (error > 0.10)")

    print("=" * 50)

    return y_predichas, exactitud_testeo, error_testeo
