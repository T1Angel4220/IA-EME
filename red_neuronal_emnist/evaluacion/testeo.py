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

    # Generar y mostrar la Matriz de Confusion
    imprimir_matriz_confusion(y_verdaderas, y_predichas)

    return y_predichas, exactitud_testeo, error_testeo


def imprimir_matriz_confusion(y_verdaderas, y_predichas, n_clases=10):
    """
    Calcula e imprime la Matriz de Confusion usando solo NumPy.

    La Matriz de Confusion muestra:
      - Filas    : clases REALES (verdaderas)
      - Columnas : clases PREDICHAS por el modelo
      - Diagonal : predicciones correctas
      - Fuera de diagonal: errores (que numero confundio con cual)

    Parametros:
        y_verdaderas: etiquetas reales del testeo
        y_predichas : etiquetas predichas por el modelo
        n_clases    : numero de clases (10 digitos)
    """
    # Calcular la matriz con numpy (sin sklearn)
    # Cada celda [i][j] = cuantas veces el real i fue predicho como j
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_verdaderas, y_predichas):
        matriz[real][pred] += 1

    print("\n" + "=" * 60)
    print("   MATRIZ DE CONFUSION (calculada con NumPy)")
    print("=" * 60)
    print("   Filas = Clase REAL | Columnas = Clase PREDICHA\n")

    # Encabezado
    encabezado = "Real\\Pred |" + "".join(f"  {j:3d}" for j in range(n_clases))
    print(encabezado)
    print("-" * len(encabezado))

    # Filas de la matriz
    for i in range(n_clases):
        fila = f"  Digito {i} |"
        for j in range(n_clases):
            if i == j:
                fila += f" [{matriz[i][j]:3d}]"   # diagonal: predicciones correctas
            else:
                fila += f"  {matriz[i][j]:3d} "
        # Precision por clase
        total_real    = np.sum(matriz[i, :])
        correctos     = matriz[i][i]
        precision_cls = (correctos / total_real * 100) if total_real > 0 else 0
        fila += f"  | {precision_cls:.1f}%"
        print(fila)

    print("-" * len(encabezado))

    # Exactitud global
    exactitud_global = np.trace(matriz) / np.sum(matriz)
    print(f"\n  Exactitud global     : {exactitud_global * 100:.2f}%")
    print(f"  Total errores        : {np.sum(matriz) - np.trace(matriz)}")
    print(f"  Total correctos      : {np.trace(matriz)}")

    # Confusiones mas frecuentes (fuera de la diagonal)
    print("\n  Top 5 confusiones mas frecuentes:")
    errores = []
    for i in range(n_clases):
        for j in range(n_clases):
            if i != j and matriz[i][j] > 0:
                errores.append((matriz[i][j], i, j))
    errores.sort(reverse=True)
    for cantidad, real, pred in errores[:5]:
        print(f"    Real={real} predicho como {pred}: {cantidad} veces")

    print("=" * 60)
