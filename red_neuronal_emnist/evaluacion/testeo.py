# pyrefly: ignore [missing-import]
import numpy as np

def evaluar_modelo(model, X_test, y_test):
    print("\n" + "=" * 50)
    print("  EVALUACION CON DATOS DE TESTEO")
    print("=" * 50)

    print("\nEjecutando model.predict(X_test)...")
    y_pred_probabilidades = model.predict(X_test)

    y_predichas = np.argmax(y_pred_probabilidades, axis=1)
    y_verdaderas = y_test

    correctas        = np.sum(y_predichas == y_verdaderas)
    total            = len(y_verdaderas)
    exactitud_testeo = correctas / total
    error_testeo     = 1 - exactitud_testeo

    print(f"\nPrimeras 10 predicciones : {y_predichas[:10]}")
    print(f"Primeras 10 reales       : {y_verdaderas[:10]}")

    print("\n" + "=" * 50)
    print("  RESULTADO FINAL EN DATOS DE TESTEO")
    print("=" * 50)
    print(f"  Total muestras testeo  : {total}")
    print(f"  Predicciones correctas : {correctas}")
    print(f"  Exactitud de testeo    : {exactitud_testeo:.4f}  ({exactitud_testeo * 100:.2f}%)")
    print(f"  Error de testeo        : {error_testeo:.4f}")

    print("\n  Verificacion del modelo:")
    if error_testeo <= 0.10:
        print("  [OK] Arquitectura ADECUADA  (error <= 0.10, exactitud >= 90%)")
    else:
        print("  [--] Arquitectura necesita ajustes (error > 0.10)")

    print("=" * 50)

    imprimir_matriz_confusion(y_verdaderas, y_predichas)

    return y_predichas, exactitud_testeo, error_testeo


def imprimir_matriz_confusion(y_verdaderas, y_predichas, n_clases=10):
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_verdaderas, y_predichas):
        matriz[real][pred] += 1

    print("\n" + "=" * 60)
    print("   MATRIZ DE CONFUSION (calculada con NumPy)")
    print("=" * 60)
    print("   Filas = Clase REAL | Columnas = Clase PREDICHA\n")

    encabezado = "Real\\Pred |" + "".join(f"  {j:3d}" for j in range(n_clases))
    print(encabezado)
    print("-" * len(encabezado))

    for i in range(n_clases):
        fila = f"  Digito {i} |"
        for j in range(n_clases):
            if i == j:
                fila += f" [{matriz[i][j]:3d}]"
            else:
                fila += f"  {matriz[i][j]:3d} "
        
        total_real    = np.sum(matriz[i, :])
        correctos     = matriz[i][i]
        precision_cls = (correctos / total_real * 100) if total_real > 0 else 0
        fila += f"  | {precision_cls:.1f}%"
        print(fila)

    print("-" * len(encabezado))

    exactitud_global = np.trace(matriz) / np.sum(matriz)
    print(f"\n  Exactitud global     : {exactitud_global * 100:.2f}%")
    print(f"  Total errores        : {np.sum(matriz) - np.trace(matriz)}")
    print(f"  Total correctos      : {np.trace(matriz)}")

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
