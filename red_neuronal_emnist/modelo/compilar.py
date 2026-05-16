"""
compilar.py
-----------
Responsabilidad: Compilar el modelo y mostrar su resumen.

Conceptos del profesor aplicados aqui:
  - model.compile() : prepara el modelo para entrenamiento
  - loss            : entropia cruzada (funcion de perdida)
  - optimizer       : adam (tasa de aprendizaje adaptativa)
                      Adam da saltos grandes al inicio y pequeños cerca del minimo
                      Analogia del profesor: como bajar una montana con los ojos vendados
  - metrics         : accuracy (exactitud del clasificador)
  - model.summary() : muestra los parametros de cada capa

Librerias usadas: tensorflow, keras
"""


def compilar_modelo(model):
    """
    Compila el modelo con los parametros del profesor.

    Parametros:
        model: modelo keras.Sequential sin compilar

    Retorna:
        model: modelo compilado listo para entrenar
    """
    print("\n" + "=" * 50)
    print("  COMPILACION DEL MODELO")
    print("=" * 50)

    # Como explico el profesor:
    # - loss     : entropia cruzada (sparse porque las etiquetas son enteros 0-9)
    # - optimizer: adam (descenso del gradiente adaptativo)
    # - metrics  : accuracy -> error = 1 - accuracy
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("\nFuncion de perdida : sparse_categorical_crossentropy (entropia cruzada)")
    print("Optimizador        : adam (tasa de aprendizaje adaptativa)")
    print("Metrica            : accuracy (exactitud)")
    print("Error de testeo    : 1 - accuracy\n")

    # model.summary() - como mostro el profesor en clase
    model.summary()

    return model
