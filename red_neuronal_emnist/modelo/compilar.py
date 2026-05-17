def compilar_modelo(model):
    print("\n" + "=" * 50)
    print("  COMPILACION DEL MODELO")
    print("=" * 50)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("\nFuncion de perdida : sparse_categorical_crossentropy (entropia cruzada)")
    print("Optimizador        : adam (tasa de aprendizaje adaptativa)")
    print("Metrica            : accuracy (exactitud)")
    print("Error de testeo    : 1 - accuracy\n")

    model.summary()

    return model
