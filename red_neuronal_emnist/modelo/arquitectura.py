from tensorflow import keras
from tensorflow.keras import layers
from config import CAPAS_OCULTAS, N_SALIDA, ACTIVACION_SALIDA

def construir_modelo(n_features):
    print("\n" + "=" * 60)
    print("  ARQUITECTURA DE LA RED NEURONAL")
    print("=" * 60)

    model = keras.Sequential()

    for i, (neuronas, activacion) in enumerate(CAPAS_OCULTAS):
        if i == 0:
            # pyrefly: ignore [unexpected-keyword]
            model.add(layers.Dense(neuronas, activation=activacion, input_shape=(n_features,)))
            print(f"Entrada   : {n_features} variables (pixeles 28x28)")
        else:
            model.add(layers.Dense(neuronas, activation=activacion))

        nota = "" if activacion == 'relu' else " (ok con pocas capas)"
        print(f"Oculta {i+1:2d}  : {neuronas:3d} neuronas | act: {activacion}{nota}")

    model.add(layers.Dense(N_SALIDA, activation=ACTIVACION_SALIDA))
    print(f"Salida    : {N_SALIDA:3d} neuronas | act: {ACTIVACION_SALIDA} (probabilidades 0-9)")

    return model
