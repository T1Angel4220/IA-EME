import os
# pyrefly: ignore [missing-import]
import numpy as np
from tensorflow import keras

GRAPHVIZ_BIN = r'C:\Program Files (x86)\Graphviz\bin'
_path = __import__('os').environ.get('PATH', '')
if GRAPHVIZ_BIN not in _path:
    __import__('os').environ['PATH'] += __import__('os').pathsep + GRAPHVIZ_BIN

def imprimir_arquitectura_ascii(model):
    print("\n" + "=" * 60)
    print("   DIAGRAMA DE LA RED NEURONAL")
    print("=" * 60)

    capas = model.layers
    ancho = 40

    for i, capa in enumerate(capas):
        nombre   = capa.name
        shape    = capa.output.shape
        params   = capa.count_params()
        config   = capa.get_config()
        activ    = config.get('activation', '-')
        neuronas = shape[-1]

        if i == 0:
            etiqueta = f"[ENTRADA]  {neuronas} neuronas | act: {activ} | params: {params}"
        elif i == len(capas) - 1:
            etiqueta = f"[SALIDA]   {neuronas} neuronas | act: {activ} | params: {params}"
        else:
            etiqueta = f"[OCULTA {i}] {neuronas} neuronas | act: {activ} | params: {params}"

        print(f"\n  {'_' * ancho}")
        print(f"  |{etiqueta.center(ancho)}|")
        print(f"  {'‾' * ancho}")

        if i < len(capas) - 1:
            print(f"  {'|':^{ancho + 2}}")
            print(f"  {'v':^{ancho + 2}}")

    total = model.count_params()
    print(f"\n  Total parametros entrenables: {total:,}")
    print("=" * 60)

def guardar_imagen_red(model, ruta_salida=r'C:\logs_emnist\arquitectura_red.png'):
    try:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

        keras.utils.plot_model(
            model,
            to_file=ruta_salida,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            rankdir='TB',
            dpi=150
        )
        print(f"\n[OK] Imagen de la red guardada en: {ruta_salida}")
        print("     Abrela con cualquier visor de imagenes.")

    except Exception as e:
        print(f"\n[!] No se pudo generar la imagen PNG.")
        print(f"    Razon: {e}")
        print("\n    Para activar esta funcion instala:")
        print("      py -3.11 -m pip install pydot")
        print("      Luego descarga Graphviz de: https://graphviz.org/download/")
        print("      e instala en Windows (agrega al PATH).")
        print("\n    El diagrama ASCII de arriba siempre funciona sin instalar nada.")
