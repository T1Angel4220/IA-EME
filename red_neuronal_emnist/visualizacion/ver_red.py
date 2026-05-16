"""
ver_red.py
----------
Responsabilidad: Visualizar la arquitectura de la red neuronal.

Opciones:
  1. keras.utils.plot_model -> genera imagen PNG de la red (requiere graphviz)
  2. Diagrama ASCII en consola -> siempre funciona, sin dependencias extra

Libreria usada: keras (parte de TensorFlow)
"""

import os
import sys

# Agregar Graphviz al PATH para que pydot pueda encontrar el ejecutable 'dot'
GRAPHVIZ_BIN = r'C:\Program Files (x86)\Graphviz\bin'
if os.path.exists(GRAPHVIZ_BIN) and GRAPHVIZ_BIN not in os.environ.get('PATH', ''):
    os.environ['PATH'] += os.pathsep + GRAPHVIZ_BIN

from tensorflow import keras


def imprimir_arquitectura_ascii(model):
    """
    Muestra la arquitectura de la red como diagrama en la consola.
    No requiere librerias adicionales.
    """
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
    """
    Genera una imagen PNG con el grafo de la red neuronal.
    Usa keras.utils.plot_model (requiere pydot + graphviz instalados).

    Si no estan instalados, muestra instrucciones para instalarlos.

    Parametros:
        model      : modelo keras entrenado
        ruta_salida: donde guardar la imagen PNG
    """
    try:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

        keras.utils.plot_model(
            model,
            to_file=ruta_salida,
            show_shapes=True,          # muestra dimensiones de cada capa
            show_layer_names=True,     # muestra nombres de capas
            show_layer_activations=True,  # muestra funcion de activacion
            rankdir='TB',              # de arriba a abajo (Top-Bottom)
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
