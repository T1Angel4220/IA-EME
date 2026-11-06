import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
from tensorflow import keras

# Deteccion dinamica de binarios de Graphviz en Windows
graphviz_paths = [
    r'C:\Program Files\Graphviz\bin',
    r'C:\Program Files (x86)\Graphviz\bin',
]
current_path = os.environ.get('PATH', '')
for gpath in graphviz_paths:
    if os.path.exists(gpath) and gpath not in current_path:
        os.environ['PATH'] += os.pathsep + gpath

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

def guardar_imagen_red(model, ruta_salida=None):
    if ruta_salida is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        ruta_salida = str(base_dir / "logs" / "arquitectura_red.png")

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
        print("\n    Para activar esta funcion opcional instala:")
        print("      pip install pydot")
        print("      Luego descarga Graphviz de: https://graphviz.org/download/")
        print("      e instala en el sistema (agregando al PATH).")
        print("\n    Nota: El diagrama ASCII en consola se genera de forma nativa sin dependencias externas.")
