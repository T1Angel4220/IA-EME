"""
tensorboard_viz.py
------------------
Responsabilidad: Configurar TensorBoard para visualizar el entrenamiento.

TensorBoard es la herramienta de visualizacion oficial de TensorFlow.
Permite ver graficamente:
  - Loss (funcion de perdida) por epoca
  - Accuracy (exactitud) por epoca
  - Grafo de la arquitectura de la red

Libreria usada: tensorflow.keras.callbacks (parte de TensorFlow)
"""

from tensorflow.keras.callbacks import TensorBoard
import os

# IMPORTANTE: TensorFlow no soporta rutas con caracteres especiales (tildes, acentos)
# Por eso usamos una ruta simple en C:\ sin acentos ni espacios
LOG_DIR = r'C:\logs_emnist'


def obtener_callback_tensorboard():
    """
    Crea el callback de TensorBoard para usarlo en model.fit().

    Usa ruta absoluta para que funcione sin importar desde donde
    se ejecute el script.

    Retorna:
        callback de TensorBoard listo para model.fit()
    """
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,    # histogramas de pesos por epoca
        write_graph=True,    # grafo de la arquitectura
        write_images=False,
        update_freq='epoch'  # actualiza metricas cada epoca
    )

    print(f"\nTensorBoard -> logs en: {log_dir}")
    print("Para ver las graficas, abre OTRA terminal y ejecuta:")
    print(f"  py -3.11 -m tensorboard.main --logdir={log_dir}")
    print("Luego abre: http://localhost:6006\n")

    return callback
