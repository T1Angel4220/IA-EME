import os
from pathlib import Path
from tensorflow.keras.callbacks import TensorBoard

# Directorio base para almacenamiento de logs
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_DIR = BASE_DIR / "logs" / "fit"
LOG_DIR = os.getenv("TENSORBOARD_LOG_DIR", str(DEFAULT_LOG_DIR))

def obtener_callback_tensorboard(log_dir=None):
    directorio = log_dir or LOG_DIR
    os.makedirs(directorio, exist_ok=True)

    callback = TensorBoard(
        log_dir=directorio,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )

    print(f"\nTensorBoard -> logs en: {directorio}")
    print("Para ver las graficas, abre OTRA terminal y ejecuta:")
    print(f"  py -3.11 -m tensorboard.main --logdir=\"{directorio}\"")
    print("Luego abre: http://localhost:6006\n")

    return callback
