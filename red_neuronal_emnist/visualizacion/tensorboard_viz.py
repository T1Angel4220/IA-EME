from tensorflow.keras.callbacks import TensorBoard

LOG_DIR = r'C:\logs_emnist'

def obtener_callback_tensorboard():
    log_dir = LOG_DIR
    __import__('os').makedirs(log_dir, exist_ok=True)

    callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )

    print(f"\nTensorBoard -> logs en: {log_dir}")
    print("Para ver las graficas, abre OTRA terminal y ejecuta:")
    print(f"  py -3.11 -m tensorboard.main --logdir={log_dir}")
    print("Luego abre: http://localhost:6006\n")

    return callback
