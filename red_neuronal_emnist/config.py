EPOCAS = 30
BATCH_SIZE = 64

CAPAS_OCULTAS = [
    (128, 'relu'),
    (64,  'relu'),
    (32,  'tanh'),
    (16,  'sigmoid'),
]

N_SALIDA = 10
ACTIVACION_SALIDA = 'softmax'
