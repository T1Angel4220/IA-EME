"""
config.py
---------
Archivo de configuracion central del proyecto.

AQUI puedes modificar todos los parametros del modelo sin tocar
ningun otro archivo. Solo cambia los valores de este archivo.

Parametros configurables:
  - EPOCAS           : cuantas veces entrena con todos los datos
  - BATCH_SIZE       : cuantos datos procesa por paso
  - CAPAS_OCULTAS    : lista de capas (neuronas, activacion)

Nota sobre vanishing gradient:
  - relu    : NO tiene vanishing gradient (derivada constante = 1 si x>0)
  - tanh    : PUEDE tenerlo, pero solo con MUCHAS capas (>50)
              Con 4 capas como las nuestras, NO es problema
  - sigmoid : similar a tanh, aceptable en pocas capas
  - softmax : SOLO para la capa de salida multiclase

Librerias usadas: ninguna (solo Python puro)
"""

# ============================================================
#   CONFIGURACION DE ENTRENAMIENTO
#   Cambia estos valores para ajustar el entrenamiento
# ============================================================

EPOCAS      = 30    # numero de epocas (el profesor pidio 30)
BATCH_SIZE  = 64    # tamanio del lote por paso de gradiente

# ============================================================
#   ARQUITECTURA: CAPAS OCULTAS
#   Cada elemento es una tupla: (neuronas, 'activacion')
#
#   Para AGREGAR una capa: agrega una linea mas a la lista
#   Para QUITAR una capa: comenta o borra la linea
#   Para CAMBIAR activacion: cambia 'relu' por 'tanh' o 'sigmoid'
#
#   Activaciones disponibles: 'relu', 'tanh', 'sigmoid'
#   NO uses 'softmax' aqui (solo va en la capa de salida)
#
#   NOTA sobre vanishing gradient:
#   Con pocas capas (< 10) tanh y sigmoid son aceptables.
#   Si usas 50+ capas, usa solo 'relu' para evitar el problema.
# ============================================================

CAPAS_OCULTAS = [
    (128, 'relu'),      # Capa 1: relu evita vanishing gradient
    (64,  'relu'),      # Capa 2: relu, extrae caracteristicas
    (32,  'tanh'),      # Capa 3: tanh, distinta activacion (como dijo el profe)
    (16,  'sigmoid'),   # Capa 4: sigmoid, otra activacion diferente
    # (8, 'relu'),      # <-- ejemplo: descomenta para agregar capa extra
]

# ============================================================
#   CONFIGURACION DE SALIDA (no cambiar normalmente)
# ============================================================

N_SALIDA    = 10    # 10 neuronas de salida (digitos 0-9)
ACTIVACION_SALIDA = 'softmax'   # siempre softmax para clasificacion multiclase
