# Red Neuronal ANN - EMNIST Digits

## Descripcion
Red neuronal artificial construida con **Keras + TensorFlow** para clasificar
digitos escritos a mano (0-9) del dataset EMNIST Digits.

Implementacion basada en la clase del Ing., siguiendo exactamente el flujo
y los conceptos explicados.

---

## Estructura del proyecto

```
red_neuronal_emnist/
│
├── main.py                    <- PUNTO DE ENTRADA: ejecutar este archivo
│
├── datos/
│   └── cargar_datos.py        <- Carga los CSV de EMNIST y normaliza pixeles
│
├── modelo/
│   ├── arquitectura.py        <- keras.Sequential() + model.add() (capas y activaciones)
│   ├── compilar.py            <- model.compile() (loss, optimizer, metrics) + summary()
│   └── entrenar.py            <- model.fit() con 30 epocas
│
└── evaluacion/
    └── testeo.py              <- model.predict() con X_test + error de testeo
```

---

## Flujo del programa (como explico el profesor)

| Paso | Archivo | Que hace |
|------|---------|---------|
| 1 | `datos/cargar_datos.py` | Lee `emnist-digits-train.csv` y `emnist-digits-test.csv` |
| 2 | `datos/cargar_datos.py` | Normaliza pixeles: [0,255] -> [0.0, 1.0] |
| 3 | `modelo/arquitectura.py` | Construye el modelo con `keras.Sequential()` |
| 4 | `modelo/compilar.py` | Compila: entropia cruzada + adam + accuracy |
| 5 | `modelo/entrenar.py` | Entrena con `model.fit(epochs=30)` |
| 6 | `evaluacion/testeo.py` | Predice con `model.predict(X_test)` y calcula error |

---

## Arquitectura de la red

```
Input (784)  ->  Dense(25, relu)  ->  Dense(15, relu)  ->  Dense(5, relu)  ->  Dense(10, softmax)
```

- **784 entradas**: cada imagen 28x28 pixeles aplanada
- **relu** en capas ocultas: evita el vanishing gradient (pesos muy pequeños)
- **softmax** en salida: probabilidad para cada digito 0-9
- Solo la **primera capa** lleva `input_shape` (como dijo el profesor)

---

## Conceptos del profesor implementados

| Concepto | Implementado en |
|----------|----------------|
| `keras.Sequential()` | `arquitectura.py` |
| `model.add(layers.Dense(...))` | `arquitectura.py` |
| Solo 1ra capa con `input_shape` | `arquitectura.py` |
| `relu` evita vanishing gradient | `arquitectura.py` |
| Cada capa puede tener distinta activacion | `arquitectura.py` |
| `model.compile(loss, optimizer, metrics)` | `compilar.py` |
| Entropia cruzada como funcion de perdida | `compilar.py` |
| Adam como optimizador (tasa de aprendizaje) | `compilar.py` |
| `model.summary()` para ver parametros | `compilar.py` |
| `model.fit(epochs=30)` | `entrenar.py` |
| Exactitud por epoca | `entrenar.py` |
| `model.predict(X_test)` solo con testeo | `testeo.py` |
| `y_test` guardadas aparte | `testeo.py` |
| Error = 1 - exactitud | `testeo.py` |
| Error no debe ser > 0.10 | `testeo.py` |

---

## Librerias usadas
- `numpy` : manejo de arrays y calculo del error
- `tensorflow` : framework de deep learning
- `keras` : API de alto nivel para construir la red

---

## Como ejecutar

```bash
cd red_neuronal_emnist
python main.py
```

---

## Dataset
- **Nombre**: EMNIST Digits
- **Ruta**: `C:\Users\ayuqu\Downloads\DB\`
- **Archivos**: `emnist-digits-train.csv` y `emnist-digits-test.csv`
- **n_features**: 784 (imagenes 28x28 pixeles)
- **Clases**: 10 digitos (0 al 9)
