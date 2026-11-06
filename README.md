# 🧠 EMNIST Digits Classifier — Red Neuronal Artificial (ANN)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Implementación modular en Python de una **Red Neuronal Artificial (Multilayer Perceptron / MLP)** profunda para la clasificación supervisada de dígitos manuscritos (0 al 9) a partir del dataset **EMNIST Digits**.

Diseñado bajo principios de ingeniería de software limpia: separación de responsabilidades, evaluación matemática propia (matriz de confusión y métricas implementadas en NumPy puro sin librerías externas de caja negra), visualización en consola y telemetría de entrenamiento con **TensorBoard**.

---

## 📋 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Dataset: EMNIST Digits](#-dataset-emnist-digits)
3. [Arquitectura de la Red y Decisiones de Diseño](#-arquitectura-de-la-red-y-decisiones-de-diseño)
4. [Estrategia de Entrenamiento y Compilación](#-estrategia-de-entrenamiento-y-compilación)
5. [Evaluación y Matriz de Confusión en NumPy](#-evaluación-y-matriz-de-confusión-en-numpy)
6. [Visualización y Monitoreo](#-visualización-y-monitoreo)
7. [Estructura del Proyecto](#-estructura-del-proyecto)
8. [Instalación y Uso](#-instalación-y-uso)
9. [Buenas Prácticas Aplicadas](#-buenas-prácticas-aplicadas)
10. [Contacto y Autor](#-contacto-y-autor)

---

## 🎯 Descripción General

El objetivo de este proyecto es construir, entrenar y evaluar una red neuronal multicapa capaz de reconocer con alta precisión dígitos manuscritos del conjunto de datos **Extended MNIST (EMNIST)**, el cual presenta mayor variabilidad y desafío de clasificación que el clásico MNIST.

El proyecto destaca por:
- **Estructura Modular:** Desacoplamiento entre la ingesta de datos, definición de arquitectura, compilación, ciclo de entrenamiento, evaluación y visualizadores.
- **Transparencia Algorítmica:** Métricas de precisión global, precisión por clase y cálculo de matriz de confusión implementadas directamente con operaciones vectorizadas de `numpy`.
- **Telemetría en Vivo:** Integración nativa con callbacks de `TensorBoard` para registrar curvas de pérdida/exactitud e histogramas de pesos por época.
- **Portabilidad y Seguridad:** Manejo dinámico de rutas relativas y variables de entorno, evitando quemar rutas absolutas o secretos en el repositorio.

---

## 📊 Dataset: EMNIST Digits

El dataset utilizado corresponde al subconjunto **Digits** de **EMNIST (Extended MNIST)**:
- **Muestras:** 240,000 imágenes de entrenamiento y 40,000 imágenes de evaluación/test.
- **Dimensiones:** Imágenes en escala de grises de $28 \times 28$ píxeles, aplanadas en vectores de **784 características (features)**.
- **Clases:** 10 dígitos numéricos (etiquetas del 0 al 9).
- **Preprocesamiento:** Normalización min-max de intensidades de píxeles dividiendo entre $255.0$, escalando los valores del rango $[0, 255]$ a $[0.0, 1.0]$. Esto previene gradientes explosivos y acelera la convergencia de descenso de gradiente.

---

## 🏗 Arquitectura de la Red y Decisiones de Diseño

La topología está construida usando la API `Sequential` de Keras, organizada en un flujo feed-forward con las siguientes capas:

| Capa | Tipo | Neuronas | Activación | Justificación Técnica |
| :--- | :--- | :---: | :---: | :--- |
| **Entrada** | `Input` | 784 | — | 784 píxeles normalizados por imagen ($28 \times 28$). |
| **Oculta 1** | `Dense` | 128 | `ReLU` | Extrae patrones y combinaciones de píxeles primarias; ReLU evita el desvanecimiento del gradiente (*vanishing gradient*). |
| **Oculta 2** | `Dense` | 64 | `ReLU` | Sintetiza trazos y curvas geométricas de mayor nivel de abstracción. |
| **Oculta 3** | `Dense` | 32 | `Tanh` | Acota la representación en el rango simétrico $[-1, 1]$, centrando las salidas en cero. |
| **Oculta 4** | `Dense` | 16 | `Sigmoid` | Transición no lineal suave que comprime las características antes del mapeo final. |
| **Salida** | `Dense` | 10 | `Softmax` | Distribución de probabilidad discreta normalizada $\sum p_i = 1$ para cada uno de los 10 dígitos. |

### Diagrama de Flujo

```
   [ Entrada: 784 px ]
           │
           ▼
 [ Densa: 128 | ReLU ]     ──▶ Extracción de patrones locales
           │
           ▼
 [ Densa:  64 | ReLU ]     ──▶ Combinación de trazos
           │
           ▼
 [ Densa:  32 | Tanh ]     ──▶ Normalización centrada en cero
           │
           ▼
 [ Densa:  16 | Sigmoid ]  ──▶ Compresión de características
           │
           ▼
 [ Salida: 10 | Softmax ]  ──▶ Vector de probabilidades (0-9)
```

---

## ⚙ Estrategia de Entrenamiento y Compilación

- **Función de Pérdida:** `sparse_categorical_crossentropy`  
  Calcula la entropía cruzada directamente sobre etiquetas enteras ($y \in \{0, \dots, 9\}$), evitando la sobrecarga de memoria de generar vectores *one-hot encoding*.
- **Optimizador:** `Adam` (Adaptive Moment Estimation)  
  Ajusta dinámicamente la tasa de aprendizaje mediante momentos de primer y segundo orden.
- **Métricas Monitoreadas:** Exactitud (`accuracy`) y Tasa de Error ($1 - \text{accuracy}$).
- **Hiperparámetros de Entrenamiento:**
  - **Épocas:** 30
  - **Batch Size:** 64
  - **Criterio de Validación:** Error en testeo $\le 10\%$ (Exactitud $\ge 90\%$).

---

## 📈 Evaluación y Matriz de Confusión en NumPy

En lugar de delegar el análisis a dependencias complejas, el módulo `evaluacion/testeo.py` implementa el diagnóstico con álgebra lineal y `NumPy`:

1. **Matriz de Confusión Cuadrada ($10 \times 10$):**
   - Filas = Dígito real.
   - Columnas = Dígito predicho.
   - Diagonal principal = Aciertos / Verdaderos positivos.
2. **Métricas por Clase:** Porcentaje de exactitud y precisión individual por dígito manuscrito.
3. **Detección de Confusiones Críticas:** Identificación y ordenamiento automático del **Top 5 de pares de mayor confusión** (e.g., similitud visual entre 4 y 9, o 3 y 5).

---

## 🖥 Visualización y Monitoreo

El proyecto incluye dos herramientas de inspección:

1. **Diagrama ASCII en Consola (`visualizacion/ver_red.py`):**  
   Imprime la estructura de capas, cantidad de neuronas, funciones de activación y parámetros entrenables en tiempo real sin dependencias externas.
2. **Dashboard de Telemetría con TensorBoard (`visualizacion/tensorboard_viz.py`):**  
   Registra el grafo computacional, evolución de exactitud y pérdida por época, e histogramas de distribución de pesos de las capas densas.

---

## 📂 Estructura del Proyecto

```
IA-EME/
│
├── .gitignore                      # Reglas de exclusion de caches, datasets y logs
├── .env.example                    # Plantilla de configuracion de entorno
├── LICENSE                         # Licencia de codigo abierto MIT
├── README.md                       # Documentacion tecnica del repositorio
├── requirements.txt                # Dependencias del proyecto
│
└── red_neuronal_emnist/            # Paquete principal
    ├── __init__.py
    ├── config.py                   # Configuracion de hiperparametros y capas
    ├── main.py                     # Script principal de ejecucion
    │
    ├── datos/
    │   ├── __init__.py
    │   └── cargar_datos.py         # Carga y normalizacion de los datasets CSV
    │
    ├── modelo/
    │   ├── __init__.py
    │   ├── arquitectura.py         # Definicion de capas y activaciones de la red
    │   ├── compilar.py             # Compilacion (loss, optimizer, metrics)
    │   └── entrenar.py             # Ciclo de entrenamiento y resumen por epoca
    │
    ├── evaluacion/
    │   ├── __init__.py
    │   └── testeo.py               # Inferencia, calculo de error y matriz de confusion
    │
    └── visualizacion/
        ├── __init__.py
        ├── tensorboard_viz.py      # Configuracion de callbacks de TensorBoard
        └── ver_red.py              # Visualizador ASCII y exportacion grafica
```

---

## 🚀 Instalación y Uso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/T1Angel4220/IA-EME.git
cd IA-EME
```

### 2. Crear y Activar Entorno Virtual

Se recomienda **Python 3.10 o 3.11** para compatibilidad con TensorFlow:

```bash
# En Windows (PowerShell)
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# En Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar el Dataset EMNIST

Los archivos CSV pueden descargarse desde [Kaggle - EMNIST Dataset](https://www.kaggle.com/datasets/crawford/emnist) o el repositorio oficial del NIST.
Coloca los archivos en la carpeta `data/`:
- `data/emnist-digits-train.csv`
- `data/emnist-digits-test.csv`

*(Opcional: puedes configurar rutas personalizadas mediante variables de entorno en un archivo `.env` tomando como base `.env.example`).*

### 5. Ejecutar Entrenamiento y Evaluación

```bash
python red_neuronal_emnist/main.py
```

### 6. Visualizar Métricas en TensorBoard

En otra terminal con el entorno virtual activo:

```bash
tensorboard --logdir=logs/fit
```

Abre tu navegador en `http://localhost:6006` para explorar las gráficas interactivas.

---

## 🛡 Buenas Prácticas Aplicadas

- **Zero Secrets / Zero Bloat:** Cero credenciales, rutas absolutas o secretos en el historial de Git. Datasets pesados (`*.csv`) y carpetas temporales (`logs/`, `__pycache__/`) ignorados preventivamente.
- **Código Modular y Reutilizable:** Arquitectura desacoplada en submódulos especializados.
- **Resiliencia de Rutas:** Resolución dinámica de directorios (`Path.resolve()`) compatible tanto con Windows como con Unix/macOS.
- **Documentación Completa:** Docstrings descriptivos y configuración tipada.

---

## 👤 Autor

Desarrollado por **Ángel Ayuquina**  
- **GitHub:** [@T1Angel4220](https://github.com/T1Angel4220)  
- **Enfoque:** Inteligencia Artificial, Deep Learning y Arquitectura de Software.

---

## 📄 Licencia

Este proyecto se distribuye bajo los términos de la licencia [MIT](LICENSE).
