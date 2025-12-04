# Deep Q-Network (DQN) – CartPole-v1

Este proyecto implementa un **agente de Reinforcement Learning** usando el algoritmo **Deep Q-Network (DQN)** para resolver el entorno **CartPole-v1** de Gymnasium.  
El agente aprende a equilibrar un poste sobre un carrito mediante *experiencias almacenadas*, *función Q aproximada con redes neuronales*, y una *red objetivo* para estabilizar el aprendizaje.

---

## 🚀 Características del proyecto

- Implementación completa de **DQN desde cero** con PyTorch.
- Uso de **Replay Memory** para romper correlación entre muestras.
- Política **ε-greedy** con decaimiento exponencial.
- Actualización suave (**Soft Update**) de la red objetivo mediante `TAU`.
- Entrenamiento automático con soporte para:
  - **CUDA (GPU NVIDIA)**
  - **MPS (GPU Apple Silicon)**
  - **CPU**
- Gráficas en tiempo real del progreso del agente.
- Código totalmente documentado paso a paso.

---

## 📂 Estructura general del algoritmo aprendido

### 1. **Transiciones (Transition)**
Cada experiencia del agente guarda:
- estado
- acción
- siguiente estado
- recompensa

Se almacena en una estructura `namedtuple`, ideal por su velocidad y simplicidad.

---

### 2. **ReplayMemory**
Gestión de memoria FIFO con capacidad limitada.  
Permite extraer batches aleatorios para entrenar la red.

Esto:
- evita que el agente aprenda solo de experiencias recientes,
- mejora la estabilidad del entrenamiento.

---

### 3. **Red neuronal (DQN)**
Tres capas lineales con 128 neuronas intermedias:


Usa activación ReLU y salida sin activación para representar los valores Q(s, a).

---

### 4. **Política ε-greedy**
El agente:
- explora (acción aleatoria) con probabilidad ε,
- explota (mejor acción) con prob. 1−ε,

donde ε decae exponencialmente desde 0.9 hasta 0.01.

---

### 5. **Cálculo de la función Q y Backpropagation**
Durante el entrenamiento se calcula:

\[
Q_{\text{target}} = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
\]

La pérdida se calcula con **SmoothL1Loss (Huber Loss)**.

---

### 6. **Soft Update de la red objetivo**
La red objetivo se actualiza lentamente:

\[
\theta_{target} \leftarrow \tau \theta_{policy} + (1-\tau)\theta_{target}
\]

Esto evita oscilaciones en el entrenamiento.

---

### 7. **Entrenamiento**
Se ejecutan entre:
- **600 episodios en GPU**,  
- **50 episodios en CPU**,  

según disponibilidad del hardware.

---

## 📊 Gráficas del rendimiento

El script dibuja la duración de cada episodio y una media móvil de 100 episodios.  
Se utiliza `matplotlib` en modo interactivo.

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
