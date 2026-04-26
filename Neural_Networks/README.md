# 🧠 Neural Networks

> Clasificación con redes neuronales artificiales usando scikit-learn — desde arquitecturas básicas hasta búsqueda de hiperparámetros.

---

## 📋 Descripción

Esta sección introduce las **redes neuronales artificiales (MLP)** aplicadas a problemas de clasificación supervisada. Se trabaja con datos simulados mediante clustering para visualizar de forma intuitiva cómo los modelos aprenden a separar clases en espacios de distintas complejidades.

---

## 📂 Contenido

| Archivo | Descripción |
|---|---|
| `Neuronas.ipynb` | Fundamentos de redes neuronales y MLPClassifier |
| `Tiny_nn.ipynb` | Implementación minimalista de una red neuronal |
| `utils.py` | Funciones auxiliares de visualización y preprocesamiento |

---

## 🏗️ Arquitectura MLP

Las redes utilizadas se basan en capas completamente conectadas (*fully connected*), donde cada capa transforma la información hasta generar una predicción final:

```
Entrada → [Capa Oculta 1] → [Capa Oculta 2] → ... → Salida
```

Se experimenta con distintas configuraciones variando:
- Número de **capas ocultas**
- Número de **neuronas por capa**
- **Función de activación**

---

## ⚙️ Parámetros Clave

| Parámetro | Descripción |
|---|---|
| `learning_rate` | Velocidad de ajuste de pesos en cada iteración |
| `epochs` | Número de pasadas completas sobre el conjunto de entrenamiento |
| `hidden_layer_sizes` | Tamaño y profundidad de la red |
| `activation` | Función de activación por capa (relu, tanh, logistic) |

---

## 📉 Función de Pérdida

Se emplea **entropía cruzada** (*cross-entropy*), estándar para clasificación multiclase:

$$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$

Mide la diferencia entre las probabilidades predichas y las etiquetas reales, guiando el ajuste de pesos mediante *backpropagation*.

---

## 🔍 Optimización de Hiperparámetros

| Técnica | Descripción |
|---|---|
| **Grid Search** | Búsqueda exhaustiva sobre una grilla de valores definida |
| **Randomized Search** | Muestreo aleatorio del espacio de hiperparámetros |
| **K-Fold CV** | Validación cruzada para estimación robusta del rendimiento |

---

## 🛠️ Herramientas

| Librería | Uso |
|---|---|
| `scikit-learn` | MLPClassifier, GridSearchCV, métricas de evaluación |
| `Matplotlib` | Visualización de fronteras de decisión y curvas de pérdida |
| `Seaborn` | Gráficos de distribución y análisis de resultados |

---

## 🏁 Objetivo

Establecer una base sólida en redes neuronales para clasificación, comprendiendo el impacto de la arquitectura, los hiperparámetros y las estrategias de validación en el rendimiento del modelo.
