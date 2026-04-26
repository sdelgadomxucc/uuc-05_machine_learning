# 🤖 Artificial Intelligence

> Técnicas de optimización evolutiva y modelos predictivos aplicados a problemas de regresión y búsqueda en espacios complejos.

---

## 📋 Descripción

Esta sección explora conceptos fundamentales de **Inteligencia Artificial**, con foco en algoritmos de optimización bioinspirados y modelos de aprendizaje supervisado para regresión.

---

## 📂 Contenido

| Archivo | Descripción |
|---|---|
| `genetic-algorithms-with-deap.ipynb` | Algoritmos genéticos con la librería DEAP |
| `Regression_Trees.ipynb` | Árboles de regresión para modelado no lineal |
| `Ridge_Lasso.ipynb` | Regularización Ridge y Lasso en regresión |
| `app.py` | Aplicación demostrativa |
| `RN.pdf / RN.tex` | Notas teóricas sobre Redes Neuronales |

---

## 🧬 Algoritmos Genéticos

Técnica de optimización inspirada en la evolución natural. Parten de una **población inicial aleatoria** de soluciones que evolucionan iterativamente mediante:

- **Selección** — se priorizan las soluciones con mejor *fitness*
- **Cruce (crossover)** — combinación de soluciones para generar descendencia
- **Mutación** — variaciones aleatorias para mantener diversidad genética

### Cuándo usarlos

| Escenario | Por qué son útiles |
|---|---|
| Sin gradiente disponible | No requieren derivadas |
| Espacio de búsqueda discreto o complejo | Exploran de forma combinatoria |
| Riesgo de óptimos locales | Mantienen diversidad poblacional |
| Alta dimensionalidad | Paralelizables y escalables |

---

## 🌳 Árboles de Regresión

Modelos que particionan el espacio de características en regiones, asignando un valor constante a cada una. Permiten capturar **relaciones no lineales** entre variables de forma interpretable.

**Ventajas:**
- Intuitivos y fácilmente visualizables
- No requieren normalización de datos
- Robustos ante valores atípicos

---

## 📉 Regresión Regularizada (Ridge & Lasso)

Extensiones de la regresión lineal que penalizan la magnitud de los coeficientes para reducir el sobreajuste:

| Método | Penalización | Efecto principal |
|---|---|---|
| **Ridge** | L2 — suma de cuadrados | Reduce coeficientes sin eliminarlos |
| **Lasso** | L1 — suma de valores absolutos | Selección automática de variables (sparse) |

---

## 🏁 Objetivo

Proveer una base sólida en métodos evolutivos y modelos predictivos, combinando herramientas de optimización global con técnicas supervisadas para resolver problemas complejos de regresión y búsqueda.
