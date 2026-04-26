## Probabilidad en Python

En esta sección se introduce el estudio de la probabilidad utilizando Python como herramienta principal para la resolución de problemas.

Se desarrollan ejemplos prácticos de cálculo de probabilidades, en los cuales se construyen funciones desde cero (sin el uso de librerías especializadas en probabilidad) para resolver distintos casos de estudio, tales como:
- Problemas de extracción de urnas  
- Problema de Newton aplicado al lanzamiento de dados  

Estos ejercicios permiten comprender paso a paso la lógica detrás de los cálculos probabilísticos, reforzando el aprendizaje mediante la implementación directa en código.

Asimismo, se incluyen ejemplos gráficos relacionados con:
- Frecuencias  
- Distribuciones de probabilidad  
- Esperanza matemática  
- Varianza  

Para ello, se hace uso de herramientas como **NumPy** y **Matplotlib**, facilitando la visualización e interpretación de los resultados.

### Muestreo aleatorio con NumPy

Se aborda el muestreo aleatorio a través del **Ensayo de Bernoulli**, utilizando el ejemplo del lanzamiento de una moneda:

- Espacio muestral: $\Omega = \{\text{águila}, \text{sol}\}$  
- Variable aleatoria: $X:\Omega \to \{0,1\}$  
- Asignación de probabilidad:  
  - $\mathbb{P}(X(\text{sol}) = 1) = p \in (0,1)$  
  - $\mathbb{P}(X(\text{águila}) = 0) = 1 - p \in (0,1)$  

Este enfoque permite simular experimentos aleatorios y analizar sus resultados mediante programación.

### Ruina del jugador

También se introduce el problema de la **ruina del jugador**, un modelo clásico de probabilidad que describe la evolución del capital de un jugador que participa en apuestas repetidas. Este problema permite analizar la probabilidad de que el jugador pierda todo su dinero (ruina) o alcance un objetivo, en función de sus probabilidades de ganar y su capital inicial.

---