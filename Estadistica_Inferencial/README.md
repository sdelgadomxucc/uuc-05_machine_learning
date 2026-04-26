## Estadística Inferencial

En esta sección se abordan conceptos de la **estadística inferencial** mediante un enfoque teórico y práctico utilizando Python.

Se trabaja con modelos probabilísticos, simulaciones y técnicas de estimación.

---

### 📌 Modelos de distribución

Se analiza una muestra aleatoria $X_1, \dots, X_n$ cuya función de densidad está dada por:

$$
f_X(x;\theta)=
\begin{cases}
\theta x^{\theta-1} & 0 < x < 1,\ \theta > 0 \\
0 & \text{en otro caso}
\end{cases}
$$

Asimismo, se explora la **distribución Beta**, utilizando herramientas como `NumPy`, `Matplotlib` y `SciPy` para:
- Generar muestras aleatorias  
- Visualizar funciones de densidad  
- Analizar el comportamiento de la distribución  

---

### 📊 Simulación de distribuciones

Se realizan simulaciones con distintas distribuciones pertenecientes a la familia exponencial, tales como:
- Distribución t de Student  
- Distribución Beta  
- Distribución Lognormal  
- Distribución Gamma  
- Distribución Poisson  
- Distribución Exponencial  

A través de estas simulaciones, se estudia el comportamiento de los datos y la evolución de la **media muestral**, mostrando empíricamente la convergencia hacia la media teórica (Ley de los Grandes Números).

---

### 📈 Visualización y análisis

Mediante gráficos generados con **Matplotlib**, se representan:
- Datos simulados  
- Trayectorias de la media muestral  
- Comparación con la media teórica  

Esto permite una comprensión visual del comportamiento estadístico de las muestras.

---

### 📐 Estimación por Máxima Verosimilitud (MLE)

Se estudia el método de **Máxima Verosimilitud** para la estimación de parámetros.

Dada una variable aleatoria discreta $X$ con distribución:

$$
\begin{align*}
P(X=0)&=\frac{2}{3}\theta,\\
P(X=1)&=\frac{1}{3}\theta,\\
P(X=2)&=\frac{2}{3}(1-\theta),\\
P(X=3)&=\frac{1}{3}(1-\theta)
\end{align*}
$$

y una muestra observada:

$$
(3, 0, 2, 1, 3, 2, 1, 0, 2, 1)
$$

Se construye la función de verosimilitud:

$$
L(\theta) = \frac{32}{243}\theta^{5}(1-\theta)^{5}
$$

A partir de la log-verosimilitud:

$$
\log L(\theta) = \log(32/243) + 5\log(\theta) + 5\log(1-\theta)
$$

Se obtiene el estimador:

$$
\hat{\theta} = 0.5
$$

Verificando que corresponde a un máximo mediante la segunda derivada.

