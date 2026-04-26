## Cálculo Estocástico

En esta sección se presenta una **ruta de aprendizaje** para el estudio del cálculo estocástico, integrando teoría de probabilidad, procesos estocásticos y herramientas matemáticas avanzadas. El contenido está basado en diversos materiales, incluyendo libros y notas especializadas.

---

### Fundamentos de probabilidad avanzada

Se comienza con conceptos esenciales que permiten construir una base sólida:

- Espacios de probabilidad $(\Omega, \mathcal{F}, P)$  
- Completitud de un espacio de probabilidad  
- Conjuntos nulos y su importancia en la teoría de la medida  

Un espacio de probabilidad es **completo** si todo subconjunto de un conjunto de probabilidad cero también es medible. Esta propiedad es fundamental para evitar inconsistencias en construcciones probabilísticas más avanzadas.

---

### Variables aleatorias y tiempos de espera

Se estudian variables aleatorias continuas, en particular:

- Distribución exponencial  
- Propiedades de los tiempos de espera  
- Relojes exponenciales independientes  

Si $T_1, \dots, T_n$ son variables aleatorias independientes con distribución exponencial, entonces:

$$
T = \min\{T_1, \dots, T_n\}
$$

representa el primer evento que ocurre en un sistema de múltiples procesos aleatorios.

---

### Cadenas de Markov en tiempo continuo

Se introduce el estudio de procesos estocásticos en tiempo continuo, incluyendo:

- Definición de cadenas de Markov en tiempo continuo  
- Interpretación mediante tiempos de espera  
- Tiempos medios de paso  
- Clasificación de estados  
- Procesos de nacimiento y muerte  
- Procesos de nacimiento puro  

Además, se analizan las **ecuaciones forward y backward**, fundamentales para describir la evolución temporal de estos procesos.

---

### Movimiento Browniano y procesos Gaussianos

Se estudia el **movimiento browniano** como uno de los procesos fundamentales del cálculo estocástico:

- Definición de movimiento browniano estándar  
- Incrementos independientes y estacionarios  
- Distribución normal de los incrementos  

Por ejemplo, si $B_t$ es un movimiento browniano, entonces:

$$
B_t - B_s \sim \mathcal{N}(0, t - s)
$$

También se analizan momentos de orden superior y propiedades de procesos gaussianos.

---

### Martingalas

Se introduce el concepto de **martingalas**, clave en teoría de probabilidad moderna:

- Definición de martingala  
- Propiedades básicas  
- Interpretación como "juegos justos"  

Las martingalas son fundamentales en finanzas, teoría de apuestas y modelado estocástico.

---

### Integración estocástica

Se construye la integral estocástica de manera rigurosa:

- Definición inicial para procesos simples (funciones escalonadas)  
- Extensión mediante límite en media cuadrática ($L^2$)  
- Motivación: superar limitaciones de definiciones clásicas  

Este enfoque permite definir integrales respecto al movimiento browniano, base del cálculo estocástico.

---

### Aplicaciones y conexión con otras áreas

El cálculo estocástico tiene aplicaciones en:

- Finanzas cuantitativas  
- Modelos de difusión  
- Inteligencia Artificial  
- Procesos de decisión  

También se estudian ejemplos aplicados como:
- Procesos de nacimiento y muerte  
- Modelos dinámicos en sistemas aleatorios  
