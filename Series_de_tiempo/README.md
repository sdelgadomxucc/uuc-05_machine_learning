## Series de Tiempo

En esta sección se introduce el análisis de **series de tiempo**, abordando tanto los fundamentos teóricos como su implementación práctica en Python para el modelado y predicción de datos temporales.

---

## 🧭 Ruta de aprendizaje

### 1. Introducción a las series de tiempo

Una serie de tiempo es una secuencia de datos u observaciones registradas en distintos momentos del tiempo, generalmente en intervalos regulares. Estos pueden ser:

- Horarios  
- Diarios  
- Semanales  
- Mensuales  
- Trimestrales  
- Anuales  

El objetivo principal es analizar el comportamiento de los datos a lo largo del tiempo para identificar patrones y realizar predicciones.

---

### 2. Análisis y visualización

El análisis de series de tiempo utiliza métodos estadísticos para:

- Identificar tendencias  
- Detectar estacionalidad  
- Analizar variabilidad  
- Extraer patrones relevantes  

Las series de tiempo suelen visualizarse mediante **gráficos de líneas**, lo que permite interpretar de forma clara la evolución de los datos.

---

### 3. Forecasting (Predicción)

La **previsión de series temporales** consiste en utilizar modelos estadísticos para predecir valores futuros basándose en datos históricos.

Este proceso es clave en áreas como:
- Finanzas  
- Economía  
- Machine Learning  
- Sistemas de monitoreo  

---

### 4. Implementación en Python

Se utiliza Python para descargar, procesar y modelar datos financieros reales.

#### 📥 Descarga de datos
Se emplea la librería `yfinance` para obtener datos históricos del índice S&P 500:

```python
import yfinance as yf

sp500 = yf.download("^GSPC", start="2000-01-01", end="2014-12-31")