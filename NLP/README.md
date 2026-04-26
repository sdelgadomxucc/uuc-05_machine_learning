## NLP

En esta sección se abordan técnicas fundamentales de **Procesamiento de Lenguaje Natural (NLP)**, enfocadas en la representación, transformación y análisis de datos textuales utilizando Python.

El texto, uno de los tipos de datos más comunes en ciencia de datos, se maneja generalmente como cadenas de caracteres (*strings*). A partir de este tipo de datos, se aplican distintas técnicas para extraer información relevante y convertir el lenguaje natural en representaciones que puedan ser procesadas por modelos computacionales.

Se estudian métodos de representación de texto como:
- **LSI (Latent Semantic Indexing)**  
- **LDA (Latent Dirichlet Allocation)**  

Estos modelos permiten identificar temas latentes dentro de un conjunto de documentos, facilitando el análisis semántico y la organización de grandes volúmenes de texto.

Asimismo, se trabaja con **vectores embebidos (word embeddings)** utilizando **FastText**, los cuales permiten representar palabras en espacios vectoriales continuos, capturando relaciones semánticas y sintácticas. Estos embeddings son utilizados en tareas como la **clasificación de texto** y el **análisis de sentimientos**.

Dentro del preprocesamiento del lenguaje, se abordan técnicas clave como:

- **Stemming (tallado)**: reduce las palabras a su raíz mediante reglas heurísticas.  
- **Lematización (lemmatization)**: transforma las palabras a su forma canónica o lema, considerando su significado gramatical.  

Ambas técnicas buscan **normalizar el texto**, permitiendo que diferentes variantes de una palabra sean tratadas como una misma unidad (token), lo que mejora el rendimiento de los modelos.

El análisis de sentimientos se presenta como una aplicación práctica relevante, donde se clasifican textos (por ejemplo, reseñas) en categorías como positivo o negativo, utilizando tanto técnicas clásicas como representaciones vectoriales modernas.

En conjunto, esta sección proporciona una base sólida en NLP, combinando técnicas de preprocesamiento, modelado semántico y aplicaciones prácticas sobre datos reales.