# 📝 NLP — Procesamiento de Lenguaje Natural

> Representación, análisis y clasificación de texto — desde preprocesamiento hasta modelos semánticos y análisis de sentimientos.

---

## 📋 Descripción

Esta sección cubre técnicas fundamentales de **Procesamiento de Lenguaje Natural (NLP)** en Python, transformando texto en representaciones computacionales que permiten extraer significado, identificar temas y clasificar opiniones.

---

## 📂 Contenido

| Archivo | Descripción |
|---|---|
| `Stemming_lemmatization.ipynb` | Normalización de texto: stemming y lematización |
| `MNA_NLP_FastText_embeddings.ipynb` | Word embeddings con FastText |
| `SENTIMENT_Movies.ipynb` | Análisis de sentimientos sobre reseñas de películas |
| `MNA_NLP_Actividad_semanas_6y7.ipynb` | Actividades integradoras de NLP |
| `NLP_ejercicios_complementarios.ipynb` | Ejercicios adicionales de práctica |
| `amazon5.txt` | Dataset de reseñas de Amazon |
| `imdb5.txt` | Dataset de reseñas de IMDB |
| `embedding_dict.pkl` | Diccionario de embeddings preentrenados |

---

## 🔄 Flujo NLP

```
Texto crudo → Preprocesamiento → Representación vectorial → Modelo → Predicción
```

---

## 🧹 Preprocesamiento de Texto

Técnicas para normalizar el texto antes del modelado:

| Técnica | Descripción | Ejemplo |
|---|---|---|
| **Stemming** | Reduce la palabra a su raíz mediante reglas heurísticas | *corriendo* → *corr* |
| **Lematización** | Transforma la palabra a su forma canónica (lema) | *corriendo* → *correr* |
| **Tokenización** | Divide el texto en unidades mínimas (tokens) | *"Hola mundo"* → `["Hola", "mundo"]` |

---

## 🗂️ Modelos de Tópicos

Métodos para descubrir temas latentes en grandes colecciones de documentos:

| Modelo | Descripción |
|---|---|
| **LSI** — Latent Semantic Indexing | Reducción de dimensionalidad semántica via SVD |
| **LDA** — Latent Dirichlet Allocation | Modelo probabilístico de mezcla de tópicos |

---

## 🔢 Word Embeddings con FastText

Representación de palabras en espacios vectoriales continuos que capturan relaciones semánticas y sintácticas:

- Maneja palabras **fuera del vocabulario** (OOV) usando subpalabras
- Útil para idiomas morfológicamente ricos como el español
- Aplicado a tareas de **clasificación de texto** y **análisis de sentimientos**

```
"rey" - "hombre" + "mujer" ≈ "reina"
```

---

## 💬 Análisis de Sentimientos

Clasificación de textos en categorías de opinión (positivo / negativo):

| Dataset | Dominio |
|---|---|
| `imdb5.txt` | Reseñas de películas |
| `amazon5.txt` | Reseñas de productos |

---

## 🛠️ Herramientas

| Librería | Uso |
|---|---|
| `NLTK` / `spaCy` | Tokenización, stemming y lematización |
| `FastText` | Generación y uso de word embeddings |
| `scikit-learn` | Vectorización (TF-IDF), clasificadores, métricas |
| `Gensim` | Modelos LSI y LDA |

---

## 🏁 Objetivo

Construir una base sólida en NLP combinando preprocesamiento, modelado semántico y aplicaciones prácticas sobre datos reales de texto, desde la normalización hasta la clasificación de opiniones.
