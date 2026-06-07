\# AI Receta: Generación Aumentada por Recuperación (RAG) para Asistencia Gastronómica Segura



Este repositorio contiene el código fuente, la infraestructura de despliegue y la documentación técnica correspondiente al Trabajo Final de Máster en Inteligencia Artificial. 



El proyecto propone una arquitectura RAG cien por cien local diseñada para la generación dinámica de recetas culinarias. Su principal innovación reside en la aplicación de un ecosistema de \*guardarraíles\* deterministas que garantizan la seguridad alimentaria, mitigando el riesgo de alucinaciones paramétricas en usuarios con restricciones dietéticas estrictas.



\## 🏗️ Arquitectura y Pila Tecnológica



La solución se fundamenta en un patrón de diseño orientado a microservicios, garantizando el aislamiento operativo y la escalabilidad de sus componentes:



\* \*\*Motor de Inferencia (LLM):\*\* Inferencia local y determinista mediante \[Ollama](https://ollama.ai/), ejecutando el modelo \*\*Qwen2.5 de 7B parámetros\*\*.

\* \*\*Base de Conocimiento y Recuperación Semántica:\*\* Indexación vectorial densa implementada con \*\*FAISS\*\* (Facebook AI Similarity Search) sobre el corpus masivo \*\*RecipeNLG\*\*.

\* \*\*Orquestador (Backend):\*\* API REST desarrollada con \*\*Flask\*\* (Python) encargada del procesamiento de lenguaje natural (NLP), validación de restricciones y ejecución de rutinas de \*Self-Healing\*.

\* \*\*Interfaz de Usuario (Frontend):\*\* Entorno interactivo desarrollado en \*\*Streamlit\*\*.

\* \*\*Infraestructura (MLOps):\*\* Contenerización total mediante \*\*Docker\*\* y \*\*Docker Compose\*\*.



\## ⚙️ Requisitos del Entorno



Para asegurar la correcta ejecución del sistema, la máquina anfitriona debe cumplir con las siguientes especificaciones:

\* \*\*Software:\*\* Docker (v20.10 o superior) y Docker Compose (v2.0 o superior) instalados y en ejecución.

\* \*\*Hardware (Memoria):\*\* Un mínimo de 16 GB de memoria RAM.

\* \*\*Hardware (Aceleración):\*\* Se recomienda encarecidamente el uso de una Unidad de Procesamiento Gráfico (GPU) compatible con soporte nativo para aceleración CUDA (ej. NVIDIA Tesla T4) para optimizar la latencia de inferencia.



\## 🚀 Instrucciones de Despliegue y Ejecución



El proceso de despliegue se encuentra completamente automatizado. Durante la primera inicialización, el orquestador descargará las imágenes base y los pesos del modelo neuronal de forma autónoma.



1\. \*\*Clonación del repositorio institucional:\*\*

&#x20;  ```bash

&#x20;  git clone \[https://github.com/tu-usuario/tu-repositorio-tfm.git](https://github.com/tu-usuario/tu-repositorio-tfm.git)

&#x20;  cd entrega

