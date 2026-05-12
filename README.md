# aXet.Flows - Clasificador de Casos de Uso

Este repositorio contiene la arquitectura del **Clasificador de Casos de Uso** de aXet.Flows, un sistema de "Log Analysis Multi-Agent RAG". Su propósito es realizar ingeniería inversa sobre ejecuciones de Node-RED (Trazas) para mapearlas a un Catálogo de Negocio (Casos de Uso), combinando minería de procesos (Process Mining) con Inteligencia Artificial Generativa.

El sistema adopta un modelo híbrido: **Determinista + Cognitivo + Explicable**.

---

## 🏗 Arquitectura del Pipeline End-to-End

El pipeline se divide en **6 Capas (Layers)** secuenciales que actúan como un embudo (funnel). Las trazas más fáciles y técnicas se resuelven rápido con 0 costo computacional, reservando el "cerebro" (SLM) solo para decisiones complejas de negocio.

### Layer 1: Data Ingestion & Context Hydration (Extracción)
*   **Objetivo:** Reconstruir la historia técnica.
*   **Proceso:** El `GraphEngine` lee la estructura estática del JSON de Node-RED. Luego, el sistema cruza estos nodos con los logs de ejecución reales (ELK) inyectando payloads, variables de entorno y configuraciones internas (URLs de APIs, queries SQL).
*   **Resultado:** Una "Traza Hidratada" rica en contexto.

### Layer 2: Deterministic Keyword Routing (Búsqueda por Reglas - *NUEVO*)
*   **Objetivo:** "Quick Wins". No usar IA si no es necesario.
*   **Proceso:** Antes de cualquier red neuronal, un motor clásico (BM25 o Regex) busca coincidencias exactas entre la traza (ej. `/api/v1/payment`) y los triggers/palabras clave del Catálogo de Casos de Uso. 
*   **Resultado:** Si hay un *match* exacto del 100%, se mapea instantáneamente y se salta el resto del pipeline (ahorrando horas de procesamiento).

### Layer 3: Cognitive Triage & Error Taxonomy (Escáner de Ruido)
*   **Objetivo:** Filtrar fontanería técnica y auditar fallos de sistema.
*   **Proceso:** La traza hidratada se envía al SLM (Phi-3) actuando como un **Scanner Agent**. El Agente clasifica en 3 vías:
    1.  `TechnicalInfrastructure`: Se marca como "System Overhead (Non-billable)" y termina.
    2.  `ErrorHandling`: **NUEVO:** El SLM extrae la taxonomía del error (ej. *Auth Failure*, *Database Timeout*, *Bad Gateway*). Ideal para métricas de salud de TI. Termina.
    3.  `BusinessLogic`: Es un proceso de negocio real. Pasa a la siguiente capa.

### Layer 4: Semantic RAG Retrieval (Búsqueda Vectorial Amplia)
*   **Objetivo:** Encontrar posibles candidatos en el Catálogo de Casos de Uso.
*   **Proceso:** Se utiliza BGE-Small (o Jina) para convertir el resumen de la traza en vectores matemáticos. En lugar de elegir el mejor resultado ciegamente, se extraen los **Top 5 Candidatos** que superen un umbral base (ej. 0.50).

### Layer 5: Cross-Encoder Re-Ranking & Explainability (LLM-as-a-Judge - *NUEVO*)
*   **Objetivo:** Toma de decisión humana y explicabilidad.
*   **Proceso:** El sistema envía al SLM un prompt complejo: *"Tengo esta Traza y estos 5 Casos de Uso candidatos. Como Arquitecto de Negocio, ¿cuál es el correcto y por qué?"*.
*   **Resultado:** 
    *   **Self-Correction:** Si ninguno de los 5 candidatos es bueno, el SLM declara "Sin Match / Requiere Revisión Humana".
    *   **Explainable AI:** Si elige uno, el SLM redacta un párrafo justificando ejecutivamente la decisión (ej. *"Se asigna el UC-0489 porque la traza demuestra la validación de un formulario seguido de una persistencia en base de datos"*).

### Layer 6: Executive Reporting & Business Impact
*   **Objetivo:** Traducir los datos técnicos en valor comercial y operativo.
*   **Proceso:** El pipeline final consolida todas las ejecuciones. Genera el JSON (o exporta a CSV/Dashboard) con el Caso de Uso, la explicación (Reasoning) y las métricas de error. Esto permite calcular el ROI en tiempo real.

---

## 🛠 Stack Tecnológico
*   **Procesamiento:** Python 3.x
*   **Graph Analysis:** Lógica de recursión propia para árboles de Node-RED.
*   **Small Language Model (SLM):** `llama-cpp-python` (Phi-3-mini 4K context) para inferencia local offline 100% segura.
*   **Vector Database / Embeddings:** `sentence-transformers` (BGE-Small / Jina Embeddings) + Búsqueda por similitud del Coseno.

---

## 🚀 Hoja de Ruta de Implementación (Roadmap)
Para migrar el código actual a esta nueva Arquitectura V2, se ejecutarán los siguientes pasos de desarrollo:
1.  **Refactor de `pipeline.py`:** Separar el código en agentes distintos (Routing, RAG, Re-ranker).
2.  **Implementación del BM25/Regex (Layer 2):** Crear un módulo ligero para *Quick Wins*.
3.  **Actualización del Prompt (Layer 3 & 5):** Dividir el mega-prompt actual en dos prompts especializados (uno para el Triage y otro para el Juez Re-ranker).
4.  **Carga del Top-K:** Modificar el motor vectorial para devolver arrays de candidatos en lugar del ganador único.
