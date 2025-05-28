# Clasificador de Péptidos de Penetración Celular (PPC)

## Descripción

Este proyecto implementa un clasificador de péptidos de penetración celular (PPC) utilizando el modelo de lenguaje Llama3.2:1b a través de la API de Ollama. El script procesa un dataset de péptidos, realiza clasificación binaria (es_PPC: 1/0) y genera métricas de evaluación.

## Requisitos

- Python 3.x
- Entorno virtual (venv)
- Librerías: pandas, ollama, numpy, scikit-learn
- Modelo Llama3.2:1b instalado en Ollama
- Archivo de datos: data/CPP_Classif_data.xlsx

## Uso

1. Crear y activar entorno virtual:
   - Windows: `python -m venv venv` → `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` → `source venv/bin/activate`
2. Instalar dependencias con el entorno activado: `pip install -r requirements.txt`
3. Asegurar que Ollama esté corriendo con el modelo correcto:
   - Comando: `ollama serve llama3.2:1b` o cambiar el nombre del modelo en el script si es necesario usar otro.
4. Ejecutar el script: `python classify_peptides.py`
5. Los resultados se guardarán en: data/classification_results.xlsx

## Dataset

El archivo de datos debe contener dos columnas:

- 'peptide': secuencias de péptidos
- 'y': etiquetas binarias (1 para CPP, 0 para no-CPP)

## Evaluación

El script muestra métricas de rendimiento (accuracy, precision, recall, F1-score) y la matriz de confusión

## resultados (Métricas del Modelo)

# Reporte de Clasificación

| Clase | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.60      | 0.41   | 0.48     | 37      |
| 1     | 0.55      | 0.73   | 0.63     | 37      |

- **Exactitud (accuracy)**: 0.5676
- **Matriz de Confusión**:

|         | CPP | non-CPP |
| ------- | --- | ------- |
| CPP     | 15  | 22      |
| non-CPP | 10  | 27      |
