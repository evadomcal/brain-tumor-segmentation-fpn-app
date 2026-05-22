# Brain Tumor UNet Segmentation App

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Dagster](https://img.shields.io/badge/Dagster-1.5+-green.svg)](https://dagster.io)
[![UNet](https://img.shields.io/badge/Architecture-UNet-orange.svg)](https://arxiv.org/abs/1505.04597)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-cyan.svg)](https://docs.astral.sh/uv)

> **Trabajo académico** - Aplicación web para segmentación de tumores cerebrales en resonancias magnéticas (MRI) con evaluación de urgencia médica.

## Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Características](#características)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración de Kaggle](#configuración-de-kaggle)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Autor](#autor)
- [Licencia](#licencia)


## Descripción General

Esta aplicación permite la segmentación automática de tumores cerebrales a partir de imágenes de resonancia magnética (MRI) en formato .tif. Utilizando una arquitectura UNet, el sistema genera:

- Máscara de segmentación del tumor
- Características tumorales (tamaño, forma, ubicación)
- Nivel de urgencia basado en análisis de imagen + datos clínicos

La aplicación incluye un formulario médico donde se pueden ingresar datos clínicos del paciente (edad, grado histológico, etc.) que se combinan con el análisis de la imagen para predecir un nivel de urgencia más preciso.

## Características

- Segmentación UNet de tumores en MRI cerebrales
- Interfaz web interactiva con Streamlit
- Formulario clínico para mejorar predicción de urgencia
- Orquestación con Dagster para flujos de trabajo reproducibles
- Procesamiento paralelo con Dask para grandes volúmenes de datos
- Análisis estadístico integrado con R
- Ficha técnica automática con métricas del modelo

## Requisitos Previos

- Python 3.12 o superior
- Git
- uv
- Cuenta de Kaggle para descargar los datos
- R
- GPU recomendada (funciona con CPU pero es más lento)

## Instalación

1. Clonar el repositorio:

git clone https://github.com/evadomcal/brain-tumor-segmentation-FPN-app.git
cd brain-tumor-segmentation-unet-app

2. Instalar uv:

pip install uv

3. Crear entorno virtual e instalar dependencias:

uv venv

Activar entorno (Windows):
.venv\Scripts\activate

Instalar dependencias:
uv sync

## Configuración de Kaggle

Para descargar los datos necesitas una API key de Kaggle:

1. Ve a Kaggle Settings (https://www.kaggle.com/settings)
2. En la sección "API", haz clic en "Create New Token"
3. Se descargará kaggle.json
4. Coloca el archivo en la raíz del proyecto

Enlace del dataset: [https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data]

## Estructura del Proyecto

brain-tumor-segmentation-unet-app/  
src/  
 &ensp; datos.py   &emsp;                      # Importación de datos desde Kaggle  
 &ensp; mapeo_archivos.py  &emsp;              # Generación del catálogo de trabajo  
 &ensp; transformar_datos.py   &emsp;          # Adaptación y preprocesamiento de imágenes  
 &ensp; Analisis/  
 &ensp;&ensp;   Analisis_imagen.py    &emsp;   # Mini-modelo para predicción de urgencia  
 &ensp;&ensp;   Analisis_test.py    &emsp;     # Análisis del conjunto de test  
 &ensp;&ensp;   Metricas_modelo.py    &emsp;   # Métricas de calidad de segmentación  
 &ensp;&ensp;   Script_R_1.R    &emsp;         # Análisis estadístico en R  
 &ensp; Aplicacion/  
 &ensp;&ensp;   streamlit.py    &emsp;         # Página principal de la app  
 &ensp;&ensp;   cliente.py   &emsp;            # Comunicación con Dagster  
 &ensp;&ensp;   pages/  
 &ensp;&ensp;&ensp;     Ficha_tecnica.py   &emsp;     # Ficha técnica del modelo  
 &ensp; orquestador/  
 &ensp;&ensp;   activos.py   &emsp;                   # Activos de Dagster para el flujo de trabajo  
 &ensp; procesamiento_datos/  
 &ensp;&ensp;   procesador_dask.py  &emsp;            # Transformación paralela con Dask  
 &ensp;&ensp;   modelo/  
 &ensp;&ensp;&ensp;     modelo_fpn.py    &emsp;       # Arquitectura FPN  
 &ensp;&ensp;&ensp;     entrenar_modelo.py    &emsp;  # Entrenamiento del modelo  
 &ensp;&ensp;&ensp;     segmentar.py   &emsp;         # Lógica de segmentación  
 &ensp;&ensp;&ensp;     caracteristicas.py  &emsp;    # Extracción de características tumorales  
.gitignore  
pyproject.toml  
uv.lock  
README.md  


## Entrenamiento del Modelo

Para entrenar el modelo, se utiliza Dagster como orquestador. Debes ejecutar los activos en el siguiente orden:

1. Catalogo_maestro
2. imagenes_procesadas
3. dividir_dataset_balanceado
4. modelo_fpn_entrenado
5. mejor_umbral
6. segmentaciones_test
7. caracteristicas_tumorales
8. analisis_datos
9. calidad_modelo_segmentacion
10. entrenar_modelo_urgencia

### Ejecución con Dagster

```bash
dagster dev
```

Luego accede a la interfaz web de Dagster (http://localhost:3000) y materializa los activos en el orden indicado.


## Uso de la Aplicación

Iniciar la aplicación Streamlit:

streamlit run Aplicacion/streamlit.py

La aplicación se abrirá en el navegador (normalmente http://localhost:8501).

Flujo de uso:


1. Rellenar y subir formulario clínico
2. Subir imagen MRI
3. Ejecutar análisis
4. Resultados: máscara de segmentación, características del tumor, nivel de urgencia, archivo .zip con los resultados


## Autor

Alejandro y Eva  
- https://github.com/alejandrooam  
- https://github.com/evadomcal  

## Licencia

Este proyecto es un trabajo académico de uso educativo. Para uso comercial, contactar al autor.
