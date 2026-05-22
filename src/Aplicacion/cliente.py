"""

CLIENTE DE COMUNICACIÓN ENTRE LA APP Y DAGSTER 

Este script gestiona el flujo completo de datos para el diagnóstico de tumores cerebrales:
- Procesamiento de imágenes de resonancia magnética (TIF)
- Segmentación con Features Pyramid Network
- Extracción de biomarcadores 
- Predicción de urgencia clínica combinando IA y datos del paciente
- Generación de informe médico 

"""


# LIBRERÍAS NECESARIAS:

import requests         # Comunicación HTTP con la API de Dagster
import json             # Manejo de datos en formato JSON para intercambio con API
import pandas as pd     # Manipulación de dataframes 
import numpy as np      # Computación numérica y manejo de arrays multidimensionales (las imágenes)
from PIL import Image   # Procesamiento básico de imágenes (convertir formatos)
import io               # Operaciones de entrada/salida en memoria (bytes)
import base64           # Codificación/decodificación Base64 para imágenes en JSON
from typing import Tuple, Dict, Any, Optional  # Para documentar tipos de datos
import streamlit as st  # Framework para crear la interfaz web interactiva
from pathlib import Path    # Manejo de rutas de archivos
import tempfile             # Creación de archivos y directorios temporales
import shutil               # Operaciones avanzadas de archivos (borrar directorios recursivamente)
import joblib               # Carga de modelos guardados con scikit-learn (formato .pkl)
from datetime import datetime    # Generación de timestamps para nombres únicos de archivos
import cv2                       # OpenCV: procesamiento avanzado de imágenes
import torch                     # PyTorch: deep learning para el modelo FPN
import sys               # Manipulación del sistema: modificar ruta de búsqueda
from skimage.measure import label, regionprops         # Etiquetado de regiones y para propiedades morfológicas
from skimage.feature import graycomatrix, graycoprops  # Matriz de co-ocurrencia y textura (grises)


# CONFIGURACIÓN DE RUTAS 

# Obtiene el directorio padre del directorio actual (sube 2 niveles desde Aplicacion a src)
src_path = Path(__file__).parent.parent

# Inserta src_path al inicio de sys.path para que Python busque módulos allí primero
sys.path.insert(0, str(src_path))


# IMPORTACIÓN DE MÓDULOS PROPIOS DEL PROYECTO
from transformar_datos import procesar_imagen_completo  # Preprocesamiento: recorte y normalización de resonancias
from procesamiento_datos.modelo.segmentar import segmentar_imagen  # Inferencia: aplicar FPN para obtener máscara
from TRABAJO.src.procesamiento_datos.modelo.modelo_fpn import FPN   # Arquitectura del modelo FPN


# CLASE PRINCIPAL: DAGSTER CLIENT
class DagsterClient:
    """
    Cliente para comunicarse con la API de Dagster y orquestar el análisis.
    Actúa como intermediario entre la aplicación web (Streamlit) y el orquestador Dagster,
    gestionando los datos temporales de cada sesión de paciente.
    """
    
    # CONSTRUCTOR
    def __init__(self, base_url: str = "http://localhost:3000"):
        """
        Inicializa el cliente con la URL base del orquestador Dagster.
        
        Args:
            base_url: URL donde corre el servidor de Dagster (puerto 3000 por defecto)
        """
        self.base_url = base_url  # Almacena URL del orquestador Dagster
    
    
    # GESTIÓN DE SESIONES
    def inicializar_sesion(self, session_id: str) -> str:
        """
        Crea un directorio temporal único para cada sesión de paciente.
        
        Args:
            session_id: Identificador único de la sesión
        
        Returns:
            Ruta del directorio temporal que hemos creado
        """
        # Construye ruta: /temp/mrai_session_{session_id}
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)  # Crea el directorio (no da error si ya existe)
        return str(temp_dir)  # Devuelve ruta como string
    

    # MANEJO DE DATOS CLÍNICOS
    def guardar_datos_clinicos(self, datos_clinicos: Dict[str, Any], session_id: str) -> str:
        """
        Guarda los datos clínicos del paciente en un CSV temporal.
        
        Args:
            datos_clinicos: Diccionario con datos del paciente
        
        Returns:
            Ruta del directorio temporal donde se guardó el CSV
        """
        # Construye ruta del directorio temporal de la sesión
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # Convierte diccionario a DataFrame (envuelto en lista para crear una fila)
        df_clinicos = pd.DataFrame([datos_clinicos])
        
        # Define ruta completa del archivo CSV
        ruta_clinicos = temp_dir / "datos_clinicos.csv"
        
        # Guarda CSV: sin índice, valores nulos como 'NULL'
        df_clinicos.to_csv(ruta_clinicos, index=False, na_rep='NULL')
        
        return str(temp_dir)


    # PIPELINE PRINCIPAL: PROCESAMIENTO DE IMAGEN + IA

    def procesar_imagen(self, imagen_tif_bytes: bytes, datos_clinicos: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de IA:
        1. Preprocesado (recorte y normalización de resonancias)
        2. Segmentación del tumor con FPN
        3. Extracción de biomarcadores
        
        Args:
            imagen_tif_bytes: Imagen de resonancia en formato TIF como bytes
            datos_clinicos: Datos del paciente 
        
        Returns:
            Tuple con (máscara_binaria, DataFrame de características)
        """
        
        # PASO 1: GUARDAR IMAGEN TEMPORAL
        # Crea directorio temporal general
        temp_dir = Path(tempfile.gettempdir()) / "mrai_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Genera nombre único con timestamp y microsegundos
        temp_img_path = temp_dir / f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.tif"
        
        # Escribe los bytes de la imagen en el archivo temporal
        # 'wb' indica lenguaje binario
        with open(temp_img_path, 'wb') as f:
            f.write(imagen_tif_bytes)
        
        # PASO 2: PREPROCESAMIENTO
        # Prepara la imagen: recorta a tamaño estándar y normaliza
        # fila_maestro simula la estructura que espera procesar_imagen_completo
        fila_maestro = {'ruta_imagen': str(temp_img_path), 'ruta_mascara': None}
        img_procesada, _ = procesar_imagen_completo(fila_maestro, entrenando=False)
        # img_procesada tiene forma (Alto, Ancho, 3) - 3 canales: FLAIR, Pre, Post
        
        # PASO 3: GUARDAR ARRAY PROCESADO
        # Guarda como .npy (formato nativo de NumPy) para que el modelo pueda leerlo
        temp_npy_path = temp_dir / f"img_procesada_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy"
        np.save(temp_npy_path, img_procesada)
        
        # PASO 4: CARGAR MODELO FPN ENTRENADO
        # Ruta a la carpeta de datos procesados (sube un nivel desde src)
        DATOS_PROCESADOS = src_path.parent / "datos_procesados"
        
        # Instancia la arquitectura FPN: 3 canales entrada (imagen), 1 salida (máscara binaria)
        # FPN usa pirámide de características para detectar tumores a diferentes escalas
        model = FPN()  # n_clases=1 para segmentación binaria (tumor vs fondo) por defecto

        # Ruta al archivo de pesos entrenados del modelo FPN
        modelo_fpn_entrenado = DATOS_PROCESADOS / "modelo_fpn_mejor.pth"

        # Carga los pesos en el modelo (map_location='cpu' fuerza CPU aunque se entrenó en GPU)
        # Esto permite cargar modelos entrenados en GPU en máquinas sin GPU
        model.load_state_dict(torch.load(modelo_fpn_entrenado, map_location='cpu'))

        # Selecciona dispositivo: GPU (CUDA) si está disponible, si no CPU
        # FPN se beneficia de GPU por tener más parámetros que U-Net
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)  # Mueve el modelo al dispositivo elegido

        # Carga el umbral óptimo para binarizar la máscara (calculado durante entrenamiento)
        # El umbral maximiza F1-Score en validación (típicamente entre 0.3 y 0.7)
        umbral = np.load(DATOS_PROCESADOS / "umbral.npy")

        #  PASO 5: SEGMENTACIÓN (INFERENCIA)
        # Obtiene máscara de probabilidad (0-1) y máscara binaria (0/1 después del umbral)
        mascara_prob, mascara_binaria = segmentar_imagen(
            model=model, 
            imagen_npy_path=str(temp_npy_path), 
            device=device, 
            umbral=umbral
        )
        
        # PASO 6: EXTRACCIÓN DE BIOMARCADORES 
        # Calcula características morfológicas, de intensidad y textura
        caracteristicas = self.extraer_caracteristicas_completas(img_procesada, mascara_binaria)
        
        # PASO 7: CONVERTIR A DATAFRAME
        # Transforma diccionario a DataFrame (1 fila al ser un paceinte, múltiples columnas)
        df_caracteristicas = pd.DataFrame([caracteristicas])
        
        # PASO 8: LIMPIEZA (ELIMINAR TEMPORALES)
        temp_img_path.unlink()  # Borra archivo TIF temporal
        temp_npy_path.unlink()  # Borra archivo NPY temporal
        
        return mascara_binaria, df_caracteristicas
    
 
    # EXTRACCIÓN DE CARACTERÍSTICAS RADIÓMICAS
    def extraer_caracteristicas_completas(self, imagen: np.ndarray, mascara_binaria: np.ndarray) -> Dict[str, Any]:
        """
        Cálculo de biomarcadores a partir de la máscara del tumor:
        - Morfología: área, perímetro, circularidad (forma)
        - Intensidad: media y mínima post-contraste, percentil 95 FLAIR
        - Textura: contraste GLCM (heterogeneidad)
        
        Args:
            imagen: Array 3D (Alto, Ancho, Canales) con las 3 secuencias (FLAIR, Pre, Post)
            mascara_binaria: Array 2D (Alto, Ancho) con valores 0 (fondo) y 1 (tumor)
        
        Returns:
            Diccionario con todas las características extraídas
        """
        
        # CASO ESPECIAL: SIN TUMOR
        # Si la máscara está vacía (suma=0), devuelve todo a cero para evitar errores
        if mascara_binaria.sum() == 0:
            return {
                'area': 0, 
                'perimetro': 0, 
                'circularidad': 0, 
                'intensidad_media_post': 0, 
                'intensidad_minima_post': 0, 
                'percentil_95_flair': 0, 
                'textura_contraste': 0
            }
        
        # SECCIÓN 1: MORFOMETRÍA (FORMA Y TAMAÑO)
        
        # Etiqueta regiones conectadas (zonas tumorales) en la máscara
        labeled = label(mascara_binaria)
        
        # Toma la primera región (la más grande, asumiendo que es el tumor principal)
        props = regionprops(labeled)[0]
        
        # Área: número de píxeles que ocupa el tumor
        area = props.area
        
        # Perímetro: longitud del borde del tumor (en píxeles)
        perimetro = props.perimeter
        
        # Circularidad: indica qué tan redondo es el tumor (1 = círculo perfecto, <1 = irregular)
        # Fórmula: 4π * área / (perímetro²)
        # Tumores malignos suelen ser más irregulares (circularidad baja)
        circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        
        # SECCIÓN 2: INTENSIDAD POST-CONTRASTE (VASCULARIZACIÓN)
        
        # Canal Post-contraste: índice 2 (canal 0=FLAIR, 1=Pre, 2=Post)
        canal_post = imagen[:, :, 2]
        
        # Extrae solo los píxeles que están dentro del tumor
        valores_post = canal_post[mascara_binaria > 0]
        
        # Intensidad media: promedio de realce del tumor (más realce = más vascularizado = más agresivo)
        intensidad_media_post = float(valores_post.mean())
        
        # Intensidad mínima: necrosis o tejido poco vascularizado
        intensidad_minima_post = float(valores_post.min())
        
        # SECCIÓN 3: PERCENTIL 95 FLAIR (EDEMA/INFILTRACIÓN)
        
        # Canal FLAIR: índice 0 (mejores para detectar edema)
        canal_flair = imagen[:, :, 0]
        valores_flair = canal_flair[mascara_binaria > 0]
        
        # Percentil 95: valor por debajo del cual está el 95% de los píxeles (ignora outliers)
        # Valores altos indican edema peritumoral significativo
        percentil_95_flair = float(np.percentile(valores_flair, 95))
        
        #  SECCIÓN 4: TEXTURA (HETEROGENEIDAD TUMORAL)
        # Obtiene el rectángulo que envuelve al tumor (bounding box)
        minr, minc, maxr, maxc = props.bbox
        
        # Recorta la región del tumor del canal post-contraste
        rdi = canal_post[minr:maxr, minc:maxc]  # ROI: Región de interés (Region of Interest)
        
        # Recorta también la máscara para la misma región
        rdi_mask = mascara_binaria[minr:maxr, minc:maxc]
        
        # Aplica la máscara (pone a 0 los píxeles fuera del tumor)
        rdi = rdi * rdi_mask
        
        # Normaliza a 8 bits (0-255) si hay valores positivos
        if rdi.sum() > 0 and rdi.max() > 0:
            rdi = (rdi / rdi.max() * 255).astype(np.uint8)
        
        # Calcula matriz de co-ocurrencia (GLCM) si la región es válida
        if rdi.sum() > 0 and rdi.shape[0] > 1 and rdi.shape[1] > 1:
            try:
                # GLCM: analiza patrones de textura (distancia=1 píxel, ángulo=0° horizontal)
                glcm = graycomatrix(rdi, distances=[1], angles=[0], levels=256, symmetric=True)
                
                # Contraste: mide la heterogeneidad (valores altos = textura irregular = tumor agresivo)
                textura_contraste = float(graycoprops(glcm, 'contrast')[0, 0])
            except:
                textura_contraste = 0  # Si falla el cálculo, asigna 0
        else:
            textura_contraste = 0
        
        # RETORNA DICCIONARIO CON TODAS LAS CARACTERÍSTICAS
        return {
            'area': area, 
            'perimetro': perimetro, 
            'circularidad': circularidad, 
            'intensidad_media_post': intensidad_media_post, 
            'intensidad_minima_post': intensidad_minima_post, 
            'percentil_95_flair': percentil_95_flair, 
            'textura_contraste': textura_contraste
        }
    
  
    # PREDICCIÓN DE URGENCIA CLÍNICA (REGRESIÓN LOGÍSTICA)
    def calcular_urgencia(self, caracteristicas_df: pd.DataFrame, datos_clinicos: Dict[str, Any]) -> float:
        """
        Predice el nivel de urgencia clínica combinando:
        - Biomarcadores (extraídas de la imagen)
        - Datos clínicos del paciente (edad, grado tumoral)
        
        Args:
            caracteristicas_df: DataFrame con las 7 características tumorales
            datos_clinicos: Diccionario con edad y grado histológico del tumor
        
        Returns:
            Probabilidad de urgencia (valor entre 0 y 1)
            - 0-0.3: Urgencia baja 
            - 0.3-0.7: Urgencia moderada
            - 0.7-1.0: Urgencia alta 
        """
        
        # Verifica si hay tumor (área > 0)
        if 'area' in caracteristicas_df.columns:
            area = caracteristicas_df['area'].iloc[0]
        
        # Si no hay tumor o es NA, urgencia = 0
        if area == 0 or pd.isna(area):
            return 0.0
        
        # PASO 1: CARGAR MODELO DE URGENCIA
        DATOS_PROCESADOS = src_path.parent / "datos_procesados"
        modelo_path = DATOS_PROCESADOS / "modelo_urgencia" / "modelo_urgencia.pkl"
        
        # Verifica que el archivo existe
        if not modelo_path.exists():
            raise FileNotFoundError(f"Error: {modelo_path} no encontrado")
        
        # Carga el modelo con joblib (contiene el modelo logístico entrenado)
        data = joblib.load(modelo_path)
        modelo = data['modelo']  # Extrae el modelo (ej: Regresión Logística)
        
        # PASO 2: PREPARAR CARACTERÍSTICAS COMBINADAS (diccionario)
        features_dict = {}
        
        # Características tumorales (7 variables)
        for col in ['area', 'perimetro', 'circularidad', 'intensidad_media_post', 
                    'intensidad_minima_post', 'percentil_95_flair', 'textura_contraste']:
            features_dict[col] = caracteristicas_df[col].iloc[0] # la primera fila será la del paciente en estudio, al ser la única
        
        # Edad del paciente (si no viene, usa 55 como valor por defecto)
        features_dict['age_at_initial_pathologic'] = datos_clinicos.get('age_at_initial_pathologic', 55) or 55
        
        # Grado histológico (convierte texto a número)
        grado = datos_clinicos.get('neoplasm_histologic_grade')
        grado_map = {
            "Grado IV": 4, "IV": 4,
            "Grado III": 3, "III": 3,
            "Grado II": 2, "II": 2,
            "Grado I": 1, "I": 1
        }
        features_dict['neoplasm_histologic_grade'] = grado_map.get(grado, 2)  # Default = Grado II
        
        # PASO 3: ORDENAR COLUMNAS (mismo orden que en entrenamiento) 
        columnas_ordenadas = [
            'area', 'perimetro', 'circularidad', 
            'intensidad_media_post', 'intensidad_minima_post', 
            'percentil_95_flair', 'textura_contraste', 
            'age_at_initial_pathologic', 'neoplasm_histologic_grade'
        ]
        
        # Crea DataFrame con las características en el orden correcto
        features_completas = pd.DataFrame([features_dict])[columnas_ordenadas]
        
        # PASO 4: PREDICCIÓN (PROBABILIDAD DE URGENCIA) 
        # predict_proba devuelve [prob_clase_0, prob_clase_1]
        # Tomamos la probabilidad de clase positiva, de que haya muerto 'death0=1' (urgencia)
        urgencia = modelo.predict_proba(features_completas)[0, 1]
        
        return float(urgencia) # me da la probabilidad de morir/urgencia
    
   

    # GENERACIÓN DE INFORME FINAL
    def guardar_todos_los_datos(self, session_id: str, mascara: np.ndarray, 
                                caracteristicas_df: pd.DataFrame, urgencia: float, 
                                datos_clinicos: Dict) -> str:
        """
        Genera el informe final del diagnóstico con todos los datos consolidados:
        - Máscara de segmentación (para visualización)
        - CSV unificado (características + datos clínicos + urgencia)
        - Informe médico 
        
        Args:
            session_id: Identificador de la sesión
            mascara: Máscara binaria del tumor (array 2D)
            caracteristicas_df: DataFrame con características tumorales
            urgencia: Probabilidad de urgencia (0-1)
            datos_clinicos: Diccionario con datos clínicos del paciente
        
        Returns:
            Ruta del directorio con todos los archivos generados
        """
        
        # Crea directorio temporal para la sesión
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # ARCHIVO 1: MÁSCARA (para visualización)
        np.save(temp_dir / "mascara.npy", mascara)
        
        # ARCHIVO 2: CSV UNIFICADO
        # Copia las características en df_completo para no machacar
        df_completo = caracteristicas_df.copy()
        
        # Añade datos clínicos como columnas adicionales
        if datos_clinicos:
            for key, value in datos_clinicos.items():
                df_completo[key] = value if value is not None else 'NULL'
        
        # Añade nivel de urgencia
        df_completo['nivel_urgencia'] = urgencia
        
        # Guarda CSV completo
        df_completo.to_csv(temp_dir / "datos_completos.csv", index=False, na_rep='NULL')
        
        # ARCHIVO 3: INFORME MÉDICO 
        with open(temp_dir / "nivel_urgencia.txt", "w", encoding="utf-8") as f:
            f.write("INFORME DE RIESGO MRAI\n")
            f.write(f"Puntuación: {urgencia:.3f}\n") # Con 3 decimales
            
            # Interpretación basada en umbrales (semáforo clínico)
            if urgencia < 0.3:
                f.write(" URGENCIA BAJA: Seguimiento ambulatorio (3-6 meses)\n")
            elif urgencia < 0.7:
                f.write(" URGENCIA MODERADA: Evaluación prioritaria (<2 semanas)\n")
            else:
                f.write(" ALTA URGENCIA: Intervención  inmediata (24-48h)\n")
        
        return str(temp_dir)
    
 
    # LIMPIEZA DE DATOS DE SESIÓN
    
    def limpiar_datos_sesion(self, session_id: str):
        """
        Elimina todos los datos temporales de la sesión al finalizar la consulta.
        Importante para garantizar la privacidad del paciente.
        
        Args:
            session_id: Identificador de la sesión a eliminar
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        
        # Si el directorio existe, lo borra con todo su contenido
        if temp_dir.exists():
            shutil.rmtree(temp_dir)  # Eliminación recursiva



# FUNCIÓN PARA STREAMLIT (CACHE) que hace que guarde al cliente en memoria y no lo recargue una y otra vez.
@st.cache_resource
def get_dagster_client():
    """
    Decorador de Streamlit que mantiene el cliente en memoria caché.
    Evita recrear el cliente en cada interacción del usuario.
    
    Returns:
        Instancia única de DagsterClient (singleton)
    """
    return DagsterClient(base_url="http://localhost:3000")