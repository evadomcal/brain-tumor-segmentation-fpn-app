# CREAMOS LA COMUNICACION ENTRE LA APLICACION Y EL DAGSTER

import requests  # se usa para llamar a APIs (enlace online entre una base de datos y mi codigo)
import json # para manejar datos en formato json
import pandas as pd
import numpy as np
from PIL import Image # para trata con imagenes
import io
import base64 # para tratar imagenes como texto, que es lo que necesita la API (no puede ver un numpy, solo texto)
from typing import Tuple, Dict, Any # para indicar tipos de datos
import streamlit as st # para la app 
from pathlib import Path

class DagsterClient:
    """Cliente para comunicarse con la API de Dagster"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        
    def ejecutar_activo_prediccion_mascara(self, imagen_tif_bytes: bytes, 
                                        datos_clinicos: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Ejecuta el activo 'prediccion_mascara' de Dagster
        
        Args:
            imagen_tif_bytes: Bytes de la imagen TIF
            datos_clinicos: Diccionario con datos del formulario
            
        Returns:
            máscara (numpy array), DataFrame con características
        """
        # Opción 1: Si los activos están en el mismo proceso Python
        if st.session_state.get("modo_directo", False):
            return self._ejecutar_local(imagen_tif_bytes, datos_clinicos)
        
        
    def _ejecutar_local(self, imagen_tif_bytes: bytes, datos_clinicos: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
        """Ejecuta los activos directamente (mismo proceso)"""
        from orquestador.activos import prediccion_mascara, analisis_datos
        
        # Convertir bytes a imagen
        imagen = Image.open(io.BytesIO(imagen_tif_bytes))
        imagen_array = np.array(imagen)
        
        # Ejecutar activo de predicción
        resultado = prediccion_mascara(imagen_array, datos_clinicos)
        mascara = resultado["mascara"]
        df_caracteristicas = resultado["caracteristicas"]
        
        return mascara, df_caracteristicas
    
    def _ejecutar_via_api(self, imagen_tif_bytes: bytes, datos_clinicos: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
        """Ejecuta via API REST de Dagster"""
        # Codificar imagen en base64
        imagen_b64 = base64.b64encode(imagen_tif_bytes).decode('utf-8')
        
        # Construir payload
        payload = {
            "imagen_b64": imagen_b64,
            "datos_clinicos": datos_clinicos
        }
        
        # Llamar al endpoint de materialización de activos
        response = requests.post(
            f"{self.base_url}/materialize/prediccion_mascara",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Error en Dagster: {response.text}")
        
        data = response.json()
        
        # Decodificar máscara
        mascara_bytes = base64.b64decode(data["mascara_b64"])
        mascara = np.load(io.BytesIO(mascara_bytes))
        
        # Cargar características
        df_caracteristicas = pd.read_json(data["caracteristicas_json"])
        
        return mascara, df_caracteristicas
    
    def ejecutar_activo_analisis(self, df_caracteristicas: pd.DataFrame, 
                                datos_clinicos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el activo 'analisis_datos' de Dagster
        """
        if st.session_state.get("modo_directo", False):
            from dagster_app.assets import analisis_datos
            resultado = analisis_datos(df_caracteristicas, datos_clinicos)
            return resultado
        else:
            # Preparar datos para API
            payload = {
                "caracteristicas_json": df_caracteristicas.to_json(),
                "datos_clinicos": datos_clinicos
            }
            
            response = requests.post(
                f"{self.base_url}/materialize/analisis_datos",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Error en análisis: {response.text}")
            
            return response.json()

    # ======================================================================
    # NUEVA CONEXIÓN: EJECUCIÓN LOCAL DEL ACTIVO DE R (SUBPROCESO R-4.5.2)
    # ======================================================================
    def ejecutar_activo_analisis_r(self, df_caracteristicas: pd.DataFrame) -> str:
        """
        Ejecuta el script de R de tu amigo usando el sistema de subprocesos
        envolviéndolo en la lógica interna del cliente de Dagster.
        """
        import subprocess

        # 1. Creamos una carpeta temporal en el mismo directorio para guardar el CSV que leerá R
        temp_dir = Path(__file__).parent / "temp_datos"
        temp_dir.mkdir(exist_ok=True)
        ruta_csv = temp_dir / "caracteristicas_actuales.csv"
        df_caracteristicas.to_csv(ruta_csv, index=False)
        
        # 2. Carpeta donde R guardará el informe txt y las 4 imágenes PNG
        output_dir = temp_dir / "resultados_r"
        output_dir.mkdir(exist_ok=True)
        
        # 3. Localizamos el script de R subiendo un nivel en la estructura de carpetas
        script_r = Path(__file__).parent.parent / "analisis_r.R"
        
        # 4. CONFIGURACIÓN ADAPTADA A TU ENTORNO LOCAL (Versión de R-4.5.2)
        cmd = [
            r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe', # <-- Tu ejecutable local
            str(script_r),
            str(ruta_csv),
            str(output_dir)
        ]
        
        # 5. Ejecutamos el pipeline de R de forma transparente
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        
        if result.returncode != 0:
            raise Exception(f"Error en el motor R-4.5.2 de Dagster: {result.stderr}")
            
        # Devolvemos la ubicación física de las gráficas creadas por R
        return str(output_dir)


# Instancia global del cliente
@st.cache_resource
def get_dagster_client():
    return DagsterClient(base_url=st.secrets.get("DAGSTER_URL", "http://localhost:3000"))