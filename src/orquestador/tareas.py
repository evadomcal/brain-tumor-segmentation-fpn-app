"""
===========================================================================
RECETARIO DE MISIONES (JOBS) - EL PANEL DE CONTROL
===========================================================================
Este archivo define qué procesos queremos ejecutar y cómo se conectan 
unos con otros. Es el "Mapa de Carreteras" del proyecto.
===========================================================================
"""

# --- IMPORTACIONES ---
# 'job' es el encargado de empaquetar varias tareas en una sola misión ejecutable.
from dagster import job

# Importamos las piezas (assets) que definimos en el otro archivo.
# El punto '.' le dice a Python: "están aquí".
# NOTA: Asegúrate de que los nombres coincidan con los de tu archivo 'activos.py'.
from .activos import catalogo_maestro, imagenes_procesadas, dividir_dataset_balanceado

# --- MISIÓN 1: EL FLUJO TOTAL ---

@job # Marcamos esta función como una misión que aparecerá en el panel de Dagster.
def pipeline_completo():
    """
    ESTRATEGIA: Ejecutar la cadena de montaje entera.
    Se usa cuando queremos actualizar todo el sistema de una sola vez.
    """
    
    # 1. Ejecutamos el inventario y guardamos el resultado en la variable 'catalogo'.
    catalogo = catalogo_maestro()
    
    # 2. Le pasamos ese 'catalogo' a los 8 obreros de Dask para que limpien las fotos.
    procesadas = imagenes_procesadas(catalogo)
    
    # 3. Cogemos esas fotos limpias ('procesadas') y las repartimos en grupos de examen.
    dividir_dataset_balanceado(procesadas)


# --- MISIÓN 2: SOLO PREPARACIÓN ---

@job 
def solo_procesamiento():
    """
    ESTRATEGIA: Solo registrar y limpiar imágenes.
    Útil si queremos probar si la cámara del hospital funciona bien sin entrenar nada.
    """
    
    # 1. Hacemos la lista de fotos nuevas.
    catalogo = catalogo_maestro()
    
    # 2. Las limpiamos y las guardamos en formato rápido (.npy). 
    # Aquí el proceso se detiene; no se dividen para la IA.
    imagenes_procesadas(catalogo)


# --- INSTRUCCIONES DE LANZAMIENTO (TERMINAL) ---

# Si quieres arrancar la misión completa desde la consola negra de tu PC sin usar la web:
# dagster job execute -f src/orquestador/tareas.py -j pipeline_completo