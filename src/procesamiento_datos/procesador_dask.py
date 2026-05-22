# PROCESAMIENTO_DATOS/PROCESADOR_DASK

# En lugar de procesar las cientos de imágenes una por una (lo que tardaría horas), este código
#  divide el trabajo en montones pequeños y los reparte entre varios núcleos para que trabajen 
# todos a la vez. Así, aprovechamos toda la potencia del ordenador para limpiar y preparar
#  las imágenes en menor tiempo.

# LIBRERÍAS NECESARIAS: 
from dask.distributed import Client, LocalCluster  # Para procesamiento en paralelo
import numpy as np    # Para guardar y leer las fotos como matrices (.npy)
import pandas as pd   # Para organizar la lista de pacientes y sus rutas
import os             # Para crear carpetas y gestionar archivos en el ordenador
import traceback      # Para que, si algo falla, nos diga exactamente en qué línea ha sido el error


# 1. Configuración del "Jefe de Equipo" (Dask)
class DaskBrainProcessor: 
    
    def __init__(self, n_workers=1): # 
        self.n_workers = n_workers   # Guardamos el número de workers en la clase
        self.cluster = LocalCluster( # Creamos un grupo de trabajo 
            n_workers=n_workers,     # Número de trabajadores que trabajarán en paralelo
            threads_per_worker=2,    # Capacidad de cada ayudante
            memory_limit='4GB'       # Tope de memoria para no bloquear el equipo
        )
        self.client = Client(self.cluster) # El canal para enviar las tareas a los trabajadores

    # 2. FUNCIÓN DE REPARTO DE TRABAJO
    def procesar_todas_imagenes(self, catalogo_df): 
        """ Organiza el reparto de imágenes y las procesa en equipo """
        
        # Creamos la carpeta donde se guardarán las imágenes procesadas
        os.makedirs("datos_procesados", exist_ok=True) 
        
        # Reparto de trabajo: Calculamos cuántas imágenes le tocan a cada trabajador
        # Como mínimo, una para cada.
        chunk_size = max(1, len(catalogo_df) // self.n_workers) 
        
        # Creamos la información de cada paciente que hay que pasarle a los trabajadores
        chunks = [] # imágenes que le corresponden a cada trabajador
        for i in range(0, len(catalogo_df), chunk_size):  # Saltos del tamaño de los lotes a repartir (chunk_size)
            
            # Vamos cortando el catalogo_df en trozos pequeños
            chunks.append(catalogo_df.iloc[i:i+chunk_size])  # Lista de listas (lotes de imágenes)
            
        # Tarea que ejecutará cada trabajador de forma independiente:
        def procesar_lote(lote_df):
            resultados = [] # Aquí cada trabajador guardará su lista de tareas terminadas
            import sys # maneja el sistema de python (librerías, rutas internas...)
            import os # para gestionar carpetas y archivos
            from pathlib import Path
            
            # 1. ORIENTACIÓN: le decimos a cada trabajador la ruta de sus imágenes, al trabajar de manera independiente

            src_path = Path(__file__).parent.parent
            sys.path.insert(0,str(src_path))
            # Importamos la función de transformación de imágenes que ya habíamos creado
            from src.transformar_datos import procesar_imagen_completo 
            
            # 2. TRABAJO EN SERIE: El trabajador recorre su lote de imágenes asignado
            # iterrows devuelves pares (indice, fila) pero solo nos interesa la fila
            # enumerate aigna un índice idx para que el trabajador no se pierda 
            for idx, (_, fila) in enumerate(lote_df.iterrows()): 
                id_paciente = fila.get('id_paciente') # Identificamos al paciente
                num_corte = fila.get('num_corte')    # Identificamos el número de corte
                    
            # 3. PROCESAMIENTO: Aplicamos la limpieza, recorte y normalización
                img, mask = procesar_imagen_completo(fila, entrenando=True) 
                    
                if img is not None:
            # 4. GUARDADO: guardamos la imagen y la máscara ya procesadas en formato .npy
                    nombre_archivo = f"datos_procesados/{id_paciente}_{num_corte}.npy"
                    np.save(nombre_archivo, img)
            
                    nombre_mascara = f"datos_procesados/{id_paciente}_{num_corte}_m.npy"
                    np.save(nombre_mascara, mask)    
                    
            # 5. REPORTE: cada trabajador guarda una lista de sus resultados 
            # lista de tantas listas como el tamaño del lote (imágenes asociadas a cada trabajador)
                    resultados.append({
                        'paciente': id_paciente,
                        'num_corte': num_corte,
                        'ruta_procesada': nombre_archivo,
                        'ruta_mascara' : nombre_mascara,
                        'tiene_tumor': fila.get('mascara_tiene_tumor', False)
                    })
            
                    
                    return resultados
            
            # ENVÍO: Repartimos los lotes de imágenes, "chunks" es una listas de listas
            futures = [] # Lista de "promesas"
            for i, chunk in enumerate(chunks):
                if len(chunk) > 0:
                    future = self.client.submit(procesar_lote, chunk) # uso de la función definida arriba
                    futures.append(future) # Guardamos las "promesas" de cumplimiento
            
            # RECOGIDA de trabajos realizaados
            resultados = [] 
            for future in futures:
                # .result(): esperamos a que todas las promesas se realicen.
                resultado = future.result() # es una lista de listas (una por cada imagen procesadas)
                
                # Unimos los resultados de este trabajador
                resultados.extend(resultado) # lista de listas, al usar extend une todos los elementos que eran listas también
            
            return resultados # Devolvemos la lista completa de todas las fotos del proyecto
    
    def shutdown(self): 
        # Al terminar, "apagamos las luces del taller" y liberamos la memoria RAM.
        # Es fundamental para que el ordenador no se quede lento después.
        self.client.close()
        self.cluster.close()