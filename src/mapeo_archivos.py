# IMPORTACIÓN, LIMPIEZA Y TRANSFORMACIÓN DE DATOS

# En esta primera parte, vamos a importar y transformar nuestros datos de modo que al final
# obtengamos un diccionario o catálogo con el que podamos empezar a analizar y construir
# el modelo en cuestión que nos permita identificar si un paciente tiene o no un tumor cerebral.

# LIBRERÍAS NECESARIAS:
import os            # Permite navegar por las carpetas del ordenador
import pandas as pd  # Para manejar tablas de datos (DataFrames)
import numpy as np   # Para cálculos matemáticos y manejar las fotos como matrices
from pathlib import Path # Ayuda a que las rutas de los archivos funcionen en Windows y Mac sin fallos por / o \
from PIL import Image    # Para abrir y manipular las imágenes reales
import warnings      # Para que Python no nos llene la pantalla de avisos
warnings.filterwarnings('ignore') # Ignorar esos avisos

# 1. CONFIGURACIÓN DE RUTAS
# 'r' antes de la comilla significa "texto en bruto" para que las barras invertidas \ no den error
ruta_base = r"Trabajo\DATOS" 

# Ruta de los datos exportados de kaggle: "data.csv"
ruta_csv = os.path.join(ruta_base, "data.csv")

# 2. CARGA INICIAL DE DATOS CLÍNICOS
# Leemos "data.csv" y lo convertimos en una tabla llamada 'df_clinico'
df_clinico = pd.read_csv(ruta_csv) 

# Mostramos cuántos pacientes tenemos (110) y las variables disponibles.
print(f"   Pacientes en data.csv: {len(df_clinico)}") 
print(f"   Columnas disponibles: {df_clinico.columns.tolist()}") 

# 3. FUNCIÓN PARA DETECTAR lOS CANALES DISPONIBLES
def detectar_secuencias_disponibles(ruta_imagen, umbral_similitud=0.95):
    """
    Esta función mira si la foto tienen los 3 canales (Pre, Flair, Post)
    o si alguna es una copia de la otra porque faltaba el dato.Partimos de que todas
    tienen Flair.
    """
    try:
        # Abrimos la foto y la convertimos en una matriz para poder trabajar
        img = np.array(Image.open(ruta_imagen)).astype(np.float32)
        
        # Las fotos médicas tienen 3 "canales" 
        # El canal 1  es la secuencia FLAIR (todas la tienen)
        # El canal 0 es la secuencia PRE-contraste: antes del líquido de contraste
        # El canal 2 es la secuencia POS-contraste: tras el líquido de contraste en sangre

        # CANAL FLAIR
        canal_flair = img[:,:,1]
        
        # Si la foto está totalmente en negro (todo ceros), asumimos TRUE para no borrarla por error
        if np.all(canal_flair == 0):
            return True, True 
        
        # CANAL PRE-CONTRASTE
        canal_pre = img[:,:,0]
        if np.all(canal_pre == 0): 
            pre_disponible = False  # Si está vacío, lo definimos como no disponible
        else:
            # Comparamos el canal PRE con el FLAIR
            # Si son casi iguales (correlación > 0.95), han copiado una encima de otra. Luego, no hay Pre.
           
            # Pasamos de 2D a 1D.
            # Convertimos la imagen en una lista larga de números para poder compararla 
            # con el canal flare, pues cada imagen es un cuadrado 256x256.
            pre_flat = canal_pre.flatten() 
            flair_flat = canal_flair.flatten()
            if np.std(pre_flat) > 0 and np.std(flair_flat) > 0: # Si hay datos, desviación típica > 0
                correlacion_pre = np.corrcoef(pre_flat, flair_flat)[0,1] # Calculamos la similitud
                pre_disponible = correlacion_pre < umbral_similitud # Si son distintos TRUE, si no, FALSE (no hay Pre)
            else:
                pre_disponible = True # Si no tiene datos, asumimos que el pre está disponible

            # RESUMEN: Este bloque es nuestro control de calidad. Como en el dataset a 
            # veces faltan secuencias y se rellenan con copias, usamos la correlación
            # estadística. Si la capa PRE es casi idéntica a la capa FLAIR, el sistema 
            # detecta que es una copia y marca que esa secuencia no está disponible 
            # realmente para ese paciente. Así evitamos que la IA aprenda con datos
            # repetidos o falsos.
        
        # CANAL POST-CONSTRASTE
        # Repetimos el procedimiento
        canal_post = img[:,:,2]
        if np.all(canal_post == 0):
            post_disponible = False
        else:
            post_flat = canal_post.flatten()
            if np.std(post_flat) > 0 and np.std(flair_flat) > 0:
                correlacion_post = np.corrcoef(post_flat, flair_flat)[0,1]
                post_disponible = correlacion_post < umbral_similitud
            else:
                post_disponible = True
        
        return pre_disponible, post_disponible # Nos devuelve: ¿Tiene Pre?, ¿Tiene Post?
        
    except Exception as e:
        # Si la foto está rota o da error al abrir, decimos que "está todo bien" para no frenar el programa
        print(f"   Error detectando secuencias: {e}")
        return True, True

# 4. FUNCIÓN PARA CREAR EL ÍNDICE 
def crear_indice_archivos(ruta_base, df_clinico, detectar=True):
    registros = [] # Aquí guardaremos la información de cada foto que encontremos
    pacientes_procesados = 0
    estadisticas = {'pre_faltan': 0, 'post_faltan': 0, 'ambos_faltan': 0}
    
    # Empezamos a entrar en cada carpeta, correspondiente a cada paciente, de la ruta base ("DATOS")
    for carpeta in os.listdir(ruta_base): 
        ruta_carpeta = os.path.join(ruta_base, carpeta) 
        
        # Nuestras carpetas empiezan por TCGA_
        # Si no es una carpeta: .isdir comprueba si es un directorio o carpeta
        # Si no empieza por "TCGA_": .startswith
        # en esos casos ignoramos y seguimos con el resto 
        if not os.path.isdir(ruta_carpeta) or not carpeta.startswith("TCGA_"):
            continue
        
        # Ej: TCGA_DU_5849_19950405
        # 1. Separo por '_':  ['TCGA', 'DU', '5849', '19950405']
        partes = carpeta.split('_') 
        # 2. Une las primeras 3 partes con '_' para formar el ID del paciente
        id_paciente = '_'.join(partes[:3])  # Resultado: 'TCGA_DU_5849'
        fecha = partes[3] # La cuarta parte es la fecha del estudio
        
        # Si el paciente cuya carpeta estamos tratando no está en data.csv (columna Patient), lo ignoro
        if id_paciente not in df_clinico['Patient'].values:
            continue
        
        pacientes_procesados += 1
        

        # De la carpeta, nos quedamos solo con los que terminan en .tif, imágenes y máscaras
        archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.tif')] 

        # Separamos las imágenes reales de las máscaras
        imagenes = [f for f in archivos if not f.endswith('_mask.tif')] 
        
        # Analizamos, con la función previa, si este paciente tiene Pre y Post.
        tiene_pre = True
        tiene_post = True

        # Si está activada la detección automática y hay al menos una imagen
        if detectar and len(imagenes) > 0:

            # Basta con mirar la primera imagen del paciente o su "primer corte"
            primer_corte = os.path.join(ruta_carpeta, imagenes[0])
            tiene_pre, tiene_post = detectar_secuencias_disponibles(primer_corte)

            # Sumamos las estadísticas para el informe final
            if not tiene_pre: 
                estadisticas['pre_faltan'] += 1
            if not tiene_post: 
                estadisticas['post_faltan'] += 1
            if not tiene_pre and not tiene_post: 
                estadisticas['ambos_faltan'] += 1
        
        # Recorremos cada imagen del paciente en análisis
        for img in imagenes:
            # Construimos el nombre de su máscara correspondiente
            mask = img.replace('.tif', '_mask.tif') 
            # Establecemos las rutas de las imagenes y sus mascaras
            ruta_img = os.path.join(ruta_carpeta, img)
            ruta_mask = os.path.join(ruta_carpeta, mask)
            
            # Sacamos el número de corte del cerebro 
            # Ej: TCGA_CS_4941_19960909_1.tif, el corte se corresponde al número 1.
            try:
                num_corte = int(img.replace('.tif', '').split('_')[-1])
            except:
                num_corte = -1  # si algo falla, ponemos -1
            
            # ANALIZAMOS LA MÁSCARA 
            try:
                mask_array = np.array(Image.open(ruta_mask))

                # Si hay algún píxel mayor que 0, hay tumor según la RM.
                tiene_tumor = np.any(mask_array > 0)
                tamaño_tumor = np.sum(mask_array > 0) if tiene_tumor else 0
               
            except:
                tiene_tumor = None
                tamaño_tumor = None 

            # Registramos la información obtenida
            registros.append({
                'id_paciente': id_paciente,
                'num_corte': num_corte,
                'ruta_imagen': ruta_img,
                'ruta_mascara': ruta_mask,
                'tiene_pre': tiene_pre,
                'tiene_post': tiene_post,
                'mascara_tiene_tumor': tiene_tumor, 
                'tamaño_tumor_pixeles': tamaño_tumor
            })
    
    # Convertimos toda la lista de registros en una tabla de Pandas
    df = pd.DataFrame(registros)
    print(f"\n  Pacientes procesados: {pacientes_procesados}")
    return df

# EJECUTAMOS LA CREACIÓN DEL ÍNDICE
df_archivos = crear_indice_archivos(ruta_base, df_clinico, detectar=True)


# 5. CREAMOS EL CATÁLOGO MAESTRO (merge)
# Pegamos df_archivos con data.csv usando el ID de paciente.

# Tabla izquierda (df_archivos): tiene una fila por cada imagen
# Tabla derecha (df_clinico): tiene una fila por cada paciente (mas pequeña)

df_catalogo = pd.merge(
    df_archivos,
    df_clinico,
    left_on='id_paciente', # df_archivos
    right_on='Patient',   # data.csv
    how='left'
)

# 6. RESUMEN FINAL
cortes_con_tumor = df_catalogo['mascara_tiene_tumor'].sum()
print(f" Cortes CON tumor detectados: {cortes_con_tumor}")
print(f" Cortes SIN tumor detectados: {len(df_catalogo)-cortes_con_tumor}")

# Guardamos todo un archivo final con el que trabajaremos a partir de ahora.
ruta_catalogo = os.path.join(ruta_base, "catalogo_maestro_final.csv")
df_catalogo.to_csv(ruta_catalogo, index=False)
print(f"   ¡Catálogo guardado con éxito!")