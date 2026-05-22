#PROCESAMIENTO_DATOS/MODELO/CARACTERISTICAS

# Este script es el traductor que convierte las imágenes y las máscaras en datos numéricos. 
# Una vez localizado el tumor, analizamos su forma, su brillo y su textura para extraer 'biomarcadores'. 
# Estos números son los que realmente permiten a un médico decidir si el tumor es agresivo o no, 
# basándose en medidas objetivas del tejido.

# LIBRERÍAS NECESARIAS:
import numpy as np         # Para realizar cálculos matemáticos sobre los píxeles (medias, percentiles, etc.)
import pandas as pd        # Para organizar los resultados en una tabla final (dataset)
from skimage.measure import regionprops, label # Herramientas para medir el tamaño y la forma del tumor 
from skimage.feature import graycomatrix, graycoprops # Algoritmos para analizar la textura y "desorden" del tejido tumoral
from pathlib import Path    # Para gestionar las rutas de las imágenes y máscaras


# Definimos la función para extraer las características del tumor
def extraer_caracteristicas(imagen, mascara_binaria):
   """
    Recibe:
    imagen: array (H, W, 3)
    mascara_binaria: array (H, W) - 0 fondo, 1 tumor

    Devuelve un diccionario con 7 biomarcadores:
    - area (píxeles)
    - perimetro (píxeles)
    - circularidad
    - intensidad_media_post
    - intensidad_minima_post
    - percentil_95_flair
    - textura_contraste

    """
    
    # CONTROL DE SEGURIDAD: Si la IA no detectó tumor, devolvemos todo a cero
   if mascara_binaria.sum() == 0:
        return {
            'area': 0, 'perimetro': 0, 'circularidad': 0,
            'intensidad_media_post': 0, 'intensidad_minima_post': 0,
            'percentil_95_flair': 0, 'textura_contraste': 0,
        }

    # 1. MORFOMÉTRICAS (Forma y Tamaño) 
   labeled = label(mascara_binaria) # Separa focos del tumor distintas 
   props = regionprops(labeled)[0]  # Analiza el foco más grande (el tumor principal)
    
   area = props.area
   perimetro = props.perimeter
    # Circularidad: mide qué tan "redondo" es. Cerca de 1 es benigno, cerca de 0 es invasivo
   circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0 

    # 2. INTENSIDAD POST-CONTRASTE (Actividad metabólica)
   canal_post = imagen[:, :, 2]  # Canal con contraste (POST)
    # Extraemos solo los píxeles donde la máscara dice que hay tumor
   valores_post = canal_post[mascara_binaria > 0] 

    # Intensidad media: indica grado de malignidad (más brillo = peor)
   intensidad_media_post = float(valores_post.mean()) 
    # Intensidad mínima: si es muy baja, hay NECROSIS (tejido muerto por falta de riego), signo de alta agresividad
   intensidad_minima_post = float(valores_post.min()) 

    # 3. EDEMA E INFILTRACIÓN (canal FLAIR) 
   canal_flair = imagen[:, :, 0] 
   valores_flair = canal_flair[mascara_binaria > 0]
    # Percentil 95: captamos la zona de mayor inflamación (edema)
   percentil_95_flair = float(np.percentile(valores_flair, 95)) 

    # 4. TEXTURA Y HETEROGENEIDAD 
    # Recortamos una "caja" (bbox) alrededor del tumor para ser más eficientes
   minr, minc, maxr, maxc = props.bbox 
   rdi = canal_post[minr:maxr, minc:maxc] # Región de interés de la imagen
   rdi_mask = mascara_binaria[minr:maxr, minc:maxc]  # Región de interés de la máscara
    
    # Normalizamos a 0-255 tonos de grises (formato imagen estándar) para poder calcular la textura
   rdi = rdi * rdi_mask  # solo donde hay tumor
   rdi = (rdi / rdi.max() * 255).astype(np.uint8) if rdi.max() > 0 else rdi.astype(np.uint8) 
    
    # Calculamos la matriz de co-ocurrencia (GLCM) para ver cómo cambian los píxeles vecinos
   if rdi.sum() > 0 and rdi.shape[0] > 1 and rdi.shape[1] > 1: # imagen no vacía
        try:
            # Compara con los píxeles a distancia 1, a la derecha (angles=0) con dirección bidimensional
            glcm = graycomatrix(rdi, distances=[1], angles=[0], levels=256, symmetric=True)

            # Contraste alto: tumor caótico/heterogéneo (malo). 
            # Contraste bajo: tumor uniforme (mejor pronóstico)
            textura_contraste = float(graycoprops(glcm, 'contrast')[0, 0]) # [0,0] porque solo usamos un ángulo y distancia
        except:
            textura_contraste = 0
   else:
        textura_contraste = 0
    
   return {
        'area': area, 'perimetro': perimetro, 'circularidad': circularidad,
        'intensidad_media_post': intensidad_media_post,
        'intensidad_minima_post': intensidad_minima_post,
        'percentil_95_flair': percentil_95_flair,
        'textura_contraste': textura_contraste,
    }

# Creamos una tabla final con las características o biomarcadores obtenidos
def generar_dataset_features(df_segmentacion, directorio_imagenes, directorio_mascaras, df_previa=None):
    """
	Cruza las imágenes, las características calculadas y el historial clínico.
    Crea una tabla final lista para análisis médico.
    """
    # Convertimos las rutas a objetos Path para que funcionen igual en Windows (\) o Linux/Mac (/).
    directorio_imagenes = Path(directorio_imagenes)
    directorio_mascaras = Path(directorio_mascaras)
    
    # Lista donde guardaremos los resultados de cada corte (cada fila de nuestra tabla)
    registros = []
    
    # Recorremos fila a fila el DataFrame de la segmentación (tras la predicción de la U-Net)
    for _, row in df_segmentacion.iterrows():
        
        # 1. Localizamos los archivos:
        # Construimos el nombre exacto de la imagen y de la máscara predicha 
        img_path = directorio_imagenes / Path(row['ruta_procesada']).name
        mask_path = directorio_mascaras / f"{row['paciente']}_{row['num_corte']}_mask.npy"
        
        # Si falta algún archivo en el disco, saltamos ese corte y seguimos
        if not img_path.exists() or not mask_path.exists():
            continue 
        
        # 2. Cargamos los datos:
        # imagen (canales FLAIR, PRE, POST) y máscara (ceros y unos) que son matrices numpy
        img = np.load(img_path)
        mascara = np.load(mask_path)
        
        # 3. Obtenemos los biomarcadores:
        features = extraer_caracteristicas(img, mascara)
        
        # 4. Añadimos quién es el paciente y qué número de corte estamos analizando
        features['paciente'] = row['paciente']
        features['num_corte'] = row['num_corte']
        
        # 5. Cruzamos con los datos útiles del paciente que el médico introduzca en la web:
        if df_previa is not None:
            # Buscamos en el historial clínico las filas del paciente en cuestión
            previa_paciente = df_previa[df_previa['id_paciente'] == row['paciente']]
            
            if len(previa_paciente) > 0:
                # Si existe, copiamos todas las columnas clínicas al registro del tumor
                for col in df_previa.columns:
                    if col != 'paciente':
                        features[f'{col}'] = previa_paciente.iloc[0][col] # iloc[0] pues solo hay una fila con ese paciente
            else:
                # Si el paciente no está en el historial clínico, rellenamos con NaN 
                for col in df_previa.columns:
                    if col != 'paciente':
                        features[f'{col}'] = np.nan
        
        # Guardamos el diccionario completo 
        # Es una lista de diccionarios, uno por corte, por tanto habrá datos repetidos al repetirse pacientes.
        registros.append(features)
    
    # Convertimos la lista de registros en una tabla de Pandas (DataFrame) para poder trabajarla
    return pd.DataFrame(registros)



# INTERPRETACION  DE LAS CARACTERÍSTICAS:

# ÁREA: un tumor pequeño tiene menor área que uno más grande
# PERÍMETRO: un tumor pequeño tiene menor perímetro que uno más grande
# CIRCULARIDAD: si es redondo es benigno, si es muy irregular es más invasivo.
# si es alta, empuja al tejido sano, si es baja, invade al tejido sano

# MEDIDAS DE INTESIDAD (post-contrast)
# El líquido de contraste (post) solo entra donde la barrera ematoncefalica esta rota, y los tumores la rompen,
# de ahi que estudiemos este canal para la intensidad.
# INTENSIDAD MEDIA: me dice si el tumor es de bajo grado o maligno

# INTENSIDAD MINIMA: si la minima es baja, hay necrosis en el tumor (crece tan rapido que a las celulas
# centrales nos les llega la sangre y mueren). El contraste se inyecta en la sangre luego no les llega nada mas
# que a las celulas malignas de la frontera de la barrera, de ahi a que sea bajo.
# Mientras más baja sea, peor es el tumor

# PERCENTIL 95 DE FLAIR: nos da el 5% mas brillante (la zona de edema)
# Flair: hace brillar el edema, la reaccion del cerebro alrededor del tumor
# Si es bajo: hay poco edema 
# Si es alto: el tumor se ha ido infiltrando y hay muchos edemas, es muy malo.

# TEXTURA (se hace con el post) cómo se organizan los píxeles. 
# CONTRASTE: Estoy contrastando tumores muy organizados (tumor igual por todas partes) con tumores que estén desorganizados
# (necrosis, cúmulos...) y que por tanto en función de la zona del tumor sera de una forma u otra.
# A mayor contraste peor, más difícil de controlar el tumor