# TRANSFORMACIÓN DE DATOS

# En esta parte preparamos las imágenes médicas para que una red neuronal pueda aprender 
# a detectar tumores. Sigue este guión:

# 1. Carga la imagen y la máscara en bruto
# 2. Recorte: Quitamos el fondo negro, ahorra tiempo y espacio en la memoria.
# 3. Normalización: todas mismo idioma, así no importa el hospital de procedencia de la resonancia.
# 4. Aumentamos datos para que la red no memorice patrones.
# 5. Redimensionamos: 256x256, la red neuronal necesita un tamaño concreto.
 
# LIBRERÍAS NECESARIAS:
import cv2   # para leer y manipular imágenes
import numpy as np  # transforma las imágenes en matrices con las que podemos trabajar
import albumentations as A  # para data augmentation, crear más variantes de las imágenes

# 1. Función para transformar las imágenes
transformacion_aumento = A.Compose([
    A.HorizontalFlip(p=0.5), # Volteo horizontal, efecto espejo (50% de probabilidad)
    A.RandomRotate90(p=0.5), # Rotación de 90 grados (50% de probabilidad)
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3), # Desplaza, hace zoom o gira la imagen con probabilidad 0.3
])

# 2. Función para cargar y preparar cada imagen
def procesar_imagen_completo(fila_maestro, entrenando=True):
    # Control de seguridad: nos aseguramos de que el registro del paciente existe
    if fila_maestro is None:
        raise ValueError("fila_maestro es None")
    
    # Verificamos que la ruta de la imagen esté presente en el catálogo maestro
    if 'ruta_imagen' not in fila_maestro:
        raise KeyError(f"fila_maestro no tiene 'ruta_imagen': {fila_maestro}")

    # CARGA DE IMAGEN: Leemos la resonancia RM con sus 3 canales (Pre, FLAIR, Post)
    # Usamos IMREAD_UNCHANGED para mantener la profundidad de bits original
    img = cv2.imread(fila_maestro['ruta_imagen'], cv2.IMREAD_UNCHANGED).astype(np.float32)

    # CARGA DE MÁSCARA: 
    if fila_maestro['ruta_mascara'] is not None:
        # La cargamos en escala de grises (blanco para tumor, negro para fondo)
        mask = cv2.imread(fila_maestro['ruta_mascara'], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        # Si el paciente no tiene máscara, creamos un lienzo negro del mismo tamaño mg.shape[:2]= 256x256
        mask = np.zeros(img.shape[:2], dtype=np.float32)


    # RECORTE AUTOMÁTICO
    # Buscamos todos los píxeles que no sean negros (valor > 0) en cualquier canal
    puntos_cerebro = np.argwhere(img.max(axis=2) > 0) #Busca en la matriz ancho x alto x 3 canales los pixeles que no son 0 en los tres canales
    
    if puntos_cerebro.size > 0:
        # Encontramos los límites: mínimo y máximo en ejes Y (filas) y X (columnas)
        y_min, x_min = puntos_cerebro.min(axis=0) #Buscamos la esquina superior izquierda que no tenga 0 
        y_max, x_max = puntos_cerebro.max(axis=0) #Buscamos la esquina inferior derecha  que no tenga 0
        
        # Recortamos tanto la imagen como la máscara usando esos límites
        img = img[y_min:y_max+1, x_min:x_max+1] #Recortamos el cuadrado de la imagen (le sumamos 1 porque empieza por 0)
        mask = mask[y_min:y_max+1, x_min:x_max+1] #Igual en la máscara

    # NORMALIZACIÓN Z-SCORE POR CANAL 
    for i in range(3): # Iteramos por los 3 canales (0, 1 y 2)
        canal = img[:, :, i]
        pixeles_validos = canal[canal > 0] # Seleccionamos solo píxeles del cerebro
        
        if pixeles_validos.size > 0:
            media = pixeles_validos.mean() # Calculamos el promedio del canal
            std = pixeles_validos.std()   # Calculamos la desviación estándar
            # Aplicamos la fórmula: (valor - media) / desviación
            img[:, :, i] = (canal - media) / (std + 1e-8)
            # Nos aseguramos que el fondo recortado siga siendo 0 absoluto
            img[canal == 0, i] = 0

    # AUMENTO DE DATOS (DATA AUGMENTATION)
    if entrenando:
        # Convertimos la máscara a binaria (0 o 1) antes de transformar
        mask = (mask > 0).astype(np.float32)
        # Aplicamos las transformaciones aleatorias a ambos archivos a la vez
        resultado = transformacion_aumento(image=img, mask=mask)
        img, mask = resultado['image'], resultado['mask']

    # REDIMENSIONADO FINAL
    # Como el recorte cambia el tamaño, redimensionamos a un estándar (ej. 256x256) para el modelo
    img = cv2.resize(img, (256, 256)) 
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) #INTER_NEAREST para que la mascara se difumine

    return img, mask