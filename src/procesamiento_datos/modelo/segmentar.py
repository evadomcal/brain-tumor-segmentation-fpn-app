# PROCESAMIENTO_DATOS/MODELO/SEGMENTAR

# Este script es el encargado de poner a trabajar a la IA ya entrenada. 
# Su función es recibir imágenes nuevas, pasárselas al modelo (la U-Net) y 
# obtener como respuesta la 'máscara', que es el dibujo en blanco y negro 
# que indica dónde está el tumor.

# LIBRERÍAS NECESARIAS:
import torch              # Permite cargar el modelo (tensores) y usar la tarjeta gráfica
import numpy as np        # Para manejar las fotos como matrices y realizar cálculos matemáticos
import pandas as pd       # Para organizar los resultados finales en tablas (DataFrames)
from pathlib import Path   # Para las rutas de las carpetas y nombres de archivos
from sklearn.metrics import f1_score, jaccard_score, precision_recall_curve # Herramientas para medir la bondad del modelo UNet


# 1. Función para que la IA analice una sola imagen
def segmentar_imagen(model, imagen_npy_path, device='cpu', umbral=0.5):
    """
    Analiza una imagen y nos dice qué probabilidad hay de que cada píxel sea tumor
    """
    # Cargamos la imagen que ya está procesada
    img = np.load(imagen_npy_path)  # (Alto, Ancho, 3 canales)
    
    # Preparamos la imagen para la IA: 
    # La convertimos en 'tensor' (formato de PyTorch), reordenamos los canales 
    # y la enviamos a la CPU o a la Tarjeta Gráfica (GPU/device)
    # EJEMPLO: entra (256,256,3) y sale (1,3,256,256)
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    model.eval() # Ponemos el modelo en "modo evaluación" (no aprende)
    with torch.no_grad(): # Le decimos que no guarde memoria extra porque no estamos entrenando
        # La UNet analiza la foto y nos da unos valores brutos (logits)
        # squeeze quita los valores que valen 1: (1,1,256,256) a (256,256), una matriz plana
        logits = model(img_tensor).squeeze() 
        # Convertimos esos valores en probabilidades del 0 al 1 usando 'sigmoid'
        mascara_prob = torch.sigmoid(logits).cpu().numpy() 
    
    # Creamos la máscara final: si la probabilidad es mayor al umbral, es tumor (1)
    mascara_binaria = (mascara_prob > umbral).astype(np.uint8) 
    
    return mascara_prob, mascara_binaria

# 2. Función para procesar un grupo entero de imágenes automáticamente
def segmentar_lote(model, lista_imagenes, output_dir, device='cpu', umbral=0.5):
    """
    Analiza muchas fotos una tras otra y guarda los dibujos de los tumores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True) # Crea la carpeta donde guardaremos las predicciones
    
    resultados = []
    for img_path in lista_imagenes:
        img_path = Path(img_path)
        # Llamamos a la función anterior para analizar la foto actual
        prob, binaria = segmentar_imagen(model, img_path, device, umbral)
        
        # Le ponemos nombre al dibujo del tumor (mascara) y lo guardamos
        # .stem le quita el formato final (ej: .npy)
        nombre_mascara = output_dir / f"{img_path.stem}_mask.npy"
        np.save(nombre_mascara, binaria)
        
        # Guardamos en una lista si la UNet ha encontrado tumor o no
        resultados.append({
            'imagen': str(img_path), # ruta de la imagen
            'mascara': str(nombre_mascara), # ruta de la máscara
            'tiene_tumor': int(binaria.sum() > 0) # Si hay píxeles blancos, es que hay tumor
        })
    
    return resultados