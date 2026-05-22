# ANALISIS/METRICAS_MODELO.PY

# Este fichero es el encargado de medir la bondad del modelo U-Net, comparando las máscaras predichas del modelo con 
# la máscara real.
# Primero, ajusta los tamaños de las imágenes para que coincidan perfectamente y las analiza.
# Después, calcula métricas de precisión de cada imagen y mide errores en los bordes con la distancia de Hausdorff. 
# Finalmente, genera medidas globales con la media y la desviación típica de las anteriores y crea un informe en
# CSV y texto con todos los resultados para el médico. 


# LIBRERÍAS NECESARIAS:
import numpy as np          # Para manejar las imágenes como matrices
import pandas as pd         # Para crear las tablas de resultados y exportarlas a CSV.
from pathlib import Path    # Para gestionar las rutas de las carpetas sin problemas de multiplataformas.

from scipy.spatial.distance import directed_hausdorff # Mide la distancia máxima entre el borde real y el predicho.
from scipy.spatial import KDTree # Permite encontrar puntos cercanos muy rápido
from typing import Dict, Tuple # Para indicar qué tipo de datos devuelven las funciones
import warnings
warnings.filterwarnings('ignore') # Para ocultar warnings

# 1. CARGA Y PREPARACIÓN DE IMÁGENES 
# Recibe las rutas de la máscara real y la predicha y devuelve una dupla de matrices numpy.
def cargar_mascaras(ruta_mascara_original: Path, ruta_mascara_predicha: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carga archivos .npy y asegura que sean binarios y del mismo tamaño"""
    original = np.load(ruta_mascara_original)
    predicha = np.load(ruta_mascara_predicha)
    
    # Binarización: Convertimos cualquier valor en 0 (fondo) o 1 (tumor)
    original = (original > 0).astype(np.uint8)
    predicha = (predicha > 0.5).astype(np.uint8)
    
    # Igualamos los tamaños: comprueba si las dimensiones (alto y ancho) son diferentes.
    if original.shape != predicha.shape:
        from scipy.ndimage import zoom  # Importamos la herramienta para cambiar el tamaño.
        
        # Calcula cuánto hay que modificar la máscara para que coincida con la original.
        zoom_factor = (original.shape[0] / predicha.shape[0], 
                    original.shape[1] / predicha.shape[1])
        
        # Redimensiona la máscara predicha. 
        # 'order=0': usa el vecino más cercano
        predicha = zoom(predicha, zoom_factor, order=0)
        
        # Aseguramos que los píxeles toman valores 0-1.
        predicha = (predicha > 0.5).astype(np.uint8)
    
    return original, predicha  # Ya podemos compararlas y ver la bondad del modelo U-Net

# 2. CÁLCULO DE MÉTRICAS POR IMAGEN 
def calcular_metricas_por_imagen(original: np.ndarray, predicha: np.ndarray, imagen_id: str) -> Dict:
    """Calcula el rendimiento (Dice, IoU, Hausdorff) para un solo corte o imagen de MRI"""
    
    # Aplanamos las matrices a una sola fila para comparar píxel a píxel
    y_true = original.flatten()
    y_pred = predicha.flatten()
    
    # Matriz de Confusión: Contamos aciertos y errores
    VP = np.sum((y_true == 1) & (y_pred == 1)) # Verdaderos Positivos
    FP = np.sum((y_true == 0) & (y_pred == 1)) # Falsos Positivos
    VN = np.sum((y_true == 0) & (y_pred == 0)) # Verdaderos Negativos
    FN = np.sum((y_true == 1) & (y_pred == 0)) # Falsos Negativos
    
    # Cálculos estadísticos: medidas de precisión del modelo
    sensibilidad = VP / (VP + FN) if (VP + FN) > 0 else 0 
    especificidad = VN / (VN + FP) if (VN + FP) > 0 else 0 
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0 
    fpr = FP / (FP + VN) if (FP + VN) > 0 else 0  # Tasa de falsos positivos
    fnr = FN / (FN + VP) if (FN + VP) > 0 else 0  # Tasa de falsos negativos
    exactitud = (VP + VN) / (VP + VN + FP + FN) if (VP + VN + FP + FN) > 0 else 0
    f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
    iou = VP / (VP + FP + FN) if (VP + FP + FN) > 0 else 0
    npv = VN / (VN + FN) if (VN + FN) > 0 else 0
    
	#Coeficiente de correlación de Matthew
    numerador = (VP * VN) - (FP * FN)
    denominador = np.sqrt((VP + FP) * (VP + FN) * (VN + FP) * (VN + FN))
    mcc = numerador / denominador if denominador > 0 else 0
	
    balanced_acc = (sensibilidad + especificidad) / 2
	
    # Distancia de Hausdorff: mide cuánto se aleja el contorno predicho del contorno real en píxeles.
    if original.sum() > 0 and predicha.sum() > 0:
        # np.where(original > 0): ubica qué pixeles son tumores y devuelve dos listas
        # Y: filas con tumor
        # X: columnas con tumor
        # np.column_stack(): lista de parejas (Y,X)
        coords_orig = np.column_stack(np.where(original > 0))
        coords_pred = np.column_stack(np.where(predicha > 0))
        
        # Cómo de lejos está el punto más alejado del tumor real de la predicción
        # Cuánto se le escapó a la IA
        h1 = directed_hausdorff(coords_orig, coords_pred)[0]  

        # Cómo de lejos está el punto más alejado de la predicción del tumor real
        # Cuánto se inventó la IA
        h2 = directed_hausdorff(coords_pred, coords_orig)[0]
        hausdorff = max(h1, h2) # Tomamos la distancia máxima entre bordes: error máximo
        
        tree_pred = KDTree(coords_pred)
        tree_orig = KDTree(coords_orig)
		
        dists_orig_to_pred = tree_pred.query(coords_orig)[0]
        dists_pred_to_orig = tree_orig.query(coords_pred)[0]
        avg_surface_distance = (np.mean(dists_orig_to_pred) + np.mean(dists_pred_to_orig)) / 2
	
    else:
        hausdorff = np.nan
        avg_surface_distance = np.nan
		
    # Empaquetamos todo en un diccionario
    resultado = {
        'imagen_id': imagen_id,
        'VP': int(VP), 'FP': int(FP), 'VN': int(VN), 'FN': int(FN),
        'sensibilidad': round(sensibilidad, 4),
        'especificidad': round(especificidad, 4),
        'precision': round(precision, 4),
        'exactitud': round(exactitud, 4),
        'f1_score': round(f1, 4),
        'dice': round(f1, 4),
        'iou': round(iou, 4),
        'mcc': round(mcc, 4),
        'npv': round(npv, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'area_real': int(original.sum()),
        'area_predicha': int(predicha.sum()),
        'hausdorff_distance': round(hausdorff, 2) if not np.isnan(hausdorff) else None,
        'avg_surface_distance': round(avg_surface_distance, 2) if not np.isnan(avg_surface_distance) else None,
    }
    
    return resultado

# 3. CÁLCULO DE MÉTRICAS GLOBALES 
def calcular_metricas_globales(df_metricas: pd.DataFrame) -> Dict:
    """Calcula la media de todos los resultados de todas las imágenes"""
    if df_metricas.empty: return {}
    
    metricas_globales = {}
    metricas_interes = ['sensibilidad', 'especificidad', 'precision', 'exactitud', 'f1_score', 'iou', 'mcc', 'balanced_accuracy']
    
    # Calculamos media y desviación estándar GLOBALES 
    for metrica in metricas_interes:
        if metrica in df_metricas.columns:
            metricas_globales[f'{metrica}_media'] = round(df_metricas[metrica].mean(), 4) # ej: sensibilidad_media
            metricas_globales[f'{metrica}_std'] = round(df_metricas[metrica].std(), 4) # ej: sensibilidad_std
		
    
    metricas_globales['VP_total'] = int(df_metricas['VP'].sum())
    metricas_globales['FP_total'] = int(df_metricas['FP'].sum())
    metricas_globales['VN_total'] = int(df_metricas['VN'].sum())
    metricas_globales['FN_total'] = int(df_metricas['FN'].sum())
    
    VP = metricas_globales['VP_total']
    FP = metricas_globales['FP_total']
    VN = metricas_globales['VN_total']
    FN = metricas_globales['FN_total']
    
    metricas_globales['global_sensibilidad'] = VP / (VP + FN) if (VP + FN) > 0 else 0
    metricas_globales['global_especificidad'] = VN / (VN + FP) if (VN + FP) > 0 else 0
    metricas_globales['global_precision'] = VP / (VP + FP) if (VP + FP) > 0 else 0
    metricas_globales['global_exactitud'] = (VP + VN) / (VP + VN + FP + FN) if (VP + VN + FP + FN) > 0 else 0
    
    global_f1 = 2 * (metricas_globales['global_precision'] * metricas_globales['global_sensibilidad']) / (metricas_globales['global_precision'] + metricas_globales['global_sensibilidad']) if (metricas_globales['global_precision'] + metricas_globales['global_sensibilidad']) > 0 else 0
    metricas_globales['global_f1'] = round(global_f1, 4)
	
	
    return metricas_globales

# 4. FUNCIÓN PRINCIPAL DE EVALUACIÓN DE LA BONDAD DEL MODELO
def evaluar_segmentacion(directorio_mascaras_originales: Path, 
                                 directorio_mascaras_predichas: Path, 
                                 guardar_resultados: bool = True):
    """Realiza la evaluación global del modelo y guarda los archivos finales"""
    
    # Localiza las máscaras reales (_m) y las de la IA (_mask).
    # 'glob' busca patrones y 'sorted' los ordena (ascendente) para que coincidan en las dos.
    originales = sorted(list(Path(directorio_mascaras_originales).glob("*_m.npy")))
    predichas = sorted(list(Path(directorio_mascaras_predichas).glob("*_mask.npy")))
    
    # Crea un diccionario para unir cada predicción con su original.
    # Usamos .stem para quitar la extensión y .replace para limpiar el nombre y que sean iguales.
    mapeo_predichas = {p.stem.replace('_mask', ''): p for p in predichas} # Diccionario: {"pacienteA": "ruta/pacienteA_mask.npy"...}
    resultados = [] # Aquí guardaremos la nota de cada imagen
    
    # Recorremos cada máscara original una por una.
    for ruta_orig in originales:
        nombre_base = ruta_orig.stem.replace('_m', '') # Sacamos el ID del paciente "pacienteA"
        
        # Si existe una predicción para este paciente, las comparamos
        if nombre_base in mapeo_predichas:
            # A) Cargamos las dos imágenes (asegurando mismo tamaño y binarias)
            orig, pred = cargar_mascaras(ruta_orig, mapeo_predichas[nombre_base])
            
            # B) Calculamos las métricas (Dice, IoU, Hausdorff) para esta imagen
            metricas = calcular_metricas_por_imagen(orig, pred, nombre_base)
            
            # C) Guardamos el resultado
            resultados.append(metricas)
    
    # Convertimos la lista de resultados en una tabla de Pandas.
    df_metricas = pd.DataFrame(resultados)

    # Calculamos las medias y std globales 
    metricas_globales = calcular_metricas_globales(df_metricas)
    
 
    # Devolvemos la tabla detallada y el resumen global
    return df_metricas, metricas_globales
	
def evaluar_calidad_segmentacion(directorio_mascaras_originales:Path,
                                 directorio_mascaras_predichas:Path,
								 archivo_test_csv:Path = None,
								 guardar_resultados:bool = True) -> Tuple[pd.DataFrame, Dict]:
				
				
    df_metricas, metricas_globales = evaluar_segmentacion(
	    directorio_mascaras_originales = directorio_mascaras_originales,
		directorio_mascaras_predichas = directorio_mascaras_predichas
	)
	
    if guardar_resultados:
        output_dir = Path(directorio_mascaras_predichas).parent / "evaluacion_calidad"
        output_dir.mkdir(exist_ok=True)
		
        df_metricas.to_csv(output_dir / "metricas_por_imagen.csv", index=False)
		
        with open(output_dir / "resumen_metricas.txt", "w", encoding='utf-8') as f:
            f.write("RESUMEN DE MÉTRICAS DE SEGMENTACIÓN\n")
            f.write("="*50 + "\n\n")
            for key, value in metricas_globales.items():
                f.write(f"{key}: {value}\n")
				
    return df_metricas, metricas_globales