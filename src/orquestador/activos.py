#SRC/ORQUESTADOR/ACTIVOS

# Este código utiliza un orquestador llamado Dagster para organizar el proyecto en "Activos" (Assets),
# que son los productos finales de cada etapa (como el catálogo, las imágenes limpias o el modelo 
# entrenado). El sistema funciona como una cadena de montaje donde cada paso depende del anterior:
# primero se preparan los datos, luego la IA aprende a localizar los tumores y, finalmente,
# se extraen medidas para predecir el riesgo del paciente. 
# Esta estructura permite que el proceso sea automático, ordenado y que se pueda actualizar
# cualquier paso sin tener que repetir todo el trabajo desde el principio.

# LIBRERÍAS NECESARIAS:
import os                       # Para gestionar las carpetas y rutas del sistema 
import json                     # Guardado y lectura de metadatos en formato de texto ligero
import torch                    # Para cargar y ejecutar la red neuronal U-Net
import pickle                   # Guardado y carga de modelos estadísticos (Regresión Logística)
import pandas as pd             # Para tablas de datos (catálogos, métricas y resultados)
import numpy as np              # Operaciones matemáticas sobre matrices de imágenes
from datetime import datetime    # Registro de la fecha y hora de ejecución de cada proceso
from pathlib import Path        # Manejo de rutas de archivos (evitar problemas Linux/Mac-Windows)

# Componentes de Dagster para definir los Activos, las salidas de datos y el registro de eventos
from dagster import asset, Output, AssetExecutionContext

# Librerías de Ciencia de Datos
from sklearn.model_selection import train_test_split  # Divide los datos en grupos de entrenamiento, validación y test
from sklearn.utils.class_weight import compute_class_weight # Calcula pesos para equilibrar clases
from sklearn.linear_model import LogisticRegression    # "Modelo de Urgencia" (predice probabilidad de fallecimiento)
from sklearn.metrics import roc_auc_score             # Métrica para medir la bondad del modelo
from sklearn.impute import SimpleImputer              # Imputar datos faltantes
from sklearn.preprocessing import StandardScaler      # Normaliza los datos

# Importación de funciones propias
from procesamiento_datos.procesador_dask import DaskBrainProcessor
from mapeo_archivos import crear_indice_archivos
from procesamiento_datos.modelo.modelo_unet import UNet
from procesamiento_datos.modelo.entrenar_modelo import entrenar_unet, encontrar_mejor_umbral
from procesamiento_datos.modelo.segmentar import segmentar_lote
from procesamiento_datos.modelo.caracteristicas import generar_dataset_features

# Configuración de rutas globales
BASE_DIR = Path("Trabajo")
DATOS_PROCESADOS = BASE_DIR / "datos_procesados"


# 1. CATÁLOGO: 
@asset
def catalogo_maestro():
    """
    ASSET: Catálogo Maestro.
    Este es el primer paso del flujo. Su objetivo es crear un inventario centralizado
    que relacione cada archivo de imagen en el disco con su información clínica.
    """
    ruta_base = r"Trabajo\DATOS"  # Donde tenemos los datos de kaggle
    
    # Cargamos el archivo CSV de datos de kaggle
    df_clinico = pd.read_csv(os.path.join(ruta_base, "data.csv"))
    
    # 'crear_indice_archivos recorre las carpetas y crea un data.frame con tantas filas como imágenes
    df_archivos = crear_indice_archivos(ruta_base, df_clinico, detectar=True)
    
    # Cruce de información (Merge).
    # Unimos ambas tablas usando el paciente como índice. Tendrá tantas filas como df_archivos
    df_catalogo = pd.merge(df_archivos, df_clinico, left_on='id_paciente', right_on='Patient', how='left')
    
    # Reemplazamos las barras invertidas '\' por '/' para que las rutas funcionen correctamente.
    df_catalogo['ruta_imagen'] = df_catalogo['ruta_imagen'].str.replace('\\', '/')
    
    # Guardamos este catálogo en un nuevo archivo CSV
    df_catalogo.to_csv(BASE_DIR / "DATOS" / "catalogo_maestro.csv", index=False)
    
    # Añadimos estadísticas que Dagster mostrará en su interfaz visual.
    return Output(
        df_catalogo, 
        metadata={
            "total_pacientes": df_catalogo['id_paciente'].nunique() 
        }
    )

# 2. PROCESAMIENTO: 

@asset
def imagenes_procesadas(catalogo_maestro):
    """
    Procesamos las imágenes trabajando en paralelo.
    """
    # Creamos la carpeta de destino si no existe 
    os.makedirs(DATOS_PROCESADOS, exist_ok=True)
    
    # Configuración:
    # Usamos Dask para trabajar con 4 trabajadores en paralelo.
    processor = DaskBrainProcessor(n_workers=4)
    
    try:
        # Ejecución del Procesamiento
        resultados = processor.procesar_todas_imagenes(catalogo_maestro)
    
    finally:

        # Cerramos los procesos de Dask para liberar la memoria.
        processor.shutdown()
        
    # Devolvemos las rutas de las imágenes procesadas la metadata.
    return Output(
        resultados, 
        metadata={
            "n_procesadas": len(resultados) # Total de cortes cerebrales listos para la IA
        }
    )


# 3. BALANCEADO DE LOS DATOS

@asset
def dividir_dataset_balanceado(imagenes_procesadas):
    """Divide en Train/Val/Test y calcula pesos para compensar la falta de tumores."""

    df = pd.DataFrame(imagenes_procesadas)
    
    # División: mantenemos el % de tumores en cada grupo.
    # 70% para entrenar y  30% temporal. Con stratify mantenemos la proporción de los que tienen tumor.
    # Fijamos la semilla 42 para reproducibilidad.
    train, temp = train_test_split(df, test_size=0.3, stratify=df['tiene_tumor'], random_state=42)

    # Del 30% temporal, 50% serán para evaluar y 50% para validar.
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['tiene_tumor'], random_state=42)
    
    # Cálculo de pesos para la Loss Function (penalizamos más el error en tumor que en fondo)
    y_train = train['tiene_tumor'].values
    pesos_imagen = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    
    # Guardado de archivos 
    train.to_csv(DATOS_PROCESADOS / "train.csv", index=False)
    np.save(DATOS_PROCESADOS / "pesos_imagen.npy", pesos_imagen)
    
    return Output(len(train),
         metadata={"train_size": len(train), "peso_tumor": float(pesos_imagen[1])})



# 4. U-NET: 

@asset
def modelo_unet_entrenado(context: AssetExecutionContext, dividir_dataset_balanceado):
    """
    Modelo U-Net: entrenamos el modelo de segmentación.
    """
    
    # PASO 1: Usamos la función 'entrenar_unet'
    model = entrenar_unet(
        train_csv=DATOS_PROCESADOS / "train.csv", # Datos de entrenamiento
        val_csv=DATOS_PROCESADOS / "val.csv",     # Datos de validación
        images_dir=DATOS_PROCESADOS,
        epochs=10,        # La red dará 10 vueltas completas a los datos
        batch_size=10,     # Procesa las imágenes en grupos de 10 
        logger=context.log 
    )
    
    #  Guardamos los pesos (state_dict) de la red neuronal.
    path = DATOS_PROCESADOS / "modelo_unet_mejor.pth"
    torch.save(model.state_dict(), path)
    
    # Calculamos el Umbral 
    # Cuál es el límite entre tumor o tejido sano.
    umbral = encontrar_mejor_umbral(model, DATOS_PROCESADOS / "val.csv", DATOS_PROCESADOS)
    
    # Guardamos el umbral en un nuevo archivo .npy
    np.save(DATOS_PROCESADOS / "umbral.npy", umbral)
    
    # Devolvemos la ruta para que la encuentre el siguiente activo.
    return str(path)


# 5. SEGMENTACIÓN Y CARACTERÍSTICAS: Extracción de biomarcadores

@asset
def caracteristicas_tumorales(modelo_unet_entrenado):
    """Usa el modelo entrenado sobre el grupo Test y obtiene características del tumor."""

    # Cargamos los datos test y el umbral calculado
    test_df = pd.read_csv(DATOS_PROCESADOS / "test.csv")
    umbral = np.load(DATOS_PROCESADOS / "umbral.npy")
    
    # Cargamos el modelo U-Net
    model = UNet(entrada=3, salida=1)
    # Le damos los pesos del modelo óptimo con 'load_state_dict'
    model.load_state_dict(torch.load(modelo_unet_entrenado, map_location='cpu'))
    
    # Segmentamos por lotes: creamos las máscaras de las imágenes test
    output_dir = DATOS_PROCESADOS / "mascaras_predichas"  # ruta de la salida
    
    segmentar_lote(
        model, 
        [DATOS_PROCESADOS / Path(r['ruta_procesada']).name for _, r in test_df.iterrows()], 
        output_dir, 
        umbral=umbral
    )
    
    # Calculamos las características tumorales
    df_caract = generar_dataset_features(test_df, DATOS_PROCESADOS, output_dir)
    
    # Guardamos el resultado en un CSV 
    df_caract.to_csv(DATOS_PROCESADOS / "caracteristicas_tumorales.csv", index=False)
    
    return str(DATOS_PROCESADOS / "caracteristicas_tumorales.csv")


# 6. MODELO DE URGENCIA: 
@asset
def entrenar_modelo_urgencia(caracteristicas_tumorales):
    """
    ASSET: Modelo de Urgencia Clínica.
    Este activo entrena un modelo estadístico para predecir el riesgo de fallecimiento,
    a paritr de la variable binaria 'death0'.
    """
    # Leemos el dataset final con las características tumorales y los datos clínicos.
    df = pd.read_csv(DATOS_PROCESADOS / "tumores_limpio.csv")
    
    # Filtramos solo los valores que no son NAs: 'death0' no vacío.
    df_conocidos = df[df['death01'].notna()].copy()
    
    # Elegimos qué factores influyen en el riesgo.
    vars_x = ['area', 'circularidad', 'intensidad_media_post', 'age_at_initial_pathologic']
    X = df_conocidos[vars_x]
    y = df_conocidos['death01'] # Variable objetivo (0 = Vivo, 1 = Fallecido)
    
    # Preparación de datos:
    # SimpleImputer: Si faltan datos, imputamos con la mediana.
    # StandardScaler: estandarizamos a la misma escala.
    pipeline_prep = SimpleImputer(strategy='median')
    X_imputado = pipeline_prep.fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputado)
    
    # Entrenamos el modelo de regresión logística
    #'class_weight=balanced': misma importancia a los fallecidos que a los supervivientes.
    modelo = LogisticRegression(class_weight='balanced', random_state=42)
    modelo.fit(X_scaled, y) # lo entrenamos
    
    # Bondad del modelo (Métrica AUC ROC):
    # Mide la capacidad del modelo para distinguir entre un paciente crítico y uno estable.
    y_probabilidades = modelo.predict_proba(X_scaled)[:, 1] # probabilidad de 1, de fallecer (urgencia)
    auc = roc_auc_score(y, y_probabilidades)  # Valores reales y las predicciones
    
    # Devuelve los resultados
    return {
        "auc_final": float(auc), 
        "modelo_guardado": True,
        "n_pacientes": len(df_conocidos)
    }