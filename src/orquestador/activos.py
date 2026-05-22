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
import pickle                   # Guardado y carga de modelos estadísticos (Regresión Logística)
import pandas as pd             # Para tablas de datos (catálogos, métricas y resultados)
import numpy as np              # Operaciones matemáticas sobre matrices de imágenes
from datetime import datetime   # Registro de la fecha y hora de ejecución de cada proceso
from pathlib import Path        # Manejo de rutas de archivos (evitar problemas Linux/Mac-Windows)
import torch                    # Formato necesario para la red neuronal (tensores)

# Componentes de Dagster para definir los Activos, las salidas de datos y el registro de eventos
from dagster import asset, Output, AssetExecutionContext

# Librerías de Ciencia de Datos
from sklearn.model_selection import train_test_split  # Divide los datos en grupos de entrenamiento, validación y test
from sklearn.utils.class_weight import compute_class_weight # Calcula pesos para equilibrar clases

# Importación de funciones propias
from src.procesamiento_datos.procesador_dask import DaskBrainProcessor
from src.mapeo_archivos import crear_indice_archivos
from src.transformar_datos import procesar_imagen_completo
from procesamiento_datos.modelo.modelo_fpn import FPN
from src.procesamiento_datos.modelo.entrenar_modelo import MRIDataset, entrenar_fpn, encontrar_mejor_umbral
from src.procesamiento_datos.modelo.segmentar import segmentar_imagen, segmentar_lote
from src.procesamiento_datos.modelo.caracteristicas import extraer_caracteristicas, generar_dataset_features


# 1. CATÁLOGO MAESTRO: 
@asset
def catalogo_maestro():
    """
    Crea un catálogo con la información inicial
    """
    Path("DATOS").mkdir(exist_ok=True) #Si no existe la 

    script_dir = Path(__file__).parent 
    proyecto_dir = script_dir.parent.parent  
    ruta_base = proyecto_dir / "DATOS"
    ruta_csv = proyecto_dir / "DATOS" / "data.csv" #Donde tenemos los datos de kaggle

    # Cargamos el archivo CSV de datos de kaggle
    df_clinico = pd.read_csv(ruta_csv)

    # 'crear_indice_archivos recorre las carpetas y crea un data.frame con tantas filas como imágenes
    df_archivos = crear_indice_archivos(ruta_base, df_clinico, detectar=True)

    # Cruce de información (Merge).
    # Unimos ambas tablas usando el paciente como índice. Tendrá tantas filas como df_archivos
    df_catalogo = pd.merge(
        df_archivos, 
        df_clinico, 
        left_on='id_paciente', 
        right_on='Patient', 
        how='left')

    # Reemplazamos las barras invertidas '\' por '/' para que las rutas funcionen correctamente.
    df_catalogo['ruta_imagen'] = df_catalogo['ruta_imagen'].str.replace('\\', '/')

    if 'ruta_mascara' in df_catalogo.columns:
        df_catalogo['ruta_mascara'] = df_catalogo['ruta_mascara'].str.replace('\\', '/')

    # Guardamos este catálogo en un nuevo archivo CSV
    df_catalogo.to_csv("TRABAJO/DATOS/catalogo_maestro.csv", index=False)

    # Añadimos estadísticas que Dagster mostrará en su interfaz visual.
    n_pacientes = df_catalogo['id_paciente'].nunique()
    n_imagenes = len(df_catalogo)
    n_tumores = df_catalogo['mascara_tiene_tumor'].sum()

    return Output(
        df_catalogo,
        metadata={
            "total_pacientes": n_pacientes,
            "total_imagenes": n_imagenes,
            "con_tumor": int(n_tumores),
            "sin_tumor": int(n_imagenes - n_tumores),
            "porcentaje_tumor": round(float(n_tumores/n_imagenes*100), 2)
        }
    )

# 2. PROCESAMIENTO: 

@asset
def imagenes_procesadas(catalogo_maestro):
    """
    Procesamos las imágenes trabajando en paralelo.
    """
    import pandas as pd
    import os
    import traceback    # para ver las rutas de los errores

    # Creamos la carpeta de destino si no existe 
    os.makedirs("datos_procesados", exist_ok=True)

    # Configuración:
    # Usamos Dask para trabajar con 4 trabajadores en paralelo.
    processor = DaskBrainProcessor(n_workers=4)

    try:
        # Ejecución del Procesamiento
        resultados = processor.procesar_todas_imagenes(catalogo_maestro)

    finally:

        # Cerramos los procesos de Dask para liberar la memoria.
        processor.shutdown()
        
    # Recapitulamos los resultados
    n_procesadas = len(resultados)
    n_tumores_proc = sum(1 for r in resultados if r.get('tiene_tumor', False))
    n_sanos_proc = n_procesadas - n_tumores_proc

    # Devuelve un objeto 'Output' con dos paquetes de información:
    return Output(
        resultados,  # Paquete 1: La lista detallada de cada imagen y su máscara.
        metadata={   # Paquete 2: El resumen estadístico o la metadata.
            "imagenes_procesadas": n_procesadas,
            "con_tumor": n_tumores_proc,
            "sin_tumor": n_sanos_proc,
            # Calcula el % de casos positivos 
            "porcentaje_tumor": round(n_tumores_proc/n_procesadas*100, 2) if n_procesadas > 0 else 0,
            # Registra la fecha y hora exacta del análisis.
            "timestamp": datetime.now().isoformat()
        }
    )


# 3. BALANCEADO DE LOS DATOS

@asset
def dividir_dataset_balanceado(imagenes_procesadas):
    """Divide en Train/Val/Test y calcula pesos para un entrenamiento balanceado."""

    # Convertimos a dataframe
    df = pd.DataFrame(imagenes_procesadas)

    # División: mantenemos el % de tumores en cada grupo.
    # 70% para entrenar y  30% temporal. Con stratify mantenemos la proporción de los que tienen tumor.
    # Fijamos la semilla 42 para reproducibilidad.
    # Con shuffle cambiamos el orden de las imágenes para evitar patrones
    train, temp = train_test_split(df,
        test_size=0.3, 
        stratify=df['tiene_tumor'], 
        random_state=42,
        shuffle=True
    )

    # Del 30% temporal, 50% serán para evaluar y 50% para validar.
    val, test = train_test_split(
        temp, 
        test_size=0.5, 
        stratify=temp['tiene_tumor'], 
        random_state=42,
        shuffle=True
    )

    # CALCULAR PESOS DE CADA CLASE para el entrenamiento (usamos Pandas)
    # Distinguimos entre píxeles fondo (sanos o fuera del cerebro) y de tumor.
    total_pixeles_fondo = 0
    total_pixeles_tumor = 0

    # Recorremos cada imagen para contar píxeles de tumor vs píxeles de fondo
    for _, fila in df.iterrows():
        path_mask = fila.ruta_mascara  # ruta de la máscara de esa fila
        mask = np.load(path_mask) # Cargamos la máscara
        
        pixeles_fondo = np.sum(mask == 0) # Sumamos píxeles negros (sanos)
        pixeles_tumor = np.sum(mask == 1) # Sumamos píxeles blancos (tumor)
        
        total_pixeles_fondo += pixeles_fondo  # los añadimos a los que teníamos ya
        total_pixeles_tumor += pixeles_tumor

    # 1. Cálculo de pesos por píxeles
    total_pixeles = total_pixeles_fondo + total_pixeles_tumor

    # Calculamos el peso: a menos píxeles de un tipo, más peso le damos
    peso_fondo = total_pixeles / (2 * total_pixeles_fondo)
    peso_tumor = total_pixeles / (2 * total_pixeles_tumor)
    pesos_clase = np.array([peso_fondo, peso_tumor]) # por píxeles

    # 2. Cálculo de pesos por imagen
    # Equilibra el hecho de que pueda haber más pacientes sanos que enfermos
    pesos_imagen = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]), 
        y=train['tiene_tumor']
    )

    # Factor de importancia: ¿Cuántas veces es más importante el tumor que el fondo?
    factor_peso = pesos_imagen[1] / pesos_imagen[0]

    # Guardamos los CSV con las rutas de entrenamiento, validación y test
    train.to_csv("datos_procesados/train.csv", index=False)
    val.to_csv("datos_procesados/val.csv", index=False)
    test.to_csv("datos_procesados/test.csv", index=False)

    # Guardamos los pesos en archivos .npy para que la red neuronal los use al entrenar
    np.save("datos_procesados/pesos_imagen.npy", pesos_imagen)
    np.save("datos_procesados/pesos_clase.npy", pesos_clase)

    # Creamos un resumen o metadata
    metadata = {
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'train_tumores': int(sum(train['tiene_tumor'])),
        'train_sanos': int(sum(train['tiene_tumor'] == 0)),
        'peso_sano': float(pesos_clase[0]),
        'peso_tumor': float(pesos_clase[1]),
        'factor_peso_tumor': float(factor_peso),
        'timestamp': datetime.now().isoformat()
    }


    return Output(metadata, metadata=metadata)

# 4.ENTRENAMOS EL MODELO U-NET: 

DATOS_PROCESADOS = Path("datos_procesados")

@asset
def modelo_unet_entrenado():
    """
    Modelo Feature Pyramid Network: entrenamos el modelo de segmentación con balanceo de clases.
    """
    train_csv = DATOS_PROCESADOS / "train.csv"
    val_csv = DATOS_PROCESADOS / "val.csv"
    pesos_clase = DATOS_PROCESADOS / "pesos_clase.npy"

    # PASO 1: Usamos la función 'entrenar_unet'
    model = entrenar_fpn(
        train_csv=train_csv,
        val_csv=val_csv,
        images_dir=DATOS_PROCESADOS,
        epochs=20,        # La red dará 20 vueltas completas a los datos
        lr=1e-3,           # learning rate: velocidad a la que aprende la IA
        batch_size=16,     # Procesa las imágenes en grupos de 16 
    )

    #  Guardamos los pesos (state_dict) de la red neuronal.
    model_path = DATOS_PROCESADOS / "modelo_fpn_mejor.pth"
    torch.save(model.state_dict(), model_path)


    # Devolvemos la ruta para que la encuentre el siguiente activo.
    return str(model_path)


# 5. MEJOR UMBRAL PARA DETERMINAR SI ES TUMOR O TEJIDO SANO
@asset  
def mejor_umbral():
    """
    Encuentra el umbral que maximiza F1
    """
    # Localizamos los datos de validación y el archivo del modelo guardado.
    val_csv = DATOS_PROCESADOS / "val.csv"
    modelo_fpn_entrenado = DATOS_PROCESADOS / "modelo_fpn_mejor.pth"

    # Creamos la red FPN.
    model = FPN()

    # Inyectamos en la red el conocimiento aprendido durante el entrenamiento.
    # 'map_location=cpu' asegura que se cargue correctamente aunque no haya una GPU disponible.
    model.load_state_dict(torch.load(modelo_fpn_entrenado, map_location='cpu'))

    # Ejecuta la función que prueba distintos umbrales (0.1, 0.2... 0.9).
    # El objetivo es encontrar el umbral que maximice el Dice Score en los datos de validación.
    umbral = encontrar_mejor_umbral(
        model, val_csv, DATOS_PROCESADOS, 
        # Selecciona automáticamente GPU (cuda) si existe, si no, usa la CPU.
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=8  #Lote de 8 imágenes
    )

    # Guarda el valor del mejor umbral en un archivo .npy.
    np.save(DATOS_PROCESADOS / "umbral.npy", umbral)

# 6. SEGMENTACIÓN: realizamos las predicciones con los datos test
@asset
def segmentaciones_test(): 
    """Segmenta todas las imágenes de test para la predicción final del modelo"""

    # Cargamos el CSV que contiene las rutas de las imágenes test.
    test_df = pd.read_csv(DATOS_PROCESADOS / "test.csv")

    # Reconstruimos la arquitectura U-Net y cargamos los mejores pesos guardados.
    model = FPN()
    modelo_unet_entrenado = DATOS_PROCESADOS / "modelo_unet_mejor.pth"
    model.load_state_dict(torch.load(modelo_unet_entrenado, map_location='cpu'))

    # Detectamos si hay GPU (cuda) para ir más rápido; si no, usamos CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) # Movemos los "pesos" del modelo a la unidad de procesamiento elegida.

    # Creamos la lista de rutas físicas de las imágenes que vamos a segmentar.
    imagenes_test = [DATOS_PROCESADOS / Path(row['ruta_procesada']).name for _, row in test_df.iterrows()]

    output_dir = DATOS_PROCESADOS / "mascaras_predichas" # dirección para la salida

    # Recuperamos el umbral óptimo calculado en el paso anterior.
    umbral = np.load(DATOS_PROCESADOS / "umbral.npy")

    # PASO CLAVE: Procesamos todas las imágenes 
    # La IA dibuja las máscaras basándose en el umbral y las guarda en 'device'.
    resultados = segmentar_lote(
        model=model,
        lista_imagenes=imagenes_test,
        output_dir=output_dir,
        device=device,
        umbral=umbral
    )

    # Guardamos un CSV con el resumen de qué archivos se generaron y dónde.
    df_resultados = pd.DataFrame(resultados)  # como dataframe
    df_resultados.to_csv(DATOS_PROCESADOS / "resultados_segmentacion.csv", index=False)

    # Devolvemos la ruta donde están las imágenes ya segmentadas.
    return str(output_dir)

# 7. CARACTERÍSTICAS: 
@asset
def caracteristicas_tumorales(segmentaciones_test: str):
    """Extrae las características clave sobre los tumores detectados"""

    # Cargamos las imágenes test.
    test_df = pd.read_csv(DATOS_PROCESADOS / "test.csv")

    # Accedemos al catálogo maestro que contiene la información clínica de los pacientes.
    DATOS = Path("DATOS")
    catalogo_path = DATOS / "catalogo_maestro.csv"
    catalogo = pd.read_csv(catalogo_path) # leemos el csv

    # Eliminamos columnas innecesarias (de la 1 a la 5) 
    # para quedarnos solo con el ID del paciente y su información clínica relevante.
    df_previa = catalogo.drop(catalogo.columns[1:6], axis=1)

    # Esta es la función clave:
    # Cruza la imagen original con la máscara que dibujó la IA para medir:
    # Tamaño, circularidad, intensidad, posición y lo combina con la clínica.
    caract_df = generar_dataset_features(
        df_segmentacion=test_df.iloc[:,:3], # Datos básicos del paciente (paciente,num_corte,ruta_imagen)
        directorio_imagenes=DATOS_PROCESADOS, # Imágenes originales
        directorio_mascaras=segmentaciones_test, # Máscaras creadas por la IA
        df_previa=df_previa # Datos clínicos limpios
    )

    # Guardamos la tabla final de características.
    # Este CSV es el que usará el médico para tomar decisiones clínicas.
    output_path = DATOS_PROCESADOS / "caracteristicas_tumorales.csv"
    caract_df.to_csv(output_path, index=False)

    # Devolvemos la ruta del archivo generado
    return str(output_path)

from Analisis.Analisis_test import limpiar_datos, analisis_descriptivo

# 8. ANALISIS GRÁFICO DE LOS DATOS TEST (CONEXIÓN CON R)
@asset
def analisis_datos():
    """Realiza un análisis estadístico profundo sobre el conjunto de test"""

    # Cargamos el CSV que contiene las 7 características extraídas en el paso anterior.
    datos = DATOS_PROCESADOS / "caracteristicas_tumorales.csv"

    # Ejecuta una limpieza para eliminar las columnas que no son necesarias o están repetidas
    datos_limpios = limpiar_datos(datos)

    # Crea el informe estadístico (gráficos, correlaciones...). Se hace en R.
    analisis_descriptivo(datos_limpios)

# 9. VALIDACIÓN DEL MODELO DE SEGMENTACIÓN
@asset
def calidad_modelo_segmentacion():
    """Calcula las métricas de precisión finales"""

    # Cargamos la función ya definida para evaluar el modelo.
    from src.Analisis.Metricas_modelo import evaluar_calidad_segmentacion

    # Definimos las rutas: 
    directorio_originales = DATOS_PROCESADOS
    directorio_predichas = DATOS_PROCESADOS / "mascaras_predichas"

    # Ejecuta las métricas:
    # Compara cada pareja de imágenes (real y predicha) y genera:
    # - df_metricas: La tabla detallada por imagen (sensibilidad, especificidad, dice, IoU y hausdorff).
    # - metricas_globales: El resumen promedio (el éxito del modelo).
    df_metricas, metricas_globales = evaluar_calidad_segmentacion(
        directorio_mascaras_originales=directorio_originales,
        directorio_mascaras_predichas=directorio_predichas,
        guardar_resultados=True
    )

    # Devuelve el resumen final a nivel global del modelo.
    return metricas_globales

# 10. MODELO DE URGENCIA: regresión logística
@asset
def entrenar_modelo_urgencia():
    """
    Entrena un modelo estadístico para predecir el riesgo de fallecimiento
    a paritr de la variable binaria 'death01'.
    """
    # Leemos el dataset final con las características tumorales y los datos clínicos.
    df = pd.read_csv(DATOS_PROCESADOS / "tumores_limpio.csv")

    # Filtramos solo los valores que no son NAs: 'death0' no vacío.
    df_conocidos = df[df['death01'].notna()].copy()

    # Seleccionamos las variables predictoras 
    variables_predictoras = [
        'area', 'perimetro', 'circularidad',
        'intensidad_media_post', 'intensidad_minima_post', 
        'percentil_95_flair', 'textura_contraste',
        'age_at_initial_pathologic', 'neoplasm_histologic_grade']

    # Filtramos solo las columnas que existen
    variables_existentes = [v for v in variables_predictoras if v in df_conocidos.columns]

    # Determinamos las variables predictoras y la variable objetivo binaria
    X = df_conocidos[variables_existentes]
    y = df_conocidos['death01'] # Variable objetivo (0 = Vivo, 1 = Fallecido)

    # Preparación de datos:
    from sklearn.impute import SimpleImputer
    # SimpleImputer: Si faltan datos, imputamos con la mediana.
    imputador = SimpleImputer(strategy='median')
    X_imputado = imputador.fit_transform(X)

    from sklearn.preprocessing import StandardScaler
    # StandardScaler: estandarizamos a la misma escala.
    escalador= StandardScaler()
    X_scaled = escalador.fit_transform(X_imputado)

    # Entrenamos el modelo de regresión logística
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # Dividimos los datos en entrenamiento y validación, manteniendo las proporciones
    # con stratify. El conjunto de test representa el 20% del
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenamos el modelo balanceando las clases,
    # 'class_weight=balanced': misma importancia a los fallecidos que a los supervivientes aunque en la muestra
    # haya más de una clase que de otra.
    modelo = LogisticRegression(
        class_weight='balanced', 
        random_state=42,
        max_iter=1000
    )
    modelo.fit(X_train, y_train) # lo entrenamos

    # Medimos la bondad del modelo para predecir la urgencia del paciente
    # (Métrica AUC ROC):
    y_pred_proba = modelo.predict_proba(X_val)[:, 1] # la probabilidad predicha de fallecer
    auc = roc_auc_score(y_val, y_pred_proba) # métrica AUC: mientras más cerca de 1 mejor modelo.

    # Creamos una tabla con la importancia de cada variable.
    # El Odds_Ratio nos dice cuánto aumenta el riesgo por cada unidad de la variable.
    coefs = pd.DataFrame({
        'Variable': variables_existentes,
        'Coeficiente': modelo.coef_[0],
        'Odds_Ratio': np.exp(modelo.coef_[0])
    }).sort_values(by='Coeficiente', key=abs, ascending=False) # Ordenamos por importancia real, dada por la magnitud del coeficiente (valor absoluto)

    # Guardamos la configuración del análisis.
    output_dir = DATOS_PROCESADOS / "modelo_urgencia" # dirección de salida
    output_dir.mkdir(exist_ok=True)  # si ya existe, no da problema y sigue

    import pickle # Para guardar resultados y permite la reproducibilidad posterior de los mismos.

    # wb: escritura en binario
    with open(output_dir / "modelo_urgencia.pkl", 'wb') as f:
        # Guardamos el pack completo: modelo + preprocesamiento (imputador y escalador) + resultados (AUC)
        pickle.dump({
            'modelo': modelo,
            'imputador': imputador,
            'escalador': escalador,
            'variables': variables_existentes,
            'auc': auc
        }, f)  # 'Asígnalos' a f, nombre temporal del archivo.

    # Enviamos las estadísticas clave al diccionario de salida.
    return Output({
        'auc': float(auc),
        'n_vivos': int((y == 0).sum()),
        'n_fallecidos': int((y == 1).sum()),
        'variables': variables_existentes,
        'coeficientes': coefs.to_dict(),
    }, metadata = {
        'auc' : float(auc),
        'variables' : variables_existentes,
        'coeficientes' : coefs.coeficiente,
        'odds_ratio' : coefs.Odds_Ratio,
    })