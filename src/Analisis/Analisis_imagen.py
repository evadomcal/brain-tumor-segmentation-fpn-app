# MODELO DE CLASIFICACIÓN DE RIESGO (URGENCIA TUMORAL)

# LIBRERÍAS NECESARIAS:
import numpy as np          # Para cálculos matemáticos y manejo de matrices (nuestras imágenes).
import pandas as pd         # Para organizar los datos de los pacientes en tablas (dataframes).
from pathlib import Path    # Para gestionar carpetas y rutas de archivos.
import pickle               # Para guardar y cargar el modelo ya entrenado.
from sklearn.impute import SimpleImputer      # Para rellenar datos faltantes (NAs).
from sklearn.preprocessing import StandardScaler # Para igualar la escala de todas las variables.
from sklearn.linear_model import LogisticRegression # El algoritmo que calcula la probabilidad de urgencia.
from sklearn.model_selection import train_test_split # Para separar datos de entrenamiento y de test.
from sklearn.metrics import roc_auc_score, f1_score  # Para el cálculo de métricas de validación

class ModeloUrgenciaTumoral:
    """
    Modelo para predecir la probabilidad de urgencia clínica, tomando como variables
    objetivo 'death01' (variable binaria).

    """
    # Configuramos el modelo
def __init__(self):
    # Definimos el imputador: si falta un dato, rellena con la mediana.
    self.imputador = SimpleImputer(strategy='median')
    # Definimos el escalador: pone todas las variables en la misma escala numérica.
    self.escalador = StandardScaler()
    self.modelo = None
    self.variables_entrenamiento = None
    self.umbral = 0.5 # Punto de corte por defecto para decidir si es urgente
# Preprocesamos los datos, rellenando NAs y estandarizando.
def preprocesar(self, X):
    """Aplica la limpieza y el escalado a los datos de entrada"""
    # Rellena huecos (imputación) y luego normaliza (escalado).
    X_imputado = self.imputador.transform(X)
    X_escalado = self.escalador.transform(X_imputado)
    return X_escalado


def entrenar(self, df, variables=None, target='death01', test_size=0.2):
        """
        PROCESO DE APRENDIZAJE:
        """
        # Elegimos qué columnas serán las variables predictoras.
        # Si variables está vacío, considera las siguientes por defecto:
        if variables is None:
            variables = ['area', 'perimetro', 'circularidad', 'intensidad_media_post', 
                         'percentil_95_flair', 'textura_contraste', 'age_at_initial_pathologic']
        
        #  Las asignamos a la clase con 'self'
        self.variables_entrenamiento = variables
        X = df[variables].copy() # copiamos en X para no machacar los datos 'df'
        y = df[target].copy().astype(int)  # variable objetivo, tipo entero
        
        # Verificar si hay suficientes fallecidos
        n_fallecidos = (y==1).sum()
        

        # 2. ESTRATEGIA DE DIVISIÓN DINÁMICA: Intentamos repartir los datos de forma inteligente.
        try:
            # Si tenemos al menos 2 fallecidos, podemos usar 'stratify=y' (mantener proporciones).
            if n_fallecidos >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
             )
            # Si solo hay 1 fallecido, es imposible repartirlo en dos grupos (estratificar).    # En ese caso, hacemos una división aleatoria simple y que la suerte decida dónde cae.
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
         
        except ValueError:
         X_train, X_val, y_train, y_val = train_test_split(
         X, y, test_size=test_size, random_state=42  # si algo falla, que no estratisfique.
    )
        
        # Entrenamos el imputador y el escalador con los datos de entrenamiento (predictoras)
        self.imputador.fit(X_train)
        self.escalador.fit(self.imputador.transform(X_train))
        
        # Preprocesamos los datos
        X_train_proc = self.preprocesar(X_train)
        X_val_proc = self.preprocesar(X_val)
        
        # PASO CLAVE: Construimos el modelo balanceado.
        # Como hay más vivos que fallecidos, usamos 'class_weight=balanced'
        # para que la IA no ignore a los pacientes fallecidos por ser pocos.
        self.modelo = LogisticRegression(
            C=1.0, # control del overfitting, C=1 busca un equilibrio
            class_weight='balanced',  # Siempre balancear para este caso
            random_state=42,
            max_iter=1000
        )
        # Entrenamos el modelo
        self.modelo.fit(X_train_proc, y_train)

        # Evaluación con los datos test
        y_pred_proba = self.modelo.predict_proba(X_val_proc)[:, 1]  # proob predicha de fallecer
        
        # Calcular AUC (solo si hay dos clases en validación)
        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, y_pred_proba)
        else:
            auc = 0.5
        
        # Encontrar el mejor umbral: solo si hay más de una clase predicha (no son todos vivos o fallecidos)
        if len(np.unique(y_val)) > 1:
            umbrales = np.arange(0.1, 0.9, 0.05)  # umbrales a probar
            mejores_f1 = 0   # Variable para guardar el mejor umbral
            mejor_umbral = 0.5 # Valor de umbral por defecto 
        for umb in umbrales:
        # Transformamos las probabilidades en (1, 0) según el umbral del bucle.
            y_pred = (y_pred_proba >= umb).astype(int)

        if len(np.unique(y_pred)) > 1:
            # El F1-Score es el equilibrio perfecto entre Sensibilidad y Precisión.
            f1 = f1_score(y_val, y_pred)
            # Si este umbral es mejor que el anterior, lo guardamos
            if f1 > mejores_f1:
                mejores_f1 = f1
                mejor_umbral = umb
    
        # Guardamos el umbral óptimo en el modelo para usarlo en el futuro.
        self.umbral = mejor_umbral
        
        # Mostrar coeficientes y odd ratios
        coefs = pd.DataFrame({
            'Variable': variables,
            'Coeficiente': self.modelo.coef_[0],
            'Odds_Ratio': np.exp(self.modelo.coef_[0])
        })

        # Reordenamos el diccionario el orden ascendente del efecto de la variable:
        # la determina la magnitud en valor absoluto del coefiente
        coefs = coefs.reindex(coefs['Coeficiente'].abs().sort_values(ascending=False).index)
        
        # Calcular riesgo base (probabilidad media de muerte)
        self.media_riesgo_base = y.mean()
        
        # Diccionario final de las métricas obtenidas
        metricas = {
            'auc': auc,
            'mejor_umbral': self.umbral,
            'coeficientes': coefs,
            'variables': variables,
            'riesgo_base': self.media_riesgo_base,
            'n_fallecidos': n_fallecidos,
            'n_total': len(y)
        }
        
        return metricas

# La siguiente función predice el nivel de urgencia del paciente
def predecir_urgencia(self, df, return_proba=True):
        """
        Predice la probabilidad de muerte (urgencia) para nuevos pacientes
        """
        # Verificar variables faltantes en la tabla de datos y las sustituimos por NAs 
        variables_faltantes = [v for v in self.variables_entrenamiento if v not in df.columns]
        if variables_faltantes:
            for v in variables_faltantes:
                df[v] = np.nan
        
        # Copiamos en X las variables predictoras
        X = df[self.variables_entrenamiento].copy()
        
        # Preprocesamos las variables (imputamos y escalamos)
        X_proc = self.preprocesar(X)
        
        # Predecir probabilidad de urgencia (fallecimie)
        prob_muerte = self.modelo.predict_proba(X_proc)[:, 1]
        
        # Resultados en función de si retun_proba está activado
        if return_proba:
            return prob_muerte  # probabilidad de fallecer o de urgencia
        else:
            return (prob_muerte >= self.umbral).astype(int)  # devuelve 1-0 según el umbral

# Funciones para guardar objetos y poder reproducir luego con las mismas configuraciones
def guardar(self, ruta):
        """Guarda el objeto completo (pesos, escalador e imputador) en disco"""
        # Abrimos el archivo en modo 'wb' (escritura binaria) 
        with open(ruta, 'wb') as f:
            # Pickle dump guarda el estado actual de 'self' (toda la clase)
            pickle.dump(self, f)

    
@classmethod 
# permite cargar directamente el objeto de la memoria sin tener que crear uno vacío primero
# De ahí que  usemos cls en lugar de self.
def cargar(cls,ruta): 
        """Recupera el modelo guardado para hacer predicciones inmediatas"""
        # Abrimos en modo 'rb' (Read Binary) para leer el archivo binario
        with open(ruta, 'rb') as f:
            # Reconstruimos el objeto Python a partir del archivo
            modelo = pickle.load(f)
        return modelo


# FUNCIÓN CLAVE: entrenamos el modelo de predicción del riesgo de urgencia del paciente
def entrenar_modelo_urgencia(df, output_dir=None):
    """Función principal que ejecuta el entrenamiento del modelo urgencia
    y genera reportes"""
    
    # Instanciamos la clase del modelo
    modelo = ModeloUrgenciaTumoral()
    
    # Ejecutamos el entrenamiento y obtenemos las métricas
    metricas = modelo.entrenar(df)
    
    # Si se indica una ruta, guardamos todo el trabajo en ella
    if output_dir:
        output_dir = Path(output_dir)  # para evitar problemas de multiplataformas

        # Creamos la carpeta de salida si no existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardamos en la dirección el archivo .pkl para uso futuro 
        modelo.guardar(output_dir / "modelo_urgencia.pkl")
        
        # Guardamos los coeficientes en un CSV para análisis estadístico
        metricas['coeficientes'].to_csv(output_dir / "coeficientes_modelo.csv", index=False)
        
        # INFORME DE TEXTO: Generamos un resumen final
        # modo escritura 'w', con código 'utf-8' (español)
        with open(output_dir / "resumen_modelo.txt", "w", encoding='utf-8') as f:
            f.write("="*60 + "\n") # repite '=' 60 veces y salto de línea
            f.write("MODELO DE URGENCIA TUMORAL - RESULTADOS FINALES\n")
            f.write("="*60 + "\n\n")
            f.write(f"AUC ROC: {metricas['auc']:.4f}\n")   # 4 decimales
            f.write(f"Mejor umbral: {metricas['mejor_umbral']:.3f}\n")
            f.write(f"Riesgo base: {metricas['riesgo_base']:.2%}\n\n")  # pasa de decimales a porcentaje
            f.write("VARIABLES MÁS IMPORTANTES (TOP 10):\n")
            f.write(metricas['coeficientes'].head(10).to_string())  # string para que puedan leerse en texto
    
    return modelo, metricas

# Función para verificar la calidad de los datos de entrada
def diagnosticar_datos(df, target='death01'):
    """Verifica la calidad de la información antes de entrenar"""

    # Encabezado visual:
    print("\n" + "="*60)
    print("DIAGNÓSTICO PREVIO DE LA CALIDAD DE DATOS")
    print("="*60)
    
    # Comprueba si la variable que queremos predecir (muerte) está en el archivo.
    if target in df.columns:
        
        # Muestra los valores únicos (ej: 0 y 1).
        print(f"   Valores detectados: {sorted(df[target].unique())}")
        
        # Cuenta cuántos casos hay de cada uno.
        print(f" Vivos: {(df[target]==0).sum()}")
        print(f" Fallecidos: {(df[target]==1).sum()}")
        
        # Datos faltantes.
        # Si hay muchos NAs en el 'target' (variable objetivo), el modelo no tendrá con qué compararse.
        print(f" Datos faltantes (NAs): {df[target].isna().sum()}")
        
        # UMBRAL DE SEGURIDAD: 
        # Si hay menos de 5 fallecidos, el modelo no tiene suficientes ejemplos para "entender" 
        # qué causa la muerte, y sus predicciones serán poco fiables.
        if (df[target]==1).sum() < 5:
            print(f" ADVERTENCIA: Muestra de fallecidos crítica. El modelo podría no generalizar bien.")
            
    else:
        print(f" La columna '{target}' no existe en el archivo proporcionado.")
    
    return