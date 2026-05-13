import torch                         # El motor principal. Maneja tensores (matrices 3D, como nuestras imagenes).
import torch.optim as optim          # Sirve para optimizar los pesos de la red neuronal
from torch.utils.data import DataLoader, Dataset # Dataset es la libreria de fotos y DataLoader el camión que las lleva.
import numpy as np                   # El lenguaje de las imágenes. Tus archivos .npy son básicamente matrices de Numpy.
import pandas as pd                  # El gestor de archivos. Lee tu CSV con la lista de pacientes y rutas.
from pathlib import Path             # El guía de caminos. Gestiona las carpetas (images_dir) de forma inteligente.
from .modelo_unet import UNet         # Tu creación. Trae el diseño de la red en "U" que definimos en el otro archivo.


# =========================================================================
# PARTE 1: EL BIBLIOTECARIO (Clase MRIDataset)
# =========================================================================
# Esta clase enseña a PyTorch a leer tus archivos .npy de resonancias.

# Creamos una clase, una "plantilla" de como leer nuestros datos, basándonos 
# en la ya creada en pyTorch, Dataset.
class MRIDataset(Dataset):
    # init es el constructor inicial, asi se comienza siempre a crear una clase
    def __init__(self, csv_path, images_dir):
        # 1. Leemos el "inventario" (CSV). 
        # Ejemplo: row 1 -> ID: TCGA_DU_6404, Ruta: 'vol_01.npy'
        self.data = pd.read_csv(csv_path)
        # Cuando entrenemos aqui metemos test.csv, train.csv y val.csv
        # 2. Guardamos la dirección de la carpeta de imágenes (Path nos da la ruta)
        self.images_dir = Path(images_dir)

    # Numero total de imagenes que tenemos que procesar
    def __len__(self):
        return len(self.data)
    
    # Definimos la siguiente funcion para extraer la informacion, que sera la del indice idx
    def __getitem__(self, idx):
        
        # 1. Localizar la fila idx en el CSV (ej: paciente 105)
        row = self.data.iloc[idx]
        
        ruta_procesada= Path(row['ruta_imagen']).name
        ruta_mascara= Path(row['ruta_mascara']).name

        # 2. Cargar la imagen : Tiene 3 canales (FLAIR,Pre,Post).
        # El formato original es una matriz de dimensiones : (Alto, Ancho, 3).
        # La barra / indica: entra dentro de 
        img = np.load(self.images_dir / ruta_procesada) # uso np.load pq mis imagenes son formato numpy
        

        # 3. Cargar la máscara : 
        # El formato es (Alto, Ancho), con 1 donde hay tumor y 0 donde no.
        mask = np.load(self.images_dir / ruta_mascara) 
        
        # --- TRANSFORMACIÓN PARA PYTORCH ---
        
        # img.permute(2, 0, 1): 
        # Pasa de (256, 256, 3) a (3, 256, 256). 
        # PyTorch quiere los "cables" (canales) al principio.
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        # mask.unsqueeze(0): 
        # Pasa de (256, 256) a (1, 256, 256). 
        # Le añade una dimensión para que parezca una "imagen" de un solo canal.
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return img, mask  # nos devuelve tanto imagen como mascara en formato tensor para darselo al modelo


# =========================================================================
# PARTE 2: EL ENTRENADOR (Función entrenar_unet)
# =========================================================================

def entrenar_unet(train_csv, val_csv, images_dir, pesos_clase_path, epochs=1, lr=1e-4, batch_size=8):
    
    # 1. CREAR EL SISTEMA DE LOGÍSTICA
    train_dataset = MRIDataset(train_csv, images_dir)  # formato tensor
    val_dataset = MRIDataset(val_csv, images_dir)  # formato tensor
    
    # DataLoader: Es el camión que lleva las fotos a la red.
    # hace una lista de listas para ir procesando de grupito en grupito y no saturar

    # batch_size=8: La IA estudia de 8 en 8 pacientes para no saturar la memoria.
    # shuffle=True: Mezcla las cartas en cada vuelta para que no se aprenda el orden.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. PREPARAR EL CEREBRO (La Red). Model es la red neuronal.
    model = UNet(entrada=3, salida=1)

    # device: Si hay una tarjeta NVIDIA (cuda), úsala (porque va mas rapido el procesamiento).
    # No todos los ordenadores la tienen,asi que si no que use el procesador (cpu).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) # Movemos el modelo a la tarjeta gráfica
    
    # 3. BALANCEO DE CLASES (pesos)
    # Tus imágenes tienen muchísimos píxeles sanos y pocos de tumor, asociamos pesos
    # pos_weight .
    pesos_clase = np.load(pesos_clase_path) # [peso_sano, peso_tumor]

    # Esta es la penalizacion de la IA, le dice que penaliza mas fuerte los falsos negativos, es decir,
    # si dice que no hay tumor cuando si que lo hay, la penalizacion es enorme
    #pos_weight = torch.tensor([pesos_clase[1]], dtype=torch.float32).to(device)
    pos_weight=torch.tensor([180.0],dtype=torch.float32).to(device)


    # 4. LAS REGLAS DEL JUEGO
    # criterion: Mide el error entre el dibujo de la IA y el real.
    # BCE: se usa pq nuestro problema es binario, se refiere a Entropia binaria
    # WithLogitsLoss evitar errores de precision
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer: El algoritmo (Adam) que ajusta los pesos de las neuronas de la red neuronal.
    # Una red neuronal es como un arbol de decision y en cada division se le da un peso a cada dato de entrada,
    # Cada division es como una nueva neurona.
    # lr es la velocidad de aprendizaje, mientras mas lento mejor. Es decir el cambio entre el peso de
    # una neurona y la siguiente tiene un tope, para que no se vuelva loca.
    # model.parameters() le estamos dando permiso para tocar lo que sea 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf') # Para guardar solo la "mejor versión" del modelo
    # sirve para ir bajando el error, si el error del modelo en la primera vuelta es menor que infinito, me quedo con el
    # y en la segunda ya comparo que sea menor que el de la etapa previa
    
 # Iniciamos el gran bucle. Cada 'epoch' es una vuelta completa a tus 3929 imágenes. Damos 50.
    for epoch in range(epochs):
        
        # A. FASE DE ESTUDIO (Train)
        # Le decimos a la red: "¡Atención! Ahora vas a aprender, activa tus mecanismos de ajuste".
        model.train() 
        
        # Aquí iremos sumando los errores de cada grupo de fotos para saber el error final.
        train_loss = 0
        
        # El DataLoader nos va dando paquetes (batches) de 8 imágenes y 8 máscaras a la vez.
        for imgs, masks in train_loader:
            
            # Mandamos las imágenes (3 canales) y las máscaras al "cerebro" de la 
            # tarjeta gráfica (GPU). Si no hacemos esto, el código dará error.
            imgs, masks = imgs.to(device), masks.to(device)
            
            # 1. zero_grad
            # Borramos los errores del grupo anterior. Si no lo hacemos, los errores se 
            # acumularían y la red se volvería loca. Empezamos de cero para este grupo.
            optimizer.zero_grad()
            
            # 2. La red neuronal hace su función (camino)
            # Metemos las 8 resonancias en la U-Net. La red hace todo el camino (bajada y subida) 
            # y nos devuelve 8 mapas de probabilidad.
            outputs = model(imgs)
            
            # 3. La penalizacion (loss)
            # Comparamos los resultados de la red (outputs) con las máscaras que tenemos (masks),
            # usando el criterio de la entropia visto arriba en 'criterion'
            # Gracias al pos_weight (pesos), si la red da un falso negativo, el número 
            # de 'loss' (error) será muy alto.
            loss = criterion(outputs, masks) # error cometido
            
            # 4. BUSCAR CULPABLES (backward)
            # La red viaja hacia atrás (de la salida a la entrada). Calcula cuánto ha 
            # contribuido cada neurona al error cometido. Es como buscar qué "detective" 
            # o qué "pintor" se ha equivocado.
            # Es FUNDAMENTAL para el deep learning, ya que nos dice que pesos de las neuronas hay que aumentar
            # o disminuir segun cuanto han contribuido al error.
            loss.backward()
            
            # 5. AJUSTAR LOS PESOS
            # El optimizador Adam ajusta los pesos de las neuronas culpables.
            # Si una neurona falló, le cambia un poco el valor para que la próxima 
            # vez no cometa el mismo error.
            optimizer.step()
            
            # Acumulamos (+= es sumale a lo anterior) el error de este grupo.
            train_loss += loss.item()


        # B. FASE DE VALIDACION
        model.eval() # Encendemos el modo "evaluación" (no aprende, solo demuestra)
        val_loss = 0
        with torch.no_grad(): # Bloqueamos el aprendizaje para ir más rápido
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device) # donde trabaja el ordenador
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item() #Vemos como se equivoca en datos con los que no ha sido entrenado
        
        
        # 5. GUARDAR EL MEJOR MODELO
        # Si el error actual es el más bajo que hemos visto, guardamos el mejor modelo obtenido.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "modelo_unet_mejor.pth")

            # Me dice que los pesos de la red neuronal los guarde en  "modelo_unet_mejor.pth"
    
    return model


# Por defecto el umbral es 0.5 y en función de los datos de validacion vamos a optimizarlo.
def encontrar_mejor_umbral(model, val_csv, images_dir,device='cpu',batch_size=8):
    """
    Encuentra el mejor umbral usando datos de validación
    """
    val_dataset = MRIDataset(val_csv, images_dir) # convierte a tensor las imagenes
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #Aquí el orden no afecta así que lo ponemos en False
    model.eval() # lista de listas de dimension batch_size (8 por defecto)
    
    # Almacenar todas las predicciones y realidades
    todas_predicciones = []
    todas_reales = []
    
    with torch.no_grad():  # no queremos entrenar
        for imgs, masks in val_loader:
            imgs = imgs.to(device) # lo manda a la memoria grafica donde esta el modelo la UNet
            outputs = model(imgs) # la red actua
            
            # Aplanar para tener todos los píxeles
            # Aplano la matriz y lo convierte en un vector muy largo con todos los pixeles
            preds_flat = outputs.cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            todas_predicciones.extend(preds_flat)  # las añado a todas_predicciones hasta tener una LISTA larguisimo
            todas_reales.extend(masks_flat)
    
    # Convertir a arrays (vectores de numpy)
    y_true = np.array(todas_reales) # vector de unos y ceros. Lo tiene guardado como float
    y_prob = np.array(todas_predicciones) # vector de probabilidades
    
    # Probar diferentes umbrales
    umbrales = np.arange(0.1, 0.95, 0.05)  # vector de 0.1 a 0.95 de 0.05 en 0.05
    resultados = []
    
    for umbral in umbrales:
        y_pred = (y_prob > umbral).astype(int) # vector de unos y ceros
        y_true = y_true.astype(int)
        sens = np.logical_and(y_pred , y_true).sum() / (y_true.sum() + 1e-8) #  da importancia a que HAYA TUMOR
        
        resultados.append({   # lista de listas (cada umbral con su sensibilidad)
            'umbral': umbral,
            'sensibilidad': sens
        })
    
    df_resultados = pd.DataFrame(resultados) # lo convertimos en dataframe
    
    # Mejor umbral segun sensibilidad (maxima)
    mejor_idx = df_resultados['sensibilidad'].idxmax()
    mejor_umbral = df_resultados.loc[mejor_idx, 'umbral']
    
    return mejor_umbral