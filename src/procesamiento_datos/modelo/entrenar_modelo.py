# Este script implementa el entrenamiento de una red FPN para segmentación de imágenes médicas.
# La pirámide de características permite detectar objetos a diferentes escalas,
# combinando contexto global (escala 1/32) con detalles finos (escala 1/4).

# LIBRERÍAS NECESARIAS
import torch  # Deep learning: tensores, GPU...
import torch.optim as optim  # Optimizadores: Adam, SGD, Adamax para actualizar pesos
from torch.utils.data import DataLoader, Dataset  # Carga eficiente de datos en batches y clase base para datasets
import numpy as np  # Computación numérica: arrays, carga .npy, métricas
import pandas as pd  # Manipulación de dataframes: lectura de CSV
from pathlib import Path  # Manejo de rutas de archivos
from .modelo_fpn import FPN  # Modelo de segmentación


# CLASE MRIDATASET - MANEJO DEL DATASET
class MRIDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes y máscaras de segmentación.
    """
    
    def __init__(self, csv_path, images_dir):
        """
        Inicializa el dataset.
        
        Args:
            csv_path: Ruta al archivo CSV con columnas 'ruta_procesada' y 'ruta_mascara'
            images_dir: Carpeta donde están almacenadas las imágenes .npy
            logger: Objeto para logging opcional
        """
        self.data = pd.read_csv(csv_path)           # Lee CSV y convierte a DataFrame de pandas
        self.images_dir = Path(images_dir)          # Guarda ruta como objeto Path
        self._diagnostic_done = False               # Bandera para diagnóstico único

    def __len__(self):
        """
        Devuelve el número total de muestras en el dataset.r.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Carga y preprocesa una muestra específica del dataset.
        
        Args:
            idx: Índice de la muestra a cargar
            
        Returns:
            img: Tensor de imagen (Canales, Alto, Ancho)
            mask: Tensor de máscara (1, Alto, Ancho)
        """
        row = self.data.iloc[idx]                               # Obtiene fila por índice
        ruta_procesada = Path(row['ruta_procesada']).name       # Extrae nombre del archivo de imagen
        ruta_mascara = Path(row['ruta_mascara']).name           # Extrae nombre del archivo de máscara
        
        # Carga las matrices .npy con NumPy
        img = np.load(self.images_dir / ruta_procesada)         # (Alto, Ancho, 3) - imagen RGB
        mask = np.load(self.images_dir / ruta_mascara)          # (Alto, Ancho) - máscara binaria
        
        
        # Convierte la máscara a valores 0/1 si es necesario
        # (algunas máscaras pueden tener valores >1 como 255)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.float32) # devuelve 0-1
        
        # Convertir a tensores de PyTorch con formato correcto
        # img: (Alto, Ancho, Canales) → (Canales, Alto, Ancho) con permute
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        # mask: (Alto, Ancho) → (1, Alto, Ancho) con unsqueeze (añade dimensión canal=1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return img, mask


# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO - UNET (VERSIÓN COMENTADA)
# ============================================================================

def entrenar_fpn(train_csv, val_csv, images_dir, pesos_clase_path, 
                  epochs=1, lr=5e-5, batch_size=8):
    """
    Entrena el modelo FPN con criterio de parada basado en F1-Score.
    
    Args:
        train_csv: Ruta al CSV con datos de entrenamiento
        val_csv: Ruta al CSV con datos de validación
        images_dir: Carpeta con imágenes .npy
        pesos_clase_path: Ruta al archivo con pesos de clase (para desbalanceo)
        epochs: Número máximo de épocas de entrenamiento
        lr: Learning rate (tasa de aprendizaje)
        batch_size: Tamaño del lote
    
    Returns:
        model: Modelo entrenado (con los mejores pesos encontrados)
    """
    
    # 1. CREAR DATASETS Y DATALOADERS
    train_dataset = MRIDataset(train_csv, images_dir)
    val_dataset = MRIDataset(val_csv, images_dir)
    
    # DataLoader: Distribuye los datos en lotes, shuffle=True mezcla para entrenamiento
    # Devuelve tensores de la forma: (n_lotes,canales, alto, ancho)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. INICIALIZAR MODELO
    model = FPN()                               # Instancia del modelo FPN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU si está disponible
    model.to(device)                            # Mueve el modelo a GPU/CPU
    
    # 3. CARGAR PESOS DE CLASE (para pérdida balanceada)
    pesos_clase = np.load(pesos_clase_path)     # Carga array con pesos [peso_fondo, peso_objeto]
    pos_weight = torch.tensor([pesos_clase[1]], dtype=torch.float32).to(device) # peso para tumor
    
    # 4. DEFINIR FUNCIONES DE PÉRDIDA 
    
    def dice_loss(inputs, target):
        """
        Dice Loss: Mide solapamiento entre predicción y máscara real.
        Fórmula: 1 - (2 * intersección + 1) / (unión + 1)
        El +1 suaviza para evitar división por cero.
        """
        inputs = torch.sigmoid(inputs)          # Convierte logits a probabilidades
        intersection = (target * inputs).sum()  # Área de intersección (verdaderos positivos)
        union = target.sum() + inputs.sum()     # Área total predicha + real
        return 1 - (2 * intersection + 1.0) / (union + 1.0) # fórmula
    
    def bce_dice_loss(inputs, target):
        """
        Pérdida combinada: Binary Cross Entropy + Dice Loss.
        BCE evalúa píxel a píxel, Dice evalúa solapamiento global.
        La combinación suele dar mejores resultados que cada una por separado.
        """
        bce = torch.nn.BCEWithLogitsLoss()(inputs, target)  # Pérdida binaria con logits
        dice = dice_loss(inputs, target)                    # Pérdida Dice
        return bce + dice                                   # Suma de ambas
    
    criterion = bce_dice_loss                   # Función de pérdida a usar


    #  5. OPTIMIZADOR 
    # Adamax: Variante de Adam que usa norma infinito, mejor para casa binario
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    
    # 6. CONFIGURACIÓN DE EARLY STOPPING 
    best_f1 = 0.0           # Mejor F1-Score alcanzado
    patience_counter = 0    # Contador de épocas sin mejora
    patience = 7            # Épocas a esperar antes de parar. Si en 7 épocas no mejora el f1, no espero más.
    best_model_state = None # Almacena los mejores pesos
    
    # 7. BUCLE DE ENTRENAMIENTO POR ÉPOCAS
    for epoch in range(epochs):
        
        # FASE DE ENTRENAMIENTO
        model.train()                               # Modo entrenamiento (activa dropout, batch norm)
        train_loss = 0
        
        for imgs, masks in train_loader:
            imgs = imgs.to(device)                  # Mueve imágenes a GPU/CPU
            masks = masks.to(device)                # Mueve máscaras a GPU/CPU
            
            optimizer.zero_grad()                   # Reinicia gradientes de la época anterior
            outputs = model(imgs)                   # Forward pass: imagen → predicción
            loss = criterion(outputs, masks)        # Calcula pérdida entre predicción y máscara real
            loss.backward()                         # Backward pass: calcula gradientes
            optimizer.step()                        # Actualiza pesos del modelo
            
            train_loss += loss.item()               # Acumula pérdida
            
        avg_train_loss = train_loss / len(train_loader)  # Pérdida promedio de la época
        
        #  FASE DE VALIDACIÓN 
        model.eval()                      # Modo evaluación (desactiva dropout)
        val_loss = 0
        
        # Almacenar todas las predicciones y valores reales para calcular métricas
        todas_predicciones = []
        todas_reales = []
        
        with torch.no_grad():                 # Desactiva cálculo de gradientes (ahorra memoria)
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                
                # Convierte logits a probabilidades con sigmoide
                probs = torch.sigmoid(outputs)
                
                # Aplana tensores para tener arrays 1D de píxeles: (128,128,1)
                preds_flat = probs.cpu().numpy().flatten()
                masks_flat = masks.cpu().numpy().flatten()
                
                todas_predicciones.extend(preds_flat)  # extend añade todos los elementos, no las listas
                todas_reales.extend(masks_flat)
        
        avg_val_loss = val_loss / len(val_loader)
        
        # CÁLCULO DE MÉTRICAS 
        y_true = np.array(todas_reales)            # Valores reales (0 o 1)
        y_prob = np.array(todas_predicciones)      # Probabilidades predichas
        
        # Prueba diferentes umbrales para encontrar el mejor F1-Score
        umbrales = np.arange(0.3, 0.8, 0.05)       # Umbrales de 0.3 a 0.75
        mejor_f1_epoch = 0
        mejor_umbral_epoch = 0.5
        mejor_sens_epoch = 0
        mejor_prec_epoch = 0
        
        for umbral in umbrales:
            # Clasifica según umbral: > umbral = 1 (objeto), ≤ umbral = 0 (fondo)
            y_pred = (y_prob > umbral).astype(int)
            y_true_bin = y_true.astype(int)
            
            # Calcular matriz de confusión
            tp = np.logical_and(y_pred == 1, y_true_bin == 1).sum()  # Verdaderos positivos
            fp = np.logical_and(y_pred == 1, y_true_bin == 0).sum()  # Falsos positivos
            fn = np.logical_and(y_pred == 0, y_true_bin == 1).sum()  # Falsos negativos
            
            # Métricas
            sensibilidad = tp / (tp + fn + 1e-8)    # Recall = TP/(TP+FN) - qué % de objetos detectó
            precision = tp / (tp + fp + 1e-8)       # Precisión = TP/(TP+FP) - qué % de detecciones son correctas
            f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad + 1e-8)  # Media armónica
            
            if f1 > mejor_f1_epoch:
                mejor_f1_epoch = f1
                mejor_umbral_epoch = umbral
                mejor_sens_epoch = sensibilidad
                mejor_prec_epoch = precision
        
        # 8. EARLY STOPPING
        # Si mejoró el F1, guarda los pesos y resetea contador
        if mejor_f1_epoch > best_f1:
            best_f1 = mejor_f1_epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Copia los mejores pesos
        else:
            patience_counter += 1  # No mejoró, incrementa contador
            
            # Si superó la paciencia, detiene entrenamiento (por defecto, 7 épocas de paciencia)
            if patience_counter >= patience:
                model.load_state_dict(best_model_state)    # Carga los mejores pesos
                break  # Sale del bucle de entrenamiento
    
    return model  # Devuelve el modelo entrenado

# FUNCIÓN PARA ENCONTRAR MEJOR UMBRAL

def encontrar_mejor_umbral(model, val_csv, images_dir, device='cpu', batch_size=8, logger=None):
    """
    Encuentra el umbral óptimo para binarizar las predicciones.
    El umbral óptimo es el que maximiza el F1-Score en datos de validación.
    
    Args:
        model: Modelo entrenado
        val_csv: CSV con datos de validación
        images_dir: Carpeta con imágenes
        device: 'cuda' o 'cpu'
        batch_size: Tamaño del lote
        logger: Objeto para logging
    
    Returns:
        mejor_umbral: Valor de umbral que maximiza F1 (ej: 0.45)
    """
    
    # Crear dataset y dataloader
    val_dataset = MRIDataset(val_csv, images_dir, logger=logger)
    # shuffle=False porque el orden no importa para evaluación
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()  # Modo evaluación
    
    # Almacenar todas las predicciones
    todas_predicciones = []
    todas_reales = []
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)  # Logits → probabilidades (0-1)
            
            # Aplanar: (batch, canales, H, W) → (batch*H*W,)
            preds_flat = probs.cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            todas_predicciones.extend(preds_flat)
            todas_reales.extend(masks_flat)
    
    # Convertir a arrays de NumPy
    y_true = np.array(todas_reales)
    y_prob = np.array(todas_predicciones)
    total_pixeles = len(y_true)
    
    # Probar diferentes umbrales
    umbrales = np.arange(0.1, 0.95, 0.05)  # De 0.1 a 0.9 en pasos de 0.05
    resultados = []
    
    for umbral in umbrales:
        y_pred = (y_prob > umbral).astype(int)  # Binarizar según umbral
        y_true_bin = y_true.astype(int)
        
        # Calcular matriz de confusión
        tp = np.logical_and(y_pred == 1, y_true_bin == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true_bin == 0).sum()
        fn = np.logical_and(y_pred == 0, y_true_bin == 1).sum()
        
        # Métricas
        sensibilidad = tp / (tp + fn + 1e-8)     # Recall
        precision = tp / (tp + fp + 1e-8)        # Precisión
        # Especificidad: TN/(TN+FP) - qué % de fondo clasificó correctamente
        especificidad = (total_pixeles - tp - fp - fn) / (total_pixeles - tp - fn + 1e-8)
        f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad + 1e-8)
        
        resultados.append({
            'umbral': umbral,
            'sensibilidad': sensibilidad,
            'precision': precision,
            'especificidad': especificidad,
            'f1': f1
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Encontrar umbral que maximiza F1-Score
    mejor_idx = df_resultados['f1'].idxmax()
    mejor_umbral = df_resultados.loc[mejor_idx, 'umbral']
    mejor_f1 = df_resultados.loc[mejor_idx, 'f1']
    mejor_sens = df_resultados.loc[mejor_idx, 'sensibilidad']
    mejor_prec = df_resultados.loc[mejor_idx, 'precision']
    mejor_espec = df_resultados.loc[mejor_idx, 'especificidad']
    
    return mejor_umbral