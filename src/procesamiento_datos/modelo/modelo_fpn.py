# MODELO DE SEGMENTACIÓN

# En esta sección definimos el modelo FPN (Feature Pyramid Network) para segmentación semántica.
# Usaremos una pirámide de características para detectar objetos a diferentes escalas,
# combinando información de alto nivel (contexto global) con detalles de bajo nivel (bordes finos).


# LIBRERÍAS NECESARIAS:
import torch              # Librería principal de deep learning: tensores, autograd, GPU
import torch.nn as nn     # Módulo de redes neuronales: capas (Conv2d, ReLU, Sequential, etc.)
import dagster as dg      # Orquestación de pipelines de datos

# ============================================================
# BLOQUE DE DOBLE CONVOLUCIÓN (PARTE CONTRACTIVA DEL CODIFICADOR)
# ============================================================
class DobleConv(nn.Module):
    """
    Aplica dos convoluciones 3x3 seguidas de ReLU.
    Es el bloque básico del encoder (parte que reduce resolución espacial).
    """
    def __init__(self, entrada, salida):
        """
        Args:
            entrada: número de canales de entrada
            salida: número de canales de salida
        """
        super().__init__()
        self.entrada = entrada
        self.salida = salida
        
        # Secuencia de dos convoluciones + ReLU
        self.doble_conv = nn.Sequential(
            # Mira grupos de 3x3 píxeles, añade una capa extra para evitar fallos en los bordes
            nn.Conv2d(entrada, salida, kernel_size=3, padding=1),  # 1ª conv (mantiene tamaño)
            nn.ReLU(inplace=True),  # Activación ReLU (inplace ahorra memoria)
            nn.Conv2d(salida, salida, kernel_size=3, padding=1),   # 2ª conv (mismos canales)
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.doble_conv(x)


# ============================================================
# BLOQUE DE CONVOLUCIÓN + UPSAMPLING (PARTE EXPANSIVA)
# ============================================================
class ConvReluUpsample(nn.Module):
    """
    Bloque que aplica: Convolución -> GroupNorm -> ReLU -> Upsampling (opcional)
    Se usa en la parte del decoder para aumentar resolución.
    """
    def __init__(self, entrada, salida, upsample=False):
        """
        Args:
            entrada: canales de entrada
            salida: canales de salida
            upsample: si True, duplica la resolución espacial
        """
        super().__init__()
        self.upsample = upsample
        
        # Capa de upsampling:
        # Duplica el tamño de la imagen
        # Interpolación bilineal: nuevos píxeles como promedio ponderado de los 4 píxeles vecinos
        # Alinea las esquinas de la imagen original y la nueva (al aumentar)
        self.make_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Bloque principal: conv -> GroupNorm -> ReLU
        # Entrada: tensor (1,3,128,128) (tamaño del lote, canales,ancho, altura)
        # Salida: tensor (1,...,128,128), no modifica el tamaño de la imagen
        self.block = nn.Sequential(
    nn.Conv2d(entrada, salida, (3, 3), stride=1, padding=1, bias=False),     # Conv 3x3, mantiene tamaño espacial, sin bias (sesgo)
    nn.GroupNorm(32, salida),                                                # Divide canales en 32 grupos y normaliza cada grupo (media=0 y d.t=1)
    nn.ReLU(inplace=True),                                                   # Activación ReLU, los negativos los vuelve 0, es decir, apaga esas neuronas (modifica el tensor original para ahorrar memoria)
)
    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = self.make_upsample(x)  # Duplica altura y anchura
        return x


# ============================================================
# BLOQUE DE SEGMENTACIÓN MULTIESCALA
# ============================================================
class SegmentationBlock(nn.Module):
    """
    Aplica varios ConvReluUpsample en serie.
    Permite construir ramas con diferente número de upsamples.
    """
    def __init__(self, entrada, salida, n_upsamples=0):
        """
        Args:
            entrada: canales de entrada
            salida: canales de salida
            n_upsamples: número de operaciones de upsampling (0, 1, 2 o 3)
        """
        super().__init__()

        # Primer bloque: puede o no tener upsampling
        blocks = [ConvReluUpsample(entrada, salida, upsample=bool(n_upsamples))]

        # Bloques adicionales: todos con upsampling activado
        if n_upsamples > 1:
            for _ in range(1, n_upsamples): # Itera desde 1 hasta n_upsamples-1
                blocks.append(ConvReluUpsample(salida, salida, upsample=True)) #añadimos nuevos bloques con upsampling a la lista

        self.block = nn.Sequential(*blocks)  # Secuencia de bloques
        # (*): desempaqueta la lista: [bloque1, bloque2, ...] → bloque1, bloque2, 
    def forward(self, x):
        return self.block(x)


# ============================================================
# RED COMPLETA: FEATURE PYRAMID NETWORK (FPN)
# ============================================================
class FPN(nn.Module):
    """
    Implementación de Feature Pyramid Network para segmentación semántica.
    Arquitectura:
    - Bottom-up (encoder): extrae características a diferentes escalas
    - Top-down + conexiones laterales: combina características
    - Bloques de segmentación: procesan cada nivel de la pirámide
    """
    
    def __init__(self, n_clases=1, 
                canales_piramide=256, 
                canales_segmentacion=256):
        """
        Args:
            n_clases: número de clases a segmentar (1 para binario)
            canales_piramide: canales de las capas laterales y top layer
            canales_segmentacion: canales en los bloques de segmentación
        """
        super().__init__()
        
        # ===== PARTE 1: BOTTOM-UP (ENCODER) =====
        # Bloques de doble convolución (cada uno reduce resolución por MaxPool)
        self.conv_down1 = DobleConv(3, 64)      # Entrada: imagen RGB (3 canales). Salida: 64 canales.
        self.conv_down2 = DobleConv(64, 128)    # Escala 1/2: 128x128 a 64x64 píxeles.
        self.conv_down3 = DobleConv(128, 256)   # Escala 1/4
        self.conv_down4 = DobleConv(256, 512)   # Escala 1/8
        self.conv_down5 = DobleConv(512, 1024)  # Escala 1/16
        self.maxpool = nn.MaxPool2d(2)          # Reduce resolución a la mitad
        
        # ===== PARTE 2: TOP LAYER =====
        # En la pirámide hace falta que todas las capas tengas los mismos canales (256)
        # Reduce canales de la característica más profunda (1024 -> 256)
        # Mantenemos la altura y ancho constantes (filtro 1x1 y se mueve de 1 en 1)
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        
        # ===== PARTE 3: SMOOTHING LAYERS =====
        # Suavizan las características después de las sumas (que pueden generar ruido innecesario)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # ===== PARTE 4: LATERAL LAYERS =====
        # Convierten características del encoder a 256 canales para poder comparar y sumar información.
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Para capa 4
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # Para capa 3
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)  # Para capa 2
        
        # ===== PARTE 5: BLOQUES DE SEGMENTACIÓN =====
        # Procesan cada nivel de la pirámide con diferente número de upsamples
        # Para alinear todas las características al mismo tamaño
        # Crea una lista con 4 bloques de segmentacion: con n_upsamples: 0,1,2,3
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(canales_piramide, canales_segmentacion, n_upsamples=n_upsamples)
            for n_upsamples in [0, 1, 2, 3]  
        ])
        
        # ===== PARTE 6: CAPA FINAL =====
        # Convierte de 256 canales a n_clases (mapa de probabilidades)
        self.last_conv = nn.Conv2d(256, n_clases, kernel_size=1, stride=1, padding=0)
        
    # ============================================================
    # FUNCIONES AUXILIARES
    # ============================================================
    
    # Aumenta x al tamaño de y, y suma la información píxel a píxel de ambas
    def upsample_add(self, x, y):
        """
        Upsample x al tamaño de y, luego suma elemento a elemento.
        Es la conexión top-down característica de FPN.
        """
        _,_,H,W = y.size()  # Dimensiones del mapa de menor resolución
        upsample = nn.Upsample(size=(H,W), mode='bilinear', align_corners=True) 
        return upsample(x) + y
    
    # Aumenta el tamaño de la imagen a las dimensiones (h,w)
    def upsample(self, x, h, w):
        """Upsample x a dimensiones específicas (h, w)"""
        sample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        return sample(x)
        
    # ============================================================
    # FORWARD PASS (FLUJO COMPLETO DE DATOS)
    # ============================================================
    def forward(self, x):
        """
        Flujo de datos:
        1. Encoder (bottom-up): extracción de características multiescala
        2. Top-down + lateral: construcción de pirámide de características
        3. Smoothing: suavizado
        4. Segmentación: cada nivel procesado y combinado
        5. Clasificación final
        """
        
        # ----- PASO 1: BOTTOM-UP (ENCODER) -----
        # Cada capa reduce resolución por maxpool después de las convoluciones.
        # Reunimos toda la información en 5 capas, partiendo desde la base hasta la cima, donde el nivel de información más abstracto.
        c1 = self.maxpool(self.conv_down1(x))  # Escala 1/2: Disminuye el tamaño a la mitad. Entran x canales y salen 32. Disminuye el tamaño a la mitad.
        c2 = self.maxpool(self.conv_down2(c1)) # Escala 1/4
        c3 = self.maxpool(self.conv_down3(c2)) # Escala 1/8
        c4 = self.maxpool(self.conv_down4(c3)) # Escala 1/16
        c5 = self.maxpool(self.conv_down5(c4)) # Escala 1/32. Pasa de 128x128 a 4x4 píxeles. Salen 1024 canales.
        
        # ----- PASO 2: TOP-DOWN + LATERAL (PIRÁMIDE) -----
        # Construcción de la pirámide de características
        # P5: nivel más alto. Tiene mucha información pero es muy abstracta.
        p5 = self.toplayer(c5)  # Nivel más profundo (más abstracto). Pasamos a 256 canales, mantenemos tamaño 4x4 píxeles.
        
        # Conexiones laterales: combina información de alto nivel (p5) con detalles de bajo nivel (c4, c3, c2)
        p4 = self.upsample_add(p5, self.latlayer1(c4))  # Escala 1/16. Pasamos con self.layer a 256 canales para poder comparar.
        p3 = self.upsample_add(p4, self.latlayer2(c3))  # Escala 1/8

        # P2: Capa base de la pirámide: información más detallada, menos abstracto.
        p2 = self.upsample_add(p3, self.latlayer3(c2))  # Escala 1/4: 32x32 píxeles, 256 canales.
        
        # ----- PASO 3: SMOOTHING -----
        # Elimina ruido innecesario después de las sumas (upsample_add)
        p4 = self.smooth1(p4)  # Escala 1/16: 8x8. 256 canales
        p3 = self.smooth2(p3)  # Escala 1/8: 16x16. 256 canales
        p2 = self.smooth3(p2)  # Escala 1/4: 32x32. 256 canales
        
        # ----- PASO 4: SEGMENTACIÓN MULTIESCALA -----
        # Cada nivel de la pirámide se procesa independientemente
        _, _, h, w = p2.size()  # [1,256,32,32]: capa base (p2). Luego (h,w)=(32,32)
        
        # Aplica bloque de segmentación a cada nivel (ajustando resoluciones)
        # seg_blocks es una lista de 4 bloques de segmentación: [bloque0, bloque1, bloque2, bloque3]
        # [p2, p3, p4, p5] son los 4 niveles de la pirámide
        # zip los empareja así:
        # (bloque0, p2)  # Primer par
        # (bloque1, p3)  # Segundo par  
        # (bloque2, p4)  # Tercer par
        # (bloque3, p5)  # Cuarto par
        

        # Al final vamos a sumar todas las características, luego necesitos mismos tamaños (ya tenemos mismos canales)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])]
        # Cada bloque procesa su nivel:
        # bloque0(p2)  # Procesa p2 (escala 1/4, necesita 0 upsamples): tamaño 32x32 
        # bloque1(p3)  # Procesa p3 (escala 1/8, necesita 1 upsamples): tamaño 32x32
        # bloque2(p4)  # Procesa p4 (escala 1/16, necesita 2 upsample): tamaño 32x32
        # bloque3(p5)  # Procesa p5 (escala 1/32, necesita 3 upsamples): tamaño 32x32

        # ----- PASO 5: CLASIFICACIÓN FINAL -----
        # Suma todas las características y aplica convolución final (tensores de la forma (1,256,32,32))
        # Upsample para volver a la resolución original (4*h= 4*32 es la resolución de entrada de p2)
        out = self.upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)
        
        return out  # devuelve un tensor: (1,1,128,128). 
    # Un solo canal donde nos da logits (valores negativos, positivos y nulos): si se le aplica
    # sigmoid lo convierte en probabilidades entre 0-1.