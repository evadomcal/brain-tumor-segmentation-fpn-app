# PROCESAMIENTO_DATOS/MODELO/MODELO_UNET

# Este script define la arquitectura de la red neuronal U-Net, que es el modelo que aprende
# a distinguir el tumor del tejido sano. La red tiene forma de 'U': 
# primero analiza la imagen para entender el contexto con mayor detalle (bajada de la U) y luego 
# reconstruye el dibujo del tumor o máscara píxel a píxel (subida de la U), conectando los detalles
# finos con la información general para ser extremadamente precisa.

# LIBRERÍAS NECESARIAS: 
import torch         # Permite crear los tensores con los que funciona la UNet
import torch.nn as nn  # Contiene las herramientas para construir la red 
import dagster as dg # Para organizar el flujo de trabajo y registrar cada paso

# 1. Definimos el bloque básico de la red: La "Doble Convolución"
class DobleConv(nn.Module):
    def __init__(self, entrada, salida, nombre=""):
        super().__init__()      # Inicializa las funciones básicas de PyTorch
        self.nombre = nombre    # Etiqueta para saber en qué parte de la "U" estamos
        self.entrada = entrada  # Cuántos datos entran 
        self.salida = salida    # Cuántos datos salen 

        # Definimos el orden en el que se procesa la información
        self.doble_conv = nn.Sequential(
            # 1. Busca patrones (bordes, curvas) usando grupitos de 3x3 píxeles
            # con padding=1, añadimos un marco de ceros para evitar problemas con los bordes
            nn.Conv2d(entrada, salida, kernel_size=3, padding=1), 
            
            # 2. Normaliza los datos para que la red aprenda de forma equilibrada
            nn.BatchNorm2d(salida), 
            
            # 3. Sustituye los valores negativos (irrelevantes) por 0 para que solo fluya la información
            # importante a la siguiente capa e introducir no linealidad (patrones complejos)
            nn.ReLU(inplace=True), 
            
            # 4. Se vuelve a filtrar para combinar los patrones encontrados antes con los nuevos
            nn.Conv2d(salida, salida, kernel_size=3, padding=1), 
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            
            # 5. Apaga el 40% de las neuronas al azar para que la IA no memorice, 
            # sino que aprenda a razonar qué es un tumor
            nn.Dropout2d(0.4) 
        )
    def forward(self, x):
        # La imagen (x) entra en el bloque y pasa por todas las capas en orden
        return self.doble_conv(x)

# 2. La arquitectura de la red: U-Net
class UNet(nn.Module):
    def __init__(self, entrada=3, salida=1):
        super().__init__()
        
        # 1. BLOQUE DE BAJADA (ENCODER): Extracción de características importantes
        # El objetivo aquí es reducir el tamaño de la imagen pero aumentar la "inteligencia" (canales)
        self.enc1 = DobleConv(entrada, 32, nombre="Enc1") # Recibe la imagen original (3 canales)
        self.enc2 = DobleConv(32, 64, nombre="Enc2")
        self.enc3 = DobleConv(64, 128, nombre="Enc3")
        self.enc4 = DobleConv(128, 256, nombre="Enc4")
        
        # MaxPool reduce el ancho y alto a la mitad (ej: de 256x256 a 128x128) para reducir la resolución
        # de la imagen y que la red pueda aprender características más globales y no detalles locales
        self.pool = nn.MaxPool2d(2) 
        
        # 2. PUNTO DE INFLEXIÓN (BOTTLENECK): 
        # Aquí la imagen es muy pequeña pero contiene la información más abstracta que hay
        self.bottleneck = DobleConv(256, 512, nombre="Bottleneck")
        
        # 4. BLOQUE DE SUBIDA (DECODER): Reconstrucción de la imagen 
        # Usamos ConvTranspose2d para duplicar el tamaño (Upsampling) y volver al original, a la vez
        # que disminuimos el número de canales. 
        
        # Paso 4: De 512 a 256 canales, duplicamos el tamaño de la imagen (stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        # El bloque recibe 512 canales (256 de la subida + 256 que vienen del lateral del Encoder 4)
        self.dec4 = DobleConv(512, 256, nombre="Dec4") 
        
        # Paso 3: De 256 a 128 canales
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DobleConv(256, 128, nombre="Dec3")
        
        # Paso 2: De 128 a 64 canales
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DobleConv(128, 64, nombre="Dec2")
        
        # Paso 1: De 64 a 32 canales
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DobleConv(64, 32, nombre="Dec1")
        
        # CAPA FINAL: El veredicto píxel a píxel
        # Filtro de 1x1 para convertir los 32 canales finales en un solo canal (1 canal = máscara)
        self.final = nn.Conv2d(32, salida, kernel_size=1)
    
    def forward(self, x):
        # CAMINO DE BAJADA, x es la imagen (256,256,3)
        e1 = self.enc1(x) # de 3 a 32 canales
        e2 = self.enc2(self.pool(e1)) # de 32 a 64 canales, de 256x256 a 128x128 píxeles
        e3 = self.enc3(self.pool(e2)) # de 64 a 128 canales, de 128x128 a 64x64 píxeles
        e4 = self.enc4(self.pool(e3)) # de 128 a 256 canales, de 64x64 a 32x32 píxeles
        
        # EL PUNTO MÁS PROFUNDO
        b = self.bottleneck(self.pool(e4)) # de 256 a 512 canales, de 32x32 a 16x16 píxeles
        
        # CAMINO DE SUBIDA (con "Skip Connections" o conexiones de salto)
        # Se une la información de subida con la de bajada para no perder nitidez
        # self.upconv4(b) reduce de 512 a 256 canales (primera subida)
        d4 = torch.cat([self.upconv4(b), e4], dim=1) # unimos 256 canales de la subida 1 + 256 del encoder 4, y duplica la imagen (16x16 a 32x32)
        d4 = self.dec4(d4) # de 512 a 256 canales
        
        d3 = torch.cat([self.upconv3(d4), e3], dim=1) # unimos 128 canales de la subida 2 + 128 del encode 3, duplica la imagen (32x32 a 64x64)
        d3 = self.dec3(d3) # de 256 a 128 canales
        
        d2 = torch.cat([self.upconv2(d3), e2], dim=1) # unimos 64 canales de la subida 3 + 64 del encode 2, duplicamos la imagen (64x64 a 128x128)
        d2 = self.dec2(d2) # de 128 a 64 canales
        
        d1 = torch.cat([self.upconv1(d2), e1], dim=1) # unimos 32 canales de la subida 4 + 32 del encode 1, duplicamos la imagen (128x128 a 256x256)
        d1 = self.dec1(d1) # de 64 canales a 32 canales
        
        return self.final(d1) # Entran 32 y devuelve 1 canal, la máscara del tumor