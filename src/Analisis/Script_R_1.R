#SCRIPTS DE ANÁLISIS ESTADÍSTICO EN R

# Este script de R es el traductor médico del proyecto: coge los resultados de la IA y los convierte en un informe
# médico. Primero, limpia los datos y etiqueta a los pacientes como "Vivos" o "Fallecidos". 
# Después, analiza si el tumor es peligroso según su forma (más irregular = más agresivo) y su textura 
# (más heterogéneo = posible necrosis). Finalmente, genera automáticamente gráficas y un archivo de texto 
# con recomendaciones para que el médico tome decisiones rápidas.


# 1. LIBRERÍAS: Instalación de librerías en R del repositorio cran
paquetes <- c("tidyverse", "corrplot", "ggplot2", "tidyr")
for (p in paquetes) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cran.rstudio.com/", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}

# 2. Permite que Python le pase las rutas de los archivos
args <- commandArgs(trailingOnly = TRUE) # Se lo damos en cmd y subprocess en Analisis_test.py
csv_path <- args[1]  # ruta_csv 
output_dir <- args[2]  # str(output_dir) 

# Validación de existencia del archivo de entrada
if (!file.exists(csv_path)) {
  stop(paste("Error: No se encuentra el archivo CSV en:", csv_path))
}

# CARGA Y FILTRADO: 

# Cargamos los datos a partir del csv enviado desde Python
datos <- read.csv(csv_path, stringsAsFactors = FALSE)

# Si no existe la carpeta de salida, la creamos permitiendo subcarpetas (recursive=TRUE)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
setwd(output_dir)  # aquí se guardará todo lo que hagamos en R

# Solo analizamos casos donde la IA realmente detectó un tumor (área > 0)
datos_tumor <- datos[datos$area > 0, ]
if (nrow(datos_tumor) == 0) stop("No hay datos de tumores para analizar.")

# 4. FORMATEO DE VARIABLES: Convertimos 0-1 en Vivo-Fallecidos (categorías)
if ("death01" %in% colnames(datos_tumor)) {
  datos_tumor$death01 <- factor(datos_tumor$death01, labels = c("Vivo", "Fallecido"))
}
if ("histological_type" %in% colnames(datos_tumor)) {
  datos_tumor$histological_type <- factor(datos_tumor$histological_type)
}
if ("institucion" %in% colnames(datos_tumor)) {
  datos_tumor$institucion <- factor(datos_tumor$institucion)
}  

# GENERACIÓN DEL INFORME (SINK): Redirige la salida de consola a un archivo .txt
sink("resultados_analisis.txt")

cat("============================================================\n")
cat("REPORTE DE MÉTRICAS MORFOLÓGICAS Y TEXTURALES DEL TUMOR\n")
cat("============================================================\n\n")
cat("Fecha:", date(), "\n")

#ANÁLISIS MORFOLÓGICO
# Interpretamos la forma: Circularidad cercana a 0 indica tumores invasivos/irregulares
cat("1. MORFOLOGÍA TUMORAL\n")
resumen_circ <- summary(datos_tumor$circularidad)
print(resumen_circ)

# Interpretación de la circularidad según la mediana
mediana_c <- median(datos_tumor$circularidad, na.rm = TRUE)
if (mediana_c < 0.4) {
  cat("Alta incidencia de tumores IRREGULARES (Posible agresividad).\n")
} else {
  cat("Tumores predominantemente redondeados/ovalados.\n")
}

# ANÁLISIS DE TEXTURA 
# El contraste mide la heterogeneidad (necrosis/hemorragia interna)
cat("\n2. TEXTURA Y HETEROGENEIDAD\n")
resumen_text <- summary(datos_tumor$textura_contraste)
print(resumen_text)

# CORRELACIONES BIOLÓGICAS
# Medimos la correlación entre el área y la circularidad, COEFICIENTE DE PEARSON (r)
if (all(c("area", "circularidad") %in% colnames(datos_tumor))) {
  cor_ac <- cor(datos_tumor$area, datos_tumor$circularidad, use = "complete.obs")  
  cat(sprintf("\n3. CORRELACIÓN ÁREA-CIRCULARIDAD: r = %.3f\n", cor_ac))
  if (cor_ac < -0.4) cat("Los tumores pierden circularidad conforme aumentan de tamaño.\n")
}

sink() # Cerramos el archivo de texto y volvemos a la consola de R

# VISUALIZACIÓN GRÁFICA: formato PNG
cat("\n Generando gráficos médicos...\n")

# Gráfico de Distribución de Forma (circularidad)
png("01_DISTRIBUCION_CIRCULARIDAD.png", width = 800, height = 600)

# Histograma de la circularidad
hist(datos_tumor$circularidad, col = "skyblue", main = "Frecuencia de Formas Tumorales (circularidad)",
     xlab = "Circularidad (0: Irregular, 1: Redondo)", ylab = "Nº de Pacientes")
abline(v = 0.4, col = "red", lwd = 2, lty = 2) # Umbral crítico de sospecha
dev.off()

# Gráfico de Dispersión: Tamaño vs Heterogeneidad (contraste)
png("02_AREA_VS_CONTRASTE.png", width = 800, height = 600)
plot(datos_tumor$area, datos_tumor$textura_contraste, 
     pch = 19, col = rgb(0.1, 0.5, 0.1, 0.5),
     main = "Relación Tamaño vs Heterogeneidad",
     xlab = "Área (Píxeles)", ylab = "Contraste Textural")
dev.off()

cat(" Proceso de R finalizado con éxito.\n")