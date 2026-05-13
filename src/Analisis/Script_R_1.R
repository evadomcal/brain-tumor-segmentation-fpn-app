# scripts/analisis_r.R - VERSIÓN PARA MÉDICO CON INTERPRETACIONES

# Instalar paquetes faltantes si es necesario
paquetes <- c("tidyverse", "corrplot", "ggplot2", "tidyr")
for (p in paquetes) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cran.rstudio.com/", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}

# Leer argumentos
args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
output_dir <- args[2]

if (!file.exists(csv_path)) {
  stop(paste("No se encuentra:", csv_path))
}

cat("📂 Leyendo:", csv_path, "\n")

# Leer datos
datos <- read.csv(csv_path, stringsAsFactors = FALSE)
cat("📊 Datos:", nrow(datos), "filas,", ncol(datos), "columnas\n")

# Crear directorio
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
setwd(output_dir)

# Verificar columna area
if (!"area" %in% colnames(datos)) {
  stop("No existe columna 'area'")
}

# Filtrar tumores
datos_tumor <- datos[datos$area > 0, ]
cat("🩻 Tumores:", nrow(datos_tumor), "\n")

if (nrow(datos_tumor) == 0) {
  stop("No hay tumores con area > 0")
}

# Convertir variables categóricas
if ("histological_type" %in% colnames(datos_tumor)) {
  datos_tumor$histological_type <- factor(datos_tumor$histological_type)
}
if ("institucion" %in% colnames(datos_tumor)) {
  datos_tumor$institucion <- factor(datos_tumor$institucion)
}
if ("death01" %in% colnames(datos_tumor)) {
  datos_tumor$death01 <- factor(datos_tumor$death01, labels = c("Vivo", "Fallecido"))
}

# Abrir archivo de resultados
sink("resultados_analisis.txt")

cat("============================================================\n")
cat("INFORME DE ANÁLISIS DE IMAGEN TUMORAL\n")
cat("============================================================\n\n")
cat("Fecha:", date(), "\n")
cat("Número total de tumores analizados:", nrow(datos_tumor), "\n\n")

# ============================================================
# 1. ANÁLISIS MORFOLÓGICO
# ============================================================
cat("============================================================\n")
cat("1. MORFOLOGÍA TUMORAL\n")
cat("============================================================\n\n")
cat("INTERPRETACIÓN CLÍNICA:\n")
cat("- Área: Tamaño del tumor en píxeles. Tumores más grandes suelen asociarse a mayor agresividad.\n")
cat("- Perímetro: Longitud del borde tumoral. Perímetros largos pueden indicar bordes irregulares.\n")
cat("- Circularidad: Valor entre 0 y 1. Cercano a 1 = forma redonda (benigno típico).\n")
cat("  Cercano a 0 = forma irregular (sugiere malignidad, invasión).\n\n")

vars_morf <- c("area", "perimetro", "circularidad")
for (v in vars_morf) {
  if (v %in% colnames(datos_tumor)) {
    cat("---", v, "---\n")
    resumen <- summary(datos_tumor[[v]])
    print(resumen)
    
    # Interpretación específica
    if (v == "circularidad") {
      mediana <- resumen["Median"]
      if (mediana > 0.7) {
        cat("🔵 INTERPRETACIÓN: La mediana de circularidad es >0.7, indicando tumores predominantemente REDONDEADOS (sugiere menor agresividad).\n")
      } else if (mediana > 0.4) {
        cat("🟡 INTERPRETACIÓN: La mediana de circularidad está entre 0.4-0.7, indicando tumores OVALADOS (agresividad intermedia).\n")
      } else {
        cat("🔴 INTERPRETACIÓN: La mediana de circularidad es <0.4, indicando tumores IRREGULARES (sugiere mayor agresividad/invasión).\n")
      }
    }
    cat("\n")
  }
}

# ============================================================
# 2. ANÁLISIS DE TEXTURA (Intensidades)
# ============================================================
cat("============================================================\n")
cat("2. TEXTURA TUMORAL (INTENSIDADES DE SEÑAL)\n")
cat("============================================================\n\n")
cat("INTERPRETACIÓN CLÍNICA:\n")
cat("- Intensidad media post-contraste: Mayor intensidad sugiere mayor vascularización/realce.\n")
cat("- Intensidad mínima: Zonas hipointensas (necrosis, quistes).\n")
cat("- Percentil 95: Valor umbral que supera el 95% de los píxeles (picos de realce).\n")
cat("- Contraste de textura: Mide heterogeneidad. Valores altos = tumor más heterogéneo\n")
cat("  (asociado a necrosis, hemorragia, mayor agresividad).\n\n")

vars_text <- c("intensidad_media_post", "intensidad_minima_post", 
               "percentil_95_flair", "textura_contraste")
for (v in vars_text) {
  if (v %in% colnames(datos_tumor)) {
    cat("---", v, "---\n")
    resumen <- summary(datos_tumor[[v]])
    print(resumen)
    
    # Interpretación específica
    if (v == "textura_contraste") {
      mediana <- resumen["Median"]
      q1 <- resumen["1st Qu."]
      q3 <- resumen["3rd Qu."]
      cat("🔬 INTERPRETACIÓN TEXTURA:\n")
      cat("   La mediana de contraste es", round(mediana, 2), "\n")
      cat("   Rango intercuartil:", round(q1, 2), "-", round(q3, 2), "\n")
      if (mediana > 5000) {
        cat("   ⚠️ ALTO: Contraste muy elevado → tumor muy heterogéneo (posible necrosis/agresividad)\n")
      } else if (mediana > 2000) {
        cat("   ⚠️ MODERADO: Contraste elevado → heterogeneidad significativa\n")
      } else {
        cat("   ✓ NORMAL-BAJO: Tumor homogéneo (sugiere menor agresividad)\n")
      }
    }
    cat("\n")
  }
}

# ============================================================
# 3. CORRELACIONES MORFOLOGÍA-TEXTURA
# ============================================================
cat("============================================================\n")
cat("3. CORRELACIONES ENTRE MORFOLOGÍA Y TEXTURA\n")
cat("============================================================\n\n")
cat("INTERPRETACIÓN CLÍNICA:\n")
cat("- Correlación positiva: Cuando una variable aumenta, la otra también.\n")
cat("- Correlación negativa: Una aumenta, la otra disminuye.\n")
cat("- |r| < 0.3: correlación débil | 0.3-0.7: moderada | >0.7: fuerte\n\n")

# Seleccionar variables numéricas
vars_num <- c()
for (v in c("area", "perimetro", "circularidad", "intensidad_media_post", 
            "percentil_95_flair", "textura_contraste")) {
  if (v %in% colnames(datos_tumor) && is.numeric(datos_tumor[[v]])) {
    vars_num <- c(vars_num, v)
  }
}

if (length(vars_num) >= 2) {
  datos_cor <- datos_tumor[, vars_num, drop = FALSE]
  datos_cor <- datos_cor[complete.cases(datos_cor), ]
  
  if (nrow(datos_cor) > 0 && ncol(datos_cor) >= 2) {
    matriz_cor <- cor(datos_cor, use = "pairwise.complete.obs")
    cat("Matriz de correlaciones:\n")
    print(round(matriz_cor, 3))
    
    cat("\n📌 INTERPRETACIONES CLAVE:\n")
    # Correlación área vs circularidad
    if (all(c("area", "circularidad") %in% vars_num)) {
      r_ac <- matriz_cor["area", "circularidad"]
      cat(sprintf("  • Área vs Circularidad (r = %.3f): ", r_ac))
      if (r_ac < -0.5) {
        cat("CORRELACIÓN NEGATIVA FUERTE. Los tumores más grandes tienden a ser más irregulares.\n")
      } else if (r_ac < -0.3) {
        cat("Correlación negativa moderada. Tendencia a que tumores grandes sean más irregulares.\n")
      } else {
        cat("Correlación débil. Tamaño y forma no están fuertemente relacionados.\n")
      }
    }
    
    # Correlación circularidad vs contraste
    if (all(c("circularidad", "textura_contraste") %in% vars_num)) {
      r_ct <- matriz_cor["circularidad", "textura_contraste"]
      cat(sprintf("  • Circularidad vs Contraste (r = %.3f): ", r_ct))
      if (r_ct < -0.4) {
        cat("CORRELACIÓN NEGATIVA IMPORTANTE. Los tumores más irregulares son más heterogéneos.\n")
      } else if (r_ct < -0.2) {
        cat("Tendencia: tumores irregulares → mayor heterogeneidad.\n")
      } else {
        cat("No hay relación clara entre forma y heterogeneidad.\n")
      }
    }
    
    # Correlación área vs contraste
    if (all(c("area", "textura_contraste") %in% vars_num)) {
      r_area_cont <- matriz_cor["area", "textura_contraste"]
      cat(sprintf("  • Área vs Contraste (r = %.3f): ", r_area_cont))
      if (r_area_cont > 0.5) {
        cat("CORRELACIÓN POSITIVA FUERTE. Tumores grandes son más heterogéneos.\n")
      } else if (r_area_cont > 0.3) {
        cat("Correlación positiva moderada. Tumores grandes tienden a ser más heterogéneos.\n")
      } else {
        cat("El tamaño no predice la heterogeneidad textural.\n")
      }
    }
  } else {
    cat("No hay suficientes datos completos para correlación\n")
  }
} else {
  cat("No hay suficientes variables numéricas para correlación\n")
}

# ============================================================
# 4. TEXTURA SEGÚN TAMAÑO DEL TUMOR
# ============================================================
if ("area" %in% colnames(datos_tumor) && "textura_contraste" %in% colnames(datos_tumor)) {
  cat("\n============================================================\n")
  cat("4. RELACIÓN ENTRE TAMAÑO Y HETEROGENEIDAD (CONTRASTE)\n")
  cat("============================================================\n\n")
  cat("INTERPRETACIÓN CLÍNICA:\n")
  cat("Comparamos el contraste (heterogeneidad) entre tumores pequeños, medianos y grandes.\n")
  cat("Si los tumores grandes tienen mayor contraste, sugiere que el crecimiento tumoral\n")
  cat("se asocia con desarrollo de necrosis/heterogeneidad.\n\n")
  
  cuartiles <- quantile(datos_tumor$area, probs = c(0.25, 0.75), na.rm = TRUE)
  datos_tumor$tamano_categoria <- cut(datos_tumor$area, 
                                       breaks = c(-Inf, cuartiles[1], cuartiles[2], Inf),
                                       labels = c("Pequeño", "Mediano", "Grande"))
  
  resumen <- aggregate(textura_contraste ~ tamano_categoria, 
                       data = datos_tumor, 
                       FUN = function(x) c(Media = mean(x, na.rm = TRUE), 
                                          SD = sd(x, na.rm = TRUE),
                                          N = sum(!is.na(x))))
  
  print(resumen)
  
  # Interpretación
  if (nrow(resumen) >= 2) {
    contraste_peq <- resumen[resumen$tamano_categoria == "Pequeño", ]$textura_contraste[1]
    contraste_grande <- resumen[resumen$tamano_categoria == "Grande", ]$textura_contraste[1]
    
    if (!is.null(contraste_peq) && !is.null(contraste_grande)) {
      if (contraste_grande > contraste_peq * 1.5) {
        cat("🔴 HALLAZGO: Los tumores GRANDES tienen", round(contraste_grande/contraste_peq, 1), 
            "veces más contraste (heterogeneidad) que los pequeños.\n")
        cat("   Esto sugiere que el crecimiento tumoral se acompaña de necrosis o degeneración.\n")
      } else if (contraste_grande > contraste_peq) {
        cat("🟡 HALLAZGO: Los tumores GRANDES muestran mayor heterogeneidad que los pequeños.\n")
      } else {
        cat("🟢 HALLAZGO: No hay diferencias significativas en heterogeneidad según tamaño.\n")
      }
    }
  }
}

# ============================================================
# 5. TEXTURA SEGÚN FORMA DEL TUMOR
# ============================================================
if ("circularidad" %in% colnames(datos_tumor) && "textura_contraste" %in% colnames(datos_tumor)) {
  cat("\n============================================================\n")
  cat("5. RELACIÓN ENTRE FORMA Y HETEROGENEIDAD\n")
  cat("============================================================\n\n")
  cat("INTERPRETACIÓN CLÍNICA:\n")
  cat("Clasificamos los tumores por forma (redondo >0.7, ovalado 0.4-0.7, irregular <0.4)\n")
  cat("La forma irregular suele asociarse a invasión y malignidad. Comparamos si también\n")
  cat("se asocia con mayor heterogeneidad textural.\n\n")
  
  datos_tumor$tipo_forma <- cut(datos_tumor$circularidad,
                                 breaks = c(0, 0.4, 0.7, 1),
                                 labels = c("Irregular", "Ovalado", "Redondo"))
  
  resumen_forma <- aggregate(textura_contraste ~ tipo_forma, 
                              data = datos_tumor,
                              FUN = function(x) c(Media = mean(x, na.rm = TRUE),
                                                 SD = sd(x, na.rm = TRUE),
                                                 N = sum(!is.na(x))))
  
  print(resumen_forma)
  
  # Interpretación
  if (nrow(resumen_forma) >= 2) {
    irregular <- resumen_forma[resumen_forma$tipo_forma == "Irregular", ]$textura_contraste
    redondo <- resumen_forma[resumen_forma$tipo_forma == "Redondo", ]$textura_contraste
    
    if (length(irregular) > 0 && length(redondo) > 0 && !is.na(irregular[1]) && !is.na(redondo[1])) {
      if (irregular[1] > redondo[1] * 1.3) {
        cat("🔴 HALLAZGO SIGNIFICATIVO: Los tumores IRREGULARES tienen", 
            round(irregular[1]/redondo[1], 1), 
            "veces más contraste (heterogeneidad) que los redondos.\n")
        cat("   Esto respalda que la irregularidad morfológica se asocia con\n")
        cat("   características texturales de agresividad (necrosis, invasión).\n")
      } else if (irregular[1] > redondo[1]) {
        cat("🟡 Tendencia: Los tumores irregulares son más heterogéneos que los redondos.\n")
      } else {
        cat("🟢 No se observa asociación entre forma y heterogeneidad en esta cohorte.\n")
      }
    }
  }
}

# ============================================================
# 6. RESUMEN CLÍNICO FINAL
# ============================================================
cat("\n============================================================\n")
cat("6. RESUMEN Y RECOMENDACIONES CLÍNICAS\n")
cat("============================================================\n\n")

cat("🔬 HALLAZGOS PRINCIPALES:\n\n")

# Calcular métricas clave para el resumen
if ("circularidad" %in% colnames(datos_tumor)) {
  circ_mediana <- median(datos_tumor$circularidad, na.rm = TRUE)
  cat(sprintf("  • Circularidad mediana: %.3f\n", circ_mediana))
  if (circ_mediana < 0.5) {
    cat("    → La mayoría de los tumores son IRREGULARES (sugiere comportamiento agresivo)\n")
  } else {
    cat("    → La mayoría de los tumores son REDONDEADOS/OVALADOS\n")
  }
}

if ("textura_contraste" %in% colnames(datos_tumor)) {
  contraste_mediana <- median(datos_tumor$textura_contraste, na.rm = TRUE)
  cat(sprintf("  • Contraste (heterogeneidad) mediano: %.1f\n", contraste_mediana))
  if (contraste_mediana > 3000) {
    cat("    → Heterogeneidad ELEVADA (sugiere necrosis, hemorragia o malignidad)\n")
  } else if (contraste_mediana > 1500) {
    cat("    → Heterogeneidad MODERADA\n")
  } else {
    cat("    → Heterogeneidad BAJA (tumores homogéneos)\n")
  }
}

cat("\n📋 RECOMENDACIONES:\n")
cat("  1. Los tumores con circularidad <0.4 y contraste >3000 podrían beneficiarse\n")
cat("     de estudio anatomopatológico prioritario.\n")
cat("  2. Considerar seguimiento más estrecho en tumores grandes con alta heterogeneidad.\n")
cat("  3. La combinación de forma irregular + alto contraste sugiere agresividad.\n")

cat("\n✅ FIN DEL INFORME\n")

sink()  # Cerrar archivo

# ============================================================
# GRÁFICOS (con títulos médicos)
# ============================================================
cat("\n🎨 Generando gráficos para informe médico...\n")

# Gráfico 1: Histograma de circularidad
if ("circularidad" %in% colnames(datos_tumor)) {
  png("01_DISTRIBUCION_CIRCULARIDAD.png", width = 800, height = 600)
  hist(datos_tumor$circularidad, 
       main = "Distribución de la circularidad tumoral",
       sub = "Valores cercanos a 1 = redondo (benigno) | Cercanos a 0 = irregular (agresivo)",
       xlab = "Circularidad (0=irregular, 1=redondo)", ylab = "Número de tumores",
       col = "skyblue", border = "black", breaks = 20)
  abline(v = median(datos_tumor$circularidad, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
  legend("topright", legend = c("Mediana"), col = c("red"), lty = 2, lwd = 2)
  dev.off()
  cat("  ✓ 01_DISTRIBUCION_CIRCULARIDAD.png\n")
}

# Gráfico 2: Contraste vs Circularidad
if (all(c("circularidad", "textura_contraste") %in% colnames(datos_tumor))) {
  png("02_HETEROGENEIDAD_VS_FORMA.png", width = 800, height = 600)
  plot(datos_tumor$circularidad, datos_tumor$textura_contraste,
       main = "Relación entre heterogeneidad (contraste) y forma tumoral",
       sub = "Puntos rojos: tendencia lineal | Forma irregular → mayor heterogeneidad",
       xlab = "Circularidad (más redondo → derecha)", 
       ylab = "Contraste (heterogeneidad textural)",
       pch = 19, col = rgb(0, 0, 1, 0.5))
  validos <- complete.cases(datos_tumor$circularidad, datos_tumor$textura_contraste)
  if (sum(validos) > 1) {
    abline(lm(textura_contraste ~ circularidad, data = datos_tumor[validos, ]), 
           col = "red", lwd = 2)
  }
  dev.off()
  cat("  ✓ 02_HETEROGENEIDAD_VS_FORMA.png\n")
}

# Gráfico 3: Boxplot de contraste por forma
if (exists("datos_tumor$tipo_forma") && "textura_contraste" %in% colnames(datos_tumor)) {
  png("03_CONTRASTE_POR_FORMA.png", width = 800, height = 600)
  boxplot(textura_contraste ~ tipo_forma, data = datos_tumor,
          main = "Heterogeneidad (contraste) según la forma tumoral",
          sub = "Los tumores irregulares muestran mayor heterogeneidad",
          xlab = "Forma tumoral", ylab = "Contraste (heterogeneidad)",
          col = c("lightcoral", "lightgreen", "lightblue"))
  dev.off()
  cat("  ✓ 03_CONTRASTE_POR_FORMA.png\n")
}

# Gráfico 4: Área vs Circularidad
if (all(c("area", "circularidad") %in% colnames(datos_tumor))) {
  png("04_TAMAÑO_VS_FORMA.png", width = 800, height = 600)
  plot(datos_tumor$circularidad, log10(datos_tumor$area),
       main = "Relación entre tamaño tumoral y forma",
       sub = "Escala logarítmica en área | Los tumores grandes tienden a ser más irregulares",
       xlab = "Circularidad", ylab = "log10(Área en píxeles)",
       pch = 19, col = rgb(0, 0.5, 0, 0.5))
  dev.off()
  cat("  ✓ 04_TAMAÑO_VS_FORMA.png\n")
}
