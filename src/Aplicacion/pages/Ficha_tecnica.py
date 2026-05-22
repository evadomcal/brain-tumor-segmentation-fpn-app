# APLICACION/PAGES/FICHA_TECNICA

# Este script es el centro de control de calidad del sistema. 
# Su función es mostrar qué tan bien funciona el modelo U-Net implementado.
# Reúne las métricas del modelo y los análisis estadísticos hechos en R, permitiendo 
# que el médico verifique si los resultados son fiables antes de usarlos, y conozca sus limitaciones.

# LIBRERÍAS UTILIZADAS:
import streamlit as st      # Para construir la interfaz web interactiva.
import pandas as pd         # Para cargar y procesar las tablas de métricas (CSV).
from datetime import datetime # Para poner fechas de auditoría en el reporte.
from pathlib import Path    # Para gestionar rutas de archivos de forma robusta y multiplataforma.
import plotly.express as px # Para crear gráficos interactivos.
import plotly.graph_objects as go # Para visualizaciones personalizadas de alto nivel.
from PIL import Image       # Para cargar y mostrar las gráficas generadas en R.

# CONFIGURACIÓN DE LA PÁGINA: 
# Definimos el título de la pestaña y aprovechamos todo el ancho de la pantalla para las gráficas.
st.set_page_config(
    page_title="Ficha Técnica - MRAI",
    layout="wide" 
)

# Definimos el título principal y una subnota con markdown 
st.title("Ficha Técnica de Calidad")
st.markdown("### Sistema MRAI - Análisis de Tumores Cerebrales")
st.markdown("---")


# GESTIÓN DE RUTAS:
BASE_PATH = Path(__file__).parent.parent.parent.parent  # Ascendemos hasta la raíz 'TRABAJO'
DATOS_PROCESADOS = BASE_PATH / "datos_procesados"
EVAL_CALIDAD = DATOS_PROCESADOS / "evaluacion_calidad"  # Donde Python guarda las métricas
RESULTADOS_R = DATOS_PROCESADOS / "resultados_r"       # Donde R guarda sus análisis estadísticos

# ============================================================
# SECCIÓN 1: MÉTRICAS DEL MODELO 
# ============================================================
st.header(" Métricas de Calidad del Modelo de Segmentación")  # Título

# Buscamos el archivo de resumen generado durante el entrenamiento de la U-Net
resumen_metricas_path = EVAL_CALIDAD / "resumen_metricas.txt"

# Comprobamos que el archivo existe 
if resumen_metricas_path.exists():
    
    # Abrimos en modo lectura ('r') y usamos 'utf-8' para leer tildes y símbolos correctamente
    with open(resumen_metricas_path, "r", encoding="utf-8") as f:
        # Volcamos el contenido en la variable 'contenido'
        contenido = f.read()
    try:
        # Creamos un diccionario vacío inicial
        metricas = {}
        
        # Dividimos el texto largo en líneas individuales para analizarlas una por una
        for linea in contenido.split('\n'):
            # Buscamos ":" para identificar qué es una métrica (ej: "Dice: 0.94")
            if ':' in linea:
                # Separamos el nombre de la métrica (key) del valor numérico (val)
                key, val = linea.split(':', 1)

                # Solo guardamos la métrica si no es un valor 'total'
                if not key.strip().endswith('_total'):
                    # Guardamos en el diccionario limpiando espacios en blanco (strip)
                    metricas[key.strip()] = val.strip()
        
        # Si el diccionario tiene datos, los mostramos en la interfaz
        if metricas:
            # Creamos 4 columnas para las métricas
            cols = st.columns(4) 
            
            # Recorremos el diccionario para crear una tarjeta (st.metric) por cada métrica
            for idx, (metrica, valor) in enumerate(metricas.items()):
                # Usamos el índice y el operador '%' para repartir las tarjetas de forma equilibrada en las 4 columnas
                with cols[idx % 4]:
                    try:
                        # Convertimos el valor a decimal
                        valor_float = float(valor)
                        # Formateamos con 4 decimales 
                        valor_formateado = f"{valor_float:.4f}"
                    except ValueError:
                        # Si el valor no es un número, lo dejamos tal cual para evitar errores
                        valor_formateado = valor
                    
                    # Dibujamos la tarjeta visual de Streamlit con el nombre y el valor final
                    st.metric(metrica, valor_formateado)
    
    # Si el archivo está mal escrito y falla, mostramos el texto en bruto
    except:
        st.text(contenido) 

# Si el archivo no existe, mostramos un aviso
else:
    st.warning(" No se ha encontrado el registro de validación (resumen_metricas.txt)")

st.markdown("---")


# ============================================================
# SECCIÓN 2: ANÁLISIS GRÁFICO
# ============================================================
st.header(" Métricas de dispersión")

# Definimos la ruta al CSV que contiene los resultados de cada imagen
metricas_csv_path = EVAL_CALIDAD / "metricas_por_imagen.csv"

if metricas_csv_path.exists():
    
    # Usamos Pandas para leer la tabla de resultados
    df_metricas = pd.read_csv(metricas_csv_path)
    
    # Mostramos el Excel completo en la web. 
    # 'use_container_width' hace que la tabla ocupe todo el ancho disponible.
    st.dataframe(df_metricas, use_container_width=True)
    
    
    # BOXPLOT: 
    # El coeficiente 'dice' mide el solapamiento entre el tumor real y la predicción:
    # De toda la zona que marcamos entre los dos, ¿cuánta tenemos en común?". Penaliza menos los errores.
    # Cerca de 1: idénticos
    # Cerca de 0: coinciden poco
    if 'dice' in df_metricas.columns:
        fig = px.box(df_metricas, y='dice', title='Distribución del Coeficiente Dice')
        # Mostramos el gráfico de Plotly de forma interactiva
        st.plotly_chart(fig, use_container_width=True)
    
    # HISTOGRAMA: 
    # El coeficiente 'IoU' es más estricto, y por tanto, suele ser más bajo.
    # Se calcula dividiendo el área de la intersección (lo común) 
    # entre el área de la unión (lo que ambos marcaron). Mide la exactitud del área.
    if 'iou' in df_metricas.columns:
        # 'nbins=20' divide los resultados en 20 barras
        fig = px.histogram(df_metricas, x='iou', nbins=20, title='Histograma de IoU')
        st.plotly_chart(fig, use_container_width=True)

# Si el CSV no aparece, lanzamos un aviso preventivo
else:
    st.warning(" Faltan datos de validación individual (archivo CSV no encontrado)")

st.markdown("---")


# ============================================================
# SECCIÓN 3: ANÁLISIS ESTADÍSTICO CON R
# ============================================================

st.header(" Análisis Estadístico (R)")

# Mostramos los plots generados en R en dos columnas
col_r1, col_r2 = st.columns(2)

with col_r1:
    # Gráfica 1: Circularidad 
    img1_path = RESULTADOS_R / "01_DISTRIBUCION_CIRCULARIDAD.png"
    if img1_path.exists():
        st.image(str(img1_path), caption="Distribución de la Circularidad del Tumor", use_container_width=True)
    
    # Gráfica 2: Relación Tamaño/Forma (Correlación)
    img3_path = RESULTADOS_R / "04_TAMAÑO_VS_FORMA.png" if not (RESULTADOS_R / "03_CONTRASTE_POR_FORMA.png").exists() else RESULTADOS_R / "03_CONTRASTE_POR_FORMA.png"
    if img3_path.exists():
        st.image(str(img3_path), caption="Relación Tamaño vs Forma", use_container_width=True)

with col_r2:
    # Gráfica 3: Heterogeneidad
    img2_path = RESULTADOS_R / "02_HETEROGENEIDAD_VS_FORMA.png"
    if img2_path.exists():
        st.image(str(img2_path), caption="Heterogeneidad vs Forma", use_container_width=True)


# ============================================================
# SECCIÓN 4: RESULTADOS DEL ANALISIS
# ============================================================

st.header("Informa de Análisis Estadístico")

resultados_txt_path = RESULTADOS_R / "resultados_analisis.txt"

if resultados_txt_path.exists():
    with open(resultados_txt_path, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    with st.expander("Ver informe completo", expanded=True):
        st.text(contenido)
else:
    st.warning("No se encuentra el archivo resultados_analisis.txt")

st.markdown("---")

# ============================================================
# SECCIÓN 5: RESUMEN GENERAL
# ============================================================

st.header("Resumen General")

col_sum1, col_sum2, col_sum3 = st.columns(3)

with col_sum1:
    if df_metricas is not None and 'dice' in df_metricas.columns:
        st.metric("Dice promedio", f"{df_metricas['dice'].mean():.3f}")
    else:
        st.metric("Dice promedio", "N/A")

with col_sum2:
    if df_metricas is not None and 'iou' in df_metricas.columns:
        st.metric("IoU promedio", f"{df_metricas['iou'].mean():.3f}")
    else:
        st.metric("IoU promedio", "N/A")

with col_sum3:
    st.metric("Fecha análisis", datetime.now().strftime("%d/%m/%Y"))

st.markdown("---")



# ============================================================
# SECCIÓN 6: LIMITACIONES DEL MODELO
# ============================================================
st.header("Limitaciones del modelo")
col_lim1, col_lim2 = st.columns(2) # dos columnas

# LIMITACIONES 
with col_lim1:
    st.subheader("Limitaciones conocidas")
    st.markdown("-  No sustituye el juicio clínico, es una herramienta de apoyo.")
    st.markdown("-  No valido para tumores pediátricos")

# RECOMENDACIONES
with col_lim2:
    st.subheader("Recomendaciones de uso")
    st.markdown("-  Confirmar siempre los resultados con un neurocirujano.")
    st.markdown("-  Garantizar que la calidad de la imagen sea óptima para evitar fallos por borrosidad.")

st.markdown("---")

# Contacto y soporte
st.header("Soporte y Contacto")

col_cont1, col_cont2, col_cont3 = st.columns(3)

with col_cont1:
    st.markdown("**Soporte técnico**")
    st.markdown(" tecnologia@mrai.com")
    st.markdown("📞 +34 900 123 456")
    st.markdown(" 24/7 para urgencias")

with col_cont2:
    st.markdown("**Validación clínica**")
    st.markdown(" clinica@mrai.com")
    st.markdown("📞 +34 900 123 457")
    st.markdown(" L-V 9:00-18:00")

with col_cont3:
    st.markdown("**Reportes y sugerencias**")
    st.markdown(" feedback@mrai.com")
    st.markdown(" www.mrai.com/support")


st.caption(f" Reporte de calidad actualizado: {datetime.now().strftime('%d/%m/%Y')}")