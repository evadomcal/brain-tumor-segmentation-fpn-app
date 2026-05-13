#APLICACION/STREAMLIT


# LIBRERÍAS NECESARIAS:
import streamlit as st              # Para construir la interfaz web interactiva
import pandas as pd                 # Para gestionar catálogos y resultados clínicos (dataframes)
import numpy as np                  # Para procesar las imágenes (matrices .npy)
import plotly.express as px         # Herramienta de visualización rápida para generar gráficos estadísticos
import plotly.graph_objects as go   # Visualización avanzada para superponer la máscara del tumor sobre la imagen
from PIL import Image               # Librería de procesamiento de imágenes para abrir y previsualizar archivos TIF/TIFF
import io                           # Gestión de datos (Input/Output) directamente en la memoria RAM
from datetime import datetime       # Generación de marcas temporales 
from cliente import get_dagster_client # Conector entre la web y el orquestador de datos Dagster
import json                         # Convierte el formulario médico en un formato que el servidor entiende
from pathlib import Path            # Para gestionar rutas de archivos de forma segura y multiplataforma


# 1. CONFIGURACIÓN DE LA PÁGINA: título y diseño ancho
st.set_page_config(
    page_title="MRAI: Análisis de tumores cerebrales",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. DISEÑO PERSONALIZADO (CSS): le damos a la app un aspecto más clínico y profesional.
# utiliza el lenguaje markdown 
st.markdown("""
<style>
    /* Estilos generales de la interfaz */
    .stApp { background-color: #f5f7fb; }           /* Fondo de la web azul muy claro */
    h1, h2, h3 { color: #1a3a5c; }                  /* títulos en un tono azul oscuro médico */
    
    .stButton > button {                            /* botones de acción */
        background-color: #2c5f8a;                  /* Color azul para el botón */
        color: white;                               /* Texto del botón en blanco */
        border-radius: 6px;                         /* Bordes ligeramente redondeados */
    }
            
    /* Diseño del 'Semáforo de Urgencia' (Tarjetas de colores) */
    .urgency-high { 
        background: linear-gradient(135deg, #dc3545, #c82333); /* Fondo rojo degradado para máxima alerta */
        color: white;                                          /* Texto blanco */
        padding: 1.5rem;                                       /* Espaciado interno para que la tarjeta sea grande */
        border-radius: 10px;                                   /* Bordes muy redondeados */
        text-align: center; }                                  /* Texto centrado */

    .urgency-moderate { 
        background: linear-gradient(135deg, #ffc107, #e0a800); /* Fondo amarillo/naranja para alerta media */
        color: #1a3a5c;                                        /* Texto oscuro para que se lea bien sobre amarillo */
        padding: 1.5rem; 
        border-radius: 10px; 
        text-align: center; }

    .urgency-low { 
        background: linear-gradient(135deg, #28a745, #1e7e34); /* Fondo verde para casos estables */
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        text-align: center; }
</style>
""", unsafe_allow_html=True) # Permite que Streamlit aplique estos estilos visuales personalizados

# 3. GESTIÓN DE SESIÓN (Session State): Esto permite que la web "recuerde" los datos
# aunque el usuario se mueva por la app. Es la memoria a corto plazo.

# Verificamos si es la primera vez que el usuario entra en la web para asignarle un ID único, 
# basándonos en el momento en que entra.
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

# Comprobamos si ya hemos guardado los datos del formulario clínico
if 'datos_clinicos_guardados' not in st.session_state:
    st.session_state.datos_clinicos_guardados = False
    
# ------------------------------------------------------------
# SIDEBAR: El panel lateral a la izquierda para meter los datos del paciente
# ------------------------------------------------------------

with st.sidebar:
    # Logotipo y título
    st.image("https://www.gruporecoletas.com/imagenes/institutos/110_tumor-cerebral-neurocirugia.png", width=80)
    st.title("Datos Clínicos")
    
    # Usamos st.form para que la página no se recargue cada vez que tocamos un botón
    with st.form("clinical_form"):
        # SECCIÓN DEMOGRÁFICA: 2 columnas 
        st.markdown("### Datos demográficos")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Género", ["", "Masculino", "Femenino"])
            age = st.number_input("Edad (años)", min_value=0, max_value=120, value=55)
            
        with col2:
            # Fundamentales para que el Modelo de Urgencia
            race = st.selectbox("Raza", ["", "Blanca", "Afroamericana", "Asiática", "Otra"])
            ethnicity = st.selectbox("Etnicidad", ["", "No Hispano", "Hispano"])
        
        # SECCIÓN PATOLÓGICA: sobre el tipo de cáncer
        st.markdown("### Datos patológicos")
        histological_type = st.selectbox("Tipo histológico", ["", "Glioblastoma", "Astrocitoma", "Oligodendroglioma", "Meningioma"])
        tumor_grade = st.selectbox("Grado histológico", ["", "Grado I", "Grado II", "Grado III", "Grado IV"])
        
        # 3. BOTÓN DE ENVÍO: 
        submitted = st.form_submit_button(" Guardar datos clínicos")
        
        if submitted:
            # Creamos un diccionario (JSON) con la información.
            # Si un campo está vacío, le asignamos 'None' (datos faltante).
            st.session_state.datos_clinicos = {
                "gender": gender if gender != "" else None,
                "age_at_initial_pathologic": age,
                "race": race if race != "" else None,
                "neoplasm_histologic_grade": tumor_grade if tumor_grade != "" else None,
            }
            # Datos guardados para poder continuar
            st.session_state.datos_clinicos_guardados = True
            st.success("✅ Datos guardados correctamente en la sesión")


# ------------------------------------------------------------
# CONTENIDO PRINCIPAL: Carga de imagen y Visualización
# ------------------------------------------------------------
st.title(" MRAI - Análisis de Tumores Cerebrales")

# Dividimos la pantalla en dos columnas: entrada y salida de resultados
col_imagen, col_resultados = st.columns([1, 1])

# COLUMNA IZQ: subida de imagen
with col_imagen:
    st.markdown("###  Cargar imagen MRI")
    # Para subir archivos:  formato TIF/TIFF 
    uploaded_file = st.file_uploader("Seleccionar imagen (TIF, TIFF)", type=["tif", "tiff"])
    
    if uploaded_file:
        # Abrimos la imagen, le damos título y que ocupe todo.
        imagen_preview = Image.open(uploaded_file)
        st.image(imagen_preview, caption="MRI original", use_container_width=True)
        
        # Botón para activar el informe de la IA
        if st.button(" Procesar imagen y evaluar urgencia", type="primary"):
            # st.spinner muestra un mensaje de carga
            with st.spinner(" La IA está analizando el tumor..."):
                # Conectamos con el cliente de Dagster para ejecutar el flujo
                client = get_dagster_client()
                # Convertimos la imagen a bytes para enviarla por la U-Net
                imagen_bytes = uploaded_file.getvalue()
                st.session_state.imagen_bytes = imagen_bytes
                
                # 1. La U-Net detecta el tumor y calculamos las características tumorales
                mascara, df_caracteristicas = client.procesar_imagen(imagen_bytes, st.session_state.datos_clinicos)
                
                # 2. El modelo de urgencia usa las características de 1. para predecir el riesgo
                urgencia = client.calcular_urgencia(df_caracteristicas, st.session_state.datos_clinicos)
                
                # Guardamos todo en la 'memoria de sesión' para que no se pierda al recargar
                st.session_state.mascara = mascara
                st.session_state.caracteristicas_df = df_caracteristicas
                st.session_state.urgencia = urgencia
                st.session_state.imagen_procesada = True
                st.rerun() # Refrescamos la página para mostrar los resultados en la columna derecha

# COLUMNA DERECHA: informe médico
with col_resultados:
    # Si la imagen ha sido procesada:
    if st.session_state.imagen_procesada:
        st.markdown("###  Segmentación del tumor")
        
        # Reconstruimos la imagen original desde los bytes guardados
        img_array = np.array(Image.open(io.BytesIO(st.session_state.imagen_bytes)))
        
        # Creamos una figura interactiva con Plotly
        fig = go.Figure()
        # Capa 1: El cerebro en escala de grises (Mapa de calor)
        fig.add_trace(go.Heatmap(z=img_array, colorscale='gray', showscale=False))
        
        # Capa 2: La máscara del tumor. 
        # np.ma.masked_where para que los píxeles donde no hay tumor sean transparentes
        mask_overlay = np.ma.masked_where(st.session_state.mascara == 0, st.session_state.mascara)
        # Pintamos el tumor en rojo (Reds) con un 50% de transparencia (opacity=0.5)
        fig.add_trace(go.Heatmap(z=mask_overlay, colorscale='Reds', opacity=0.5, showscale=False))
        
        # Mostramos el gráfico en la web, ocupando todo el espacio
        st.plotly_chart(fig, use_container_width=True)

        # SEMÁFORO DE URGENCIA: usando las predicciones de la regresión logística
        urg = st.session_state.urgencia
        if urg < 0.3:
            # Riesgo bajo: Color Verde
            # Se aplica el estilo verde ('urgency-low'). 
            # El formato {urg:.1%} convierte el decimal (0.15) en un porcentaje (15.0%).
            st.markdown(f'<div class="urgency-low"><h2> BAJA ({urg:.1%})</h2></div>', unsafe_allow_html=True)
        elif urg < 0.7:
            # Riesgo moderado: Color Amarillo
            # Se aplica el estilo amarillo/naranja ('urgency-moderate').
            st.markdown(f'<div class="urgency-moderate"><h2> MODERADA ({urg:.1%})</h2></div>', unsafe_allow_html=True)
        else:
            # Riesgo alto: Color Rojo
            # Se aplica el estilo rojo ('urgency-high').
            st.markdown(f'<div class="urgency-high"><h2> ALTA ({urg:.1%})</h2></div>', unsafe_allow_html=True)

        # Finalmente, mostramos los datos exactos (Área, Circularidad, etc.) en una tabla
        st.markdown("###  Características extraídas")
        st.dataframe(st.session_state.caracteristicas_df)