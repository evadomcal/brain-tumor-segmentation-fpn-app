# APLICACION/STREAMLIT

# Esta aplicación web permite a los médicos cargar resonancias magnéticas y obtener
# un análisis automático del tumor, incluyendo segmentación, características radiomicas
# y un nivel de urgencia clínica.

# ========================
# LIBRERÍAS NECESARIAS:
# ========================

import streamlit as st              # Para construir la interfaz web interactiva
import pandas as pd                 # Para gestionar características del tumor y resultados
import numpy as np                  # Para procesar las imágenes (matrices .npy)
import plotly.express as px         # Herramienta de visualización rápida para gráficos
import plotly.graph_objects as go   # Visualización avanzada para superponer la máscara del tumor
from PIL import Image               # Para abrir y previsualizar archivos TIF/TIFF
import io                           # Gestión de datos en memoria RAM
from datetime import datetime       # Generación de marcas temporales para sesiones
from cliente import get_dagster_client  # Conector entre la web y el orquestador Dagster
import json                         # Para convertir datos clínicos a formato JSON
from pathlib import Path            # Para gestionar rutas de archivos multiplataforma
import tempfile                     # Para crear carpetas temporales de resultados
import zipfile                      # Para empaquetar todos los resultados en un ZIP


# ========================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ========================
# Establecemos el título, icono, diseño ancho y estado del menú lateral

st.set_page_config(
    page_title="MRAI: Análisis de tumores cerebrales",  # Título que aparece en la pestaña del navegador
    page_icon="🧠",                                     # Emoji como icono de pestaña
    layout="wide",                                      # Usar todo el ancho de la pantalla
    initial_sidebar_state="expanded",                  # Menú lateral visible desde el inicio
)


# ========================
# 2. DISEÑO PERSONALIZADO (CSS)
# ========================
# Aplicamos estilos profesionales con aspecto clínico usando Markdown + CSS

st.markdown("""
<style>
    /* Estilo general */
    .main {
        background-color: #f5f7fb;      /* Fondo azul muy claro */
    }
    .stApp {
        background-color: #f5f7fb;      /* Fondo principal de la app */
    }
    
    /* Estilos para títulos */
    h1, h2, h3 {
        color: #1a3a5c;                 /* Azul marino oscuro (tono médico) */
        font-weight: 600;               /* Negrita semi-fuerte */
        letter-spacing: -0.5px;         /* Espaciado ligeramente reducido */
    }
    
    /* Botones de acción */
    .stButton > button {
        background-color: #2c5f8a;      /* Azul */
        color: white;                   /* Texto blanco */
        border-radius: 6px;             /* Bordes ligeramente redondeados */
        border: none;                   /* Sin borde adicional */
        padding: 0.6rem 1.2rem;         /* Espaciado interno cómodo */
        font-weight: 500;               /* Texto semi-negrita */
        transition: all 0.2s;           /* Animación suave al pasar el ratón */
    }
    .stButton > button:hover {
        background-color: #1a3a5c;      /* Color más oscuro al hover */
        transform: translateY(-1px);    /* Efecto de elevación sutil */
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); /* Sombra para destacar */
    }
    
    /* Tarjetas para métricas (características del tumor) */
    .metric-card {
        background-color: white;         /* Fondo blanco para contraste */
        border-radius: 10px;            /* Bordes muy redondeados */
        padding: 1rem;                  /* Espaciado interno */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Sombra ligera */
        border: 1px solid #e0e4e8;      /* Borde sutil gris */
    }
    
    /* Cuadro informativo para consejos/ayuda */
    .info-box {
        background-color: #e8f0f8;      /* Azul muy pálido */
        border-left: 4px solid #2c5f8a; /* Barra lateral azul (destaca el texto) */
        padding: 1rem;                  /* Espaciado interno */
        border-radius: 4px;             /* Bordes redondeados */
        margin: 1rem 0;                 /* Separación vertical */
    }
    
    /* SEMÁFORO DE URGENCIA - Tarjetas de colores según nivel de riesgo */
    .urgency-high { 
        background: linear-gradient(135deg, #dc3545, #c82333); /* Rojo degradado para máxima alerta */
        color: white;                                           /* Texto blanco */
        padding: 1.5rem;                                        /* Espaciado interno grande */
        border-radius: 10px;                                    /* Bordes redondeados */
        text-align: center;                                     /* Texto centrado */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);                  /* Sombra para destacar */
    }

    .urgency-moderate { 
        background: linear-gradient(135deg, #ffc107, #e0a800); /* Amarillo/naranja para alerta media */
        color: #1a3a5c;                                        /* Texto oscuro */
        padding: 1.5rem; 
        border-radius: 10px; 
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .urgency-low { 
        background: linear-gradient(135deg, #28a745, #1e7e34); /* Verde */
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Línea separadora */
    hr {
        margin: 2rem 0;                 /* Separación vertical amplia */
        border-color: #e0e4e8;          /* Gris */
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True permite que Streamlit aplique estos estilos


# ========================
# 3. GESTIÓN DE SESIÓN (Session State)
# ========================
# El session_state permite que la web "recuerde" los datos aunque el usuario navegue.
# Es como la memoria RAM de la aplicación.

# Inicializamos todas las variables de estado si es la primera vez que el usuario entra
if 'session_id' not in st.session_state:
    # Creamos un ID único para la sesión basado en la fecha y hora exacta (incluye microsegundos)
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

if 'datos_clinicos_guardados' not in st.session_state:
    st.session_state.datos_clinicos_guardados = False  # Indica si el médico ya guardó el formulario

if 'datos_clinicos' not in st.session_state:
    st.session_state.datos_clinicos = {}  # Diccionario (JSON) con los datos del paciente

if 'imagen_procesada' not in st.session_state:
    st.session_state.imagen_procesada = False  # Flag para saber si ya se analizó la imagen

if 'mascara' not in st.session_state:
    st.session_state.mascara = None  # Matriz binaria (0=fondo, 1=tumor)

if 'caracteristicas_df' not in st.session_state:
    st.session_state.caracteristicas_df = None  # DataFrame con área, circularidad, etc.

if 'urgencia' not in st.session_state:
    st.session_state.urgencia = None  # Probabilidad de urgencia (0.0 a 1.0)

if 'imagen_bytes' not in st.session_state:
    st.session_state.imagen_bytes = None  # Datos binarios de la MRI subida

if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None  # Carpeta temporal con resultados guardados

# Forzamos modo directo (ignoramos modo desarrollo)
st.session_state.modo_directo = True
st.session_state.modo_desarrollo = False


# ========================
# 4. FUNCIONES AUXILIARES
# ========================

def crear_zip_descargable(ruta_carpeta: str) -> bytes:
    """
    Crea un archivo ZIP con todos los archivos de una carpeta.
    
    Args:
        ruta_carpeta: Ruta de la carpeta que contiene los resultados
        
    Returns:
        bytes: Datos binarios del archivo ZIP listo para descargar
    """
    zip_buffer = io.BytesIO()  # Buffer en memoria (no se guarda en disco)
    
    # Creamos el archivo ZIP con compresión DEFLATED
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        carpeta_path = Path(ruta_carpeta)
        for archivo in carpeta_path.iterdir():  # Iteramos sobre cada archivo
            if archivo.is_file():
                # Añadimos el archivo al ZIP usando solo su nombre (sin ruta completa)
                zip_file.write(archivo, arcname=archivo.name)
    
    zip_buffer.seek(0)  # Volvemos al inicio del buffer para leerlo
    return zip_buffer.getvalue()  # Devolvemos los bytes del ZIP


def limpiar_sesion():
    """
    Limpia todos los datos de la sesión actual para empezar un nuevo caso.
    Útil cuando el médico quiere analizar otro paciente diferente.
    """
    client = get_dagster_client()  # Conectamos con Dagster
    client.limpiar_datos_sesion(st.session_state.session_id)  # Limpiamos en el servidor
    
    # Reiniciamos todas las variables de estado a sus valores por defecto
    st.session_state.datos_clinicos_guardados = False
    st.session_state.imagen_procesada = False
    st.session_state.mascara = None
    st.session_state.caracteristicas_df = None
    st.session_state.urgencia = None
    st.session_state.imagen_bytes = None
    st.session_state.temp_dir = None
    st.session_state.datos_clinicos = {}
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Nuevo ID
    
    st.success("Sesión limpiada correctamente")  # Mensaje de confirmación
    st.rerun()  # Recargamos la página para mostrar el estado limpio


# ========================
# 5. SIDEBAR: PANEL LATERAL CON DATOS CLÍNICOS
# ========================
# El médico introduce los datos demográficos y patológicos del paciente

with st.sidebar:
    # Logotipo de la aplicación (imagen de neurocirugía)
    st.image("https://www.gruporecoletas.com/imagenes/institutos/110_tumor-cerebral-neurocirugia.png", width=80)
    st.title("Datos Clínicos")
    st.markdown("*Campos opcionales - pueden dejarse vacíos*")  # No son obligatorios para el análisis
    
    # Usamos st.form para que la página NO se recargue cada vez que se toca un campo
    with st.form("clinical_form"):
        st.markdown("### Datos demográficos")
        col1, col2 = st.columns(2)  # Dos columnas para organizar mejor
        
        with col1:
            gender = st.selectbox("Género", ["", "Masculino", "Femenino"])
            age = st.number_input("Edad (años)", min_value=0, max_value=120, value=55, 
                                  help="Puede dejar el valor por defecto si no lo sabe")
            
        with col2:
            race = st.selectbox("Raza", ["", "Blanca", "Afroamericana", "Asiática", "Otra"])
            ethnicity = st.selectbox("Etnicidad", ["", "No Hispano", "Hispano"])
        
        st.markdown("### Datos patológicos")
        histological_type = st.selectbox(
            "Tipo histológico",
            ["", "Glioblastoma", "Astrocitoma", "Oligodendroglioma", "Meningioma"]
        )
        tumor_grade = st.selectbox("Grado histológico", ["", "Grado I", "Grado II", "Grado III", "Grado IV"])
        tumor_location = st.selectbox("Localización", ["", "Frontal", "Temporal", "Parietal", "Occipital", "Cerebelo"])
        
        st.markdown("### Clusters moleculares")
        with st.expander("Datos genómicos (opcional)"):  # Sección desplegable
            rnaseq = st.text_input("RNASeqCluster", placeholder="Ej: Cluster_1", 
                                   help="Puede dejarlo vacío")
            methylation = st.text_input("MethylationCluster", placeholder="Ej: Methyl_high")
            mirna = st.text_input("miRNACluster", placeholder="Ej: miR-21")
            cn = st.text_input("CNCluster", placeholder="Ej: CN_amp")
        
        # BOTÓN DE ENVÍO DEL FORMULARIO
        submitted = st.form_submit_button("Guardar datos clínicos", use_container_width=True)
        
        if submitted:
            # Guardamos TODOS los datos en un diccionario (formato JSON)
            # Los campos vacíos se guardan como None (datos faltantes)
            st.session_state.datos_clinicos = {
                "gender": gender if gender and gender != "" else None,
                "age_at_initial_pathologic": age if age > 0 else None,
                "race": race if race and race != "" else None,
                "ethnicity": ethnicity if ethnicity and ethnicity != "" else None,
                "histological_type": histological_type if histological_type and histological_type != "" else None,
                "neoplasm_histologic_grade": tumor_grade if tumor_grade and tumor_grade != "" else None,
                "tumor_location": tumor_location if tumor_location and tumor_location != "" else None,
                "RNASeqCluster": rnaseq if rnaseq and rnaseq != "" else None,
                "MethylationCluster": methylation if methylation and methylation != "" else None,
                "miRNACluster": mirna if mirna and mirna != "" else None,
                "CNCluster": cn if cn and cn != "" else None,
            }
            
            # Enviamos los datos al orquestador Dagster para su procesamiento
            with st.spinner("Procesando datos clínicos..."):
                try:
                    client = get_dagster_client()  # Conectamos con Dagster
                    # Guardamos los datos y obtenemos la ruta de la carpeta temporal
                    temp_dir = client.guardar_datos_clinicos(
                        st.session_state.datos_clinicos,
                        st.session_state.session_id
                    )
                    st.session_state.temp_dir = temp_dir
                    st.session_state.datos_clinicos_guardados = True
                    st.success("Datos clínicos guardados correctamente")
                except Exception as e:
                    st.error(f"Error al guardar datos clínicos: {str(e)}")
    
    # Indicador visual del estado del formulario
    if st.session_state.datos_clinicos_guardados:
        st.info("Datos clínicos guardados")
    else:
        st.info("Puede procesar imágenes sin rellenar el formulario")
    
    # FICHA TÉCNICA (información de calidad del modelo)
    st.markdown("---")
    st.markdown("### Documentación")
    with st.expander("Ficha técnica de calidad", expanded=False):
        st.markdown("""
        **MRAI - Sistema de apoyo diagnóstico**
        
        **Métricas de calidad:**
        - Precisión: 82.7%
        - Sensibilidad: 83.6%
        - Especificidad: 99.7%

        
        > *Sistema de apoyo diagnóstico - Validado para investigación clínica*
        """)
        
        # Botón para ver la ficha técnica completa (navega a otra página)
        if st.button("Ver ficha técnica completa", use_container_width=True):
            st.switch_page("pages/ficha_tecnica.py")
    
    # Botón para limpiar toda la sesión (empezar de cero)
    st.markdown("---")
    if st.button("Limpiar todos los datos", use_container_width=True):
        limpiar_sesion()


# ========================
# 6. CONTENIDO PRINCIPAL
# ========================

st.title("MRAI - Análisis de Tumores Cerebrales")
st.markdown("**Segmentación automática | Análisis de forma | Evaluación de urgencia**")

# Dividimos la pantalla en dos columnas iguales
col_imagen, col_resultados = st.columns([1, 1])


# ========================
# 7. COLUMNA IZQUIERDA: CARGA Y PROCESAMIENTO DE IMAGEN
# ========================

with col_imagen:
    st.markdown("### Cargar imagen MRI")
    
    # Componente para subir archivos (solo formato TIF/TIFF)
    uploaded_file = st.file_uploader(
        "Seleccionar imagen (TIF, TIFF)",
        type=["tif", "tiff"],
        help="Resonancia magnética cerebral en formato estándar"
    )
    
    if uploaded_file is not None:
        # Mostramos una previsualización de la imagen original
        imagen_preview = Image.open(uploaded_file)
        st.image(imagen_preview, caption="MRI original", use_container_width=True)
        
        # Botón principal de procesamiento
        procesar = st.button("Procesar imagen y evaluar urgencia", type="primary", use_container_width=True)
        
        if procesar and not st.session_state.imagen_procesada:
            with st.spinner("Procesando imagen..."):
                try:
                    client = get_dagster_client()  # Conectamos con el orquestador
                    
                    # Guardamos los bytes de la imagen para usarlos después
                    imagen_bytes = uploaded_file.getvalue()
                    st.session_state.imagen_bytes = imagen_bytes
                    
                    # Si el médico no guardó datos clínicos, creamos unos por defecto (todos None)
                    if not st.session_state.datos_clinicos_guardados:
                        st.info("No hay datos clínicos guardados. Se procederá con valores por defecto.")
                        st.session_state.datos_clinicos = {
                            "gender": None, "age_at_initial_pathologic": None,
                            "race": None, "ethnicity": None, "histological_type": None,
                            "neoplasm_histologic_grade": None, "tumor_location": None,
                            "RNASeqCluster": None, "MethylationCluster": None,
                            "miRNACluster": None, "CNCluster": None
                        }
                        # Guardamos también los datos vacíos
                        client.guardar_datos_clinicos(
                            st.session_state.datos_clinicos,
                            st.session_state.session_id
                        )
                        st.session_state.datos_clinicos_guardados = True
                    
                    # PASO 1: Segmentación del tumor (U-Net/FPN)
                    with st.spinner("Segmentando tumor..."):
                        mascara, df_caracteristicas = client.procesar_imagen(
                            imagen_bytes,
                            st.session_state.datos_clinicos
                        )
                        st.session_state.mascara = mascara  # Matriz binaria del tumor
                        st.session_state.caracteristicas_df = df_caracteristicas  # Área, forma, etc.
                    
                    # PASO 2: Cálculo del nivel de urgencia (Regresión Logística)
                    with st.spinner("Evaluando nivel de urgencia..."):
                        urgencia = client.calcular_urgencia(df_caracteristicas, st.session_state.datos_clinicos)
                        st.session_state.urgencia = urgencia  # Probabilidad 0-1
                    
                    # PASO 3: Guardamos todos los resultados en el servidor
                    with st.spinner("Guardando resultados..."):
                        temp_dir = client.guardar_resultados_imagen(
                            st.session_state.session_id,
                            mascara,
                            df_caracteristicas,
                            urgencia
                        )
                        st.session_state.temp_dir = temp_dir
                    
                    st.session_state.imagen_procesada = True  # Marcamos como procesado
                    st.success("Procesamiento completado")
                    st.rerun()  # Recargamos para mostrar resultados en columna derecha
                    
                except Exception as e:
                    st.error(f"Error en el procesamiento: {str(e)}")
                    st.exception(e)  # Muestra el traceback completo para debugging
        
        elif procesar and st.session_state.imagen_procesada:
            st.warning("Esta imagen ya fue procesada. Use 'Limpiar sesión' para procesar una nueva.")


# ========================
# 8. COLUMNA DERECHA: RESULTADOS DEL ANÁLISIS
# ========================

with col_resultados:
    # Solo mostramos resultados si la imagen ya fue procesada
    if st.session_state.imagen_procesada and st.session_state.mascara is not None:
        st.markdown("### Segmentación del tumor")
        
        # Mostramos la máscara del tumor superpuesta sobre la MRI original
        if st.session_state.mascara is not None and st.session_state.imagen_bytes is not None:
            # Cargamos la imagen original desde los bytes guardados
            img_original = np.array(Image.open(io.BytesIO(st.session_state.imagen_bytes)))
            
            # Si la imagen tiene color (RGB), la convertimos a escala de grises
            if len(img_original.shape) == 3:
                img_original = img_original.mean(axis=2)  # Promedio de canales
            
            # REDIMENSIONAR MÁSCARA AL TAMAÑO DE LA IMAGEN ORIGINAL
            import cv2 
            mascara_redim = cv2.resize(
            st.session_state.mascara.astype(np.float32),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
    )
    


            # Creamos una figura interactiva con Plotly
            fig = go.Figure()
            
            # Capa 1: MRI original en escala de grises
            fig.add_trace(go.Heatmap(z=img_original, colorscale='gray', showscale=False))
            
            # Capa 2: Máscara del tumor superpuesta (rojo transparente)
            mask_superpuesta = np.ma.masked_where(mascara_redim == 0, mascara_redim)
            fig.add_trace(go.Heatmap(z=mask_superpuesta, colorscale='Reds', opacity=0.5, showscale=False))
            
            # Configuramos el tamaño y ocultamos los ejes para mejor visualización
            fig.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_visible=False,
                yaxis_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # SEMÁFORO DE URGENCIA: Mostramos el nivel de riesgo según la probabilidad
        st.markdown("### Nivel de urgencia clínica")
        urgencia = st.session_state.urgencia
        
        if urgencia < 0.3:
            # Riesgo BAJO: Color verde con mensaje de seguimiento programado
            st.markdown(f"""
            <div class="urgency-low">
                <h2>URGENCIA BAJA</h2>
                <h3 style="font-size: 3rem; margin: 0;">{urgencia:.1%}</h3>
                <p style="margin-top: 1rem;">Seguimiento programado normal</p>
            </div>
            """, unsafe_allow_html=True)
        elif urgencia < 0.7:
            # Riesgo MODERADO: Color amarillo con prioridad en consulta especializada
            st.markdown(f"""
            <div class="urgency-moderate">
                <h2>URGENCIA MODERADA</h2>
                <h3 style="font-size: 3rem; margin: 0;">{urgencia:.1%}</h3>
                <p style="margin-top: 1rem;">Priorizar atención</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Riesgo ALTO: Color rojo con recomendación de intervención inmediata
            st.markdown(f"""
            <div class="urgency-high">
                <h2>ALTA URGENCIA</h2>
                <h3 style="font-size: 3rem; margin: 0;">{urgencia:.1%}</h3>
                <p style="margin-top: 1rem;">Requiere intervención inmediata</p>
            </div>
            """, unsafe_allow_html=True)
        
        # CARACTERÍSTICAS RADIÓMICAS DEL TUMOR
        st.markdown("### Características radiomicas")
        
        if st.session_state.caracteristicas_df is not None:
            df_features = st.session_state.caracteristicas_df
            
            # Mostramos en 2 columnas para aprovechar el espacio
            cols = st.columns(2)
            for idx, (feature_name, value) in enumerate(df_features.iloc[0].items()):
                with cols[idx % 2]:  # Alternamos entre columna 0 y 1
                    # Formateamos el valor según su tipo (número o texto)
                    if isinstance(value, (int, float)):
                        display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                    else:
                        display_value = str(value) if pd.notna(value) else 'N/A'
                    
                    # Mostramos cada característica en una tarjeta
                    st.markdown(f"""
                    <div class="metric-card">
                        <small style="color: #6c757d;">{feature_name}</small>
                        <h3 style="margin: 0; color: #2c5f8a;">{display_value}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        
        # DATOS CLÍNICOS DEL PACIENTE (si existen y no están vacíos)
        if st.session_state.datos_clinicos and any(v is not None for v in st.session_state.datos_clinicos.values()):
            with st.expander("Datos clínicos del paciente"):
                for key, value in st.session_state.datos_clinicos.items():
                    if value:  # Solo mostramos campos con valor (ignoramos None)
                        st.text(f"{key}: {value}")
                if not any(v is not None for v in st.session_state.datos_clinicos.values()):
                    st.info("No se proporcionaron datos clínicos")
        
        # EXPORTACIÓN DE RESULTADOS
        st.markdown("---")
        st.markdown("### Exportar resultados")
        
        if st.session_state.temp_dir:
            # Creamos un ZIP con todos los resultados (máscara, características, etc.)
            zip_data = crear_zip_descargable(st.session_state.temp_dir)
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # Botón para descargar el ZIP completo
                st.download_button(
                    label="Descargar todos los datos (ZIP)",
                    data=zip_data,
                    file_name=f"mrai_paciente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.caption("Incluye: máscara, características, urgencia y datos clínicos")
            
            with col_download2:
                # Botón para limpiar y empezar un nuevo caso
                if st.button("Limpiar y empezar nueva sesión", use_container_width=True):
                    limpiar_sesion()


# ========================
# 9. FOOTER (PIE DE PÁGINA)
# ========================
# Información legal y de responsabilidad

st.markdown("---")
st.caption("""   
**MRAI - Sistema de apoyo diagnóstico basado en inteligencia artificial**  
*Siempre confirmar los resultados con un especialista.*
""")