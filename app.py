import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(
    page_title="IA-Stats Studio | Proyecto Desarrollo",
    layout="wide"
)

# Estilo basico para mejorar la visualizacion
st.markdown("""
    <style>
    .main { background-color: #fdfefe; }
    </style>
    """, unsafe_allow_html=True)

# --- TITULO E INFORMACION DEL ALUMNO ---
st.title("Asistente Estadistico con IA Generativa")
st.markdown("""
**Materia:** Desarrollo de Software e IA  
**Objetivo:** Documentar el proceso de desarrollo y las limitaciones de la IA.
""")

st.divider()

# --- BARRA LATERAL (CONFIGURACION) ---
with st.sidebar:
    st.header("Panel de Control")
    
    opcion_carga = st.radio(
        "Origen de los datos:",
        ["Muestra Sintetica", "Subir CSV Personalizado"]
    )
    
    st.divider()
    
    st.subheader("Parametros Prueba Z")
    mu_h0 = st.number_input("Hipotesis Nula (H0: mu =)", value=100.0)
    alpha = st.select_slider("Significancia (alpha)", options=[0.01, 0.05, 0.10], value=0.05)
    
    tipo_cola = st.selectbox(
        "Tipo de Prueba",
        ["Bilateral (diff)", "Cola Derecha (>)", "Cola Izquierda (<)"]
    )
    
    # Barra de progreso para el reporte
    st.divider()
    st.subheader("Progreso del Proyecto")
    # El progreso aumentara segun las fases completadas
    progreso_val = 0
    bar_progreso = st.progress(progreso_val)

# --- GESTION DE DATOS (FASE 2) ---
df = None

if opcion_carga == "Muestra Sintetica":
    col_n, col_s = st.columns(2)
    with col_n:
        n = st.number_input("Tamano (n >= 30)", min_value=30, value=100)
    with col_s:
        sigma = st.number_input("Sigma (poblacional)", min_value=0.1, value=15.0)
    
    # Generacion de datos con semilla para reproducibilidad
    np.random.seed(42)
    datos = np.random.normal(loc=mu_h0 + 5, scale=sigma, size=n)
    df = pd.DataFrame(datos, columns=["Variable_Principal"])
    st.success(f"Datos sinteticos generados (n={n})")
    progreso_val = 25

else:
    archivo = st.file_uploader("Carga tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("Archivo cargado exitosamente")
        progreso_val = 25
    else:
        st.info("Esperando carga de archivo CSV...")

# --- VISUALIZACION Y EDA (FASE 3) ---
if df is not None:
    # Actualizar barra de progreso al cargar datos
    bar_progreso.progress(50)
    
    with st.expander("Vista previa de la tabla de datos"):
        st.dataframe(df, use_container_width=True)
    
    st.header("1. Analisis Visual de la Distribucion")
    
    # Filtrado de columnas numericas
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if cols_num:
        col_target = st.selectbox("Selecciona la variable para analizar:", cols_num)
        datos_limpios = df[col_target].dropna()
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Histograma y KDE")
            fig_h, ax_h = plt.subplots()
            sns.histplot(datos_limpios, kde=True, color="steelblue", ax=ax_h)
            ax_h.set_title(f"Distribucion de {col_target}")
            st.pyplot(fig_h)
            
        with c2:
            st.subheader("Boxplot (Outliers)")
            fig_b, ax_b = plt.subplots()
            sns.boxplot(x=datos_limpios, color="coral", ax=ax_b)
            ax_b.set_title(f"Caja y Bigotes de {col_target}")
            st.pyplot(fig_b)
            
        # Cuestionario de autodiagnostico (Requisito rubrica)
        st.markdown("### Autodiagnostico")
        q1 = st.radio("¿La distribucion parece normal?", ["Si", "No", "Incierto"])
        q2 = st.radio("¿Se observan valores atipicos (outliers)?", ["Si", "No"])
        st.text_area("Describa el sesgo observado:")
        
    else:
        st.error("El dataset no contiene columnas numericas.")
# --- 5. VISUALIZACION DE DISTRIBUCIONES ---
if df is not None:
    st.divider()
    st.header("📊 2. Análisis Visual de la Distribución")
    
    # Filtrado de columnas numéricas
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if cols_num:
        col_target = st.selectbox("Selecciona la variable para analizar:", cols_num)
        
        # Preparación de datos para graficar
        datos_limpios = df[col_target].dropna()
        
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.subheader("Histograma y KDE")
            fig_h, ax_h = plt.subplots()
            sns.histplot(datos_limpios, kde=True, color="steelblue", ax=ax_h)
            ax_h.set_title("Distribución de Frecuencias")
            st.pyplot(fig_h)
            
        with col_graf2:
            st.subheader("Boxplot (Outliers)")
            fig_b, ax_b = plt.subplots()
            sns.boxplot(x=datos_limpios, color="coral", ax=ax_b)
            ax_b.set_title("Detección de Valores Atípicos")
            st.pyplot(fig_b)
            
        # Sección de autodiagnóstico
        st.info("Responde las siguientes preguntas según lo observado en las gráficas:")
        q1 = st.radio("¿La distribución parece ser Normal?", ["Sí", "No", "Incierto"])
        q2 = st.radio("¿Se detectan valores atípicos (outliers)?", ["Sí", "No"])
        
        # Actualizar progreso
        progreso = 50
    else:
        st.error("No se detectaron columnas numéricas para graficar.")