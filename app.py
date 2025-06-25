import os
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
import joblib
import json

import time
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops
import plotly.graph_objects as go
# ——————————————————————————————
# Parámetros globales y configuración
# ——————————————————————————————
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="wide")

# GLCM settings
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 8

# LBP settings
LBP_RADIUS     = 1
LBP_POINTS     = 8 * LBP_RADIUS

# Morfología / segmentación
MORPH_OPEN_RADIUS  = 3
MORPH_CLOSE_RADIUS = 5
MIN_LESION_AREA    = 100


# =====================
# Carga de recursos (cacheado)
# =====================
@st.cache_resource
def load_all_resources():
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    preprocessor = joblib.load("preprocessor_metadata.pkl")
    label_encoder = joblib.load("labelencoder_class.pkl")
    model_hybrid = load_model("modelo_hibrido_entrenadoCW.keras", compile=False)
    model_img    = load_model("modelo_imagenes_entrenado2.keras", compile=False)
    return feature_cols, preprocessor, label_encoder, model_hybrid, model_img

# =====================
# Funciones de segmentación y extracción de features
# =====================

def segment_lesion(gray_img):
    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg, bg = gray_img[mask==255], gray_img[mask==0]
    if fg.size and bg.size and fg.mean() < bg.mean():
        mask = cv2.bitwise_not(mask)
    mask_bool = opening(mask>0, disk(MORPH_OPEN_RADIUS))
    mask_bool = closing(mask_bool, disk(MORPH_CLOSE_RADIUS))
    labels = label(mask_bool)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    regions = regionprops(labels)
    max_r = max(regions, key=lambda r: r.area)
    if max_r.area < MIN_LESION_AREA:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == max_r.label).astype(np.uint8) * 255

def compute_glcm_features(gray_roi, mask_roi):
    """Calcula GLCM multi-distancia/ángulo en ROI enmascarada."""
    ys, xs = np.where(mask_roi==255)
    if ys.size==0:
        return {f'glcm_{p}': np.nan for p in ['contrast','dissimilarity','homogeneity','energy','ASM','correlation']}
    bins = max(1, 256 // GLCM_LEVELS)
    quant = (gray_roi // bins).astype(np.uint8)
    quant[mask_roi==0] = 0
    try:
        glcm = graycomatrix(quant, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                             levels=GLCM_LEVELS, symmetric=True, normed=True)
    except Exception:
        return {f'glcm_{p}': np.nan for p in ['contrast','dissimilarity','homogeneity','energy','ASM','correlation']}
    feats = {}
    for prop in ['contrast','dissimilarity','homogeneity','energy','ASM','correlation']:
        try:
            feats[f'glcm_{prop}'] = float(graycoprops(glcm, prop=prop).mean())
        except Exception:
            feats[f'glcm_{prop}'] = np.nan
    return feats

def compute_lbp_features(gray_roi, mask_roi):
    """Calcula histograma LBP 'uniform' dentro de ROI."""
    ys, xs = np.where(mask_roi==255)
    if ys.size==0:
        return {}
    lbp = local_binary_pattern(gray_roi, LBP_POINTS, LBP_RADIUS, method='uniform')
    vals = lbp[mask_roi==255].ravel()
    if vals.size==0:
        return {}
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(vals, bins=n_bins, range=(0,n_bins), density=True)
    return {f'lbp_{i}': float(hist[i]) for i in range(n_bins)}

def extract_features_from_array(img_rgb, gray):
    """
    Extrae todas las features de una imagen (RGB uint8 + gris), 
    devuelve (feats_raw, segmentation_mask_uint8).
    """
    mask = segment_lesion(gray)
    # Si no detecta lesión, devolvemos NaNs
    if not np.any(mask==255):
        return { }, mask
    # Contorno principal
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return { }, mask
    c = max(cnts, key=cv2.contourArea)
    lesion_mask = np.zeros_like(mask); cv2.drawContours(lesion_mask, [c], -1, 255, -1)

    feats = {}
    # Estadísticos color
    for i, col in enumerate(['R','G','B']):
        pix = img_rgb[:,:,i][lesion_mask==255].astype(float)
        feats[f'mean_{col}'] = float(pix.mean()) if pix.size else np.nan
        feats[f'std_{col}']  = float(pix.std())  if pix.size else np.nan

    # Forma
    area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
    hull = cv2.convexHull(c); hull_area = cv2.contourArea(hull)
    solidity = float(area/hull_area) if hull_area>0 else np.nan
    x,y,w,h = cv2.boundingRect(c)
    extent = float(area/(w*h)) if w*h>0 else np.nan
    feats.update({
        'lesion_area': float(area),
        'lesion_perimeter': float(peri),
        'solidity': solidity,
        'extent': extent
    })

    # GLCM & LBP sobre ROI
    gray_roi = gray[y:y+h, x:x+w]
    mask_roi = lesion_mask[y:y+h, x:x+w]
    feats.update(compute_glcm_features(gray_roi, mask_roi))
    feats.update(compute_lbp_features(gray_roi, mask_roi))

    return feats, mask

# =====================
# Funciones de preprocesamiento de la imagen para el modelo
# =====================

def center_crop_to_square(img):
    h,w = img.shape[:2]
    if h==w: return img.copy()
    if h>w:
        d,h_diff = h-w, (h-w)//2
        return img[h_diff:h_diff+w, :]
    d,w_diff = w-h, (w-h)//2
    return img[:, w_diff:w_diff+h]


def crop_non_black_region(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys,xs = np.where(mask)
    if ys.size ==0: return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]


def preprocess_image_for_model(image_file, target_size=224):
    # Carga y conversión a BGR
    img = Image.open(image_file).convert('RGB')
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # Elimina fondo negro y recorta al cuadrado
    crop = crop_non_black_region(bgr, thresh=10)
    square = center_crop_to_square(crop)
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # Convertir BGR a RGB para visualización
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Preprocesar para EfficientNetV2
    img_array = np.expand_dims(rgb, axis=0).astype(np.float32)
    img_array = effnet_preprocess(img_array)
    return img_array, rgb.astype(np.uint8)




# =====================
# Interfaz de Streamlit
# =====================


# --- ESTILO VISUAL Y CSS ---
def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
            .stApp {
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                background-attachment: fixed;
            }
            .st-emotion-cache-18ni7ap, .st-emotion-cache-1d391kg {
                background: rgba(255, 255, 255, 0.5);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.18);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            }
            .stTabs [data-baseweb="tab-list"] { gap: 24px; }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 8px;
                padding: 10px 15px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #FFFFFF;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stButton>button {
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                border: none;
                background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
                color: white;
                transition: all 0.3s ease-in-out;
            }
            .stButton>button:hover {
                box-shadow: 0 0 20px #6a11cb80;
                transform: scale(1.02);
            }
            .stButton>button:disabled { background: #cccccc; color: #666666; }
            .stMetric {
                background-color: #FFFFFF;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)


# --- CONFIGURACIÓN DE PÁGINA E INICIALIZACIÓN DE ESTADO ---
st.set_page_config(
    page_title="Skin-AI | Clasificador de Lesiones",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

# Inicialización del estado de la sesión para el historial y el nombre de la predicción
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pred_name' not in st.session_state:
    st.session_state.pred_name = f"Pred_{time.strftime('%Y%m%d_%H%M%S')}"

# --- CARGA DE RECURSOS (MODELO, ETC.) ---
if 'resources_loaded' not in st.session_state:
    try:
        _, preproc, le_class, model_hybrid, model_img = load_all_resources()
        st.session_state.resources = {
            "preproc": preproc, "le_class": le_class,
            "model_hybrid": model_hybrid, "model_img": model_img
        }
        st.session_state.resources_loaded = True
    except FileNotFoundError as e:
        st.error(f"Error crítico al cargar recursos: {e}. La aplicación no puede continuar.")
        st.stop()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>📋 Historial</h1>", unsafe_allow_html=True)
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.history = []
        st.success("Historial eliminado.")
        st.rerun()
    st.markdown("---")
    if not st.session_state.history:
        st.info("Aún no hay predicciones guardadas.")
    else:
        for record in reversed(st.session_state.history):
            with st.expander(f"📌 {record['name']} ({record['timestamp'].split(' ')[0]})"):
                st.image(record['original'], use_container_width=True, caption="Imagen Original")
                st.markdown(f"**Lesión:** `{record['label']}`")
                st.markdown(f"**Confianza:** `{record['confidence']:.2%}`")
                st.markdown(f"**Modelo:** `{record['model']}`")
                if record.get('meta'):
                    st.markdown("**Metadatos:**")
                    for k, v in record['meta'].items():
                        st.markdown(f"- **{k.capitalize()}:** {v}")

# --- CONTENIDO PRINCIPAL ---
st.title("🩺 Skin-AI: Asistente de Clasificación de Lesiones Cutáneas")
st.caption("Una herramienta de IA para la clasificación preliminar de lesiones en la piel. Desarrollado como TFG.")

tab_inicio, tab_prediccion, tab_info = st.tabs(["🏠 Inicio", "🧪 Nueva Predicción", "📚 Sobre la App"])

with tab_inicio:
    st.markdown("### ¡Bienvenido a Skin-AI!")
    st.markdown("""
    Esta aplicación utiliza un modelo de Red Neuronal Convolucional para analizar imágenes de lesiones cutáneas y predecir a cuál de las siguientes categorías podría pertenecer:
    - Melanoma (MEL).
    - Nevus melanocítico (NV).
    - Carcinoma de células basales (BCC).
    - Queratosis actínica (AK).
    - Queratosis benigna (BKL).
    - Dermatofibroma (DF).
    - Lesión vascular (VASC).
    - Carcinoma escamocelular (SCC)     
                
    **¿Cómo empezar?**
    1.  Ve a la pestaña **"🧪 Nueva Predicción"**.
    2.  Sube una imagen, elige el modelo y completa los datos requeridos.
    3.  Asigna un nombre único a tu predicción.
    4.  Haz clic en "Realizar Predicción" y explora los resultados.
    """)
    st.warning("**Aviso Importante:** Esta es una herramienta experimental y **NO** un dispositivo de diagnóstico médico.")

with tab_prediccion:
    col_config, col_display = st.columns([0.4, 0.6], gap="large")

    with col_config:
        st.markdown("### 1. Carga y Configuración")
        with st.container(border=True):
            model_choice = st.radio("Selecciona el modelo:", ("Híbrido (imagen + metadatos)", "Solo imagen"), horizontal=True)
            uploaded = st.file_uploader("Sube una imagen:", type=["jpg", "jpeg", "png"], label_visibility="visible")
            
            meta = {}
            if model_choice.startswith("Híbrido"):
                st.markdown("##### Metadatos del Paciente")
                meta['edad'] = st.slider("Edad:", 1, 100, 50)
                meta['sexo'] = st.selectbox("Sexo:", ["male", "female", "unknown"])
                meta['zona'] = st.selectbox("Zona anatómica:", ["anterior torso","head/neck","lateral torso","lower extremity","upper extremity","oral/genital","palms/soles","posterior torso","unknown"])
                meta['dataset'] = st.selectbox("Fuente del dataset:", ["BCN_nan","HAM_vidir_molemax","HAM_vidir_modern","HAM_rosendahl","MSK4nan","HAM_vienna_dias"])
            
            # CAMBIO: Se usa un `key` para vincular el input al estado de la sesión.
            
            st.text_input("Nombre para este registro:", key="pred_name")
            submitted = st.button("🔍 Realizar Predicción", use_container_width=True, disabled=(uploaded is None))

    with col_display:
        st.markdown("### 2. Visualización y Resultados")
        if not submitted:
            if uploaded:
                st.image(uploaded, caption="Imagen cargada. Lista para analizar.", use_container_width=True)
            else:
                st.info("Esperando que subas una imagen para comenzar el análisis.")

        # CAMBIO: La lógica de guardado y predicción solo se activa con el botón
        if submitted and uploaded:
            # El nombre se toma directamente del estado de la sesión, que fue actualizado por el input del usuario
            current_pred_name = st.session_state.pred_name

            with st.spinner(f'🧠 Analizando "{current_pred_name}"...'):
                original = Image.open(uploaded).convert('RGB')
                img_batch, img_vis = preprocess_image_for_model(uploaded)
                
                if model_choice.startswith("Híbrido"):
                    img_vis_array = np.array(img_vis)
                    gray = cv2.cvtColor(img_vis_array, cv2.COLOR_RGB2GRAY)
                    feats_raw, _ = extract_features_from_array(img_vis_array, gray)
                    grp = ('young' if meta['edad'] <= 35 else 'adult' if meta['edad'] <= 65 else 'senior')
                    age_sex_interaction = f"{meta['sexo']}_{grp}"
                    full_meta_dict = {"age_approx": meta['edad'], "sex": meta['sexo'], "anatom_site_general": meta['zona'], "dataset": meta['dataset'], "age_sex_interaction": age_sex_interaction, **feats_raw}
                    df_meta = pd.DataFrame([full_meta_dict])
                    X_meta = st.session_state.resources["preproc"].transform(df_meta)
                    inputs = [img_batch, X_meta]
                    model = st.session_state.resources["model_hybrid"]
                else:
                    inputs = img_batch
                    model = st.session_state.resources["model_img"]
                
                le_class = st.session_state.resources["le_class"]
                pred = model.predict(inputs, verbose=0)
                idx = int(np.argmax(pred, axis=1)[0])
                conf = float(np.max(pred))
                label = le_class.inverse_transform([idx])[0]

                st.markdown(f"#### Resultados para: *{current_pred_name}*")
                with st.container(border=True):
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric(label="Diagnóstico Principal", value=label)
                        st.metric(label="Nivel de Confianza", value=f"{conf:.2%}")
                    with res_col2:
                        st.image(original, caption="Imagen Analizada", use_container_width=True)
                    
                    dfp = pd.DataFrame({"Lesión": le_class.classes_, "Probabilidad": pred.flatten()})
                    fig = go.Figure(data=go.Scatterpolar(r=dfp['Probabilidad'], theta=dfp['Lesión'], fill='toself'))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=350, margin=dict(l=40, r=40, t=40, b=40))
                    st.plotly_chart(fig, use_container_width=True)

                # Guardar en historial usando el nombre correcto
                st.session_state.history.append({
                    'name': current_pred_name, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'original': original, 'model': model_choice, 'label': label,
                    'confidence': conf, 'meta': meta if meta else None
                })

                # CAMBIO: Reiniciar el nombre para la siguiente predicción
                #st.session_state.pred_name = f"Pred_{time.strftime('%Y%m%d_%H%M%S')}"
                
                st.success(f'Análisis "{current_pred_name}" completado y guardado en el historial.')
                
                
                st.balloons()
                
                # Forzamos un rerun para limpiar el formulario y actualizar el historial visualmente
                time.sleep(2) # Pequeña pausa para que el usuario vea el mensaje de éxito
                st.rerun()

with tab_info:
    st.markdown("### 📚 Sobre la Aplicación")
    st.markdown("""
    **Skin-AI** es un proyecto demostrativo para mostrar las capacidades de los modelos de Deep Learning en dermatología computacional, utilizando modelos entrenados con Aprendizaje Federado.
    - **Tecnologías:** Python, TensorFlow/Keras, Scikit-learn, OpenCV, Streamlit.
    """)
    st.warning("**Aviso Importante:** Esta herramienta es una prueba de concepto académica y **NO** debe ser utilizada para autodiagnóstico.")