import os
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
import tensorflow as tf
import joblib
import json
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

# ——————————————————————————————
# Configuración de la página y estilo
# ——————————————————————————————
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="wide", initial_sidebar_state="expanded")

# Custom CSS para mejorar apariencia
st.markdown(
    """
    <style>
    /* Ajuste del fondo y tipografía */
    body {background-color: #f5f5f5;}
    .stApp {font-family: 'Arial', sans-serif;}
    /* Cartas y contenedores con sombra y bordes redondeados */
    .card {background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 1rem; margin-bottom: 1rem;}
    /* Encabezados personalizados */
    h1, h2, h3, h4 {color: #333333;}
    /* Ocultar menú Streamlit y pie de página */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Rutas y parámetros globales (ajustar según entorno)
FOLDERS_BASE    = os.getenv("FOLDERS_BASE", "/kaggle/working/dataset_sin_peloTODO")
METADATA_PATH   = os.getenv("METADATA_PATH", "/kaggle/input/d/antonioortizmoreno/metadatafl/ISIC_2019_Training_Metadata_FL.csv")
OUTPUT_CSV_PATH = os.getenv("OUTPUT_CSV_PATH", "/kaggle/working/enriched_ISIC_2019_Metadata_improved.csv")

# GLCM y LBP
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 8
LBP_RADIUS     = 1
LBP_POINTS     = 8 * LBP_RADIUS
MORPH_OPEN_RADIUS  = 3
MORPH_CLOSE_RADIUS = 5
MIN_LESION_AREA    = 100

# =====================
# Carga de recursos (cacheado)
# =====================
@st.cache_resource
def load_all_resources():
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    preprocessor = joblib.load("preprocessor_metadata.pkl")
    label_encoder = joblib.load("labelencoder_class.pkl")
    model = load_model("modelo_hibrido_entrenadoCW.keras", compile=False)
    return feature_columns, preprocessor, label_encoder, model


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
st.markdown("<div class='card'><h1>🧠 Clasificador de Lesiones Cutáneas</h1><p>Sube una imagen de una lesión y completa los metadatos para predecir su tipo.</p></div>", unsafe_allow_html=True)

# Sidebar para carga y metadatos
with st.sidebar:
    st.markdown("<div class='card'><h2>1. Sube la Imagen</h2></div>", unsafe_allow_html=True)
    tile = st.file_uploader("Selecciona un JPG/PNG", type=["jpg","jpeg","png"])
    st.markdown("<div class='card'><h2>2. Introduce los Metadatos</h2></div>", unsafe_allow_html=True)
    with st.form("metadata_form_sidebar"):
        edad = st.number_input("Edad aproximada", min_value=0, max_value=120, value=50)
        sexo = st.selectbox("Sexo", ["male","female","unknown"])
        site = st.selectbox("Zona anatómica", [
            "anterior torso","head/neck","lateral torso","lower extremity",
            "upper extremity","oral/genital","palms/soles","posterior torso","unknown"
        ])
        dataset = st.selectbox("Fuente del dataset", [
            "BCN_nan","HAM_vidir_molemax","HAM_vidir_modern",
            "HAM_rosendahl","MSK4nan","HAM_vienna_dias"
        ])
        submit_button = st.form_submit_button("Realizar Predicción")

# Carga recursos
try:
    feature_columns, preprocessor, le_class, model = load_all_resources()
except FileNotFoundError as e:
    st.error(f"Falta un archivo necesario: {e}")
    st.stop()

# Área principal para resultado y diagnóstico
if tile and submit_button:
    # Indicador de procesamiento
    with st.spinner('Procesando imagen y metadatos...'):
        img_batch, img_vis = preprocess_image_for_model(tile)
        gray = cv2.cvtColor(img_vis, cv2.COLOR_RGB2GRAY)
        feats_raw, mask = extract_features_from_array(img_vis, gray)

    # Resultados en contenedor con sombra
    st.markdown("<div class='card'><h2>3. Análisis y Predicción</h2></div>", unsafe_allow_html=True)
    # Visualización lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>Imagen Procesada</h3></div>", unsafe_allow_html=True)
        st.image(img_vis, caption="Imagen 224×224", use_column_width=True)
        st.markdown("<div class='card'><h3>Máscara de Lesión</h3></div>", unsafe_allow_html=True)
        st.image(mask, caption="Máscara de lesión", use_column_width=True)
    with col2:
        st.markdown("<div class='card'><h3>Características Extraídas</h3></div>", unsafe_allow_html=True)
        df_feats = pd.DataFrame([feats_raw]).fillna("NaN")
        st.dataframe(df_feats)

    # Preprocesado de metadatos
    if edad <= 35:
        grp = "young"
    elif edad <= 65:
        grp = "adult"
    else:
        grp = "senior"
    df_meta = pd.DataFrame([{
        "age_approx": edad,
        "sex": sexo,
        "anatom_site_general": site,
        "dataset": dataset,
        "age_sex_interaction": f"{sexo}_{grp}",
        **feats_raw
    }])
    try:
        X_meta = preprocessor.transform(df_meta)
    except Exception as e:
        st.error(f"Error en preprocesado: {e}")
        st.stop()

    with st.expander("🔬 Diagnóstico: Preprocesamiento Metadatos", expanded=False):
        st.subheader("Antes de transformar")
        st.dataframe(df_meta.fillna("NaN"))
        st.subheader("Después de transformar")
        arr = X_meta.toarray() if hasattr(X_meta, "toarray") else X_meta
        st.dataframe(pd.DataFrame(arr, columns=feature_columns if 'feature_columns' in locals() else None))

    # Predicción
    with st.spinner('Realizando predicción con el modelo híbrido...'):
        pred = model.predict([img_batch, X_meta], verbose=0)
    idx = int(np.argmax(pred, axis=1)[0])
    conf = float(np.max(pred))
    label = le_class.inverse_transform([idx])[0]

    # Mostrar resultado principal
    st.success(f"**Clase Predicha:** {label}  |  **Confianza:** {conf:.2%}")
    dfp = pd.DataFrame({
        "Clase": le_class.classes_,
        "Probabilidad": pred.flatten()
    }).set_index("Clase").sort_values("Probabilidad", ascending=False)
    # Gráfico de barras horizontal
    st.markdown("<div class='card'><h3>Distribución de Probabilidades</h3></div>", unsafe_allow_html=True)
    st.bar_chart(dfp)

else:
    st.markdown("<div class='card'><p>Sube una imagen y completa el formulario en la barra lateral para predecir.</p></div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("TFG – Clasificador híbrido con diagnóstico de características.")
