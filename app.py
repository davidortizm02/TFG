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


# Custom CSS for styling
def local_css():
    st.markdown(
        """
        <style>
            .app-header {display: flex; align-items: center;}
            .app-header img {margin-right: 10px;}
            .metric-label {font-size: 18px; color: #333;}
            .history-item {padding: 10px; border-bottom: 1px solid #eee;}
        </style>
        """, unsafe_allow_html=True
    )

# Página y tema
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🧠",
    layout="wide",
)
local_css()

# Sidebar: historial
st.sidebar.title("Historial de Predicciones")
if 'history' not in st.session_state:
    st.session_state.history = []

# Selector de historial
if st.session_state.history:
    # Asegurar que cada registro tenga un 'name'
    options = []
    for i, h in enumerate(st.session_state.history):
        if 'name' not in h or not h['name']:
            default_name = f"Predicción_{i+1}_{h.get('timestamp','')}
"
            h['name'] = default_name
        options.append(h['name'])
    sel = st.sidebar.selectbox(
        "Ver resultados guardados:",
        options=options
    )
    sel_idx = options.index(sel)
    record = st.session_state.history[sel_idx]
    with st.sidebar.expander("Detalles de la predicción", expanded=True):
        st.image(record['original'], use_column_width=True)
        st.markdown(f"**Nombre:** {record['name']}")
        st.markdown(f"**Timestamp:** {record['timestamp']}")
        st.markdown(f"**Modelo:** {record['model']}")
        st.markdown(f"**Lesión:** {record['label']}")
        st.markdown(f"**Confianza:** {record['confidence']:.2%}")
        if record.get('meta'):
            st.markdown("**Metadatos:**")
            for k, v in record['meta'].items():
                st.markdown(f"- **{k.capitalize()}:** {v}")

# Título principal
st.markdown("<div class='app-header'><h1>Clasificador de Lesiones Cutáneas</h1></div>", unsafe_allow_html=True)
st.markdown("---")

# Carga de recursos
try:
    feature_cols, preproc, le_class, model_hybrid, model_img = load_all_resources()
except FileNotFoundError as e:
    st.error(f"Falta un archivo necesario: {e}")
    st.stop()

# Área de predicción
timestamp_default = time.strftime('%Y-%m-%d_%H-%M-%S')
col_config, col_display = st.columns([1, 2], gap="large")
with col_config:
    st.subheader("1. Configuración")
    model_choice = st.radio("Modelo:", ("Híbrido (imagen + metadatos)", "Solo imagen"))
    uploaded = st.file_uploader("Sube JPG/PNG:", type=["jpg", "jpeg", "png"])
    pred_name = st.text_input("Nombre del registro:", f"Predicción_{timestamp_default}")

    # Metadatos dinámicos
    meta = {}
    if model_choice.startswith("Híbrido"):
        st.subheader("2. Metadatos")
        meta['edad'] = st.number_input("Edad aproximada:", 0, 120, 50)
        meta['sexo'] = st.selectbox("Sexo:", ["male", "female", "unknown"])
        meta['zona'] = st.selectbox("Zona anatómica:", [
            "anterior torso","head/neck","lateral torso","lower extremity",
            "upper extremity","oral/genital","palms/soles","posterior torso","unknown"
        ])
        meta['dataset'] = st.selectbox("Fuente del dataset:", [
            "BCN_nan","HAM_vidir_molemax","HAM_vidir_modern",
            "HAM_rosendahl","MSK4nan","HAM_vienna_dias"
        ])
    submitted = st.button("🔍 Realizar Predicción")

with col_display:
    if uploaded and submitted:
        with st.spinner('Procesando...'):
            # Preprocesamiento
            img_batch, img_vis = preprocess_image_for_model(uploaded)
            original = Image.open(uploaded).convert('RGB')
            # Visualización
            with st.expander("Visualización de Imágenes", expanded=True):
                if model_choice.startswith("Híbrido"):
                    gray = cv2.cvtColor(np.array(img_vis), cv2.COLOR_RGB2GRAY)
                    _, mask = extract_features_from_array(np.array(img_vis), gray)
                    cols = st.columns(3)
                    cols[0].image(original, caption="Original", use_container_width=True)
                    cols[1].image(img_vis, caption="Procesada", use_container_width=True)
                    cols[2].image(mask, caption="Máscara", use_container_width=True)
                else:
                    cols = st.columns(2)
                    cols[0].image(original, caption="Original", use_container_width=True)
                    cols[1].image(img_vis, caption="Procesada (224×224)", use_container_width=True)
            # Preparar datos para predicción
            if model_choice.startswith("Híbrido"):
                feats_raw, _ = extract_features_from_array(np.array(img_vis), gray)
                grp = ('young' if meta['edad']<=35 else 'adult' if meta['edad']<=65 else 'senior')
                df_meta = pd.DataFrame([{**{
                    "age_approx": meta['edad'],
                    "sex": meta['sexo'],
                    "anatom_site_general": meta['zona'],
                    "dataset": meta['dataset'],
                    "age_sex_interaction": f"{meta['sexo']}_{grp}"
                }, **feats_raw}])
                X_meta = preproc.transform(df_meta)
                inputs = [img_batch, X_meta]
                model = model_hybrid
            else:
                inputs = img_batch
                model = model_img
            # Predicción
            pred = model.predict(inputs, verbose=0)
            idx = int(np.argmax(pred, axis=1)[0])
            conf = float(np.max(pred))
            label = le_class.inverse_transform([idx])[0]
            # Resultados
            st.markdown("---")
            r1, r2 = st.columns([1,2])
            with r1:
                st.metric(label="Lesión Predicha", value=label)
                st.metric(label="Confianza", value=f"{conf:.2%}")
            with r2:
                dfp = pd.DataFrame({"Lesión": le_class.classes_, "Probabilidad": pred.flatten()})
                dfp = dfp.set_index("Lesión").sort_values("Probabilidad", ascending=False)
                st.bar_chart(dfp)
            # Guardar en historial
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.history.append({
                'name': pred_name,
                'timestamp': timestamp,
                'original': original,
                'model': model_choice,
                'label': label,
                'confidence': conf,
                'meta': meta if meta else None
            })
    else:
        st.info("Sube una imagen y configura la predicción para ejecutar.")

st.markdown("---")
st.caption("TFG – Interfaz mejorada y guardado de historial.")
