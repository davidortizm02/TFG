import os
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import joblib
import json

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

# ——————————————————————————————
# Parámetros globales y configuración
# ——————————————————————————————
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="wide")
FOLDERS_BASE    = "/kaggle/working/dataset_sin_peloTODO"
METADATA_PATH   = "/kaggle/input/d/antonioortizmoreno/metadatafl/ISIC_2019_Training_Metadata_FL.csv"
OUTPUT_CSV_PATH = "/kaggle/working/enriched_ISIC_2019_Metadata_improved.csv"

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
    """Carga modelos y preprocesadores."""
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    preprocessor = joblib.load("preprocessor_metadata.pkl")
    label_encoder = joblib.load("labelencoder_class.pkl")
    # Asegúrate de que el nombre coincide con tu archivo real
    model = load_model("modelo_hibrido_entrenadoCW.keras", compile=False)
    return feature_columns, preprocessor, label_encoder, model

# =====================
# Funciones de segmentación y extracción de features
# =====================

def segment_lesion(gray_img):
    """Segmenta la lesión con Otsu + opening/closing + CC más grande."""
    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = gray_img[mask == 255]
    bg = gray_img[mask == 0]
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
        d=h-w; t=d//2
        return img[t:t+w, :]
    d=w-h; l=d//2
    return img[:, l:l+h]

def crop_and_resize(img, target_size=224):
    sq = center_crop_to_square(img)
    return cv2.resize(sq, (target_size,target_size), interpolation=cv2.INTER_AREA)

def crop_non_black_region(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys,xs = np.where(mask)
    if not ys.size: return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

def preprocess_image_for_model(image_file):
    img = Image.open(image_file).convert('RGB')
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cropped = crop_non_black_region(bgr)
    resized = crop_and_resize(cropped, 224)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype('float32')/255.0

# =====================
# Interfaz de Streamlit
# =====================

st.title("🧠 Clasificador de Lesiones Cutáneas")
st.markdown("Sube una imagen de una lesión y completa los metadatos para predecir su tipo.")

# Carga recursos
try:
    feature_columns, preprocessor, le_class, model = load_all_resources()
except FileNotFoundError as e:
    st.error(f"Falta un archivo necesario: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.header("1. Sube la Imagen")
    tile = st.file_uploader("Selecciona un JPG/PNG", type=["jpg","jpeg","png"])
    st.header("2. Introduce los Metadatos")
    with st.form("metadata_form"):
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
if tile and submit_button:
    with col2:
        st.header("3. Análisis y Predicción")
        img_model = preprocess_image_for_model(tile)
        img_uint8 = (img_model * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        feats_raw, mask = extract_features_from_array(img_uint8, gray)

        with st.expander("🔍 Diagnóstico: Segmentación y Features", expanded=True):
            st.image(img_model, caption="Imagen 224×224", use_container_width=True)
            st.image(mask, caption="Máscara de lesión", use_container_width=True)
            st.dataframe(pd.DataFrame([feats_raw]).fillna("NaN"))

        # Codificar age_group
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

        # Transformar metadatos y forzar vector a ceros (para test)
        try:
            X_meta = preprocessor.transform(df_meta)
        except Exception as e:
            st.error(f"Error en preprocesado: {e}")
            st.stop()

        # Convertimos a array denso si es necesario
        if hasattr(X_meta, "toarray"):
            arr_meta = X_meta.toarray()
        else:
            arr_meta = X_meta.copy()

        # Sobrescribir todo a ceros
        arr_zero = np.zeros_like(arr_meta, dtype=arr_meta.dtype)
        X_meta = arr_zero  # ahora el vector de metadatos es todo ceros

        with st.expander("🔬 Diagnóstico: Preprocesamiento Metadatos", expanded=True):
            st.subheader("Antes de transformar")
            st.dataframe(df_meta.fillna("NaN"))
            st.subheader("Después de transformar (vector a ceros)")
            st.dataframe(pd.DataFrame(arr_zero))

        # Predicción con vector de metadatos cero
        pred_img = np.expand_dims(img_model, axis=0)
        pred = model.predict([pred_img, X_meta], verbose=0)
        idx = np.argmax(pred, axis=1)[0]
        conf = float(np.max(pred))
        label = le_class.inverse_transform([idx])[0]

        st.success(f"**Clase:** {label}  |  **Confianza:** {conf:.2%}")
        dfp = pd.DataFrame({
            "Clase": le_class.classes_,
            "Probabilidad": pred.flatten()
        }).set_index("Clase").sort_values("Probabilidad", ascending=False)
        st.bar_chart(dfp)
else:
    with col2:
        st.info("Sube una imagen y completa el formulario para predecir.")

st.markdown("---")
st.caption("TFG – Clasificador híbrido con diagnóstico de características.")
