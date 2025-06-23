import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import closing, opening, disk

# =====================
# Implementación personalizada de Focal Loss
# =====================
class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = tf.math.pow(1 - y_pred, self.gamma) * cross_entropy
        if self.alpha is not None:
            alpha_tensor = tf.constant(self.alpha, dtype=y_pred.dtype)
            alpha_factor = y_true * alpha_tensor
            loss = alpha_factor * loss
        return tf.reduce_sum(loss, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits
        })
        return config

# =====================
# Parámetros para extracción de características
# =====================
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 8
LBP_RADIUS     = 1
LBP_POINTS     = 8 * LBP_RADIUS

# =====================
# Funciones de extracción
# =====================
def segment_lesion(gray_img):
    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if gray_img[mask == 255].mean() < gray_img[mask == 0].mean():
        mask = cv2.bitwise_not(mask)
    mask = closing(mask, disk(5))
    mask = opening(mask, disk(3))
    return mask

def compute_glcm_features(gray_roi, mask_roi):
    quant = (gray_roi // (256 // GLCM_LEVELS)).astype(np.uint8)
    quant[mask_roi == 0] = 0
    glcm = graycomatrix(
        quant,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True
    )
    feats = {}
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'ASM', 'correlation']
    for prop in props:
        vals = graycoprops(glcm, prop=prop)
        feats[f'glcm_{prop}'] = vals.mean()
    return feats

def compute_lbp_features(gray_roi, mask_roi):
    lbp = local_binary_pattern(gray_roi, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_masked = lbp[mask_roi == 255].ravel()
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    # Se definen histogramas de 0 a n_bins-2 (uniform), igual que en entrenamiento
    return {f'lbp_{i}': hist[i] for i in range(n_bins-1)}

# CAMBIO: extraer características devolviendo siempre dict con todas las columnas esperadas
def extract_features_from_array(img_rgb, gray, feature_columns):
    mask = segment_lesion(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats = {}
    if contours:
        c = max(contours, key=cv2.contourArea)
        lesion_mask = np.zeros_like(mask)
        cv2.drawContours(lesion_mask, [c], -1, 255, -1)
        # Color stats
        for i, col in enumerate(['R','G','B']):
            pix = img_rgb[:,:,i][lesion_mask==255].astype(float)
            feats[f'mean_{col}'] = pix.mean() if pix.size else np.nan
            feats[f'std_{col}']  = pix.std()  if pix.size else np.nan
        # Shape metrics
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        x,y,w,h = cv2.boundingRect(c)
        rect_area = w*h
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        feats.update({
            'lesion_area': area,
            'lesion_perimeter': peri,
            'bbox_area': rect_area,
            'solidity': area/hull_area if hull_area>0 else np.nan,
            'extent': area/rect_area if rect_area>0 else np.nan,
        })
        # ROI para GLCM y LBP
        roi_gray = gray[y:y+h, x:x+w]
        mask_roi = lesion_mask[y:y+h, x:x+w]
        feats.update(compute_glcm_features(roi_gray, mask_roi))
        feats.update(compute_lbp_features(roi_gray, mask_roi))
    # Construir dict completo con todas las columnas, rellenando NaN donde no haya valor
    full = {col: np.nan for col in feature_columns}
    for k, v in feats.items():
        if k in full:
            full[k] = v
        # si hay claves inesperadas, se ignoran
    return full

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Clasificador de Imágenes y Metadatos", layout="centered")
st.title("🧠 Clasificador de Lesiones Cutáneas")
st.markdown("Sube una imagen de una lesión y completa los metadatos para predecir su clase y extraer características.")

# CLASES y METADATA
CLASSES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
# CAMBIO: no fit aquí; cargaremos LabelEncoder guardado
# le_class = LabelEncoder(); le_class.fit(CLASSES)

# CAMBIO: cargar lista de columnas, pipeline y labelencoder
import joblib, json

@st.cache_resource
def load_feature_columns():
    with open("feature_columns.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor_metadata.pkl")

@st.cache_resource
def load_labelencoder():
    return joblib.load("labelencoder_class.pkl")

@st.cache_resource
def load_trained_model():
    model = load_model(
        "modelo_hibrido_entrenado.h5",
        custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy},
        compile=False
    )
    model.compile(
        optimizer="adam",
        loss=CategoricalFocalCrossentropy(gamma=2.0, alpha=None, from_logits=False),
        metrics=["accuracy"]
    )
    return model

# Cargar recursos
feature_columns = load_feature_columns()  # lista de 26 columnas
preprocessor = load_preprocessor()
le_class = load_labelencoder()
model = load_trained_model()

# Preprocesamiento imagen (sin cambios)
def center_crop_to_square(img):
    h, w = img.shape[:2]
    if h == w: return img.copy()
    if h > w:
        diff = h - w; top = diff // 2
        return img[top:top + w, :]
    diff = w - h; left = diff // 2
    return img[:, left:left + h]

def crop_and_resize_to_224(img, target_size=224):
    square = center_crop_to_square(img)
    return cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def crop_non_black_region(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    return img[y0:y1 + 1, x0:x1 + 1]

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped = crop_non_black_region(img_cv2)
    resized = crop_and_resize_to_224(cropped)
    processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return processed.astype('float32') / 255.0

# Formulario metadatos
st.subheader("📋 Introduce los metadatos")
edad = st.number_input("Edad aproximada", min_value=0, max_value=100, value=50)
sexo = st.selectbox("Sexo", options=["male", "female"])
# Asegúrate de que estos strings coincidan exactamente con los usados en entrenamiento
site = st.selectbox("Zona anatómica", options=["head/neck", "torso", "lower extremity", "upper extremity", "palms/soles", "oral/genital", "unknown"])
dataset = st.selectbox("Fuente del dataset", options=["BCN_nan", "HAM_vidir_molemax", "HAM_vidir_modern", "HAM_rosendahl", "MSK4nan", "HAM_vienna_dias"])

# CAMBIO: función para construir DataFrame de metadatos + features
def build_metadata_df(edad, sexo, site, dataset, feature_dict, feature_columns):
    if edad <= 35:
        age_group = "young"
    elif edad <= 65:
        age_group = "adult"
    else:
        age_group = "senior"
    age_sex_interaction = f"{sexo}_{age_group}"
    row = {
        "age_approx": edad,
        "sex": sexo,
        "anatom_site_general": site,
        "dataset": dataset,
        "age_sex_interaction": age_sex_interaction,
    }
    # Añadir columnas de features de imagen
    for col in feature_columns:
        row[col] = feature_dict.get(col, np.nan)
    return pd.DataFrame([row])

# Eliminamos manual_metadata_encoding()

# Subida y procesamiento
tile = st.file_uploader("Sube una imagen de piel", type=["jpg", "jpeg", "png"])
if tile is not None:
    st.image(tile, caption="Imagen original", use_container_width=True)
    proc_img = preprocess_image(tile)
    img_input = np.expand_dims(proc_img, axis=0)

    # Extracción de características
    img_np = (proc_img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    feats_raw = extract_features_from_array(img_np, gray, feature_columns)  # CAMBIO: pasar feature_columns
    df_feats = pd.DataFrame([feats_raw])
    st.subheader("📑 Características extraídas")
    st.table(df_feats)
    # Advertencia si todo NaN
    if all(pd.isna(v) for v in feats_raw.values()):
        st.warning("No se detectó lesión; se usarán valores de imputación para características de imagen.")

    # Construir DataFrame de metadatos + features
    df_meta_input = build_metadata_df(edad, sexo, site, dataset, feats_raw, feature_columns)
    st.subheader("📋 Metadata + Features (antes de transformación)")
    st.write(df_meta_input.T)

    # Transformar con pipeline
    try:
        X_meta = preprocessor.transform(df_meta_input)
    except Exception as e:
        st.error(f"Error en preprocesamiento de metadatos: {e}")
        st.stop()
    st.write(f"Forma de entrada de metadatos tras preprocesar: {X_meta.shape}")

    # Predicción
    prediction = model.predict([img_input, X_meta])
    pred_class = np.argmax(prediction, axis=1)[0]
    class_label = le_class.inverse_transform([pred_class])[0]
    confidence = float(np.max(prediction)) * 100
    st.subheader("📊 Resultado de la predicción")
    st.write(f"**Clase predicha:** {class_label}")
    st.write(f"**Confianza:** {confidence:.2f}%")

    st.markdown("---")
    st.caption("Hecho con ❤️ usando Streamlit para el TFG")
