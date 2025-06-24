import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import closing, opening, disk
import joblib
import json
import keras

# ====================================================================
# Implementación de Focal Loss (si es necesaria para cargar el modelo)
# ====================================================================
class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=0.5, alpha=None, from_logits=False, **kwargs):
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
# Parámetros para extracción de características (GLCM, LBP) - Basados en tu script
# =====================
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 8  # cuantización a 8 niveles
LBP_RADIUS     = 1
LBP_POINTS     = 8 * LBP_RADIUS

# =======================================================================
# NUEVAS FUNCIONES DE EXTRACCIÓN DE CARACTERÍSTICAS (PROPORCIONADAS POR TI)
# =======================================================================

def segment_lesion(gray_img):
    """
    Segmenta la lesión mediante Otsu + operaciones morfológicas.
    Devuelve máscara binaria (255=lesión).
    """
    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invertir si detecta fondo como lesión
    if np.any(mask == 255) and np.any(mask == 0): # Evitar error si la máscara es toda blanca o negra
      if gray_img[mask == 255].mean() < gray_img[mask == 0].mean():
          mask = cv2.bitwise_not(mask)
    # Limpieza morfológica
    # Nota: se convierten a bool para skimage y de vuelta a uint8 para opencv
    mask = closing(mask.astype(bool), disk(5)).astype(np.uint8) * 255
    mask = opening(mask.astype(bool), disk(3)).astype(np.uint8) * 255
    return mask


def compute_glcm_features(gray_roi, mask_roi):
    """
    Calcula propiedades GLCM multi-distancia y multi-ángulo sobre ROI enmascarado.
    """
    # Cuantización
    quant = (gray_roi // (256 // GLCM_LEVELS)).astype(np.uint8)
    # Asegurar que background se ignora
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
    """
    Calcula histogram LBP uniform patterns dentro de la ROI.
    """
    lbp = local_binary_pattern(gray_roi, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_masked = lbp[mask_roi == 255].ravel()
    
    if lbp_masked.size == 0:
        return {} # Devuelve vacío si no hay pixeles en la máscara

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    # Se retornan los primeros n_bins-1 bins (omitir último si es ruido)
    return {f'lbp_{i}': hist[i] for i in range(n_bins-1)}


def extract_features_from_array(img_rgb_uint8, gray_uint8, feature_columns):
    """
    Extrae features para una sola imagen dada como arrays, usando la lógica de tu script.
    - img_rgb_uint8: imagen RGB uint8 (shape HxWx3).
    - gray_uint8: imagen en gris uint8 (shape HxW).
    - feature_columns: lista de nombres de columnas esperadas en el pipeline.
    Retorna:
      feats_aligned: dict con todas las columnas, rellenando con np.nan si no se encuentra.
      segmentation_mask: máscara binaria 2D uint8 (0 o 255).
    """
    feats = {}

    # 1) Segmentar lesión (usando tu nueva función)
    mask = segment_lesion(gray_uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Si no hay contornos, devuelve NaNs y una máscara negra.
        feats_aligned = {col: np.nan for col in feature_columns}
        return feats_aligned, np.zeros_like(mask)

    c = max(contours, key=cv2.contourArea)
    lesion_mask = np.zeros_like(mask)
    cv2.drawContours(lesion_mask, [c], -1, 255, -1)

    # 2) Estadísticos de color en RGB
    for i, col in enumerate(['R','G','B']):
        pix = img_rgb_uint8[:,:,i][lesion_mask==255].astype(float)
        feats[f'mean_{col}'] = pix.mean() if pix.size else np.nan
        feats[f'std_{col}']  = pix.std()  if pix.size else np.nan

    # 3) Métricas de forma
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x,y,w,h = cv2.boundingRect(c)
    rect_area = w*h
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    
    feats.update({
        'lesion_area': float(area),
        'lesion_perimeter': float(peri),
        'bbox_area': float(rect_area),
        'solidity': float(area/hull_area) if hull_area > 0 else np.nan,
        'extent': float(area/rect_area) if rect_area > 0 else np.nan,
    })

    # 4) GLCM y LBP en ROI recortado
    # Solo proceder si el ROI tiene tamaño
    if w > 0 and h > 0:
        roi_gray = gray_uint8[y:y+h, x:x+w]
        mask_roi = lesion_mask[y:y+h, x:x+w]
        
        # GLCM
        try:
            feats.update(compute_glcm_features(roi_gray, mask_roi))
        except Exception as e:
            print(f"Error en GLCM: {e}")
            # Rellenar con NaN si falla
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'ASM', 'correlation']
            for prop in props: feats[f'glcm_{prop}'] = np.nan

        # LBP
        try:
            feats.update(compute_lbp_features(roi_gray, mask_roi))
        except Exception as e:
            print(f"Error en LBP: {e}")
            # No es necesario rellenar LBP con NaN, el alineado lo hará

    # 5) Alineamos al listado feature_columns: si falta, np.nan
    feats_aligned = {}
    for col in feature_columns:
        feats_aligned[col] = feats.get(col, np.nan)
        
    return feats_aligned, lesion_mask

# =====================
# Carga de recursos (cacheado)
# =====================
@st.cache_resource
def load_all_resources():
    """Carga todos los modelos y preprocesadores de una vez."""
    with open("feature_columnsDEF.json", "r") as f:
        feature_columns = json.load(f)
    
    preprocessor = joblib.load("preprocessor_metadataDEF.pkl")
    label_encoder = joblib.load("labelencoder_classDEF.pkl")
    
    model = load_model(
        "modelo_hibrido_entrenado.h5",
        compile=False  # Para predicción no es necesario recompilar
    )
    return feature_columns, preprocessor, label_encoder, model

# =====================
# Funciones de preprocesamiento de imagen para el modelo
# =====================
def center_crop_to_square(img):
    h, w = img.shape[:2]
    if h == w: return img.copy()
    if h > w:
        diff = h - w; top = diff // 2
        return img[top:top + w, :]
    diff = w - h; left = diff // 2
    return img[:, left:left + h]

def crop_and_resize(img, target_size=224):
    square = center_crop_to_square(img)
    return cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)

def crop_non_black_region(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    return img[y0:y1 + 1, x0:x1 + 1]

def preprocess_image_for_model(image_file):
    """Prepara una imagen subida para la entrada del modelo."""
    img = Image.open(image_file).convert('RGB')
    img_np_original = np.array(img)
    img_cv2 = cv2.cvtColor(img_np_original, cv2.COLOR_RGB2BGR)
    
    cropped = crop_non_black_region(img_cv2)
    resized = crop_and_resize(cropped, target_size=224)
    
    processed_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return processed_rgb.astype('float32') / 255.0

# =====================
# Interfaz de Streamlit
# =====================
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="wide")
st.title("🧠 Clasificador de Lesiones Cutáneas")
st.markdown("Sube una imagen de una lesión y completa los metadatos para predecir su tipo. Esta versión incluye herramientas de diagnóstico.")

# Cargar recursos
try:
    feature_columns, preprocessor, le_class, model = load_all_resources()
except FileNotFoundError as e:
    st.error(f"Error al cargar un archivo necesario: {e}. Asegúrate de que `feature_columns.json`, `preprocessor_metadata.pkl`, `labelencoder_class.pkl` y `modelo_hibrido_entrenadoCW.keras` están en la misma carpeta que la app.")
    st.stop()

# --- Layout de la App ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Sube la Imagen")
    tile = st.file_uploader("Selecciona un archivo de imagen", type=["jpg", "jpeg", "png"])

    st.header("2. Introduce los Metadatos")
    with st.form("metadata_form"):
        edad = st.number_input("Edad aproximada", min_value=0, max_value=120, value=50)
        sexo = st.selectbox("Sexo", options=["male", "female", "unknown"])
        site = st.selectbox("Zona anatómica", options=[
            "anterior torso", "head/neck", "lateral torso", "lower extremity",
            "upper extremity", "oral/genital", "palms/soles", "posterior torso", "unknown"
        ])
        dataset = st.selectbox("Fuente del dataset (si se conoce)", options=[
            "BCN_nan", "HAM_vidir_molemax", "HAM_vidir_modern", 
            "HAM_rosendahl", "MSK4nan", "HAM_vienna_dias"
        ])
        submit_button = st.form_submit_button(label='Realizar Predicción')

# --- Procesamiento y Predicción ---
if tile is not None and submit_button:
    with col2:
        st.header("3. Análisis y Predicción")
        
        # --- Preprocesamiento de la imagen ---
        img_for_model = preprocess_image_for_model(tile)
        img_for_features = (img_for_model * 255).astype(np.uint8)
        gray_for_features = cv2.cvtColor(img_for_features, cv2.COLOR_RGB2GRAY)
        
        # --- Extracción de características ---
        feats_raw, segmentation_mask = extract_features_from_array(img_for_features, gray_for_features, feature_columns)
        
        with st.expander("🔍 Diagnóstico: Extracción de Características", expanded=True):
            st.info("Aquí puedes ver el resultado de la segmentación de la lesión y las características numéricas extraídas de ella.")
            
            c1, c2 = st.columns(2)
            c1.image(img_for_model, caption="Imagen Procesada (224x224)", use_container_width=True)
            c2.image(segmentation_mask, caption="Máscara de Lesión Segmentada", use_container_width=True)
            st.caption("Si la máscara es negra o no resalta la lesión, las características serán incorrectas (NaNs) y el modelo dependerá solo de los metadatos.")

            if all(pd.isna(v) for v in feats_raw.values()):
                st.warning("No se detectó ninguna lesión. Todas las características de la imagen son NaN y serán imputadas por el preprocesador.")
            
            st.subheader("Características numéricas extraídas (raw)")
            st.dataframe(pd.DataFrame([feats_raw]))
            
        # --- Preparación de metadatos ---
        if edad <= 35:
            age_group = "young"
        elif edad <= 65:
            age_group = "adult"
        else:
            age_group = "senior"
        
        df_meta_input = pd.DataFrame([{
            "age_approx": edad,
            "sex": sexo,
            "anatom_site_general": site,
            "dataset": dataset,
            "age_sex_interaction": f"{sexo}_{age_group}",
            **feats_raw
        }])
        
        # --- Preprocesamiento de metadatos ---
        try:
            X_meta = preprocessor.transform(df_meta_input)
        except Exception as e:
            st.error(f"Error al transformar los metadatos con el pipeline: {e}")
            st.stop()
        
        with st.expander("🔬 Diagnóstico: Preprocesamiento de Metadatos", expanded=True):
            st.info("Estos son los datos que entran al pipeline y la matriz final que recibe la red neuronal.")
            st.subheader("Datos ANTES de la transformación")
            st.dataframe(df_meta_input)

            st.subheader("Datos DESPUÉS de la transformación (Entrada final al modelo)")
            st.caption(f"Esta es la matriz numérica (shape: {X_meta.shape}) que realmente recibe la red.")
            X_meta_display = X_meta.toarray() if hasattr(X_meta, "toarray") else X_meta
            st.dataframe(pd.DataFrame(X_meta_display))
        
        # --- Predicción del modelo ---
        img_input_batch = np.expand_dims(img_for_model, axis=0)
        prediction = model.predict([img_input_batch, X_meta])
        
        with st.container():
            st.header("📊 Resultado Final")
            pred_class_index = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            class_label = le_class.inverse_transform([pred_class_index])[0]
            
            st.success(f"**Clase Predicha:** {class_label} (Confianza: {confidence:.2%})")

            st.subheader("Probabilidades por Clase")
            df_preds = pd.DataFrame({
                'Clase': le_class.classes_,
                'Probabilidad': prediction.flatten()
            }).sort_values(by='Probabilidad', ascending=False).set_index('Clase')
            
            st.bar_chart(df_preds)
else:
    with col2:
        st.info("Sube una imagen y rellena el formulario para ver la predicción.")

st.markdown("---")
st.caption("Aplicación para TFG. Versión con herramientas de diagnóstico.")