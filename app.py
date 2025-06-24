import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import closing, opening, disk
from skimage.measure import label, regionprops
import joblib
import json
import keras
# =====================
# Constantes de segmentación
# =====================
# Ajusta estos valores según tu caso: radios de apertura/cierre y área mínima de lesión
MORPH_OPEN_RADIUS = 5
MORPH_CLOSE_RADIUS = 5
MIN_LESION_AREA    = 100  # en píxeles; descarta regiones muy pequeñas

# ====================================================================
# Implementación de Focal Loss (si es necesaria para cargar el modelo)
# Aunque compile=False, es buena práctica tenerla por si acaso.
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
# Parámetros para extracción de características (GLCM, LBP)
# =====================
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 8
LBP_RADIUS     = 1
LBP_POINTS     = 8 * LBP_RADIUS

# =====================
# Funciones de extracción de características
# =====================
def segment_lesion(gray_img):
    """
    Segmenta la lesión mediante Otsu + apertura/cierre + selección de la CC más grande.
    Devuelve máscara binaria uint8 con valores 0 o 255.
    gray_img: array 2D uint8.
    """
    # 1) Otsu con desenfoque
    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) Invertir si fuese necesario: comparamos medias, solo si hay suficientes pixeles en fg/bg
    fg = gray_img[mask == 255]
    bg = gray_img[mask == 0]
    if fg.size > 0 and bg.size > 0:
        # Si la media de la región binarizada (fg) es menor que la del fondo, invertimos:
        if fg.mean() < bg.mean():
            mask = cv2.bitwise_not(mask)

    # 3) Convertir a booleano y aplicar apertura (opening) y cierre (closing)
    mask_bool = mask > 0
    mask_bool = opening(mask_bool, disk(MORPH_OPEN_RADIUS))
    mask_bool = closing(mask_bool, disk(MORPH_CLOSE_RADIUS))

    # 4) Seleccionar la CC más grande
    labels = label(mask_bool)
    if labels.max() == 0:
        # No hay regiones
        return np.zeros_like(mask, dtype=np.uint8)
    regions = regionprops(labels)
    max_region = max(regions, key=lambda r: r.area)
    if max_region.area < MIN_LESION_AREA:
        # Región demasiado pequeña: descartamos
        return np.zeros_like(mask, dtype=np.uint8)
    # Máscara solo de la región más grande
    mask_out = (labels == max_region.label).astype(np.uint8) * 255
    return mask_out

def compute_glcm_features(gray_roi, mask_roi):
    """
    Calcula propiedades GLCM multi-distancia y multi-ángulo sobre ROI enmascarado.
    gray_roi: ROI recortado de la imagen en gris.
    mask_roi: correspondiente máscara (0 o 255) del mismo tamaño que gray_roi.
    Devuelve diccionario con keys: glcm_contrast, glcm_dissimilarity, etc.
    Si roi demasiado pequeño o error, devuelve NaNs en los props.
    """
    ys, xs = np.where(mask_roi == 255)
    if ys.size == 0:
        return {f'glcm_{prop}': np.nan for prop in ['contrast','dissimilarity','homogeneity','energy','ASM','correlation']}

    # Cuantización a GLCM_LEVELS niveles: cuidado con división
    bins = 256 // GLCM_LEVELS
    if bins < 1:
        bins = 1
    quant = (gray_roi // bins).astype(np.uint8)
    # Forzar background=0
    quant[mask_roi == 0] = 0

    try:
        glcm = graycomatrix(
            quant,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True
        )
    except Exception:
        return {f'glcm_{prop}': np.nan for prop in ['contrast','dissimilarity','homogeneity','energy','ASM','correlation']}

    feats = {}
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'ASM', 'correlation']
    for prop in props:
        try:
            vals = graycoprops(glcm, prop=prop)
            feats[f'glcm_{prop}'] = float(vals.mean())
        except Exception:
            feats[f'glcm_{prop}'] = np.nan
    return feats

def compute_lbp_features(gray_roi, mask_roi):
    """
    Calcula histograma LBP (método 'uniform') dentro de la ROI enmascarada.
    gray_roi: ROI en escala de grises.
    mask_roi: máscara 0/255 del mismo tamaño.
    Devuelve dict con claves 'lbp_0', 'lbp_1', ..., hasta n_bins-1.
    """
    ys, xs = np.where(mask_roi == 255)
    if ys.size == 0:
        return {}
    lbp = local_binary_pattern(gray_roi, LBP_POINTS, LBP_RADIUS, method='uniform')
    masked = lbp[mask_roi == 255].ravel()
    if masked.size == 0:
        return {}
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(masked, bins=n_bins, range=(0, n_bins), density=True)
    return {f'lbp_{i}': float(hist[i]) for i in range(n_bins)}

def extract_features_from_array(img_rgb_uint8, gray_uint8, feature_columns):
    """
    Extrae features para una sola imagen dada como arrays:
    - img_rgb_uint8: imagen RGB uint8 (shape HxWx3).
    - gray_uint8: imagen en gris uint8 (shape HxW), normalmente derivada de img_rgb.
    - feature_columns: lista de nombres de columnas esperadas en el pipeline.
    Retorna:
      feats_aligned: dict con todas las columnas de feature_columns, rellenando con np.nan cuando no exista.
      segmentation_mask: máscara binaria 2D uint8 (0 o 255) del mismo tamaño que gray_uint8.
    """
    feats = {}
    # 1) Segmentar lesión
    mask = segment_lesion(gray_uint8)  # máscara 0/255, 2D
    # Si no se segmenta nada, devolvemos todos NaN
    ys_all = np.where(mask == 255)[0]
    if ys_all.size == 0:
        # Retornar diccionario con sólo NaNs para cada feature column (excepto no incluimos 'image' aquí)
        feats_aligned = {col: np.nan for col in feature_columns}
        return feats_aligned, mask

    # 2) Encontrar contorno principal de la máscara final
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        feats_aligned = {col: np.nan for col in feature_columns}
        return feats_aligned, mask
    c = max(contours, key=cv2.contourArea)
    lesion_mask = np.zeros_like(mask)
    cv2.drawContours(lesion_mask, [c], -1, 255, -1)

    # 3) Estadísticos de color en RGB dentro de la lesión
    ys, xs = np.where(lesion_mask == 255)
    if ys.size > 0:
        pix_R = img_rgb_uint8[:,:,0][lesion_mask==255].astype(float)
        pix_G = img_rgb_uint8[:,:,1][lesion_mask==255].astype(float)
        pix_B = img_rgb_uint8[:,:,2][lesion_mask==255].astype(float)
        feats['mean_R'] = float(pix_R.mean())
        feats['std_R']  = float(pix_R.std())
        feats['mean_G'] = float(pix_G.mean())
        feats['std_G']  = float(pix_G.std())
        feats['mean_B'] = float(pix_B.mean())
        feats['std_B']  = float(pix_B.std())
    else:
        feats['mean_R'] = feats['std_R'] = np.nan
        feats['mean_G'] = feats['std_G'] = np.nan
        feats['mean_B'] = feats['std_B'] = np.nan

    # 4) Métricas de forma basadas en el contorno c
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    rect_area = w * h
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / hull_area) if hull_area > 0 else np.nan
    extent = float(area / rect_area) if rect_area > 0 else np.nan

    feats.update({
        'lesion_area': float(area),
        'lesion_perimeter': float(peri),
        'bbox_x': int(x),
        'bbox_y': int(y),
        'bbox_width': int(w),
        'bbox_height': int(h),
        'bbox_area': float(rect_area),
        'solidity': solidity,
        'extent': extent,
    })

    # 5) GLCM y LBP en ROI recortado al bounding box
    gray_roi = gray_uint8[y:y+h, x:x+w]
    mask_roi = lesion_mask[y:y+h, x:x+w]

    # GLCM
    glcm_feats = compute_glcm_features(gray_roi, mask_roi)
    feats.update(glcm_feats)
    # LBP
    lbp_feats = compute_lbp_features(gray_roi, mask_roi)
    feats.update(lbp_feats)

    # 6) Alineamos al listado feature_columns: si falta, np.nan; si hay en feats que no esté en feature_columns, lo ignoramos.
    feats_aligned = {}
    for col in feature_columns:
        feats_aligned[col] = feats.get(col, np.nan)
    return feats_aligned, mask

# =====================
# Carga de recursos (cacheado)
# =====================

@st.cache_resource
def load_all_resources():
    """Carga todos los modelos y preprocesadores de una vez."""
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    
    preprocessor = joblib.load("preprocessor_metadata.pkl")
    label_encoder = joblib.load("labelencoder_class.pkl")
    keras.config.enable_unsafe_deserialization()

    model = load_model(
        "modelo_hibrido_entrenado.h5",
        custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy},
        compile=False  # Para predicción no es necesario recompilar
    )
    return feature_columns, preprocessor, label_encoder, model

# =====================
# Funciones de preprocesamiento de imagen
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
    
    # La imagen final para el modelo es RGB y normalizada
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
    st.error(f"Error al cargar un archivo necesario: {e}. Asegúrate de que `feature_columns.json`, `preprocessor_metadata.pkl`, `labelencoder_class.pkl` y `modelo_hibrido_entrenado.h5` están en la misma carpeta que la app.")
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
        img_for_model = preprocess_image_for_model(tile)  # float32 [0,1], shape (224,224,3)
        img_for_features = (img_for_model * 255).astype(np.uint8)
        gray_for_features = cv2.cvtColor(img_for_features, cv2.COLOR_RGB2GRAY)
        
        # --- Extracción de características ---
        feats_raw, segmentation_mask = extract_features_from_array(img_for_features, gray_for_features, feature_columns)
        
        with st.expander("🔍 Diagnóstico: Extracción de Características", expanded=True):
            st.info("Aquí puedes ver el resultado de la segmentación de la lesión y las características numéricas extraídas de ella.")
            
            c1, c2 = st.columns(2)
            c1.image(img_for_model, caption="Imagen Procesada (224x224)", use_container_width=True)
            # Mostrar la máscara en escala de grises: se puede normalizar para mostrar
            # convertimos a 3 canales para mejor visualización si se quiere, pero st.image muestra bien 2D.
            c2.image(segmentation_mask, caption="Máscara de Lesión Segmentada", use_container_width=True)
            st.caption("Si la máscara es negra o no resalta la lesión, las características serán incorrectas (NaNs) y el modelo dependerá solo de los metadatos.")

            # Verificar si todas las características son NaN
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
            st.caption(f"Esta es la matriz numérica (shape: {X_meta.shape}) que realmente recibe la red. **Si esta matriz es siempre la misma para diferentes imágenes, has encontrado la causa del problema.**")
            X_meta_display = X_meta.toarray() if hasattr(X_meta, "toarray") else X_meta
            st.dataframe(pd.DataFrame(X_meta_display))
        
        # --- Predicción del modelo ---
        img_input_batch = np.expand_dims(img_for_model, axis=0)
        # Muchas arquitecturas híbridas esperan lista [img, meta], asegúrate de que tu modelo acepta esta entrada
        prediction = model.predict([img_input_batch, X_meta])
        #prediction = model.predict(img_input_batch)
        
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
