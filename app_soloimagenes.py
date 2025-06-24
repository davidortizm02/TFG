import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

# ——————————————————————————————
# Configuración de la página
# ——————————————————————————————
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas (Sólo Imagen)", layout="wide")

# =====================
# Carga de recursos (cacheado)
# =====================
@st.cache_resource
def load_model_and_labels():
    """
    Carga el modelo entrenado únicamente con imagen
    y el LabelEncoder para mapear índices a etiquetas.
    """
    # Sustituye por la ruta a tu modelo sólo-imagen
    model = load_model("modelo_imagenes_entrenado2.keras", compile=False)
    # Si utilizas un LabelEncoder guardado con joblib:
    # import joblib
    # le = joblib.load("labelencoder_class_imagen_only.pkl")
    # Si tus clases son fijas, puedes definirlas manualmente:
    from tensorflow.keras.utils import get_file
    # Ejemplo de clases:
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']  # ajústalas a tus clases reales
    return model, class_names

model, class_names = load_model_and_labels()

# =====================
# Segmentación y extracción de características (opcional para diagnóstico)
# =====================
# Parámetros
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 8
LBP_RADIUS     = 1
LBP_POINTS     = 8 * LBP_RADIUS
MORPH_OPEN_RADIUS  = 3
MORPH_CLOSE_RADIUS = 5
MIN_LESION_AREA    = 100

def segment_lesion(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invertir si hace falta
    fg, bg = gray[mask==255], gray[mask==0]
    if fg.size and bg.size and fg.mean() < bg.mean():
        mask = cv2.bitwise_not(mask)
    mask = opening(mask>0, disk(MORPH_OPEN_RADIUS))
    mask = closing(mask, disk(MORPH_CLOSE_RADIUS))
    labels = label(mask)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    props = regionprops(labels)
    largest = max(props, key=lambda p: p.area)
    if largest.area < MIN_LESION_AREA:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == largest.label).astype(np.uint8) * 255

def extract_features(img_rgb, gray):
    mask = segment_lesion(gray)
    feats = {}
    if mask.max() == 0:
        return feats, mask

    # Contorno principal
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    lesion_mask = np.zeros_like(mask)
    cv2.drawContours(lesion_mask, [c], -1, 255, -1)

    # Estadísticos color
    for i, col in enumerate(['R','G','B']):
        pix = img_rgb[:,:,i][lesion_mask==255].astype(float)
        feats[f"mean_{col}"] = float(pix.mean()) if pix.size else np.nan
        feats[f"std_{col}"]  = float(pix.std())  if pix.size else np.nan

    # Forma
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    feats.update({
        "lesion_area": float(area),
        "lesion_perimeter": float(peri),
        "solidity": float(area/hull_area) if hull_area>0 else np.nan,
        "extent": float(area/(cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3])) if all(cv2.boundingRect(c)[2:]) else np.nan
    })

    # GLCM
    y,x,w,h = cv2.boundingRect(c)
    roi_gray = gray[y:y+h, x:x+w]
    roi_mask = lesion_mask[y:y+h, x:x+w]
    quant = (roi_gray // (256//GLCM_LEVELS)).astype(np.uint8)
    quant[roi_mask==0] = 0
    glcm = graycomatrix(quant, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    for prop in ['contrast','dissimilarity','homogeneity','energy','ASM','correlation']:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())

    # LBP
    lbp = local_binary_pattern(roi_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    vals = lbp[roi_mask==255].ravel()
    if vals.size:
        hist, _ = np.histogram(vals, bins=int(lbp.max()+1), range=(0, lbp.max()+1), density=True)
        for i in range(len(hist)):
            feats[f"lbp_{i}"] = float(hist[i])

    return feats, mask

# =====================
# Preprocesamiento de imagen para el modelo
# =====================
def center_crop_to_square(img):
    h,w = img.shape[:2]
    if h==w: return img
    if h>w:
        t = (h-w)//2
        return img[t:t+w, :]
    l = (w-h)//2
    return img[:, l:l+h]

def crop_non_black(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(m)
    if not ys.size:
        return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

def preprocess_image(image_file, size=224):
    img = Image.open(image_file).convert('RGB')
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cropped = crop_non_black(bgr)
    sq = center_crop_to_square(cropped)
    resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype('float32') / 255.0

# =====================
# Interfaz de Streamlit
# =====================
st.title("🧠 Clasificador de Lesiones (Sólo Imagen)")
st.markdown("Sube una imagen de una lesión para predecir su clase usando únicamente la imagen.")

uploaded = st.file_uploader("Selecciona un JPG/PNG", type=["jpg","jpeg","png"])
if uploaded:
    img_input = preprocess_image(uploaded)
    img_uint8 = (img_input * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    feats, mask = extract_features(img_uint8, gray)

    with st.expander("🔍 Segmentación y Features", expanded=True):
        st.image(img_input, caption="Imagen 224×224", use_container_width=True)
        st.image(mask, caption="Máscara de lesión", use_container_width=True)
        if feats:
            import pandas as pd
            st.dataframe(pd.DataFrame([feats]).fillna("NaN"))
        else:
            st.write("No se detectó lesión para extraer features.")

    # Predicción
    batch = np.expand_dims(img_input, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx]
    conf = float(preds[idx])

    st.success(f"**Clase predicha:** {label}  |  **Confianza:** {conf:.2%}")

    # Gráfico de probabilidades
    dfp = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    st.bar_chart(dfp)
else:
    st.info("Sube una imagen para comenzar la predicción.")
