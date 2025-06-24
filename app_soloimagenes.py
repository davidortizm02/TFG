import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
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

    st.image(img_input, caption="Imagen 224×224", use_container_width=True)
    img_array = image.img_to_array(img_input)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Dependiendo del modelo, a veces puedes omitir esto
    # Predicción
    batch = np.expand_dims(img_array, axis=0)
    #st.image(batch, caption="Antes de predecir", use_container_width=True)
    preds = model.predict(batch, verbose=0)
    st.info(preds)
    idx = int(np.argmax(preds))
    label = class_names[idx]
    conf = float(preds[idx])

    st.success(f"**Clase predicha:** {label}  |  **Confianza:** {conf:.2%}")

    # Gráfico de probabilidades
    dfp = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    st.bar_chart(dfp)
else:
    st.info("Sube una imagen para comenzar la predicción.")
