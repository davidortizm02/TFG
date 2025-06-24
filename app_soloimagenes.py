import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
import matplotlib.pyplot as plt

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
    Carga el modelo entrenado únicamente con imagen y devuelve el modelo y las etiquetas.
    """
    model = load_model("/kaggle/input/model_images/keras/default/1/modelo_imagenes_entrenado2.keras", compile=False)
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    return model, class_names

model, class_names = load_model_and_labels()

# =====================
# Preprocesamiento de imagen para el modelo
# =====================
def center_crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img.copy()
    if h > w:
        diff = h - w
        top = diff // 2
        return img[top : top + w, :]
    diff = w - h
    left = diff // 2
    return img[:, left : left + h]


def crop_non_black_region(img: np.ndarray, thresh: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return img[y0:y1 + 1, x0:x1 + 1]


def preprocess_image(image_file, target_size: int = 224) -> np.ndarray:
    # Carga y conversión a BGR
    img = Image.open(image_file).convert('RGB')
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # Elimina fondo negro y recorta
    cropped = crop_non_black_region(bgr, thresh=10)
    # Recorte central y resize
    square = center_crop_to_square(cropped)
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # Convertir a RGB para mostrar
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb

# =====================
# Interfaz de Streamlit
# =====================
st.title("🧠 Clasificador de Lesiones (Sólo Imagen)")
st.markdown("Sube una imagen de una lesión para predecir su clase usando únicamente la imagen.")
uploaded = st.file_uploader("Selecciona un JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    # Preprocesar imagen y mostrar
    img_rgb = preprocess_image(uploaded)
    st.image(img_rgb, caption="Imagen preprocesada 224×224", use_container_width=True)

    # Preparar para EfficientNetV2
    img_array = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    img_array = effnet_preprocess(img_array)

    # Predicción
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    label = class_names[idx]
    conf = float(preds[0][idx])

    # Mostrar resultados
    st.success(f"**Clase predicha:** {label}  |  **Confianza:** {conf:.2%}")

    # Gráfico de probabilidades
    prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    st.bar_chart(prob_dict)
else:
    st.info("Sube una imagen para comenzar la predicción.")
