import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

# =====================
# Cargar el modelo entrenado
# =====================
@st.cache_resource

def load_trained_model():
    model = load_model("final_global_fedavg_model (1).keras")  # Asegúrate de subir modelo.h5 al repo
    return model

model = load_trained_model()

# =====================
# Preprocesamiento de imagen
# =====================

def center_crop_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img.copy()
    if h > w:
        diff = h - w
        top = diff // 2
        return img[top:top + w, :]
    else:
        diff = w - h
        left = diff // 2
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
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return img[y0:y1 + 1, x0:x1 + 1]

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped = crop_non_black_region(img_cv2)
    resized = crop_and_resize_to_224(cropped)
    processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    processed = processed.astype('float32') / 255.0
    return processed

# =====================
# Interfaz Streamlit
# =====================
st.set_page_config(page_title="Clasificador de Imágenes", layout="centered")
st.title("🧠 Clasificador con Red Neuronal")
st.markdown("Sube una imagen para predecir su clase con el modelo entrenado.")

uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen original
    st.image(uploaded_file, caption="Imagen original", use_column_width=True)

    # Preprocesar
    processed_img = preprocess_image(uploaded_file)
    input_img = np.expand_dims(processed_img, axis=0)

    # Predicción
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100

    st.subheader("📊 Resultado")
    st.write(f"**Clase predicha:** {predicted_class}")
    st.write(f"**Confianza:** {confidence:.2f}%")

    st.markdown("---")
    st.markdown("Hecho con ❤️ usando Streamlit para el TFG")
