import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

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
# CONFIGURACIÓN INICIAL
# =====================
st.set_page_config(page_title="Clasificador de Imágenes y Metadatos", layout="centered")
st.title("🧠 Clasificador de Lesiones Cutáneas")
st.markdown("Sube una imagen de una lesión y completa los metadatos para predecir su clase.")

# =====================
# CLASES y METADATA
# =====================
CLASSES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
le_class = LabelEncoder()
le_class.fit(CLASSES)

# =====================
# Cargar el modelo
# =====================
@st.cache_resource
def load_trained_model():
    # 1) Carga sin compilar
    model = load_model(
        "modelo_hibrido_entrenado.h5",
        custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy},
        compile=False
    )
    # 2) Re-compila con tu loss
    model.compile(
        optimizer="adam",
        loss=CategoricalFocalCrossentropy(gamma=2.0, alpha=None, from_logits=False),
        metrics=["accuracy"]
    )
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
# Formulario de metadatos (opcional)
# =====================
st.subheader("📋 Introduce los metadatos")

edad = st.number_input("Edad aproximada", min_value=0, max_value=100, value=50)
sexo = st.selectbox("Sexo", options=["male", "female"])
site = st.selectbox("Zona anatómica", options=["head/neck", "torso", "lower extremity", "upper extremity", "palms/soles", "oral/genital", "unknown"])
dataset = st.selectbox("Fuente del dataset", options=["vidir", "msk", "unknown"])

if edad <= 35:
    age_group = "young"
elif edad <= 65:
    age_group = "adult"
else:
    age_group = "senior"
interaction = f"{sexo}_{age_group}"

columns = ["age_approx", "sex", "anatom_site_general", "dataset", "age_sex_interaction"]

def manual_metadata_encoding():
    df = pd.DataFrame([{
        "age_approx": edad,
        "sex": sexo,
        "anatom_site_general": site,
        "dataset": dataset,
        "age_sex_interaction": interaction
    }])
    return np.zeros((1, 26), dtype=np.float32)  # Sustituye por tu preprocesamiento real

# =====================
# Clasificación
# =====================
st.subheader("📷 Imagen")

uploaded_file = st.file_uploader("Sube una imagen de piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen original", use_column_width=True)
    processed_img = preprocess_image(uploaded_file)
    img_input = np.expand_dims(processed_img, axis=0)

    meta_input = manual_metadata_encoding()

    prediction = model.predict([img_input, meta_input])
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_label = le_class.inverse_transform([predicted_class])[0]
    confidence = float(np.max(prediction)) * 100

    st.subheader("📊 Resultado de la predicción")
    st.write(f"**Clase predicha:** {class_label}")
    st.write(f"**Confianza:** {confidence:.2f}%")

    st.markdown("---")
    st.caption("Hecho con ❤️ usando Streamlit para el TFG")
