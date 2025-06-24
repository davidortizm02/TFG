import os
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input  # Ajustar según el backbone real
import joblib
import json

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

# ——————————————————————————————
# Parámetros globales y configuración
# ——————————————————————————————
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="wide")

# =====================
# Carga de recursos (cacheado)
# =====================
@st.cache_resource
def load_all_resources():
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    preprocessor = joblib.load("preprocessor_metadata.pkl")
    label_encoder = joblib.load("labelencoder_class.pkl")
    model = load_model("modelo_hibrido_entrenadoCW.keras", compile=False)
    return feature_columns, preprocessor, label_encoder, model

# =====================
# Preprocesamiento de la imagen para el modelo
# =====================
def preprocess_image_for_model(image_file):
    img = Image.open(image_file).convert('RGB')
    arr = np.array(img)
    # Recortar bordes negros
    gray_tmp = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, mask_tmp = cv2.threshold(gray_tmp, 10, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask_tmp)
    if ys.size:
        arr = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    # Crop cuadrado centrado y resize
    h, w = arr.shape[:2]
    if h != w:
        if h > w:
            diff = (h - w) // 2
            arr = arr[diff:diff+w, :]
        else:
            diff = (w - h) // 2
            arr = arr[:, diff:diff+h]
    arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
    # Convertir a array flotante y aplicar preprocess_input
    x = np.expand_dims(arr, axis=0).astype('float32')
    x = preprocess_input(x)
    return x, arr  # devolvemos también arr para mostrar

# =====================
# Código de interfaz Streamlit
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
    upload = st.file_uploader("Selecciona un JPG/PNG", type=["jpg","jpeg","png"])
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
        submit = st.form_submit_button("Realizar Predicción")

if upload and submit:
    with col2:
        st.header("3. Análisis y Predicción")
        # Preprocesamos imagen y obtenemos array para mostrar
        x_img, img_show = preprocess_image_for_model(upload)
        st.image(img_show, caption="Imagen 224×224 preprocesada", use_container_width=True)

        # Extraer features clásicos (usa img_show convertida)
        gray = cv2.cvtColor(img_show, cv2.COLOR_RGB2GRAY)
        feats_raw, mask = extract_features_from_array(img_show, gray)

        with st.expander("🔍 Diagnóstico: Segmentación y Features", expanded=True):
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

        # Transformación y asegurar array denso
        X_meta = preprocessor.transform(df_meta)
        if hasattr(X_meta, "toarray"):
            X_meta = X_meta.toarray()

        with st.expander("🔬 Diagnóstico: Preprocesamiento Metadatos", expanded=True):
            st.subheader("Antes de transformar")
            st.dataframe(df_meta.fillna("NaN"))
            st.subheader("Después de transformar")
            st.dataframe(pd.DataFrame(X_meta, columns=feature_columns))

        # Predicción con inputs correctos
        pred = model.predict([x_img, X_meta], verbose=0)
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
