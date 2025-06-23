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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
# Configuración de preprocesamiento de metadatos
# =====================

def get_age_sex_interaction(row):
    age = row['age_approx']
    sex = row['sex']
    if age <= 35:
        age_group = 'young'
    elif age <= 65:
        age_group = 'adult'
    else:
        age_group = 'senior'
    return f"{sex}_{age_group}"

@st.cache_resource
def load_metadata_preprocessor(path_csv):
    df = pd.read_csv(path_csv)
    # Anatomía como NaN inicial
    df['anatom_site_general'] = df['anatom_site_general'].fillna(np.nan)
    # Crear interaccion edad-sex
    df['age_sex_interaction'] = df.apply(get_age_sex_interaction, axis=1)
    # Asegurar tipo string
    for col in ['sex','dataset']:
        df[col] = df[col].astype(object)
    # Columnas a usar
    categorical_cols = ['sex','anatom_site_general','dataset','age_sex_interaction']
    numerical_continuous = ['age_approx']
    color_texture_cols = [col for col in df.columns if col.startswith(('mean_','std_','glcm_','lbp_','lesion_','bbox_'))]
    # Imputadores y pipelines
    cat_imp = SimpleImputer(strategy='constant', fill_value='unknown')
    num_scale_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    cat_pipe = make_pipeline(
        cat_imp,
        OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    )
    pre = ColumnTransformer(
        transformers=[
            ('num_scale', num_scale_pipe, numerical_continuous + color_texture_cols),
            ('cat', cat_pipe, categorical_cols)
        ], verbose_feature_names_out=False, remainder='drop'
    )
    # Fit solo sobre entrenamiento si existe columna 'subset'
    if 'subset' in df.columns:
        train_df = df[df['subset']=='train']
    else:
        train_df = df
    pre.fit(train_df)
    return pre

# Cargar preprocesador y label encoder de clases
preprocessor = load_metadata_preprocessor('metadata.csv')  # Ajusta la ruta a tu CSV
le_classes = LabelEncoder()
le_classes.fit(['AK','BCC','BKL','DF','MEL','NV','SCC','VASC'])

# Cargar modelo
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

model = load_trained_model()

# Preprocesamiento imagen
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
df_meta = pd.DataFrame(columns=['age_approx','sex','anatom_site_general','dataset','age_sex_interaction'])
edad = st.number_input("Edad aproximada", min_value=0, max_value=100, value=50)
sexo = st.selectbox("Sexo", options=["male","female"])
site = st.selectbox("Zona anatómica", options=["head/neck","torso","lower extremity","upper extremity","palms/soles","oral/genital","unknown"])
dataset = st.selectbox("Fuente del dataset", options=["BCN_nan","HAM_vidir_molemax","HAM_vidir_modern","HAM_rosendahl","MSK4nan","HAM_vienna_dias"])
interaction = get_age_sex_interaction({'age_approx':edad,'sex':sexo})
# Preparar DataFrame de entrada
row = {'age_approx':edad,'sex':sexo,'anatom_site_general':site,'dataset':dataset,'age_sex_interaction':interaction}
meta_df = pd.DataFrame([row])
meta_input = preprocessor.transform(meta_df)

# Subida y procesamiento de imagen
tile = st.file_uploader("Sube una imagen de piel", type=["jpg","jpeg","png"])
if tile is not None:
    st.image(tile, caption="Imagen original", use_container_width=True)
    proc_img = preprocess_image(tile)
    img_input = np.expand_dims(proc_img, axis=0)

    feats = extract_features_from_array((proc_img*255).astype(np.uint8), cv2.cvtColor((proc_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    df_feats = pd.DataFrame([feats])
    st.subheader("📑 Características extraídas")
    st.table(df_feats)

    # Predicción
    prediction = model.predict([img_input, meta_input])
    pred_class = np.argmax(prediction, axis=1)[0]
    class_label = le_classes.inverse_transform([pred_class])[0]
    confidence = float(np.max(prediction))*100
    st.subheader("📊 Resultado de la predicción")
    st.write(f"**Clase predicha:** {class_label}")
    st.write(f"**Confianza:** {confidence:.2f}%")

    st.markdown("---")
    st.caption("Hecho con ❤️ usando Streamlit para el TFG")
