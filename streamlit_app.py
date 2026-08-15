import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "improved_unet2.keras"
IMG_SIZE = (256, 256)


# ============================================================
# Custom metric / loss used by the notebook
# ============================================================
def dice_coef(y_true, y_pred, smooth=100):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true) + K.sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred, smooth=100):
    return 1 - dice_coef(y_true, y_pred, smooth)


def iou_coef(y_true, y_pred, smooth=100):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_m(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def recall_m(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


# ============================================================
# Load model once
# ============================================================
@st.cache_resource
def load_segmentation_model():
    return load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coef": dice_coef,
            "dice_loss": dice_loss,
            "iou_coef": iou_coef,
            "precision_m": precision_m,
            "recall_m": recall_m,
        },
        compile=False,
    )


# ============================================================
# Prediction
# ============================================================
def predict_mask(model, image, threshold):
    image_rgb = np.array(image.convert("RGB"))

    resized = cv2.resize(image_rgb, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0

    prediction = model.predict(
        np.expand_dims(normalized, axis=0),
        verbose=0
    )[0, :, :, 0]

    binary_mask = (prediction > threshold).astype(np.uint8)

    # Create red tumor overlay, matching the notebook visualization.
    overlay = resized.copy()
    overlay[binary_mask == 1] = [255, 0, 0]

    # Blend original image with prediction overlay.
    blended = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)

    return resized, prediction, binary_mask, blended


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor Segmentation")
st.caption("U-Net based MRI brain tumor segmentation")

st.markdown(
    """
    Upload an MRI brain image and the trained U-Net model will generate
    a pixel-level tumor segmentation mask.
    """
)

with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
    )

    st.markdown("---")
    st.info(
        "Model input size: 256 × 256\n\n"
        "Architecture: Improved U-Net"
    )

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if uploaded_file is not None:

    if not Path(MODEL_PATH).exists():
        st.error(
            f"Model file `{MODEL_PATH}` was not found. "
            "Place the trained .keras model in the same folder as this app."
        )
        st.stop()

    try:
        image = Image.open(uploaded_file)

        with st.spinner("Loading model..."):
            model = load_segmentation_model()

        with st.spinner("Segmenting tumor..."):
            original, probability, mask, overlay = predict_mask(
                model, image, threshold
            )

        st.success("Segmentation completed successfully.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original MRI")
            st.image(original, use_container_width=True)

        with col2:
            st.subheader("Tumor Segmentation")
            st.image(mask * 255, clamp=True, use_container_width=True)

        st.subheader("Prediction Overlay")
        st.image(overlay, use_container_width=True)

        # Simple prediction statistics.
        tumor_pixels = int(np.sum(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        tumor_percentage = (tumor_pixels / total_pixels) * 100

        m1, m2 = st.columns(2)
        m1.metric("Detected Tumor Pixels", f"{tumor_pixels:,}")
        m2.metric("Tumor Area", f"{tumor_percentage:.2f}%")

        with st.expander("Show Probability Mask"):
            st.image(
                probability,
                caption="Model prediction probability",
                clamp=True,
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("👆 Upload an MRI image to start segmentation.")

st.markdown("---")
st.caption("Brain Tumor Segmentation • Improved U-Net • TensorFlow/Keras")
