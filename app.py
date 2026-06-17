from pathlib import Path

import streamlit as st
from PIL import Image

from utils import predict as yolo_predict
from utils.predict import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
)


st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_UNET_WEIGHTS_PATH = PROJECT_ROOT / "model" / "best_unet_weights.weights.h5"
DEFAULT_UNET_THRESHOLD = 0.5


@st.cache_resource(show_spinner="Loading YOLO26L-Seg model...")
def get_yolo_model(model_path: str):
    return yolo_predict.load_model(Path(model_path))


@st.cache_resource(show_spinner="Loading Attention U-Net model...")
def get_unet_model(weights_path: str):
    from utils import predict_unet as unet_predict

    return unet_predict.load_model(Path(weights_path))


st.title("Brain Tumor Segmentation")

with st.sidebar:
    model_choice = st.radio(
        "Prediction model",
        options=["YOLO26l-Seg", "Attention_Unet"],
        horizontal=False,
    )

    if model_choice == "YOLO26l-Seg":
        model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
        confidence = st.slider(
            "Confidence",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_CONFIDENCE),
            step=0.05,
        )
        image_size = st.select_slider(
            "Image size",
            options=[320, 416, 512, 640, 768, 896, 1024],
            value=DEFAULT_IMAGE_SIZE,
        )
    else:
        weights_path = st.text_input(
            "Weights path",
            value=str(DEFAULT_UNET_WEIGHTS_PATH),
        )
        threshold = st.slider(
            "Mask threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_UNET_THRESHOLD),
            step=0.05,
        )

uploaded_file = st.file_uploader("MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is None:
    st.info("Select an MRI image to run inference.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

try:
    if model_choice == "YOLO26l-Seg":
        model = get_yolo_model(model_path)
        result = yolo_predict.predict(model, image, conf=confidence, imgsz=image_size)
        summary = yolo_predict.summary_from_result(result)
        annotated = yolo_predict.render_result(result)
    else:
        from utils import predict_unet as unet_predict

        model = get_unet_model(weights_path)
        result = unet_predict.predict(model, image)
        summary = unet_predict.summary_from_prediction(result, threshold=threshold)
        annotated = unet_predict.render_result(image, result, threshold=threshold)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(image, use_column_width=True)

with right:
    st.subheader("Prediction")
    st.image(annotated, use_column_width=True)

if model_choice == "YOLO26l-Seg":
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Detections", summary["detections"])
    metric_b.metric("Masks", summary["masks"])
    metric_c.metric("Top confidence", f"{summary['top_confidence']:.2f}")

    if summary["items"]:
        rows = [
            {
                "Class": item["class_name"],
                "Confidence": round(item["confidence"], 4),
                "Box xyxy": [round(value, 2) for value in item["box_xyxy"]],
            }
            for item in summary["items"]
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
else:
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Tumor area", f"{summary['area_percent']:.2f}%")
    metric_b.metric("Tumor pixels", summary["tumor_pixels"])
    metric_c.metric("Max probability", f"{summary['max_probability']:.2f}")
