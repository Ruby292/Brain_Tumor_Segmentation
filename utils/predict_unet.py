from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model, layers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "model" / "best_unet_weights_inference.weights.h5"
IMG_SIZE = 256
NUM_CHANNELS = 3
NUM_CLASSES = 1
DEFAULT_THRESHOLD = 0.5


ImageSource = Union[np.ndarray, Image.Image]


def conv_block(x: Any, filters: int) -> Any:
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(x: Any, filters: int) -> tuple[Any, Any]:
    skip = conv_block(x, filters)
    pooled = layers.MaxPooling2D(2)(skip)
    return skip, pooled


def attention_gate(x: Any, g: Any, filters: int) -> Any:
    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    phi_g = layers.Conv2D(filters, 1, padding="same")(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add)
    psi = layers.Conv2D(1, 1, padding="same")(act)
    alpha = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([x, alpha])


def decoder_block(x: Any, skip: Any, filters: int) -> Any:
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    skip_attended = attention_gate(skip, x, filters // 2)
    x = layers.Concatenate()([x, skip_attended])
    x = conv_block(x, filters)
    return x


def build_attention_unet(
    input_shape: tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS),
) -> Model:
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b = conv_block(p4, 1024)

    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation="sigmoid")(d4)
    return Model(inputs, outputs, name="Attention-UNet")


def load_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH) -> Model:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    model = build_attention_unet()
    model.load_weights(str(path))
    return model


def preprocess_image(image: ImageSource) -> np.ndarray:
    if isinstance(image, Image.Image):
        image_array = np.array(image.convert("RGB"))
    else:
        image_array = np.asarray(image)

    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    return np.expand_dims(resized, axis=0)


def predict(model: Model, image: ImageSource) -> np.ndarray:
    prediction = model.predict(preprocess_image(image), verbose=0)
    return np.squeeze(prediction).astype(np.float32)


def mask_from_prediction(prediction: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return prediction >= threshold


def render_result(
    image: Image.Image,
    prediction: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    alpha: float = 0.45,
) -> np.ndarray:
    original = np.array(image.convert("RGB"))
    mask = mask_from_prediction(prediction, threshold).astype(np.uint8)
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = original.copy()
    overlay[mask == 1] = np.array([255, 64, 64], dtype=np.uint8)
    blended = cv2.addWeighted(original, 1.0 - alpha, overlay, alpha, 0)
    return blended


def summary_from_prediction(
    prediction: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    mask = mask_from_prediction(prediction, threshold)
    tumor_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    area_percent = (tumor_pixels / total_pixels) * 100.0 if total_pixels else 0.0

    return {
        "tumor_pixels": tumor_pixels,
        "area_percent": area_percent,
        "max_probability": float(prediction.max()) if prediction.size else 0.0,
        "mean_probability": float(prediction.mean()) if prediction.size else 0.0,
    }
