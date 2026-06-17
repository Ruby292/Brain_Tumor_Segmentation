from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "yolo26l_brain_tumor_final.pt"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
CLASS_NAMES = {0: "tumor"}


ImageSource = Union[str, Path, np.ndarray, Image.Image]


def select_device() -> Union[int, str]:
    """Match the notebook's GPU-first behavior, with CPU fallback."""
    if not torch.cuda.is_available():
        return "cpu"

    for index in range(torch.cuda.device_count()):
        try:
            sample = torch.tensor([1.0]).to(f"cuda:{index}")
            _ = sample * 2
            return index
        except Exception:
            continue

    return "cpu"


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> YOLO:
    """Load the trained YOLO26L-Seg model from the local .pt checkpoint."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    original_torch_load = torch.load

    def torch_load_trusted_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_trusted_checkpoint
    try:
        model = YOLO(str(path))
    finally:
        torch.load = original_torch_load

    try:
        model.names = CLASS_NAMES
    except Exception:
        try:
            model.model.names = CLASS_NAMES
        except Exception:
            pass

    return model


def predict(
    model: YOLO,
    source: ImageSource,
    conf: float = DEFAULT_CONFIDENCE,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    device: Optional[Union[int, str]] = None,
) -> Any:
    """Run YOLO segmentation inference with the same defaults as the notebook."""
    selected_device = select_device() if device is None else device
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        device=selected_device,
        verbose=False,
    )

    return results[0] if results else None


def render_result(result: Any) -> np.ndarray:
    """Return the annotated prediction image in RGB format for display."""
    plotted_bgr = result.plot()
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


def detections_from_result(result: Any) -> List[Dict[str, Any]]:
    """Convert Ultralytics boxes to simple dictionaries for UI/API use."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    names = getattr(result, "names", None) or CLASS_NAMES

    detections: List[Dict[str, Any]] = []
    for box, confidence, class_id in zip(xyxy, confidences, classes):
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(class_id)),
                "confidence": float(confidence),
                "box_xyxy": [float(value) for value in box],
            }
        )

    return detections


def summary_from_result(result: Any) -> Dict[str, Any]:
    """Build a compact prediction summary."""
    detections = detections_from_result(result)
    mask_count = 0

    if result is not None and result.masks is not None:
        mask_count = int(result.masks.data.shape[0])

    top_confidence = max((item["confidence"] for item in detections), default=0.0)

    return {
        "detections": len(detections),
        "masks": mask_count,
        "top_confidence": top_confidence,
        "items": detections,
    }
