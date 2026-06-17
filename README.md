# Brain Tumor Segmentation App

A Streamlit application for brain tumor segmentation on MRI images. The app supports two inference backends: a YOLO26l segmentation model for detection-style outputs and an Attention U-Net model for binary tumor mask prediction.

## Features

- Upload MRI images in JPG, JPEG, PNG, or BMP format.
- Choose between `YOLO26l-Seg` and `Attention_Unet` in the sidebar.
- Visualize predicted tumor regions directly on the uploaded image.
- Review model-specific metrics such as detections, mask count, confidence, tumor area, and probability.

## Project Structure

```text
app.py                  Streamlit user interface
utils/predict.py        YOLO segmentation inference utilities
utils/predict_unet.py   Attention U-Net architecture and inference utilities
model/                  Trained model weights
notebook/               Training notebooks
requirements.txt        Python dependencies
```

## Model Weights

Expected model files:

```text
model/yolo26l_brain_tumor_final.pt
model/best_unet_weights_inference.weights.h5
```

The U-Net inference weights are saved without optimizer state to reduce file size. If you use GitHub, track large model files with Git LFS.

## Installation

```bash
pip install -r requirements.txt
```

## Download Models

The model weights are hosted on Hugging Face. You can download them by running the following script:

```bash
python scripts/download_model.py
```

This will download the necessary model files into the `model/` directory.

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser, upload an MRI image, and select the prediction model from the sidebar.

## Notes

This project is intended for research and educational use. The predictions should not be used as a substitute for clinical diagnosis or professional medical review.
