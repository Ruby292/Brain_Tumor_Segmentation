import os
import shutil
from huggingface_hub import hf_hub_download

def download_models():
    """
    Downloads the model files from the Hugging Face Hub and places them directly in the 'model' folder.
    """
    # Ensure the target directory exists
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    repo_id = "OSAS-AI-Lab/brain-tumor-segmentation"
    files_to_download = [
        "best_unet_weights_inference.weights.h5",
        "yolo26l_brain_tumor_final.pt"
    ]

    for file_name in files_to_download:
        print(f"Downloading {file_name}...")
        # Download the file and get its cached path
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            local_dir_use_symlinks=False
        )
        
        # Construct the destination path
        destination_path = os.path.join(model_dir, file_name)
        
        # Move the file to the desired directory
        shutil.move(downloaded_file_path, destination_path)
        
        print(f"Moved {file_name} to {destination_path}")

if __name__ == "__main__":
    download_models()
