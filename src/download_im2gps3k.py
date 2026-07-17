import os
import urllib.request
import zipfile

def validate_dataset(dataset_dir: str, expected_min_images: int = 2990) -> bool:
    """Validates if the dataset exists and contains the expected number of images."""
    if not os.path.exists(dataset_dir):
        return False
    
    # Check if we have enough images in the directory
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_count = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_count += 1
                
    return image_count >= expected_min_images

def download_and_extract():
    os.makedirs("data/im2gps3k", exist_ok=True)
    zip_path = "data/im2gps3k/im2gps3ktest.zip"
    url = "https://huggingface.co/datasets/Wendy-Fly/AAAI-2026/resolve/main/im2gps3ktest.zip"
    target_dir = "data/im2gps3k/images_real"
    dataset_content_dir = os.path.join(target_dir, "im2gps3ktest")

    print("[LOG] Checking Im2GPS3k dataset cache...")
    
    # Validation step
    if validate_dataset(target_dir):
        print(f"[LOG] Dataset found in '{target_dir}' and validated successfully. Skipping download.")
        return
        
    print(f"[LOG] Dataset not found or corrupted in '{target_dir}'. Proceeding with download...")

    if not os.path.exists(zip_path):
        print(f"[LOG] Downloading Im2GPS3k from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("[LOG] Download complete.")
    else:
        print(f"[LOG] Found existing zip file at '{zip_path}'. Skipping download.")
        
    print("[LOG] Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("[LOG] Extraction complete. Dataset ready.")

if __name__ == "__main__":
    download_and_extract()
