import os
import urllib.request
import zipfile

def download_and_extract():
    os.makedirs("data/im2gps3k", exist_ok=True)
    zip_path = "data/im2gps3k/im2gps3ktest.zip"
    url = "https://huggingface.co/datasets/Wendy-Fly/AAAI-2026/resolve/main/im2gps3ktest.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading Im2GPS3k from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/im2gps3k/images_real")
    print("Extraction complete. Dataset ready.")

if __name__ == "__main__":
    download_and_extract()
