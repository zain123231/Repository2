import os
import sys
import zipfile
import urllib.request
import pandas as pd
import torch
import numpy as np
import faiss
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP

def download_and_extract_iraq(data_dir="data"):
    zip_url = "https://download.geonames.org/export/dump/IQ.zip"
    zip_path = os.path.join(data_dir, "IQ.zip")
    txt_path = os.path.join(data_dir, "IQ.txt")
    csv_path = os.path.join(data_dir, "iraq_locations.csv")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        if not os.path.exists(txt_path):
            print(f"[LOG] Downloading {zip_url}...")
            urllib.request.urlretrieve(zip_url, zip_path)
            
            print("[LOG] Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("IQ.txt", data_dir)
                
            os.remove(zip_path) # Clean up zip
            
        print("[LOG] Parsing GeoNames TSV data for Iraq...")
        cols_to_use = [2, 4, 5, 8]
        col_names = ["City", "LAT", "LON", "CountryCode"]
        
        df = pd.read_csv(txt_path, sep='\t', header=None, usecols=cols_to_use, names=col_names, low_memory=False)
        df = df.dropna(subset=["LAT", "LON"])
        
        print(f"[LOG] Parsed {len(df)} locations in Iraq. Saving to {csv_path}...")
        df.to_csv(csv_path, index=False)
    else:
        print(f"[LOG] Found existing {csv_path}. Skipping download.")
        df = pd.read_csv(csv_path)
        
    return df

def build_iraq_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    df = download_and_extract_iraq()
    
    # Load Model
    model = GeoCLIP().to(device)
    model.eval()
    print("[LOG] Loaded baseline GeoCLIP weights.")
        
    lats = df["LAT"].values
    lons = df["LON"].values
    coords = np.stack([lats, lons], axis=1) # (N, 2)
    
    batch_size = 4096
    num_batches = int(np.ceil(len(coords) / batch_size))
    
    print(f"[LOG] Generating embeddings for {len(coords)} locations in Iraq...")
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing Iraq Locations"):
            batch_coords = coords[i * batch_size : (i + 1) * batch_size]
            batch_tensor = torch.tensor(batch_coords, dtype=torch.float32).to(device)
            
            features = model.location_encoder(batch_tensor)
            features = torch.nn.functional.normalize(features, dim=1)
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"[LOG] Embeddings shape: {all_features.shape}")
    print("[LOG] Building FAISS Index...")
    
    # Use Hybrid System v0.1: Coarse Classification (Quantization) -> Fine Retrieval
    nlist = 256  # Number of coarse cells for Iraq (smaller dataset)
    quantizer = faiss.IndexFlatIP(512)
    index = faiss.IndexIVFFlat(quantizer, 512, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print(f"[LOG] Training Coarse Classifier (nlist={nlist}) on {len(all_features)} locations...")
    index.train(all_features)
    print("[LOG] Distributing locations into coarse cells...")
    index.add(all_features)
    
    faiss_path = "data/iraq_index.faiss"
    faiss.write_index(index, faiss_path)
    
    print(f"[LOG] Iraq FAISS Index saved to {faiss_path}")
    print("[LOG] Done!")

if __name__ == "__main__":
    build_iraq_index()
