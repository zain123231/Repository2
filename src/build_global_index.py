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

def download_and_extract_geonames(data_dir="data"):
    zip_url = "https://download.geonames.org/export/dump/cities500.zip"
    zip_path = os.path.join(data_dir, "cities500.zip")
    txt_path = os.path.join(data_dir, "cities500.txt")
    csv_path = os.path.join(data_dir, "global_cities.csv")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        if not os.path.exists(txt_path):
            print(f"[LOG] Downloading {zip_url}...")
            urllib.request.urlretrieve(zip_url, zip_path)
            
            print("[LOG] Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("cities500.txt", data_dir)
                
            os.remove(zip_path) # Clean up zip
            
        print("[LOG] Parsing GeoNames TSV data...")
        # GeoNames format: geonameid, name, asciiname, alternatenames, latitude, longitude, ...
        # Columns are 0-indexed. 2: asciiname, 4: latitude, 5: longitude, 8: country code
        cols_to_use = [2, 4, 5, 8]
        col_names = ["City", "LAT", "LON", "CountryCode"]
        
        df = pd.read_csv(txt_path, sep='\t', header=None, usecols=cols_to_use, names=col_names, low_memory=False)
        
        # Drop any NaNs
        df = df.dropna(subset=["LAT", "LON"])
        
        print(f"[LOG] Parsed {len(df)} cities. Saving to {csv_path}...")
        df.to_csv(csv_path, index=False)
    else:
        print(f"[LOG] Found existing {csv_path}. Skipping download.")
        df = pd.read_csv(csv_path)
        
    return df

def build_global_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    df = download_and_extract_geonames()
    
    # --- ADD DENSE GLOBAL GRID ---
    print("[LOG] Generating Dense Global Grid (0.2 degree resolution)...")
    lats_grid = np.arange(-90, 90, 0.2)
    lons_grid = np.arange(-180, 180, 0.2)
    lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)
    
    grid_lats = lat_mesh.flatten()
    grid_lons = lon_mesh.flatten()
    
    # Filter out points too far South (Antarctica) to save some space, or keep them all.
    # Let's keep them all for a true global grid.
    grid_df = pd.DataFrame({
        "City": ["Global Grid Point"] * len(grid_lats),
        "LAT": grid_lats,
        "LON": grid_lons,
        "CountryCode": ["GRID"] * len(grid_lats)
    })
    
    df = pd.concat([df, grid_df], ignore_index=True)
    print(f"[LOG] Total coordinates (Cities + Grid): {len(df)}")
    
    # Save the combined DataFrame
    df.to_csv("data/global_cities.csv", index=False)
    # -----------------------------
    
    # Load Model
    model = GeoCLIP().to(device)
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        if isinstance(checkpoint['logit_scale'], torch.Tensor):
            model.logit_scale.data = checkpoint['logit_scale']
        print("[LOG] Loaded trained GeoCLIP checkpoint.")
    else:
        print("[WARNING] No trained checkpoint found. Using baseline weights.")
        
    model.eval()
    
    lats = df["LAT"].values
    lons = df["LON"].values
    coords = np.stack([lats, lons], axis=1) # (N, 2)
    
    batch_size = 4096
    num_batches = int(np.ceil(len(coords) / batch_size))
    
    print(f"[LOG] Generating embeddings for {len(coords)} cities...")
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing Cities"):
            batch_coords = coords[i * batch_size : (i + 1) * batch_size]
            batch_tensor = torch.tensor(batch_coords, dtype=torch.float32).to(device)
            
            features = model.location_encoder(batch_tensor)
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0) # (N, 512)
    
    print(f"[LOG] Embeddings shape: {all_features.shape}")
    print("[LOG] Building FAISS Index...")
    
    # Use L2 normalized vectors for Inner Product (cosine similarity)
    # Actually GeoCLIP location encoder outputs aren't strictly normalized by default,
    # but L2 index is fine. Let's use IndexFlatL2 as standard.
    index = faiss.IndexFlatL2(512)
    index.add(all_features)
    
    faiss_path = "data/global_index.faiss"
    faiss.write_index(index, faiss_path)
    
    print(f"[LOG] Global FAISS Index saved to {faiss_path}")
    print("[LOG] Done!")

if __name__ == "__main__":
    build_global_index()
