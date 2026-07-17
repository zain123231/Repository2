import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import faiss
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geoclip import GeoCLIP
from src.metrics import haversine_distance
from src.inference import InferenceEngine
from src.reporting import generate_report

def evaluate_all(csv_file, img_dir, global_cities_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    # 1. Load Model
    model = GeoCLIP(from_pretrained=True).to(device)
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        print("[LOG] Loaded trained GeoCLIP model.")
    else:
        print("[WARNING] No trained checkpoint found. Using baseline weights.")
    model.eval()
    
    # 2. Build FAISS Index for the Gallery
    print("[LOG] Loading Global Cities...")
    cities_df = pd.read_csv(global_cities_path, low_memory=False)
    
    coords = np.stack([cities_df['LAT'].values, cities_df['LON'].values], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    
    print("[LOG] Building FAISS Index...")
    batch_size = 10000
    features_list = []
    with torch.no_grad():
        for i in range(0, len(coords_tensor), batch_size):
            batch = coords_tensor[i:i+batch_size].to(device)
            feats = model.location_encoder(batch)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            features_list.append(feats.cpu().numpy())
            
    all_features = np.vstack(features_list)
    index = faiss.IndexFlatL2(512)
    index.add(all_features)
    
    # Load OCR Reader for A4
    import easyocr
    print("[LOG] Loading OCR Reader...")
    ocr_reader = easyocr.Reader(['en', 'ar'], gpu=torch.cuda.is_available())
    
    engine = InferenceEngine(model, device, index, cities_df, ocr_reader)
    
    # 3. Load Test Data
    test_df = pd.read_csv(csv_file)
    variants = ["A1", "A2", "A3", "A4"]
    
    systems_results = {v: [] for v in variants}
    detailed_csv_rows = []
    
    print(f"[LOG] Evaluating on {len(test_df)} images...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = os.path.join(img_dir, row['IMG_PATH']) if 'IMG_PATH' in row else os.path.join(img_dir, f"{row['IMG_ID']}.jpg")
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            
            for var in variants:
                preds = engine.predict(image, variant=var, top_k=1)
                if preds:
                    pred_lat = preds[0]['lat']
                    pred_lon = preds[0]['lon']
                    dist = haversine_distance(
                        torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32),
                        torch.tensor([[row['LAT'], row['LON']]], dtype=torch.float32)
                    ).item()
                    
                    systems_results[var].append({'error': dist})
                    
                    if var == "A4": # Save A4 as the detailed prediction
                        detailed_csv_rows.append({
                            'IMG_PATH': os.path.basename(img_path),
                            'TRUE_LAT': row['LAT'],
                            'TRUE_LON': row['LON'],
                            'PRED_LAT': pred_lat,
                            'PRED_LON': pred_lon,
                            'ERROR_KM': dist,
                            'CITY': preds[0]['name'],
                            'COUNTRY': preds[0]['country']
                        })
        except Exception as e:
            print(f"[ERROR] Failed {img_path}: {e}")
            
    print("[LOG] Generating Report...")
    res_dir = generate_report(systems_results, len(test_df), dataset_name="Im2GPS3k")
    
    # Save detailed CSV
    detailed_df = pd.DataFrame(detailed_csv_rows)
    detailed_csv_path = os.path.join(res_dir, "csv", "detailed_predictions.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"[LOG] Detailed predictions saved to {detailed_csv_path}")
    
    # Run spatial analysis
    print("[LOG] Running Spatial Analysis...")
    import subprocess
    subprocess.run([sys.executable, "src/spatial_analysis.py", "--csv", detailed_csv_path, "--outdir", res_dir])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/im2gps3k_test.csv")
    parser.add_argument("--img-dir", type=str, default="data/im2gps3k/images")
    parser.add_argument("--cities", type=str, default="data/global_cities.csv")
    args = parser.parse_args()
    
    # Fix seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    import random
    random.seed(42)
    
    evaluate_all(args.csv, args.img_dir, args.cities)
