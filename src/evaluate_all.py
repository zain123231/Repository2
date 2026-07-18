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
    
    # 2. We skip the custom global FAISS index as per advisor review
    # The original GeoCLIP model.predict() uses its own coordinates_100K.csv automatically.
    print("[LOG] Using official GeoCLIP coordinates_100K gallery for evaluation.")
    
    # 3. Load Test Data
    test_df = pd.read_csv(csv_file)
    # Postpone ablation variants (A2, A3, A4) until A1 reaches gateway numbers
    variants = ["A1"] 
    
    systems_results = {v: [] for v in variants}
    detailed_csv_rows = []
    
    print(f"[LOG] Evaluating on {len(test_df)} images...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = str(row['IMG_ID'])
        if img_id.endswith('.jpg'):
            img_filename = img_id
        else:
            img_filename = f"{img_id}.jpg"
            
        img_path = os.path.join(img_dir, row.get('IMG_PATH', img_filename))
        if not os.path.exists(img_path):
            continue
            
        try:
            for var in variants:
                if var == "A1":
                    # Use official model.predict() for A1 baseline
                    top_pred_gps, top_pred_prob = model.predict(img_path, top_k=1)
                    pred_lat, pred_lon = top_pred_gps[0]
                    
                    dist = haversine_distance(
                        torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32),
                        torch.tensor([[row['LAT'], row['LON']]], dtype=torch.float32)
                    ).item()
                    
                    systems_results[var].append({'error': dist})
                    
                    detailed_csv_rows.append({
                        'IMG_PATH': os.path.basename(img_path),
                        'TRUE_LAT': row['LAT'],
                        'TRUE_LON': row['LON'],
                        'PRED_LAT': pred_lat.item() if hasattr(pred_lat, 'item') else pred_lat,
                        'PRED_LON': pred_lon.item() if hasattr(pred_lon, 'item') else pred_lon,
                        'ERROR_KM': dist,
                        'CITY': "GeoCLIP 100K Point",
                        'COUNTRY': "Unknown"
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
