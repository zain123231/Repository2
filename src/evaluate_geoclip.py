import os
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from geoclip import GeoCLIP

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import haversine_distance, calculate_metrics

def evaluate_geoclip(dataset_dir: str, metadata_csv: str, mock: bool = False):
    print(f"[LOG] Loading GeoCLIP model...")
    model = GeoCLIP()
    model.eval()
    
    if mock:
        print("[LOG] MOCK MODE: Running sanity check on random inputs.")
        preds = torch.tensor([[40.7128, -74.0060], [34.0522, -118.2437]])
        targets = torch.tensor([[40.7128, -74.0060], [34.0522, -118.2437]])
        distances = haversine_distance(preds, targets)
        metrics = calculate_metrics(distances)
        print("Mock Metrics:", metrics)
        return metrics

    if not os.path.exists(dataset_dir) or not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Dataset not found! Expected images in {dataset_dir} and metadata in {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    # Expected columns: IMG_ID, LAT, LON
    
    all_preds = []
    all_targets = []
    
    print(f"[LOG] Starting evaluation on {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['IMG_ID']
        lat_true = row['LAT']
        lon_true = row['LON']
        
        # Determine image extension (try jpg, png)
        img_path = os.path.join(dataset_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_dir, f"{img_id}") # maybe it already has extension
            if not os.path.exists(img_path):
                continue
                
        # Predict using GeoCLIP
        with torch.no_grad():
            top_pred_gps, top_pred_prob = model.predict(img_path, top_k=1)
            pred_lat, pred_lon = top_pred_gps[0]
            
        all_preds.append([pred_lat, pred_lon])
        all_targets.append([lat_true, lon_true])
        
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)
    
    print("[LOG] Calculating distances...")
    distances = haversine_distance(preds_tensor, targets_tensor)
    
    print("[LOG] Calculating metrics...")
    metrics = calculate_metrics(distances)
    
    print("="*40)
    print("GeoCLIP Evaluation Results")
    print("="*40)
    print(f"Median Error: {metrics['median_error_km']:.2f} km")
    for tau in [1, 25, 200, 750, 2500]:
        acc = metrics.get(f'acc@{tau}km', 0.0)
        print(f"Acc@{tau}km: {acc*100:.1f}%")
    print("="*40)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoCLIP on Im2GPS3k/YFCC4k")
    parser.add_argument("--data-dir", type=str, default="../data/im2gps3k/images", help="Path to images")
    parser.add_argument("--csv", type=str, default="../data/im2gps3k/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--mock", action="store_true", help="Run a quick mock evaluation")
    args = parser.parse_args()
    
    evaluate_geoclip(args.data_dir, args.csv, args.mock)
