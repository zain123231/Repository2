import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP
from src.reporting import generate_report

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    # Load Model (Zero-Shot)
    print("[LOG] Initializing GeoCLIP official pretrained model...")
    model = GeoCLIP(from_pretrained=True).to(device)
    model.eval()
    
    print(f"[LOG] Loading dataset from {args.csv}")
    df = pd.read_csv(args.csv)
    
    img_dir = args.img_dir
    results = []
    errors_km = []
    
    print("[LOG] Running Zero-Shot Evaluation Loop...")
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Handle Im2GPS3k csv format which uses IMG_ID or IMG_PATH
        img_filename = row['IMG_ID'] if 'IMG_ID' in row else row['IMG_PATH']
        gt_lat = row['LAT']
        gt_lon = row['LON']
        
        img_path = os.path.join(img_dir, img_filename)
        # Check subfolder if zip extracted weirdly
        if not os.path.exists(img_path):
            img_path_alt = os.path.join(img_dir, "im2gps3ktest", img_filename)
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                continue
            
        # Zero-shot prediction using the official independent gallery
        try:
            top_pred_gps, top_pred_prob = model.predict(img_path, top_k=1)
            pred_lat, pred_lon = top_pred_gps[0].cpu().numpy()
        except Exception as e:
            print(f"\n[ERROR] Failed to predict for {img_path}: {e}")
            continue
            
        dist = haversine_distance(gt_lat, gt_lon, pred_lat, pred_lon)
        errors_km.append(dist)
        
        results.append({
            "IMG_ID": img_filename,
            "GT_LAT": gt_lat,
            "GT_LON": gt_lon,
            "PRED_LAT": pred_lat,
            "PRED_LON": pred_lon,
            "ERROR_KM": dist
        })
        
    end_time = time.time()
    total_runtime_sec = end_time - start_time
    
    errors_km = np.array(errors_km)
    if len(errors_km) == 0:
        print("[ERROR] No images were successfully processed!")
        return

    # Trigger automated scientific reporting
    generate_report(errors_km, results, total_runtime_sec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/im2gps3k.csv")
    parser.add_argument("--img_dir", type=str, default="data/im2gps3k/images_real")
    args = parser.parse_args()
    
    evaluate(args)
