import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def plot_cdf(errors_km, output_path="results/cdf_curve.png"):
    if len(errors_km) == 0:
        return
    sorted_errors = np.sort(errors_km)
    p = 1. * np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(sorted_errors, p, marker='', color='b', linewidth=2.5, label='GeoCLIP Zero-Shot')
    
    thresholds = [1, 25, 200, 750, 2500]
    for t in thresholds:
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        
    plt.xscale('symlog')
    plt.xlim(left=0, right=20000)
    plt.ylim(0, 1.05)
    
    plt.xlabel("Localization Error (km)", fontsize=14, fontweight='bold')
    plt.ylabel("Fraction of Images", fontsize=14, fontweight='bold')
    plt.title("Cumulative Distribution of Localization Error", fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[LOG] Saved CDF curve to {output_path}")

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
        
    errors_km = np.array(errors_km)
    if len(errors_km) == 0:
        print("[ERROR] No images were successfully processed!")
        return

    median_error = np.median(errors_km)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS (Official Im2GPS3k)")
    print("="*40)
    print(f"Median Error: {median_error:.2f} km")
    
    thresholds = [1, 25, 200, 750, 2500]
    for t in thresholds:
        acc = np.mean(errors_km <= t) * 100
        print(f"Acc @ {t}km : {acc:.2f}%")
        
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_csv = "results/geoclip_im2gps3k.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[LOG] Saved detailed predictions to {out_csv}")
    
    plot_cdf(errors_km)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/im2gps3k.csv")
    parser.add_argument("--img_dir", type=str, default="data/im2gps3k/images_real")
    args = parser.parse_args()
    
    evaluate(args)
