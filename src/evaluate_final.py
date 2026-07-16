import os
import sys
import torch
import faiss
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP
from src.dataset import GeoDataset
from src.loss import pairwise_haversine_distance

def plot_cdf(errors_km, output_path="results/cdf_curve.png"):
    """
    Plots the Cumulative Distribution Function (CDF) of localization errors.
    Saves at 300 DPI for publication quality.
    """
    sorted_errors = np.sort(errors_km)
    p = 1. * np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(sorted_errors, p, marker='', color='b', linewidth=2.5, label='Hybrid System v0.1')
    
    # Plot standard thresholds
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
    
    # Save as PNG (300 dpi) and PDF (Vector)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[LOG] Saved CDF curve to {output_path} (PNG and PDF @ 300 DPI)")

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    # Load Model
    model = GeoCLIP().to(device)
    model.eval()
    
    # Load Dataset
    print(f"[LOG] Loading dataset from {args.csv}")
    dataset = GeoDataset(csv_file=args.csv, img_dir="data/images", mock=args.mock, size=100 if args.mock else None)
    
    # Mocking a FAISS index for evaluation if needed
    print("[LOG] Setting up FAISS index...")
    # For a real run, you would load the global index. Here we generate a subset for evaluation speed.
    ref_coords = []
    for i in range(len(dataset)):
        _, coord = dataset[i]
        ref_coords.append(coord.numpy())
    ref_coords = np.array(ref_coords)
    
    ref_tensor = torch.tensor(ref_coords, dtype=torch.float32).to(device)
    with torch.no_grad():
        ref_features = model.location_encoder(ref_tensor)
        import torch.nn.functional as F
        ref_features = F.normalize(ref_features, dim=1)
    
    nlist = min(16, len(dataset))
    quantizer = faiss.IndexFlatIP(512)
    index = faiss.IndexIVFFlat(quantizer, 512, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(ref_features.cpu().numpy())
    index.add(ref_features.cpu().numpy())
    index.nprobe = min(4, nlist)
    
    errors_km = []
    
    print("[LOG] Running Evaluation Loop...")
    for i in range(len(dataset)):
        image, gt_coord = dataset[i]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feature = model.image_encoder(image)
            img_feature = F.normalize(img_feature, dim=1)
            
        _, I = index.search(img_feature.cpu().numpy(), 1)
        pred_coord = ref_coords[I[0][0]]
        
        # Calculate Haversine distance
        gt_t = torch.tensor(gt_coord.numpy()).unsqueeze(0)
        pred_t = torch.tensor(pred_coord).unsqueeze(0)
        dist = pairwise_haversine_distance(gt_t, pred_t).item()
        
        errors_km.append(dist)
        
    errors_km = np.array(errors_km)
    
    # Metrics calculation
    median_error = np.median(errors_km)
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Median Error: {median_error:.2f} km")
    
    thresholds = [1, 25, 200, 750, 2500]
    for t in thresholds:
        acc = np.mean(errors_km <= t) * 100
        print(f"Acc @ {t}km : {acc:.2f}%")
        
    # Generate Visuals
    plot_cdf(errors_km)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/test.csv")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    args = parser.parse_args()
    
    evaluate(args)
