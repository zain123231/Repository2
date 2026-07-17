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
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geoclip import GeoCLIP
from src.metrics import haversine_distance
from src.inference import InferenceEngine
from src.reporting import generate_report

def plot_coverage_map(df, name, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Subsample for plotting if too large
    if len(df) > 100000:
        plot_df = df.sample(100000, random_state=42)
    else:
        plot_df = df
        
    ax.scatter(plot_df['LON'], plot_df['LAT'], s=0.1, color='blue', alpha=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    map_path = os.path.join(out_dir, f"coverage_{name}.png")
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[LOG] Saved coverage map for {name} to {map_path}")

def evaluate_galleries(csv_file, img_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    model = GeoCLIP(from_pretrained=True).to(device)
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        print("[LOG] Loaded trained GeoCLIP model.")
    model.eval()
    
    galleries = {
        'G1_GeoNames': ('data/galleries/G1_raw.csv', 'data/galleries/G1_raw.faiss'),
        'G2_PopWeighted': ('data/galleries/G2_pop.csv', 'data/galleries/G2_pop.faiss'),
        'G3_Grid_Land': ('data/galleries/G3_grid.csv', 'data/galleries/G3_grid.faiss'),
        'G4_Hybrid': ('data/galleries/G4_hybrid.csv', 'data/galleries/G4_hybrid.faiss')
    }
    
    test_df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.read_csv("data/im2gps3k.csv")
    
    systems_results = {}
    os.makedirs("results/gallery_maps", exist_ok=True)
    
    for gal_name, (csv_path, faiss_path) in galleries.items():
        if not os.path.exists(csv_path) or not os.path.exists(faiss_path):
            print(f"[WARNING] Skipping {gal_name} as files are missing. Run src/build_galleries.py first.")
            continue
            
        print(f"\n[LOG] Evaluating Gallery: {gal_name}")
        df = pd.read_csv(csv_path, low_memory=False)
        index = faiss.read_index(faiss_path)
        
        plot_coverage_map(df, gal_name, "results/gallery_maps")
        
        engine = InferenceEngine(model, device, index, df)
        
        systems_results[gal_name] = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_id = str(row.get('IMG_ID', ''))
            img_path = os.path.join(img_dir, row.get('IMG_PATH', f"{img_id}.jpg" if img_id else ""))
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, img_id)
                if not os.path.exists(img_path):
                    continue
                    
            try:
                image = Image.open(img_path).convert("RGB")
                preds = engine.predict(image, variant="A1", top_k=1)
                if preds:
                    pred_lat = preds[0]['lat']
                    pred_lon = preds[0]['lon']
                    dist = haversine_distance(
                        torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32),
                        torch.tensor([[row['LAT'], row['LON']]], dtype=torch.float32)
                    ).item()
                    systems_results[gal_name].append({'error': dist})
            except Exception as e:
                pass
                
    if systems_results:
        print("\n[LOG] Generating comparative report for galleries...")
        generate_report(systems_results, len(test_df), dataset_name="Galleries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/im2gps3k_test.csv")
    parser.add_argument("--img-dir", type=str, default="data/im2gps3k/images")
    args = parser.parse_args()
    
    # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    evaluate_galleries(args.csv, args.img_dir)
