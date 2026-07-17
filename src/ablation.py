import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "geo-clip"))
from geoclip.model.GeoCLIP import GeoCLIP
from src.metrics import calculate_metrics, haversine_distance
from src.inference import InferenceEngine
import faiss

def evaluate_variants(model, device="cpu"):
    # Load real dataset
    print("[LOG] Loading Global FAISS Index and cities...")
    cities_df = pd.read_csv("data/global_cities.csv", low_memory=False)
    global_index_path = "data/global_index.faiss"
    if os.path.exists(global_index_path):
        index = faiss.read_index(global_index_path)
    else:
        print("[ERROR] FAISS index not found. Run build_global_index.py first.")
        sys.exit(1)
    
    engine = InferenceEngine(model, device, index, cities_df)
    
    import easyocr
    engine.ocr_reader = easyocr.Reader(['en', 'ar'], gpu=torch.cuda.is_available())
    
    print("[LOG] Loading Test Dataset...")

    print("[LOG] Loading Test Dataset...")
    if os.path.exists("data/im2gps3k_test.csv"):
        test_df = pd.read_csv("data/im2gps3k_test.csv")
    else:
        test_df = pd.read_csv("data/im2gps3k.csv")
    img_dir = "data/im2gps3k/im2gps3ktest/im2gps3ktest"
    
    variants = ["A1", "A2", "A3", "A4"]
    systems_results = {v: [] for v in variants}
    
    for idx, row in test_df.iterrows():
        img_id = str(row.get('IMG_ID', ''))
        img_path = os.path.join(img_dir, str(row.get('IMG_PATH', f"{img_id}.jpg")))
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id)
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            target = torch.tensor([[row['LAT'], row['LON']]], dtype=torch.float32)
            
            for var in variants:
                preds = engine.predict(img, variant=var, top_k=1)
                if preds:
                    pred_lat = preds[0]['lat']
                    pred_lon = preds[0]['lon']
                    pred_tensor = torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32)
                    dist = haversine_distance(pred_tensor, target).item()
                    systems_results[var].append({'error': dist})
        except Exception as e:
            pass
            
    from src.reporting import generate_report
    generate_report(systems_results, len(test_df), dataset_name="Im2GPS3k")
    
    return systems_results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    model = GeoCLIP(from_pretrained=True).to(device)
    checkpoint_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        if isinstance(checkpoint['logit_scale'], torch.Tensor):
            model.logit_scale.data = checkpoint['logit_scale']
        print("[LOG] Loaded trained checkpoint.")
    else:
        print("[WARNING] Checkpoint not found, falling back to pretrained baseline.")
        
    model.eval()
    
    evaluate_variants(model, device)

