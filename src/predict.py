import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
import pandas as pd
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP
from torchvision import transforms

def predict_image(image_path, top_k=3, use_iraq=False):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Using Device: {device}")
    
    # Load Model
    model = GeoCLIP().to(device)
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        if isinstance(checkpoint['logit_scale'], torch.Tensor):
            model.logit_scale.data = checkpoint['logit_scale']
        print("[LOG] Loaded trained GeoCLIP model.")
    else:
        print("[WARNING] No trained checkpoint found. Using baseline weights.")
        
    model.eval()

    # Load Global FAISS index
    if use_iraq:
        global_index_path = "data/iraq_index.faiss"
        global_cities_path = "data/iraq_locations.csv"
        print("[LOG] Loading Iraq FAISS index...")
    else:
        global_index_path = "data/global_index.faiss"
        global_cities_path = "data/global_cities.csv"
        print("[LOG] Loading Global FAISS index (170,000+ cities)...")
    
    if not os.path.exists(global_index_path) or not os.path.exists(global_cities_path):
        if use_iraq:
            print("[ERROR] Iraq index not found. Please run 'python src/build_iraq_index.py' first.")
        else:
            print("[ERROR] Global index not found. Please run 'python src/build_global_index.py' first.")
        sys.exit(1)
        
    print("[LOG] Loading Global FAISS index (170,000+ cities)...")
    index = faiss.read_index(global_index_path)
    cities_df = pd.read_csv(global_cities_path, low_memory=False)
    
    city_names = cities_df["City"].values
    country_codes = cities_df["CountryCode"].values
    lats = cities_df["LAT"].values
    lons = cities_df["LON"].values

    from src.inference import InferenceEngine
    image = Image.open(image_path).convert("RGB")
    
    print(f"\n[LOG] Analyzing image: {os.path.basename(image_path)} (with TTA and Micro-Grid)")
    engine = InferenceEngine(model, device, index, cities_df)
    
    # We use A3 variant (TTA + Micro-Grid) for predict.py
    results = engine.predict(image, variant="A3", top_k=top_k)
    
    if len(results) == 0:
        print("[WARNING] No predictions found.")
        return

    print("\n" + "="*40)
    print("FINAL REFINED PREDICTION")
    print("="*40)
    best = results[0]
    final_lat, final_lon = best['lat'], best['lon']
    print(f"Nearest Anchor City: {best['name']}, {best['country']}")
    print(f"Refined Coordinates: ({final_lat:.6f}, {final_lon:.6f})")
    print(f"Confidence (Softmax): {best.get('confidence_prob', 0):.1f}%")
    print(f"Google Maps: https://www.google.com/maps?q={final_lat},{final_lon}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict location of a single image using GeoCLIP")
    parser.add_argument("image", type=str, help="Path to the input image (e.g., my_photo.jpg)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument("--iraq", action="store_true", help="Use the Iraq-specific dataset instead of the global one")
    parser.add_argument("--forensic", action="store_true", help="Enable EXIF metadata extraction")
    args = parser.parse_args()
    
    predict_image(args.image, args.top_k, args.iraq)
