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

    # Process Input Image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    
    print(f"\n[LOG] Analyzing image: {os.path.basename(image_path)}")
    with torch.no_grad():
        img_feature = model.image_encoder(image_tensor)
        
    # Search Top-K
    D, I = index.search(img_feature.cpu().numpy(), top_k)
    
    print("\n" + "="*40)
    print(f"TOP {top_k} PREDICTIONS FOR THIS IMAGE")
    print("="*40)
    
    for rank in range(top_k):
        idx = I[0][rank]
        city = city_names[idx]
        country = country_codes[idx]
        lat = lats[idx]
        lon = lons[idx]
        print(f"#{rank+1}: {city}, {country}")
        print(f"   Coordinates: ({lat:.4f}, {lon:.4f})")
        print(f"   Google Maps: https://www.google.com/maps?q={lat},{lon}")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict location of a single image using GeoCLIP")
    parser.add_argument("image", type=str, help="Path to the input image (e.g., my_photo.jpg)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument("--iraq", action="store_true", help="Use the Iraq-specific dataset instead of the global one")
    args = parser.parse_args()
    
    predict_image(args.image, args.top_k, args.iraq)
