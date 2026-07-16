import os
import argparse
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geoclip import GeoCLIP
from src.metrics import calculate_metrics, haversine_distance

def evaluate_retrieval(query_dir: str, query_meta: str, index_path: str, index_meta: str, mock: bool = False):
    print(f"[LOG] Loading FAISS index from {index_path}...")
    if not os.path.exists(index_path) or not os.path.exists(index_meta):
        print(f"[ERROR] FAISS index or metadata not found. Run build_index.py first.")
        return
        
    index = faiss.read_index(index_path)
    ref_df = pd.read_csv(index_meta)
    
    print("[LOG] Loading GeoCLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCLIP().to(device)
    model.eval()
    
    queries = []
    true_lats = []
    true_lons = []
    
    if mock:
        print("[LOG] MOCK MODE: Generating random query images.")
        os.makedirs(query_dir, exist_ok=True)
        for i in range(50):
            img_path = os.path.join(query_dir, f"mock_query_{i}.jpg")
            if not os.path.exists(img_path):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(img_path)
            queries.append(img_path)
            true_lats.append(np.random.uniform(-90, 90))
            true_lons.append(np.random.uniform(-180, 180))
    else:
        print(f"[LOG] Loading query metadata from {query_meta}...")
        df = pd.read_csv(query_meta)
        for _, row in df.iterrows():
            img_path = os.path.join(query_dir, f"{row['IMG_ID']}.jpg")
            if os.path.exists(img_path):
                queries.append(img_path)
                true_lats.append(row['LAT'])
                true_lons.append(row['LON'])
                
    print(f"[LOG] Evaluating {len(queries)} query images...")
    
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    from transformers import AutoProcessor
    import warnings
    warnings.filterwarnings("ignore")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=False)
    
    # Load countries for Zero-Shot Prediction
    cities_df = pd.read_csv("data/mock_cities.csv")
    countries = cities_df['country'].unique().tolist()
    country_prompts = [f"A photo taken in {c}" for c in countries]
    print(f"[LOG] Preparing {len(countries)} country prompts for Zero-Shot fusion...")
    
    with torch.no_grad():
        country_inputs = processor(text=country_prompts, return_tensors="pt", padding=True).to(device)
        country_text_features = model.image_encoder.CLIP.get_text_features(**country_inputs)
        country_text_features = country_text_features / country_text_features.norm(dim=-1, keepdim=True)
    
    distances_km = []
    
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(queries)):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                
                # 1. GeoCLIP Image Feature (512 dim)
                img_emb = model.image_encoder(img_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                
                # 2. CLIP Image Feature for Zero-Shot Classification (768 dim)
                clip_img_input = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
                clip_img_emb = model.image_encoder.CLIP.get_image_features(clip_img_input)
                clip_img_emb = clip_img_emb / clip_img_emb.norm(dim=-1, keepdim=True)
                
                # Zero-Shot Predict Country
                similarities = (clip_img_emb @ country_text_features.t()).squeeze(0)
                best_country_idx = similarities.argmax().item()
                
                # 3. Selected Text Feature (768 dim)
                text_emb = country_text_features[best_country_idx].unsqueeze(0)
                
                # 4. Fusion (512 + 768 = 1280 dim)
                fused_emb = torch.cat([img_emb, text_emb], dim=-1)
                fused_emb = fused_emb / fused_emb.norm(dim=-1, keepdim=True)
                
                fused_np = fused_emb.cpu().numpy().astype('float32')
                
                # FAISS search
                k = 1 # Top-1 retrieval
                D, I = index.search(fused_np, k)
                
                # Retrieve coordinate from reference metadata
                top_idx = I[0][0]
                pred_lat = ref_df.iloc[top_idx]['LAT']
                pred_lon = ref_df.iloc[top_idx]['LON']
                
                # Calculate distance
                gt_lat = true_lats[i]
                gt_lon = true_lons[i]
                dist = haversine_distance(
                    torch.tensor([[pred_lat, pred_lon]]),
                    torch.tensor([[gt_lat, gt_lon]])
                )
                distances_km.append(dist.item())
                
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
                
    if distances_km:
        metrics = calculate_metrics(torch.tensor(distances_km))
        print(f"\nRetrieval Metrics (FAISS Top-1): {metrics}")
    else:
        print("[WARNING] No valid distances computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval baseline")
    parser.add_argument("--query-dir", type=str, default="data/im2gps3k/images", help="Path to query images")
    parser.add_argument("--query-meta", type=str, default="data/im2gps3k/metadata.csv", help="Path to query metadata")
    parser.add_argument("--index-path", type=str, default="data/reference.faiss", help="Path to FAISS index")
    parser.add_argument("--index-meta", type=str, default="data/reference_meta.csv", help="Path to index metadata")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    
    args = parser.parse_args()
    evaluate_retrieval(args.query_dir, args.query_meta, args.index_path, args.index_meta, args.mock)
