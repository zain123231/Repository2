import os
import argparse
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import calculate_metrics, haversine_distance

def extract_clip_features(img_paths, model, preprocess, device, batch_size=32):
    features = []
    model.eval()
    
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch_tensors = []
        valid_indices = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(preprocess(img))
                valid_indices.append(j)
            except:
                pass
                
        if not batch_tensors:
            # All failed, insert zeros
            features.extend([np.zeros(768)] * len(batch_paths))
            continue
            
        with torch.no_grad():
            img_tensor = torch.stack(batch_tensors).to(device)
            # Use OpenAI CLIP from HuggingFace
            inputs = {"pixel_values": img_tensor}
            img_emb = model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            emb_np = img_emb.cpu().numpy()
            
        # Reconstruct with failures
        batch_feats = []
        v_idx = 0
        for j in range(len(batch_paths)):
            if j in valid_indices:
                batch_feats.append(emb_np[v_idx])
                v_idx += 1
            else:
                batch_feats.append(np.zeros(768))
        features.extend(batch_feats)
        
    return np.array(features, dtype=np.float32)

def evaluate_image_retrieval(query_csv, query_dir, ref_csv, ref_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LOG] Device: {device}")
    
    print("[LOG] Loading CLIP model for Image-to-Image retrieval...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Custom preprocess to match our transforms if needed, or use processor
    def preprocess(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    
    print("[LOG] Loading Reference Gallery...")
    ref_df = pd.read_csv(ref_csv)
    ref_paths = [os.path.join(ref_dir, str(row.get('IMG_PATH', row.get('IMG_ID', '')))) for _, row in ref_df.iterrows()]
    # Fallback to appending .jpg if not there
    ref_paths = [p if p.endswith('.jpg') else p + '.jpg' for p in ref_paths]
    
    print(f"[LOG] Extracting features for {len(ref_paths)} reference images...")
    ref_features = extract_clip_features(ref_paths, model, preprocess, device)
    
    print("[LOG] Building FAISS Index...")
    index = faiss.IndexFlatL2(768)
    index.add(ref_features)
    
    print("[LOG] Loading Query Images...")
    query_df = pd.read_csv(query_csv)
    query_paths = [os.path.join(query_dir, str(row.get('IMG_PATH', row.get('IMG_ID', '')))) for _, row in query_df.iterrows()]
    query_paths = [p if p.endswith('.jpg') else p + '.jpg' for p in query_paths]
    query_lats = query_df['LAT'].values
    query_lons = query_df['LON'].values
    
    print(f"[LOG] Evaluating {len(query_paths)} queries...")
    query_features = extract_clip_features(query_paths, model, preprocess, device)
    
    D, I = index.search(query_features, 1)
    
    distances_km = []
    for i in range(len(query_paths)):
        if np.sum(np.abs(query_features[i])) == 0:
            continue # Skipped failed image
            
        top_idx = I[i][0]
        pred_lat = ref_df.iloc[top_idx]['LAT']
        pred_lon = ref_df.iloc[top_idx]['LON']
        
        gt_lat = query_lats[i]
        gt_lon = query_lons[i]
        
        dist = haversine_distance(
            torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32),
            torch.tensor([[gt_lat, gt_lon]], dtype=torch.float32)
        ).item()
        distances_km.append(dist)
        
    if distances_km:
        metrics = calculate_metrics(torch.tensor(distances_km))
        print(f"\n================ IMAGE RETRIEVAL BASELINE ================")
        print(f"Evaluated Images: {len(distances_km)}")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")
    else:
        print("[WARNING] No valid distances computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image-to-Image Retrieval Baseline")
    parser.add_argument("--query-csv", type=str, default="data/im2gps3k_test.csv")
    parser.add_argument("--query-dir", type=str, default="data/im2gps3k/images")
    parser.add_argument("--ref-csv", type=str, default="data/mp16_train.csv")
    parser.add_argument("--ref-dir", type=str, default="data/mp16/images")
    args = parser.parse_args()
    
    evaluate_image_retrieval(args.query_csv, args.query_dir, args.ref_csv, args.ref_dir)
