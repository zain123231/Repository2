import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "geo-clip"))
from geoclip.model.GeoCLIP import GeoCLIP
from src.metrics import calculate_metrics, haversine_distance

def extract_features(model, images, processor, texts=None, device="cpu"):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    embeddings = []
    with torch.no_grad():
        for i, img in enumerate(images):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            img_emb = model.image_encoder(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            if texts is not None:
                text_inputs = processor(text=[texts[i]], return_tensors="pt", padding=True).to(device)
                text_emb = model.image_encoder.CLIP.get_text_features(**text_inputs)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                
                fused_emb = torch.cat([img_emb, text_emb], dim=-1)
                fused_emb = fused_emb / fused_emb.norm(dim=-1, keepdim=True)
                embeddings.append(fused_emb)
            else:
                embeddings.append(img_emb)
                
    return torch.cat(embeddings, dim=0)

def evaluate_model_in_memory(model, processor, device="cpu"):
    model.eval()
    
    # 1. Generate Mock Data (Refs and Queries)
    np.random.seed(42) # For fair comparison
    ref_images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(100)]
    ref_lats = np.random.uniform(-90, 90, 100)
    ref_lons = np.random.uniform(-180, 180, 100)
    
    cities_df = pd.read_csv("data/mock_cities.csv")
    countries = cities_df['country'].unique().tolist()
    ref_texts = [f"A photo taken in {np.random.choice(countries)}" for _ in range(100)]
    
    query_images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(50)]
    query_lats = np.random.uniform(-90, 90, 50)
    query_lons = np.random.uniform(-180, 180, 50)
    
    # 2. Extract Reference Features (Fused)
    ref_feats = extract_features(model, ref_images, processor, ref_texts, device) # (100, 1280)
    
    # 3. Extract Query Features (Fused via Zero-Shot)
    country_prompts = [f"A photo taken in {c}" for c in countries]
    with torch.no_grad():
        country_inputs = processor(text=country_prompts, return_tensors="pt", padding=True).to(device)
        country_text_features = model.image_encoder.CLIP.get_text_features(**country_inputs)
        country_text_features = country_text_features / country_text_features.norm(dim=-1, keepdim=True)
    
    query_fused_feats = []
    with torch.no_grad():
        for img in query_images:
            q_img_emb = extract_features(model, [img], processor, None, device) # (1, 512)
            
            # Zero-Shot Classification
            clip_img_input = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            clip_img_emb = model.image_encoder.CLIP.get_image_features(clip_img_input)
            clip_img_emb = clip_img_emb / clip_img_emb.norm(dim=-1, keepdim=True)
            
            similarities = (clip_img_emb @ country_text_features.t()).squeeze(0)
            best_idx = similarities.argmax().item()
            
            q_text_emb = country_text_features[best_idx].unsqueeze(0)
            
            q_fused = torch.cat([q_img_emb, q_text_emb], dim=-1)
            q_fused = q_fused / q_fused.norm(dim=-1, keepdim=True)
            query_fused_feats.append(q_fused)
            
    query_feats = torch.cat(query_fused_feats, dim=0) # (50, 1280)
    
    # 4. Compute Similarities and Top-1
    similarities = query_feats @ ref_feats.t() # (50, 100)
    top1_indices = similarities.argmax(dim=-1) # (50,)
    
    # 5. Compute Metrics
    pred_lats = torch.tensor(ref_lats[top1_indices.cpu().numpy()])
    pred_lons = torch.tensor(ref_lons[top1_indices.cpu().numpy()])
    preds = torch.stack([pred_lats, pred_lons], dim=1)
    
    targets = torch.tensor(np.stack([query_lats, query_lons], axis=1), dtype=torch.float32)
    preds = preds.type(torch.float32)
    
    distances_km = haversine_distance(preds, targets)
    metrics = calculate_metrics(distances_km)
    
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=False)
    
    print("\n--- BASELINE MODEL ---")
    baseline_model = GeoCLIP(from_pretrained=True).to(device)
    baseline_metrics = evaluate_model_in_memory(baseline_model, processor, device)
    
    print("\n--- TRAINED MODEL (With Geographic Loss) ---")
    trained_model = GeoCLIP(from_pretrained=True).to(device)
    # Load weights
    checkpoint_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        trained_model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        trained_model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        if isinstance(checkpoint['logit_scale'], torch.Tensor):
            trained_model.logit_scale.data = checkpoint['logit_scale']
        print("[LOG] Loaded trained checkpoint.")
    else:
        print("[WARNING] Checkpoint not found, falling back to pretrained.")
    
    trained_metrics = evaluate_model_in_memory(trained_model, processor, device)
    
    print("\n================ ABLATION RESULTS ================")
    print(f"{'Metric':<20} | {'Baseline':<15} | {'Trained':<15}")
    print("-" * 55)
    for k in baseline_metrics.keys():
        b_val = baseline_metrics[k]
        t_val = trained_metrics[k]
        print(f"{k:<20} | {b_val:<15.4f} | {t_val:<15.4f}")
