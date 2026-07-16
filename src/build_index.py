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

def build_index(data_dir: str, metadata_csv: str, out_index: str, out_meta: str, mock: bool = False):
    print("[LOG] Loading GeoCLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCLIP().to(device)
    model.eval()
    
    images = []
    lats = []
    lons = []
    texts = []
    
    if mock:
        print("[LOG] MOCK MODE: Generating random reference images.")
        os.makedirs(data_dir, exist_ok=True)
        # Load mock cities to assign random countries
        cities_df = pd.read_csv("data/mock_cities.csv")
        countries = cities_df['country'].unique().tolist()
        
        for i in range(100):
            img_path = os.path.join(data_dir, f"mock_ref_{i}.jpg")
            if not os.path.exists(img_path):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(img_path)
            images.append(img_path)
            lats.append(np.random.uniform(-90, 90))
            lons.append(np.random.uniform(-180, 180))
            texts.append(f"A photo taken in {np.random.choice(countries)}")
    else:
        print(f"[LOG] Loading reference metadata from {metadata_csv}...")
        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            # Assuming CSV has 'IMG_ID', 'LAT', 'LON'
            img_path = os.path.join(data_dir, f"{row['IMG_ID']}.jpg")
            if os.path.exists(img_path):
                images.append(img_path)
                lats.append(row['LAT'])
                lons.append(row['LON'])
            else:
                print(f"[WARNING] Image not found: {img_path}")

    print(f"[LOG] Extracting fused embeddings (Image + Text) for {len(images)} reference images...")
    embeddings = []
    
    from transformers import AutoProcessor
    import warnings
    warnings.filterwarnings("ignore")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=False)
    
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(images)):
            try:
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                         std=[0.26862954, 0.26130258, 0.27577711])
                ])
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                
                # 1. Image Features (GeoCLIP - 512 dim)
                img_emb = model.image_encoder(img_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                
                # 2. Text Features (CLIP - 768 dim)
                text_inputs = processor(text=[texts[i]], return_tensors="pt", padding=True).to(device)
                text_emb = model.image_encoder.CLIP.get_text_features(**text_inputs)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                
                # 3. Fusion (Concatenation -> 1280 dim)
                fused_emb = torch.cat([img_emb, text_emb], dim=-1)
                fused_emb = fused_emb / fused_emb.norm(dim=-1, keepdim=True)
                
                embeddings.append(fused_emb.cpu().numpy().astype('float32'))
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
                
    if not embeddings:
        print("[ERROR] No embeddings extracted. Exiting.")
        return
        
    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]
    
    print(f"[LOG] Building FAISS IndexFlatIP (Dimension: {dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    print(f"[LOG] Saving FAISS index to {out_index}...")
    faiss.write_index(index, out_index)
    
    print(f"[LOG] Saving reference metadata to {out_meta}...")
    meta_df = pd.DataFrame({
        'IMG_PATH': images,
        'LAT': lats,
        'LON': lons,
        'TEXT': texts
    })
    meta_df.to_csv(out_meta, index=False)
    print("[LOG] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for reference images")
    parser.add_argument("--data-dir", type=str, default="data/im2gps3k/images", help="Path to reference images")
    parser.add_argument("--csv", type=str, default="data/im2gps3k/metadata.csv", help="Path to reference metadata")
    parser.add_argument("--out-index", type=str, default="data/reference.faiss", help="Output path for FAISS index")
    parser.add_argument("--out-meta", type=str, default="data/reference_meta.csv", help="Output path for metadata")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    
    args = parser.parse_args()
    build_index(args.data_dir, args.csv, args.out_index, args.out_meta, args.mock)
