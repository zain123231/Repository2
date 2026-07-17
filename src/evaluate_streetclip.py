import os
import argparse

# إزالة متغيرات البيئة التي قد تسبب مشاكل ValueError في مكتبة huggingface_hub
os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HEADERS", None)

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, AutoProcessor

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import haversine_distance, calculate_metrics

def evaluate_streetclip(dataset_dir: str, metadata_csv: str, cities_csv: str, mock: bool = False):
    print("[LOG] Loading StreetCLIP model...")
    
    # استخدام المسار المحلي المباشر لتخطي أي فحص عبر الإنترنت نهائياً
    import os
    local_streetclip = r"C:\Users\Msi\.cache\huggingface\hub\models--geolocal--StreetCLIP\snapshots\e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
    model_id = local_streetclip if os.path.exists(local_streetclip) else "geolocal/StreetCLIP"
    
    try:
        model = CLIPModel.from_pretrained(model_id, use_safetensors=True, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    except Exception as e:
        print(f"[ERROR] Failed to load StreetCLIP: {e}")
        print("[LOG] Falling back to standard CLIP (openai/clip-vit-base-patch32) for demonstration if StreetCLIP is unavailable...")
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id, use_safetensors=True, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"[LOG] Loading hierarchical cities database from {cities_csv}")
    cities_df = pd.read_csv(cities_csv)
    
    # Pre-compute country prompts
    unique_countries = cities_df['country'].unique().tolist()
    country_prompts = [f"a street photo taken in {country}" for country in unique_countries]
    print(f"[LOG] Encoded {len(unique_countries)} unique countries.")
    
    # Tokenize countries
    country_inputs = processor(text=country_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        country_features = model.get_text_features(**country_inputs)
        country_features /= country_features.norm(dim=-1, keepdim=True)

    if mock:
        print("[LOG] MOCK MODE: Running sanity check on randomly generated images.")
        # Instead of real images, we'll just mock 10 predictions
        all_preds = []
        all_targets = []
        
        for i in range(10):
            # mock image embedding
            img_features = torch.randn(1, model.config.projection_dim).to(device)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            # Country prediction
            country_similarity = (100.0 * img_features @ country_features.T).softmax(dim=-1)
            best_country_idx = country_similarity.argmax().item()
            pred_country = unique_countries[best_country_idx]
            
            # City prediction inside the country
            country_cities_df = cities_df[cities_df['country'] == pred_country]
            city_prompts = [f"a street photo taken in {row['city']}, {pred_country}" for _, row in country_cities_df.iterrows()]
            
            city_inputs = processor(text=city_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                city_features = model.get_text_features(**city_inputs)
                city_features /= city_features.norm(dim=-1, keepdim=True)
                
            city_similarity = (100.0 * img_features @ city_features.T).softmax(dim=-1)
            best_city_idx = city_similarity.argmax().item()
            best_city_row = country_cities_df.iloc[best_city_idx]
            
            pred_lat, pred_lon = best_city_row['lat'], best_city_row['lon']
            
            all_preds.append([pred_lat, pred_lon])
            # Random target from the mock cities
            target_row = cities_df.sample(1).iloc[0]
            all_targets.append([target_row['lat'], target_row['lon']])
            
        distances = haversine_distance(torch.tensor(all_preds), torch.tensor(all_targets))
        metrics = calculate_metrics(distances)
        print("Mock Metrics for StreetCLIP:", metrics)
        return metrics

    if not os.path.exists(dataset_dir) or not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Dataset not found! Expected images in {dataset_dir} and metadata in {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    
    all_preds = []
    all_targets = []
    
    print(f"[LOG] Starting hierarchical Zero-Shot evaluation on {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['IMG_ID']
        lat_true = row['LAT']
        lon_true = row['LON']
        
        img_path = os.path.join(dataset_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_dir, f"{img_id}")
            if not os.path.exists(img_path):
                continue
                
        image = Image.open(img_path).convert("RGB")
        image_input = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            img_features = model.get_image_features(**image_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            # Country level
            country_similarity = (100.0 * img_features @ country_features.T).softmax(dim=-1)
            best_country_idx = country_similarity.argmax().item()
            pred_country = unique_countries[best_country_idx]
            
            # City level
            country_cities_df = cities_df[cities_df['country'] == pred_country]
            city_prompts = [f"a street photo taken in {c_row['city']}, {pred_country}" for _, c_row in country_cities_df.iterrows()]
            
            city_inputs = processor(text=city_prompts, return_tensors="pt", padding=True).to(device)
            city_features = model.get_text_features(**city_inputs)
            city_features /= city_features.norm(dim=-1, keepdim=True)
                
            city_similarity = (100.0 * img_features @ city_features.T).softmax(dim=-1)
            best_city_idx = city_similarity.argmax().item()
            best_city_row = country_cities_df.iloc[best_city_idx]
            
            pred_lat, pred_lon = best_city_row['lat'], best_city_row['lon']
            
        all_preds.append([pred_lat, pred_lon])
        all_targets.append([lat_true, lon_true])
        
    distances = haversine_distance(torch.tensor(all_preds), torch.tensor(all_targets))
    metrics = calculate_metrics(distances)
    
    print("="*40)
    print("StreetCLIP Hierarchical Zero-Shot Results")
    print("="*40)
    print(f"Median Error: {metrics['median_error_km']:.2f} km")
    for tau in [1, 25, 200, 750, 2500]:
        acc = metrics.get(f'acc@{tau}km', 0.0)
        print(f"Acc@{tau}km: {acc*100:.1f}%")
    print("="*40)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate StreetCLIP on Im2GPS3k/YFCC4k")
    parser.add_argument("--data-dir", type=str, default="../data/im2gps3k/images_real", help="Path to images")
    parser.add_argument("--csv", type=str, default="../data/im2gps3k/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--cities", type=str, default="data/global_cities.csv", help="Path to cities database CSV")
    parser.add_argument("--mock", action="store_true", help="Run a quick mock evaluation")
    args = parser.parse_args()
    
    evaluate_streetclip(args.data_dir, args.csv, args.cities, args.mock)
