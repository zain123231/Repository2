import os
import argparse
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geoclip import GeoCLIP
from src.metrics import haversine_distance, calculate_metrics
from src.inference import InferenceEngine

def evaluate_gated_refinement(val_csv, img_dir, global_cities_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    model = GeoCLIP(from_pretrained=True).to(device)
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
    model.eval()
    
    cities_df = pd.read_csv(global_cities_path, low_memory=False)
    global_index_path = "data/global_index.faiss"
    if os.path.exists(global_index_path):
        print(f"[LOG] Loading existing FAISS Index from {global_index_path}...")
        index = faiss.read_index(global_index_path)
    else:
        print(f"[ERROR] FAISS index not found at {global_index_path}. Please run src/build_global_index.py first.")
        import sys
        sys.exit(1)
    
    engine = InferenceEngine(model, device, index, cities_df)
    
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"[ERROR] Validation CSV {val_csv} not found! Dataset isolation rule violated. Please build it first.")
    else:
        val_df = pd.read_csv(val_csv)
        
    print(f"[LOG] Validation set size: {len(val_df)}")
    
    Rs = [50, 100, 500, 1000]
    top_ks = [2, 3, 5]
    criteria_list = ["distance", "margin", "confidence", "dispersion"]
    
    results_grid = []
    
    # 1. First extract features and compute coarse search for all validation images
    # to avoid re-running the model for every hyperparameter combination.
    print("[LOG] Pre-computing features and coarse searches...")
    val_data = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        img_id = str(row.get('IMG_ID', ''))
        img_path = os.path.join(img_dir, row.get('IMG_PATH', f"{img_id}.jpg" if img_id else ""))
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id)
            if not os.path.exists(img_path):
                continue
                
        try:
            image = Image.open(img_path).convert("RGB")
            # We use A3 variant features (TTA)
            img_feature = engine.extract_image_features(image, use_tta=True)
            # Get max top_k needed
            coarse_candidates = engine.coarse_search(img_feature, top_k=max(top_ks))
            target = torch.tensor([[row['LAT'], row['LON']]], dtype=torch.float32)
            
            # Precompute refinement for dispersion/confidence checks
            refined_candidates = engine.micro_grid_refinement(img_feature, coarse_candidates)
            
            val_data.append({
                'feature': img_feature,
                'coarse': coarse_candidates,
                'refined': refined_candidates,
                'target': target
            })
        except Exception as e:
            pass
            
    print("[LOG] Sweeping gating criteria, R, and top-k...")
    for crit in criteria_list:
      for k in top_ks:
        for r in Rs:
            errors = []
            num_refined = 0
            for item in val_data:
                coarse_candidates = item['coarse'][:k]
                
                # Check criteria
                should_refine = False
                if crit == "distance":
                    should_refine = engine.conditional_refinement(coarse_candidates, radius_km=r)
                elif crit == "margin":
                    # Margin between top 1 and top 2 coarse distance/similarity
                    should_refine = abs(coarse_candidates[1][4] - coarse_candidates[0][4]) < (r / 1000.0)
                elif crit == "confidence":
                    # Softmax prob of top network
                    should_refine = item['refined'][0]['confidence_prob'] < (r / 10.0) # r acting as threshold sweep
                elif crit == "dispersion":
                    # Dispersion of top scores in grid (proxy: difference between rank 1 and 5 in refined grid)
                    should_refine = (item['refined'][0]['score'] - item['refined'][min(4, len(item['refined'])-1)]['score']) < (r / 1000.0)
                
                if should_refine:
                    num_refined += 1
                    pred_lat, pred_lon = item['refined'][0]['lat'], item['refined'][0]['lon']
                else:
                    pred_lat, pred_lon = coarse_candidates[0][0], coarse_candidates[0][1]
                    
                pred_tensor = torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32)
                dist = haversine_distance(pred_tensor, item['target']).item()
                errors.append(dist)
                
            if not errors: continue
            med_error = np.median(errors)
            acc_1km = np.mean(np.array(errors) <= 1) * 100
            acc_2500km = np.mean(np.array(errors) <= 2500) * 100
            pct_refined = (num_refined / len(val_data)) * 100
            
            results_grid.append({
                'Criterion': crit,
                'top_k': k,
                'Threshold/R': r,
                'Refined_%': pct_refined,
                'Median_Error': med_error,
                'Acc@1km': acc_1km,
                'Acc@2500km': acc_2500km
            })
            
    df_grid = pd.DataFrame(results_grid)
    print("\n--- Hyperparameter Sweep Results ---")
    print(df_grid.to_string(index=False))
    
    # Select best combination (e.g., highest Acc@1km with lowest Median Error)
    best_row = df_grid.sort_values(by=['Acc@1km', 'Median_Error'], ascending=[False, True]).iloc[0]
    best_k = int(best_row['top_k'])
    best_r = float(best_row['R_km'])
    print(f"\n[LOG] Selected Best Hyperparameters: top_k={best_k}, R={best_r}km")
    
    # 2. Compare No Refinement vs Always Refine vs Conditional Refine using best params
    print("\n[LOG] Running Final Comparison...")
    comparison = {'No Refinement': [], 'Always Refine': [], 'Conditional': []}
    
    for item in val_data:
        coarse_candidates = item['coarse'][:best_k]
        
        # No Refinement
        pred_no = torch.tensor([[coarse_candidates[0][0], coarse_candidates[0][1]]], dtype=torch.float32)
        comparison['No Refinement'].append(haversine_distance(pred_no, item['target']).item())
        
        # Always Refine
        refined_always = engine.micro_grid_refinement(item['feature'], coarse_candidates)
        pred_always = torch.tensor([[refined_always[0]['lat'], refined_always[0]['lon']]], dtype=torch.float32)
        comparison['Always Refine'].append(haversine_distance(pred_always, item['target']).item())
        
        # Conditional
        should_refine = engine.conditional_refinement(coarse_candidates, radius_km=best_r)
        if should_refine:
            comparison['Conditional'].append(haversine_distance(pred_always, item['target']).item())
        else:
            comparison['Conditional'].append(haversine_distance(pred_no, item['target']).item())
            
    # Format the results and save using reporting
    from src.reporting import generate_report
    systems_results = {k: [{'error': e} for e in v] for k, v in comparison.items()}
    generate_report(systems_results, len(val_data), dataset_name="Val_Set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", type=str, default="data/val.csv")
    parser.add_argument("--img-dir", type=str, default="data/im2gps3k/images")
    parser.add_argument("--cities", type=str, default="data/global_cities.csv")
    args = parser.parse_args()
    
    # Fix seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    import random
    random.seed(42)
    
    evaluate_gated_refinement(args.val_csv, args.img_dir, args.cities)
