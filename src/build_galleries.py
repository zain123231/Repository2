import os
import argparse
import torch
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geoclip import GeoCLIP

def build_index(model, device, lats, lons, output_faiss):
    coords = np.stack([lats, lons], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    
    batch_size = 1024
    features_list = []
    
    print(f"Building index for {len(coords)} points...")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(coords_tensor), batch_size)):
            batch = coords_tensor[i:i+batch_size].to(device)
            feats = model.location_encoder(batch)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            features_list.append(feats.cpu().numpy())
            
    all_features = np.vstack(features_list)
    index = faiss.IndexFlatIP(512)
    index.add(all_features)
    faiss.write_index(index, output_faiss)
    print(f"Saved FAISS index to {output_faiss}")

def is_on_land(lons, lats):
    """
    Check if coordinates are on land using Geopandas Natural Earth lowres dataset.
    """
    if not HAS_GEOPANDAS:
        print("[WARNING] geopandas not installed! Skipping land mask and using full grid.")
        return np.ones(len(lons), dtype=bool)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    print("Filtering points using Natural Earth land mask (vectorized)...")
    df_pts = pd.DataFrame({'LON': lons, 'LAT': lats})
    gdf_pts = gpd.GeoDataFrame(df_pts, geometry=gpd.points_from_xy(df_pts.LON, df_pts.LAT), crs=world.crs)
    
    joined = gpd.sjoin(gdf_pts, world, how="inner", predicate="intersects")
    valid = np.zeros(len(lons), dtype=bool)
    valid[joined.index] = True
    return valid

def generate_galleries(global_cities_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCLIP(from_pretrained=True).to(device)
    
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        
    os.makedirs("data/galleries", exist_ok=True)
    
    # 1. Raw GeoNames (Current Baseline)
    # We assume global_cities.csv exists and has LAT, LON, population
    print("\n--- Gallery 1: Raw GeoNames Density ---")
    df_raw = pd.read_csv(global_cities_path, low_memory=False)
    df_raw.to_csv("data/galleries/G1_raw.csv", index=False)
    build_index(model, device, df_raw['LAT'].values, df_raw['LON'].values, "data/galleries/G1_raw.faiss")
    
    # 2. Population-weighted GeoNames
    print("\n--- Gallery 2: Population Weighted ---")
    # GeoNames has population. We can filter top N or use population to sample.
    # We will filter to keep only populated places > 1000 to drastically change distribution
    if 'population' in df_raw.columns:
        # True population weighting: Sample proportional to population
        df_raw['pop_weight'] = df_raw['population'].clip(lower=1)
        df_pop = df_raw.sample(n=min(100000, len(df_raw)), weights='pop_weight', random_state=42, replace=True).drop_duplicates(subset=['LAT', 'LON'])
    else:
        # Mock population weight if column missing
        print("[WARNING] 'population' column missing, simulating pop-weight by dropping 50% random")
        df_pop = df_raw.sample(frac=0.5, random_state=42)
        
    df_pop.to_csv("data/galleries/G2_pop.csv", index=False)
    build_index(model, device, df_pop['LAT'].values, df_pop['LON'].values, "data/galleries/G2_pop.faiss")
    
    # 3. Pure Regular Grid (0.2 degree) with Natural Earth land mask
    print("\n--- Gallery 3: Pure Regular Grid (Land Masked) ---")
    lats = np.arange(-90, 90, 0.2)
    lons = np.arange(-180, 180, 0.2)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    flat_lons, flat_lats = grid_lon.ravel(), grid_lat.ravel()
    
    land_mask = is_on_land(flat_lons, flat_lats)
    grid_lats_land = flat_lats[land_mask]
    grid_lons_land = flat_lons[land_mask]
    
    df_grid = pd.DataFrame({
        'LAT': grid_lats_land,
        'LON': grid_lons_land,
        'City': ['GridPoint'] * len(grid_lats_land),
        'CountryCode': ['UNK'] * len(grid_lats_land)
    })
    df_grid.to_csv("data/galleries/G3_grid.csv", index=False)
    build_index(model, device, df_grid['LAT'].values, df_grid['LON'].values, "data/galleries/G3_grid.faiss")
    
    # 4. Hybrid (Grid + Populated Places)
    print("\n--- Gallery 4: Hybrid (Grid + Pop) ---")
    df_hybrid = pd.concat([df_grid, df_pop]).drop_duplicates(subset=['LAT', 'LON'])
    df_hybrid.to_csv("data/galleries/G4_hybrid.csv", index=False)
    build_index(model, device, df_hybrid['LAT'].values, df_hybrid['LON'].values, "data/galleries/G4_hybrid.faiss")
    
    print("\n[LOG] All galleries successfully built.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", type=str, default="data/global_cities.csv")
    args = parser.parse_args()
    
    # Using fixed seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    generate_galleries(args.cities)
