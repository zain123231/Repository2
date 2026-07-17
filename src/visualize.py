import os
import sys
import torch
import numpy as np
import folium
import argparse
import faiss
import pandas as pd

# Ensure correct path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP
from src.dataset import GeoDataset

def create_map(gt_coords, pred_coords, output_path="error_map.html"):
    """
    Creates an interactive map using folium showing Ground Truth vs Predicted locations.
    """
    print("[LOG] Generating interactive map...")
    
    # Center map roughly at the mean of the ground truth coordinates
    center_lat = np.mean([lat for lat, lon in gt_coords])
    center_lon = np.mean([lon for lat, lon in gt_coords])
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    for i, ((gt_lat, gt_lon), (pred_lat, pred_lon)) in enumerate(zip(gt_coords, pred_coords)):
        # Ground Truth Marker (Green)
        folium.Marker(
            [gt_lat, gt_lon], 
            popup=f"Sample {i}: Ground Truth", 
            icon=folium.Icon(color="green", icon="ok")
        ).add_to(m)
        
        # Prediction Marker (Red)
        folium.Marker(
            [pred_lat, pred_lon], 
            popup=f"Sample {i}: Prediction", 
            icon=folium.Icon(color="red", icon="remove")
        ).add_to(m)
        
        # Line connecting the two to show error distance
        folium.PolyLine(
            locations=[(gt_lat, gt_lon), (pred_lat, pred_lon)],
            color="blue",
            weight=2.5,
            opacity=0.6,
            dash_array='5'
        ).add_to(m)
        
    m.save(output_path)
    print(f"[LOG] Map successfully saved to {output_path}")
    print("[LOG] Open the HTML file in any browser to view the interactive map.")

def main():
    parser = argparse.ArgumentParser(description="GeoCLIP Error Visualization")
    parser.add_argument("--csv", type=str, default="data/train.csv", help="Path to test CSV metadata")
    parser.add_argument("--mock", action="store_true", help="Use mock dataset instead of real data")
    parser.add_argument("--global_search", action="store_true", help="Search using the 130k+ global cities database")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    
    if not args.mock and not os.path.exists(args.csv):
        print(f"\n[ERROR] Dataset not found at {args.csv}")
        print("Please ensure you have downloaded the dataset. See src/README.md for instructions.")
        print("If you want to run a quick test with fake data, use the --mock flag.")
        sys.exit(1)

    # Load Model
    model = GeoCLIP().to(device)
    
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        if isinstance(checkpoint['logit_scale'], torch.Tensor):
            model.logit_scale.data = checkpoint['logit_scale']
        print("[LOG] Loaded trained model checkpoint.")
    else:
        print("[LOG] Trained checkpoint not found. Using baseline model.")
        
    model.eval()

    if args.mock:
        print(f"[LOG] Generating mock data ({args.samples} samples)...")
        # For visualization, we need reference points for the FAISS index to choose from.
        # Let's generate a gallery of reference points (e.g., 500 cities worldwide)
        np.random.seed(42) # For reproducible random locations
        ref_coords = np.random.uniform(low=[-90.0, -180.0], high=[90.0, 180.0], size=(500, 2))
        
        # Extract location features for FAISS
        ref_coords_tensor = torch.tensor(ref_coords, dtype=torch.float32).to(device)
        with torch.no_grad():
            ref_features = model.location_encoder(ref_coords_tensor)
        
        # Build a temporary FAISS index
        index = faiss.IndexFlatL2(512)
        index.add(ref_features.cpu().numpy())
        
        # Now generate a few mock query images and ground truth coords
        dataset = GeoDataset(csv_file="", mock=True, size=args.samples)
        
        gt_coords = []
        pred_coords = []
        
        print("[LOG] Running inference...")
        for i in range(args.samples):
            image, gt_coord = dataset[i]
            # Predict image features
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                img_feature = model.image_encoder(image)
                
            # Search in FAISS index
            D, I = index.search(img_feature.cpu().numpy(), 1)
            pred_idx = I[0][0]
            pred_coord = ref_coords[pred_idx]
            
            gt_coords.append(gt_coord.numpy())
            pred_coords.append(pred_coord)
            
        create_map(gt_coords, pred_coords, output_path="error_map_mock.html")
        
    else:
        print(f"[LOG] Loading real data from {args.csv}...")
        
        # Load the real dataset
        dataset = GeoDataset(csv_file=args.csv, img_dir="data/images", mock=False)
        
        print("[LOG] Loading Global FAISS index and cities...")
        global_index_path = "data/global_index.faiss"
        global_cities_path = "data/global_cities.csv"
        
        if not os.path.exists(global_index_path) or not os.path.exists(global_cities_path):
            print("[ERROR] Global index not found. Run 'python src/build_global_index.py' first.")
            sys.exit(1)
            
        index = faiss.read_index(global_index_path)
        cities_df = pd.read_csv(global_cities_path, low_memory=False)
        
        ref_coords = np.stack([cities_df["LAT"].values, cities_df["LON"].values], axis=1)
        city_names = cities_df["City"].values
        country_codes = cities_df["CountryCode"].values
            
        gt_coords = []
        pred_coords = []
        
        num_samples = min(args.samples, len(dataset))
        print(f"[LOG] Running inference on {num_samples} samples...")
        
        for i in range(num_samples):
            image, gt_coord = dataset[i]
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                img_feature = model.image_encoder(image)
                
            D, I = index.search(img_feature.cpu().numpy(), 1)
            pred_idx = I[0][0]
            pred_coord = ref_coords[pred_idx]
            
            gt_coords.append(gt_coord.numpy())
            pred_coords.append(pred_coord)
            
            print(f"Sample {i+1}: Predicted City: {city_names[pred_idx]}, {country_codes[pred_idx]}")
            
        create_map(gt_coords, pred_coords, output_path="error_map_real.html")

if __name__ == "__main__":
    main()
