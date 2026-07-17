import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import math
import os

# CLIP Normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class InferenceEngine:
    def __init__(self, model, device, index=None, cities_df=None, ocr_reader=None):
        self.model = model
        self.device = device
        self.index = index
        self.cities_df = cities_df
        self.ocr_reader = ocr_reader
        
        # Pre-process for A1 (Single Crop)
        self.transform_a1 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
        
        # Pre-process for A2/A3/A4 (TTA 10 crops)
        self.transform_tta_base = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224)
        ])
        self.transform_tta_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])

    def extract_image_features(self, image, use_tta=False):
        """Extracts features from an image, with or without TTA."""
        if use_tta:
            crops = self.transform_tta_base(image)
            img_tensor = torch.stack([self.transform_tta_tensor(crop) for crop in crops]).to(self.device)
            with torch.no_grad():
                img_features_batch = self.model.image_encoder(img_tensor)
                # Average the features across all 10 crops
                img_feature = img_features_batch.mean(dim=0, keepdim=True)
                img_feature = F.normalize(img_feature, dim=-1)
        else:
            img_tensor = self.transform_a1(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feature = self.model.image_encoder(img_tensor)
                img_feature = F.normalize(img_feature, dim=-1)
        return img_feature

    def coarse_search(self, img_feature, top_k=3):
        """Perform coarse search using FAISS index."""
        if self.index is None or self.cities_df is None:
            raise ValueError("Index or cities_df not loaded.")
            
        distances, indices = self.index.search(img_feature.cpu().numpy(), top_k)
        
        candidates = []
        for i in range(top_k):
            idx = indices[0][i]
            dist = distances[0][i]
            row = self.cities_df.iloc[idx]
            
            name = row.get('City', 'Unknown')
            if pd.isna(name): name = "Grid Location"
            country = row.get('CountryCode', 'Unknown')
            lat = row['LAT']
            lon = row['LON']
            candidates.append((lat, lon, name, country, dist))
            
        return candidates

    def conditional_refinement(self, coarse_candidates, radius_km=500):
        """
        Confidence-Conditioned Refinement logic.
        Checks if the top candidates agree within a certain radius.
        """
        from src.metrics import haversine_distance
        
        # If there's only 1 candidate, we can't check agreement, so we might as well refine
        if len(coarse_candidates) <= 1:
            return True
            
        # Check distance between top-1 and top-2
        lat1, lon1 = coarse_candidates[0][:2]
        lat2, lon2 = coarse_candidates[1][:2]
        
        p1 = torch.tensor([[lat1, lon1]], dtype=torch.float32)
        p2 = torch.tensor([[lat2, lon2]], dtype=torch.float32)
        dist = haversine_distance(p1, p2).item()
        
        # If they agree within radius, we are confident, so we refine
        # If they disagree, we might not refine
        return dist <= radius_km

    def micro_grid_refinement(self, img_feature, coarse_candidates, grid_steps=40, offset_range=0.5):
        """Perform Micro-Grid refinement around coarse candidates with geometric correction."""
        results = []
        
        for rank, (coarse_lat, coarse_lon, name, country, dist) in enumerate(coarse_candidates):
            # Geometric Correction: Longitude offset should be divided by cos(latitude)
            lat_rad = math.radians(coarse_lat)
            # Clip lat_rad to avoid division by zero near poles (e.g. max 89 degrees)
            lat_rad = max(min(lat_rad, math.radians(89)), math.radians(-89))
            cos_lat = math.cos(lat_rad)
            lon_offset_range = offset_range / cos_lat
            
            lat_offsets = np.linspace(-offset_range, offset_range, grid_steps)
            lon_offsets = np.linspace(-lon_offset_range, lon_offset_range, grid_steps)
            
            lat_grid, lon_grid = np.meshgrid(coarse_lat + lat_offsets, coarse_lon + lon_offsets)
            micro_grid = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
            micro_grid_tensor = torch.tensor(micro_grid, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                loc_features = self.model.location_encoder(micro_grid_tensor)
                loc_features = F.normalize(loc_features, dim=-1)
                
                logit_scale = self.model.logit_scale.exp()
                similarity = logit_scale * (img_feature @ loc_features.T)
                
                # Softmax over all micro-grid points for confidence calibration
                probs = F.softmax(similarity, dim=-1)
                
                best_local_idx = similarity.argmax().item()
                best_local_score = similarity[0, best_local_idx].item()
                best_local_prob = probs[0, best_local_idx].item()
                best_local_coord = micro_grid[best_local_idx]
                
                results.append({
                    "name": name,
                    "country": country,
                    "lat": best_local_coord[0],
                    "lon": best_local_coord[1],
                    "score": best_local_score,
                    "confidence_prob": best_local_prob * 100.0,
                    "coarse_dist": dist
                })
                
        # Sort results by the new refined score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def predict(self, image, variant="A3", top_k=3, conditional_refine=False, refine_radius=500):
        """
        Main prediction method handling variants A1, A2, A3, A4.
        Returns a list of dictionaries with predictions.
        """
        use_tta = variant in ["A2", "A3", "A4"]
        use_micro_grid = variant in ["A3", "A4"]
        use_ocr = variant == "A4"
        
        # 1. Feature Extraction
        img_feature = self.extract_image_features(image, use_tta=use_tta)
        
        # 2. OCR Integration (A4)
        detected_texts = []
        if use_ocr and self.ocr_reader is not None:
            img_np = np.array(image)
            ocr_results = self.ocr_reader.readtext(img_np)
            detected_texts = [res[1] for res in ocr_results if res[2] > 0.35]
            
        # 3. Coarse Search
        coarse_candidates = self.coarse_search(img_feature, top_k=top_k)
        
        # 4. Refinement
        if use_micro_grid:
            should_refine = True
            if conditional_refine:
                should_refine = self.conditional_refinement(coarse_candidates, radius_km=refine_radius)
                
            if should_refine:
                refined_results = self.micro_grid_refinement(img_feature, coarse_candidates)
                # Cap to top_k
                refined_results = refined_results[:top_k]
                for res in refined_results:
                    res["variant"] = variant
                    res["ocr_text"] = detected_texts
                return refined_results
        
        # If no micro-grid (A1, A2) or condition failed
        final_results = []
        for lat, lon, name, country, dist in coarse_candidates[:top_k]:
            final_results.append({
                "name": name,
                "country": country,
                "lat": lat,
                "lon": lon,
                "score": dist,
                "confidence_prob": 0.0, # Unavailable for coarse
                "coarse_dist": dist,
                "variant": variant,
                "ocr_text": detected_texts
            })
            
        return final_results
