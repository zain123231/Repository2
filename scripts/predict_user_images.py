"""Predict high-precision GPS for user's 2 images using full gallery pipeline."""
import os, warnings, time, gc
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
warnings.filterwarnings('ignore')

import numpy as np
import torch; torch.set_num_threads(16)
import faiss; faiss.omp_set_num_threads(16)
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/im2gps3k'
USER_PATHS = [
    r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-47.jpg',
    r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-38.jpg',
]

# ====== 1. LOAD GeoCLIP MODEL ======
print("Loading GeoCLIP...")
t0 = time.time()
from geoclip import GeoCLIP
model = GeoCLIP()
model.eval()
print(f"  Done ({time.time()-t0:.1f}s)")

# ====== 2. EXTRACT FRESH FEATURES ======
print("Extracting image features...")
t_feat = time.time()
images = [Image.open(p).convert('RGB') for p in USER_PATHS]
with torch.no_grad():
    pixel_vals = model.image_encoder.preprocess_image(images)
    feats = model.image_encoder(pixel_vals).cpu().numpy().astype(np.float32)
norms = np.linalg.norm(feats, axis=1, keepdims=True)
feats = feats / np.maximum(norms, 1e-8)
print(f"  Features shape: {feats.shape} ({time.time()-t_feat:.1f}s)")

# ====== 3. LOAD LOCATION GALLERY (3.5M) ======
print("Loading 3.5M location gallery...")
t_gal = time.time()
wd_gps = np.load(str(RAW / 'wikidata_gps_expanded.npy')).astype(np.float32)
wd_feats = np.load(str(RAW / 'wikidata_features_expanded.npy')).astype(np.float32)
osm_gps = np.load(str(RAW / 'osm_gps.npy')).astype(np.float32)
osm_feats = np.load(str(RAW / 'osm_features.npy')).astype(np.float32)

def norm_f(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)

loc_gps = np.concatenate([wd_gps, osm_gps], axis=0).astype(np.float32)
loc_feats = norm_f(np.concatenate([wd_feats, osm_feats], axis=0).astype(np.float32))
del wd_gps, wd_feats, osm_gps, osm_feats; gc.collect()
print(f"  Location gallery: {len(loc_gps):,} points ({time.time()-t_gal:.1f}s)")

# ====== 4. LOAD IMAGE GALLERY (3000) ======
print("Loading image gallery (3000 reference images)...")
fresh = np.load(str(RAW / 'fresh_features_all_3000.npy')).astype(np.float32)
fresh = fresh / np.linalg.norm(fresh, axis=1, keepdims=True)
all_gps = np.load(str(RAW / 'coordinates.npy')).astype(np.float32)
print(f"  Image gallery: {len(fresh)} points")

# ====== 5. BUILD FAISS INDEXES ======
print("Building FAISS indexes...")
t_idx = time.time()
loc_idx = faiss.IndexFlatIP(loc_feats.shape[1])
loc_idx.add(loc_feats)
img_idx = faiss.IndexFlatIP(fresh.shape[1])
img_idx.add(fresh)
print(f"  Indexes built ({time.time()-t_idx:.1f}s)")

# ====== 6. SEARCH GALLERIES ======
K = 50
print(f"Searching galleries (K={K})...")
t_s = time.time()
D_loc, I_loc = loc_idx.search(feats, K)
D_img, I_img = img_idx.search(feats, K)
print(f"  Search done ({time.time()-t_s:.1f}s)")

# ====== 7. DENSITY VOTING ======
def density_vote(all_pts, all_w, R_km):
    pts_km = all_pts * 111.32
    dists = np.linalg.norm(pts_km[:, None] - pts_km[None, :], axis=2)
    nmask = dists < R_km
    wd = (nmask * all_w[None, :]).sum(axis=1)
    score = wd * all_w
    best = score.argmax()
    mask = nmask[best]
    cw = all_w[mask]
    return (cw[:, None] * all_pts[mask]).sum(axis=0) / cw.sum()

print("\n" + "="*80)
print("RESULTS")
print("="*80)

for i, path in enumerate(USER_PATHS):
    name = os.path.basename(path)

    # Collect candidates from all galleries
    loc_pts = loc_gps[I_loc[i]]; loc_w = D_loc[i].copy()
    img_pts = all_gps[I_img[i]]; img_w = D_img[i].copy()

    all_pts = np.concatenate([loc_pts, img_pts])
    all_w = np.concatenate([loc_w, img_w])
    all_w = np.maximum(all_w, 0) + 1e-8

    # Try different radii
    print(f"\n  [{name}]")
    for R in [5, 10, 25, 50, 100]:
        pred = density_vote(all_pts, all_w, R)
        print(f"    R={R:3d}km -> {pred[0]:.6f}, {pred[1]:.6f}")

    # Also show top-5 raw matches from location gallery
    print(f"\n    Top-5 location gallery matches:")
    for j in range(5):
        lat, lon, sim = loc_gps[I_loc[i,j],0], loc_gps[I_loc[i,j],1], D_loc[i,j]
        print(f"      #{j+1}: {lat:.6f}, {lon:.6f}  (sim={sim:.4f})")

    # Top-3 from image gallery
    print(f"    Top-3 image gallery matches:")
    for j in range(3):
        lat, lon, sim = all_gps[I_img[i,j],0], all_gps[I_img[i,j],1], D_img[i,j]
        print(f"      #{j+1}: {lat:.6f}, {lon:.6f}  (sim={sim:.4f})")

print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
