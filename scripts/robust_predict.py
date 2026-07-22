"""
Final robust geolocation pipeline — multi-gallery, confidence-aware.
"""
import os, warnings, time, gc
os.environ['OMP_NUM_THREADS'] = '8'; os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings('ignore')
import numpy as np; import torch; torch.set_num_threads(8)
import faiss; faiss.omp_set_num_threads(8)
from pathlib import Path; from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/im2gps3k'; OSV = ROOT / 'data/raw/osv5m_clean'

def norm_f(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)

def density_vote(pts, w, R_km):
    if len(pts) < 2: return pts[0] if len(pts) > 0 else np.array([0., 0.])
    pts_km = pts * 111.32
    dists = np.linalg.norm(pts_km[:, None] - pts_km[None, :], axis=2)
    nmask = dists < R_km
    wd = (nmask * w[None, :]).sum(axis=1)
    best = (wd * w).argmax()
    mask = nmask[best]
    cw = w[mask]
    return (cw[:, None] * pts[mask]).sum(axis=0) / cw.sum() if cw.sum() > 0 else pts[best]

print("Loading GeoCLIP...")
t0 = time.time()
from geoclip import GeoCLIP
gc_model = GeoCLIP(); gc_model.eval()
print(f"  Done ({time.time()-t0:.1f}s)")

print("Loading CLIP ViT-B/32...")
t0 = time.time()
import open_clip
clip_model, _, clip_pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()
print(f"  Done ({time.time()-t0:.1f}s)")

print("Loading location gallery (3.5M)...")
t0 = time.time()
loc_gps = np.concatenate([
    np.load(str(RAW / 'wikidata_gps_expanded.npy')).astype(np.float32),
    np.load(str(RAW / 'osm_gps.npy')).astype(np.float32)], axis=0)
loc_feats = norm_f(np.concatenate([
    np.load(str(RAW / 'wikidata_features_expanded.npy')).astype(np.float32),
    np.load(str(RAW / 'osm_features.npy')).astype(np.float32)], axis=0))
print(f"  {len(loc_gps):,} points ({time.time()-t0:.1f}s)")

print("Loading image gallery (3000)...")
img_fts = norm_f(np.load(str(RAW / 'fresh_features_all_3000.npy')).astype(np.float32))
img_gps = np.load(str(RAW / 'coordinates.npy')).astype(np.float32)

print("Loading OSV-5M (50K)...")
osv_fts = np.load(str(OSV / 'clip_features_00.npy')).astype(np.float32)
osv_gps = np.load(str(OSV / 'clip_gps_00.npy')).astype(np.float32)

print("Building indexes...")
loc_idx = faiss.IndexFlatIP(loc_feats.shape[1]); loc_idx.add(loc_feats)
img_idx = faiss.IndexFlatIP(img_fts.shape[1]); img_idx.add(img_fts)
osv_idx = faiss.IndexFlatIP(osv_fts.shape[1]); osv_idx.add(osv_fts)

def extract_features(img):
    with torch.no_grad():
        px = gc_model.image_encoder.preprocess_image([img])
        gc_f = gc_model.image_encoder(px)
        gc_f = (gc_f / gc_f.norm()).cpu().numpy().astype(np.float32)
        t = clip_pre(img).unsqueeze(0)
        cl_f = clip_model.encode_image(t)
        cl_f = (cl_f / cl_f.norm()).cpu().numpy().astype(np.float32)
    return gc_f, cl_f

images = [
    (r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-47.jpg', 'Karbala'),
    (r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-38.jpg', 'Giza'),
]

for img_path, img_name in images:
    print(f"\n{'='*70}")
    print(f"  {img_name}")
    print(f"{'='*70}")
    
    img = Image.open(img_path).convert('RGB')
    gc_f, cl_f = extract_features(img)
    
    K = 100
    D_l, I_l = loc_idx.search(gc_f, K)
    D_i, I_i = img_idx.search(gc_f, 50)
    D_o, I_o = osv_idx.search(cl_f, K)
    
    loc_pts = loc_gps[I_l[0]]; loc_w = np.maximum(D_l[0], 0) + 1e-8
    img_pts = img_gps[I_i[0]]; img_w = np.maximum(D_i[0], 0) + 1e-8
    osv_pts = osv_gps[I_o[0]]; osv_w = np.maximum(D_o[0], 0) + 1e-8
    
    top_sim = float(loc_w.max())
    spread = float(np.linalg.norm(loc_pts[:10].std(axis=0) * 111.32))
    
    print(f"\n  GeoCLIP Location Gallery Top-10:")
    for j in range(10):
        lat, lon = loc_gps[I_l[0,j],0], loc_gps[I_l[0,j],1]
        print(f"    #{j+1}: {lat:.4f}, {lon:.4f}  (sim={D_l[0,j]:.4f})")
    
    print(f"\n  GeoCLIP Image Gallery Top-5:")
    for j in range(5):
        lat, lon = img_gps[I_i[0,j],0], img_gps[I_i[0,j],1]
        print(f"    #{j+1}: {lat:.4f}, {lon:.4f}  (sim={D_i[0,j]:.4f})")
    
    print(f"\n  CLIP OSV-5M Top-5:")
    for j in range(5):
        lat, lon = osv_gps[I_o[0,j],0], osv_gps[I_o[0,j],1]
        print(f"    #{j+1}: {lat:.4f}, {lon:.4f}  (sim={D_o[0,j]:.4f})")
    
    # Density vote at multiple radii
    print(f"\n  Density Voting:")
    for R in [5, 10, 25, 50, 100]:
        p = density_vote(loc_pts, loc_w, R)
        print(f"    GC-Loc R={R:3d}km: {p[0]:.6f}, {p[1]:.6f}")
    for R in [5, 10, 25, 50, 100]:
        p = density_vote(osv_pts, osv_w, R)
        print(f"    CLIP-OSV R={R:3d}km: {p[0]:.6f}, {p[1]:.6f}")
    
    # Combined: GC-loc + image gallery
    comb_pts = np.concatenate([loc_pts, img_pts])
    comb_w = np.concatenate([loc_w / loc_w.max(), img_w / img_w.max()]) + 1e-8
    for R in [5, 10, 25, 50, 100]:
        p = density_vote(comb_pts, comb_w, R)
        print(f"    GC-Comb R={R:3d}km: {p[0]:.6f}, {p[1]:.6f}")
    
    # Confidence assessment
    if spread < 50 and top_sim > 0.45:
        conf = "HIGH"
        pred = density_vote(loc_pts, loc_w, 25)
    elif spread < 200 and top_sim > 0.38:
        conf = "MEDIUM"
        pred = density_vote(loc_pts, loc_w, 25)
    else:
        conf = "LOW"
        # Try CLIP-based + GeoCLIP ensemble
        gc_pred = density_vote(loc_pts, loc_w, 50)
        osv_pred = density_vote(osv_pts, osv_w, 50)
        # Weight by confidence
        gc_conf = min(top_sim / 0.5, 1.0)
        osv_conf = min(osv_w.max() / 0.8, 1.0)
        pred = (gc_pred * gc_conf + osv_pred * osv_conf) / (gc_conf + osv_conf + 1e-8)
    
    print(f"\n  >> FINAL: {pred[0]:.6f}, {pred[1]:.6f}")
    print(f"  >> Confidence: {conf} (spread={spread:.0f}km, top_sim={top_sim:.4f})")

print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
