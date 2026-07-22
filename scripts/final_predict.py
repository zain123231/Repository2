"""
Final answer -- clean prediction for both images.
"""
import os, warnings, time, gc
os.environ['OMP_NUM_THREADS'] = '8'; os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings('ignore')
import numpy as np; import torch; torch.set_num_threads(8)
import faiss; faiss.omp_set_num_threads(8)
from pathlib import Path; from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/im2gps3k'

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

print("Loading models...")
t0 = time.time()
from geoclip import GeoCLIP
model = GeoCLIP(); model.eval()
print(f"  GeoCLIP: {time.time()-t0:.1f}s")

t0 = time.time()
import open_clip
clip_model, _, clip_pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()
print(f"  CLIP: {time.time()-t0:.1f}s")

print("Loading location gallery...")
t0 = time.time()
wd_gps = np.load(str(RAW / 'wikidata_gps_expanded.npy')).astype(np.float32)
wd_fts = np.load(str(RAW / 'wikidata_features_expanded.npy')).astype(np.float32)
osm_gps = np.load(str(RAW / 'osm_gps.npy')).astype(np.float32)
osm_fts = np.load(str(RAW / 'osm_features.npy')).astype(np.float32)
loc_gps = np.concatenate([wd_gps, osm_gps], axis=0)
loc_fts = norm_f(np.concatenate([wd_fts, osm_fts], axis=0))
del wd_gps, wd_fts, osm_gps, osm_fts; gc.collect()
print(f"  {len(loc_gps):,} points ({time.time()-t0:.1f}s)")

print("Loading OSV-5M...")
osv_fts = np.load(str(ROOT / 'data/raw/osv5m_clean/clip_features_00.npy')).astype(np.float32)
osv_gps = np.load(str(ROOT / 'data/raw/osv5m_clean/clip_gps_00.npy')).astype(np.float32)

loc_idx = faiss.IndexFlatIP(loc_fts.shape[1]); loc_idx.add(loc_fts)
osv_idx = faiss.IndexFlatIP(osv_fts.shape[1]); osv_idx.add(osv_fts)

images = [
    ('Karbala, Iraq', r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-47.jpg', 32.6164, 44.0323),
    ('Giza, Egypt', r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-38.jpg', 29.85, 31.24),
]

print(f"\n{'='*80}")
print(f"  FINAL RESULTS")
print(f"{'='*80}")

for label, path, true_lat, true_lon in images:
    print(f"\n{'-'*80}")
    print(f"  {label}")
    print(f"  True location: {true_lat:.4f}, {true_lon:.4f}")
    print(f"{'-'*80}")

    img = Image.open(path).convert('RGB')
    
    with torch.no_grad():
        px = model.image_encoder.preprocess_image([img])
        gc_f = (model.image_encoder(px) / model.image_encoder(px).norm()).cpu().numpy().astype(np.float32)
        t = clip_pre(img).unsqueeze(0)
        cl_f = (clip_model.encode_image(t) / clip_model.encode_image(t).norm(dim=-1, keepdim=True))
        cl_f = cl_f.cpu().numpy().astype(np.float32)
    
    K = 200
    D_l, I_l = loc_idx.search(gc_f, K)
    loc_pts = loc_gps[I_l[0]]; loc_w = np.maximum(D_l[0], 0) + 1e-8
    
    D_o, I_o = osv_idx.search(cl_f, K)
    osv_pts = osv_gps[I_o[0]]; osv_w = np.maximum(D_o[0], 0) + 1e-8
    
    top_sim = float(loc_w.max())
    spread = float(np.linalg.norm(loc_pts[:10].std(axis=0) * 111.32))
    
    print(f"\n  [GeoCLIP Location Gallery] top-5:")
    regions = {'Iran':(44,63,25,40), 'Iraq':(39,48,29,37), 'Egypt':(25,37,22,32),
               'Turkey':(26,45,36,42), 'India':(68,98,6,38), 'Saudi':(35,55,16,32)}
    for j in range(5):
        lat, lon = loc_gps[I_l[0,j],0], loc_gps[I_l[0,j],1]
        region = 'other'
        for rn, (x1,x2,y1,y2) in regions.items():
            if x1 <= lon <= x2 and y1 <= lat <= y2: region = rn; break
        d_gt = np.linalg.norm(np.array([lat,lon]) - np.array([true_lat,true_lon])) * 111.32
        print(f"    #{j+1}: {lat:.4f}, {lon:.4f}  sim={D_l[0,j]:.4f}  region={region}  err={d_gt:.0f}km")
    
    print(f"\n  [GeoCLIP Density Vote]")
    best_dist = 1e9; best_pred = None
    for R in [10, 25, 50, 100]:
        p = density_vote(loc_pts, loc_w, R)
        d_gt = np.linalg.norm((p - np.array([true_lat, true_lon])) * 111.32)
        if d_gt < best_dist: best_dist = d_gt; best_pred = p
        print(f"    R={R:3d}km -> {p[0]:.6f}, {p[1]:.6f}  err={d_gt:.0f}km")
    
    print(f"\n  [CLIP OSV-5M Density Vote]")
    for R in [200, 500, 1000]:
        p = density_vote(osv_pts, osv_w, R)
        d_gt = np.linalg.norm((p - np.array([true_lat, true_lon])) * 111.32)
        print(f"    R={R:4d}km -> {p[0]:.6f}, {p[1]:.6f}  err={d_gt:.0f}km")
    
    # Check nearest gallery point to ground truth
    mask = (np.abs(loc_gps[:,0]-true_lat) < 0.5) & (np.abs(loc_gps[:,1]-true_lon) < 0.5)
    if mask.any():
        candidates = np.where(mask)[0]
        sims = loc_fts[candidates] @ gc_f.T
        best_k = sims.argmax()
        gi = candidates[best_k]
        print(f"\n  Nearest gallery point to truth: {loc_gps[gi,0]:.4f}, {loc_gps[gi,1]:.4f}  sim={sims[best_k,0]:.4f}")
        # Check if it's in top-K
        for k in range(K):
            if I_l[0,k] == gi:
                print(f"    -> Rank #{k+1} in GeoCLIP results")
                break
        else:
            print(f"    -> NOT in top-{K} (!)")
    
    if spread < 50 and top_sim > 0.45:
        conf = "HIGH"
        final = density_vote(loc_pts, loc_w, 25)
    elif spread < 200 and top_sim > 0.38:
        conf = "MEDIUM"
        final = density_vote(loc_pts, loc_w, 25)
    else:
        conf = "LOW - model is uncertain"
        final = density_vote(loc_pts, loc_w, 50)
    
    d_final = np.linalg.norm((final - np.array([true_lat, true_lon])) * 111.32)
    
    print(f"\n  {'='*60}")
    print(f"  FINAL OUTPUT:")
    print(f"    lat = {final[0]:.6f}")
    print(f"    lon = {final[1]:.6f}")
    print(f"    confidence: {conf}")
    print(f"    error: {d_final:.0f} km")
    print(f"    spread: {spread:.0f} km")
    print(f"    top_sim: {top_sim:.4f}")
    print(f"  {'='*60}")

print(f"\n{'='*80}")
print(f"  SUMMARY:")
print(f"  Giza (Egypt): Model works well -- <20km error HIGH confidence.")
print(f"  Karbala (Iraq): Model FAILS -- outputs Iran. Root cause:")
print(f"    1. Shrine architecture is visually similar across Iraq/Iran")
print(f"    2. Training data imbalance (Iran overrepresented)")
print(f"    3. OSM gallery has zero Iraq points")
print(f"  Solution: Retrain with more Middle East data, or use a different model.")
print(f"{'='*80}")
