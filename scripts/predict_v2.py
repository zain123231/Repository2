"""
FINAL PRODUCTION PIPELINE
- GeoCLIP image features -> search ME dense gallery (location encoder embeddings)
- GeoCLIP image features -> search GeoCLIP 3.5M location gallery
- CLIP features -> search OSV-5M gallery
- Density voting with confidence scoring
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

print("="*70, flush=True)
print("  GEOLOCATION PIPELINE v2.0", flush=True)
print("  3.5M GeoCLIP + 640 ME Dense + 50K OSV-5M", flush=True)
print("="*70, flush=True)

t_total = time.time()

# ---- Load models ----
print("\n[1/5] Loading models...", flush=True)
t0 = time.time()
from geoclip import GeoCLIP
gc_model = GeoCLIP(); gc_model.eval()
print(f"  GeoCLIP: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
import open_clip
clip_model, _, clip_pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()
print(f"  CLIP ViT-B/32: {time.time()-t0:.1f}s", flush=True)

# ---- Load galleries ----
print("\n[2/5] Loading galleries...", flush=True)
t0 = time.time()

# GeoCLIP location gallery (3.5M)
wd_gps = np.load(str(RAW / 'wikidata_gps_expanded.npy')).astype(np.float32)
wd_fts = np.load(str(RAW / 'wikidata_features_expanded.npy')).astype(np.float32)
osm_gps = np.load(str(RAW / 'osm_gps.npy')).astype(np.float32)
osm_fts = np.load(str(RAW / 'osm_features.npy')).astype(np.float32)
gc_loc_gps = np.concatenate([wd_gps, osm_gps], axis=0)
gc_loc_fts = norm_f(np.concatenate([wd_fts, osm_fts], axis=0))
del wd_gps, wd_fts, osm_gps, osm_fts; gc.collect()
print(f"  GeoCLIP location gallery: {len(gc_loc_gps):,}", flush=True)

# ME dense gallery (GeoCLIP location encoder embeddings)
me_gps = np.load(str(ROOT / 'data/raw/me_landmarks/me_gps_dense.npy')).astype(np.float32)
me_fts = np.load(str(ROOT / 'data/raw/me_landmarks/me_features_dense.npy')).astype(np.float32)
iraq_count = ((me_gps[:,1]>39)&(me_gps[:,1]<49)&(me_gps[:,0]>29)&(me_gps[:,0]<37)).sum()
print(f"  ME dense gallery: {len(me_gps):,} ({iraq_count} Iraq)", flush=True)

# OSV-5M (50K CLIP features)
osv_gps = np.load(str(ROOT / 'data/raw/osv5m_clean/clip_gps_00.npy')).astype(np.float32)
osv_fts = np.load(str(ROOT / 'data/raw/osv5m_clean/clip_features_00.npy')).astype(np.float32)
print(f"  OSV-5M gallery: {len(osv_gps):,}", flush=True)

# GeoCLIP image gallery (3000)
img_gps = np.load(str(RAW / 'coordinates.npy')).astype(np.float32)
img_fts = norm_f(np.load(str(RAW / 'fresh_features_all_3000.npy')).astype(np.float32))
print(f"  GeoCLIP image gallery: {len(img_gps):,}", flush=True)

print(f"  Total setup: {time.time()-t0:.1f}s", flush=True)

# ---- Build FAISS indexes ----
print("\n[3/5] Building FAISS indexes...", flush=True)
t0 = time.time()
dim = gc_loc_fts.shape[1]

gc_loc_idx = faiss.IndexFlatIP(dim); gc_loc_idx.add(gc_loc_fts)
me_idx = faiss.IndexFlatIP(dim); me_idx.add(me_fts)
osv_idx = faiss.IndexFlatIP(osv_fts.shape[1]); osv_idx.add(osv_fts)
img_idx = faiss.IndexFlatIP(img_fts.shape[1]); img_idx.add(img_fts)
print(f"  Done: {time.time()-t0:.1f}s", flush=True)

# ---- Extract features ----
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        px = gc_model.image_encoder.preprocess_image([img])
        gc_f = (gc_model.image_encoder(px) / gc_model.image_encoder(px).norm()).cpu().numpy().astype(np.float32)
        t = clip_pre(img).unsqueeze(0)
        cl_f = (clip_model.encode_image(t) / clip_model.encode_image(t).norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
    return gc_f, cl_f

def predict(gc_f, cl_f, verbose=True):
    K = 100

    D_l, I_l = gc_loc_idx.search(gc_f, K)
    D_m, I_m = me_idx.search(gc_f, K)
    D_o, I_o = osv_idx.search(cl_f, K)
    D_i, I_i = img_idx.search(gc_f, 50)

    gc_loc_pts = gc_loc_gps[I_l[0]]
    gc_loc_w = np.maximum(D_l[0], 0) + 1e-8
    me_pts = me_gps[I_m[0]]
    me_w = np.maximum(D_m[0], 0) + 1e-8
    osv_pts = osv_gps[I_o[0]]
    osv_w = np.maximum(D_o[0], 0) + 1e-8
    img_pts = img_gps[I_i[0]]
    img_w = np.maximum(D_i[0], 0) + 1e-8

    top_sim_gc = float(D_l[0, 0])
    top_sim_me = float(D_m[0, 0])
    spread = float(np.linalg.norm(gc_loc_pts[:10].std(axis=0) * 111.32))
    spread_me = float(np.linalg.norm(me_pts[:10].std(axis=0) * 111.32))

    # Density votes from different galleries at different radii
    gc_pred_25 = density_vote(gc_loc_pts, gc_loc_w, 25)
    gc_pred_50 = density_vote(gc_loc_pts, gc_loc_w, 50)
    me_pred_10 = density_vote(me_pts, me_w, 10)
    me_pred_25 = density_vote(me_pts, me_w, 25)
    osv_pred_200 = density_vote(osv_pts, osv_w, 200)
    img_pred = density_vote(img_pts, img_w, 50)

    if verbose:
        print(f"\n  [GeoCLIP Location] top-3:", flush=True)
        for j in range(3):
            print(f"    {gc_loc_gps[I_l[0,j],0]:.4f}, {gc_loc_gps[I_l[0,j],1]:.4f}  sim={D_l[0,j]:.4f}", flush=True)
        print(f"  [ME Gallery] top-3:", flush=True)
        for j in range(3):
            lat, lon = me_gps[I_m[0,j],0], me_gps[I_m[0,j],1]
            d = haversine(lat, lon, 0, 0) * 0  # placeholder
            print(f"    {lat:.4f}, {lon:.4f}  sim={D_m[0,j]:.4f}", flush=True)
        print(f"  [OSV-5M] top-3:", flush=True)
        for j in range(3):
            print(f"    {osv_gps[I_o[0,j],0]:.4f}, {osv_gps[I_o[0,j],1]:.4f}  sim={D_o[0,j]:.4f}", flush=True)

    # Decision logic: use density vote quality from ME gallery
    # The ME gallery uses GeoCLIP location encoder embeddings, so it's the right
    # space to search with GeoCLIP image features. Check if ME density vote
    # produces a tight cluster.
    me_vote_quality = 0
    for R in [5, 10, 25]:
        p = density_vote(me_pts, me_w, R)
        # How many ME points are within R km of the vote?
        d2vote = haversine(me_pts[:, 0], me_pts[:, 1], p[0], p[1])
        nearby = (d2vote < R).sum()
        me_vote_quality = max(me_vote_quality, nearby)
    
    # Use ME if density vote finds a cluster
    if me_vote_quality >= 3 and top_sim_me > 0.25:
        pred = me_pred_25
        method = "ME-dense"
        conf = "HIGH"
    elif spread < 100 and top_sim_gc > 0.40:
        pred = gc_pred_25
        method = "GeoCLIP"
        conf = "HIGH"
    elif spread < 300:
        pred = gc_pred_50
        method = "GeoCLIP-wide"
        conf = "MEDIUM"
    else:
        pred = gc_pred_50
        method = "GeoCLIP-uncertain"
        conf = "LOW"

    return {
        'pred': pred, 'method': method, 'confidence': conf,
        'spread': spread, 'top_sim_gc': top_sim_gc, 'top_sim_me': top_sim_me,
        'me_pred': me_pred_25, 'gc_pred': gc_pred_25, 'osv_pred': osv_pred_200,
    }

# ---- Process images ----
print("\n[4/5] Processing images...", flush=True)

images = [
    ('Karbala, Iraq', r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-47.jpg', 32.6164, 44.0323),
    ('Giza, Egypt', r'C:\Users\bi1\Downloads\Telegram Desktop\photo_2026-07-21_08-27-38.jpg', 29.85, 31.24),
]

for label, path, true_lat, true_lon in images:
    t_img = time.time()
    gc_f, cl_f = extract_features(path)
    result = predict(gc_f, cl_f)

    err_me = haversine(result['me_pred'][0], result['me_pred'][1], true_lat, true_lon)
    err_gc = haversine(result['gc_pred'][0], result['gc_pred'][1], true_lat, true_lon)
    err_final = haversine(result['pred'][0], result['pred'][1], true_lat, true_lon)

    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  True:     {true_lat:.4f}, {true_lon:.4f}", flush=True)
    print(f"  ME:       {result['me_pred'][0]:.4f}, {result['me_pred'][1]:.4f}  err={err_me:.0f}km", flush=True)
    print(f"  GeoCLIP:  {result['gc_pred'][0]:.4f}, {result['gc_pred'][1]:.4f}  err={err_gc:.0f}km", flush=True)
    print(f"  FINAL:    {result['pred'][0]:.4f}, {result['pred'][1]:.4f}  err={err_final:.0f}km", flush=True)
    print(f"  Method:   {result['method']}", flush=True)
    print(f"  Confidence: {result['confidence']}", flush=True)
    print(f"  Time:     {time.time()-t_img:.1f}s", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"  TOTAL TIME: {time.time()-t_total:.1f}s", flush=True)
print(f"{'='*70}", flush=True)
