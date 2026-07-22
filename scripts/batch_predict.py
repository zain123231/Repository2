"""
Batch predict all user images.
"""
import os, warnings, time, gc, sys
os.environ['OMP_NUM_THREADS'] = '8'; os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings('ignore')
import numpy as np; import torch; torch.set_num_threads(8)
import faiss; faiss.omp_set_num_threads(8)
from pathlib import Path; from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/im2gps3k'
ME_DIR = ROOT / 'data/raw/me_landmarks'

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

# ---- Load models ----
print("Loading models...", flush=True)
t0 = time.time()
from geoclip import GeoCLIP
gc_model = GeoCLIP(); gc_model.eval()
import open_clip
clip_model, _, clip_pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()
print(f"  Models loaded: {time.time()-t0:.1f}s", flush=True)

# ---- Load galleries ----
print("Loading galleries...", flush=True)
t0 = time.time()
wd_gps = np.load(str(RAW / 'wikidata_gps_expanded.npy')).astype(np.float32)
wd_fts = norm_f(np.load(str(RAW / 'wikidata_features_expanded.npy')).astype(np.float32))
osm_gps = np.load(str(RAW / 'osm_gps.npy')).astype(np.float32)
osm_fts = norm_f(np.load(str(RAW / 'osm_features.npy')).astype(np.float32))
gc_loc_gps = np.concatenate([wd_gps, osm_gps], axis=0)
gc_loc_fts = np.concatenate([wd_fts, osm_fts], axis=0)
del wd_gps, wd_fts, osm_gps, osm_fts; gc.collect()

me_gps = np.load(str(ME_DIR / 'me_gps_dense.npy')).astype(np.float32)
me_fts = np.load(str(ME_DIR / 'me_features_dense.npy')).astype(np.float32)

osv_gps = np.load(str(ROOT / 'data/raw/osv5m_clean/clip_gps_00.npy')).astype(np.float32)
osv_fts = np.load(str(ROOT / 'data/raw/osv5m_clean/clip_features_00.npy')).astype(np.float32)

img_gps = np.load(str(RAW / 'coordinates.npy')).astype(np.float32)
img_fts = norm_f(np.load(str(RAW / 'fresh_features_all_3000.npy')).astype(np.float32))
print(f"  Galleries loaded: {time.time()-t0:.1f}s", flush=True)

# ---- Build FAISS indexes ----
dim = gc_loc_fts.shape[1]
gc_loc_idx = faiss.IndexFlatIP(dim); gc_loc_idx.add(gc_loc_fts)
me_idx = faiss.IndexFlatIP(dim); me_idx.add(me_fts)
osv_idx = faiss.IndexFlatIP(osv_fts.shape[1]); osv_idx.add(osv_fts)
img_idx = faiss.IndexFlatIP(img_fts.shape[1]); img_idx.add(img_fts)

# ---- All images ----
gallery_dir = Path(r'C:\Users\bi1\Downloads\Telegram Desktop')
all_images = sorted(gallery_dir.glob('photo_*.jpg'))
print(f"\nFound {len(all_images)} images to process\n", flush=True)

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        px = gc_model.image_encoder.preprocess_image([img])
        gc_f = (gc_model.image_encoder(px) / gc_model.image_encoder(px).norm()).cpu().numpy().astype(np.float32)
        t = clip_pre(img).unsqueeze(0)
        cl_f = (clip_model.encode_image(t) / clip_model.encode_image(t).norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    K = 100
    D_l, I_l = gc_loc_idx.search(gc_f, K)
    D_m, I_m = me_idx.search(gc_f, K)
    D_o, I_o = osv_idx.search(cl_f, K)

    gc_pts = gc_loc_gps[I_l[0]]; gc_w = np.maximum(D_l[0], 0) + 1e-8
    me_pts = me_gps[I_m[0]]; me_w = np.maximum(D_m[0], 0) + 1e-8
    osv_pts = osv_gps[I_o[0]]; osv_w = np.maximum(D_o[0], 0) + 1e-8

    top_sim_gc = float(D_l[0, 0])
    top_sim_me = float(D_m[0, 0])
    spread = float(np.linalg.norm(gc_pts[:10].std(axis=0) * 111.32))

    gc_pred = density_vote(gc_pts, gc_w, 25)
    me_pred = density_vote(me_pts, me_w, 25)
    osv_pred = density_vote(osv_pts, osv_w, 500)

    # ME decision: count nearby points in density vote
    me_vote_quality = 0
    for R in [5, 10, 25]:
        p = density_vote(me_pts, me_w, R)
        d2vote = haversine(me_pts[:, 0], me_pts[:, 1], p[0], p[1])
        nearby = (d2vote < R).sum()
        me_vote_quality = max(me_vote_quality, nearby)

    if me_vote_quality >= 3 and top_sim_me > 0.25:
        pred = me_pred; method = "ME-dense"; conf = "HIGH"
    elif spread < 100 and top_sim_gc > 0.40:
        pred = gc_pred; method = "GeoCLIP"; conf = "HIGH"
    elif spread < 300:
        pred = gc_pred; method = "GeoCLIP-wide"; conf = "MEDIUM"
    else:
        pred = gc_pred; method = "GeoCLIP-uncertain"; conf = "LOW"

    # Find nearest country
    lat, lon = pred[0], pred[1]
    if 39 < lon < 49 and 29 < lat < 37: region = "Iraq"
    elif 25 < lon < 37 and 22 < lat < 32: region = "Egypt"
    elif 44 < lon < 63 and 25 < lat < 40: region = "Iran"
    elif 26 < lon < 45 and 36 < lat < 42: region = "Turkey"
    elif 35 < lon < 43 and 12 < lat < 18: region = "Yemen"
    elif 50 < lon < 56 and 23 < lat < 26: region = "Saudi/Emirates"
    elif 49 < lon < 54 and 24 < lat < 27: region = "Qatar/Bahrain/Kuwait"
    elif 41 < lon < 45 and 32 < lat < 35: region = "Syria"
    elif 34 < lon < 39 and 29 < lat < 34: region = "Lebanon/Jordan"
    elif 100 < lon < 130 and 5 < lat < 25: region = "SE Asia"
    elif 68 < lon < 90 and 6 < lat < 35: region = "India"
    elif -10 < lon < 3 and 35 < lat < 47: region = "Europe"
    elif -50 < lon < -30 and -50 < lat < 10: region = "S. America"
    elif -130 < lon < -60 and 15 < lat < 50: region = "N. America"
    elif 25 < lon < 40 and 0 < lat < 15: region = "Africa"
    else: region = "unknown"

    return {
        'lat': lat, 'lon': lon, 'method': method, 'confidence': conf,
        'spread': spread, 'top_gc': top_sim_gc, 'top_me': top_sim_me,
        'region': region, 'gc_pred': gc_pred, 'me_pred': me_pred,
    }

# ---- Process all ----
results = []
for i, img_path in enumerate(all_images):
    t1 = time.time()
    r = predict_image(img_path)
    r['name'] = img_path.name
    r['time'] = time.time() - t1
    results.append(r)
    print(f"[{i+1}/{len(all_images)}] {img_path.name}: {r['lat']:.4f}, {r['lon']:.4f} ({r['region']}) [{r['method']}, {r['confidence']}] {r['time']:.1f}s", flush=True)

# ---- Summary ----
print(f"\n{'='*80}", flush=True)
print(f"  SUMMARY ({len(results)} images)", flush=True)
print(f"{'='*80}", flush=True)
print(f"{'Name':<45} {'Lat':>9} {'Lon':>9}  {'Region':<15} {'Conf':<5} {'Method':<15}", flush=True)
print(f"{'-'*45} {'-'*9} {'-'*9}  {'-'*15} {'-'*5} {'-'*15}", flush=True)
for r in results:
    print(f"{r['name']:<45} {r['lat']:>9.4f} {r['lon']:>9.4f}  {r['region']:<15} {r['confidence']:<5} {r['method']:<15}", flush=True)
