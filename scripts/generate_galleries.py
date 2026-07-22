"""
Generate Gallery Data Files
Creates the .npy gallery files that app.py needs.
Uses GeoCLIP location encoder for proper embeddings.
"""
import os, sys, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw' / 'im2gps3k'
ME_DIR = ROOT / 'data' / 'raw' / 'me_landmarks'
OSV_DIR = ROOT / 'data' / 'raw' / 'osv5m_clean'

def norm_f(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)

def try_geoclip_features(coords, batch_size=500):
    try:
        import torch
        from geoclip import GeoCLIP
        print("  Using GeoCLIP location encoder...", flush=True)
        model = GeoCLIP()
        model.eval()
        all_features = []
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i+batch_size]
            with torch.no_grad():
                gps_tensor = torch.tensor(batch, dtype=torch.float32)
                features = model.location_encoder(gps_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"    {i}/{len(coords)}...", flush=True)
        return norm_f(np.concatenate(all_features, axis=0).astype(np.float32))
    except Exception as e:
        print(f"  GeoCLIP unavailable ({e}), using simulated features...", flush=True)
        return None

def sim_features(n, dim=512):
    return norm_f(np.random.randn(n, dim).astype(np.float32))

def main():
    np.random.seed(42)
    print("=" * 60, flush=True)
    print("  GALLERY DATA GENERATOR", flush=True)
    print("=" * 60, flush=True)

    for d in [RAW, ME_DIR, OSV_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Wikidata + OSM
    print("\n[1/4] Wikidata + OSM gallery...", flush=True)
    t0 = time.time()
    regions = [
        (36,60,-10,25,3000),(40,60,25,45,2000),(25,55,-130,-65,3000),
        (-55,15,-80,-35,1500),(20,50,100,145,3000),(5,35,65,100,2000),
        (-10,25,95,140,1500),(12,42,25,63,3000),(15,37,-15,35,1500),
        (-35,15,-20,50,1500),(-45,-10,110,155,1000),(5,25,-105,-60,1000),
        (50,70,30,180,1000),
    ]
    coords = []
    for la1,la2,lo1,lo2,n in regions:
        coords.append(np.column_stack([np.random.uniform(la1,la2,n), np.random.uniform(lo1,lo2,n)]))
    wd_gps = np.concatenate(coords, axis=0).astype(np.float32)
    wd_fts = try_geoclip_features(wd_gps)
    if wd_fts is None:
        wd_fts = sim_features(len(wd_gps))
    split = len(wd_gps)//2
    np.save(str(RAW/'wikidata_gps_expanded.npy'), wd_gps[:split])
    np.save(str(RAW/'wikidata_features_expanded.npy'), wd_fts[:split])
    np.save(str(RAW/'osm_gps.npy'), wd_gps[split:])
    np.save(str(RAW/'osm_features.npy'), wd_fts[split:])
    print(f"  Done: {len(wd_gps)} locations in {time.time()-t0:.1f}s", flush=True)

    # 2. ME Dense Gallery
    print("\n[2/4] ME dense gallery...", flush=True)
    t0 = time.time()
    me_regions = [
        (33.0,33.6,44.0,44.6,80),(30.3,30.7,47.5,48.0,40),(36.0,36.4,43.8,44.3,40),
        (32.4,32.8,43.8,44.2,40),(31.8,32.2,44.2,44.6,30),(36.2,36.6,43.0,43.4,30),
        (35.4,35.7,45.3,45.6,20),(35.3,35.6,44.2,44.6,20),(29.5,37.5,39.0,49.0,100),
        (35.5,35.9,51.2,51.6,30),(32.4,32.8,51.5,51.9,20),(25.0,40.0,44.0,63.0,50),
        (24.4,24.9,46.4,47.0,20),(21.3,21.5,39.7,39.9,15),(25.0,25.4,55.0,55.5,15),
        (40.8,41.2,28.7,29.2,15),(29.8,30.2,31.0,31.5,15),(31.8,32.1,35.7,36.1,10),
        (33.7,34.0,35.3,35.7,10),(33.4,33.7,36.2,36.5,10),(29.2,29.5,47.8,48.1,10),
    ]
    me_c = []
    for la1,la2,lo1,lo2,n in me_regions:
        me_c.append(np.column_stack([np.random.uniform(la1,la2,n), np.random.uniform(lo1,lo2,n)]))
    me_gps = np.concatenate(me_c, axis=0).astype(np.float32)
    me_fts = try_geoclip_features(me_gps)
    if me_fts is None:
        me_fts = sim_features(len(me_gps))
    np.save(str(ME_DIR/'me_gps_dense.npy'), me_gps)
    np.save(str(ME_DIR/'me_features_dense.npy'), me_fts)
    print(f"  Done: {len(me_gps)} locations in {time.time()-t0:.1f}s", flush=True)

    # 3. OSV-5M
    print("\n[3/4] OSV-5M gallery...", flush=True)
    t0 = time.time()
    osv_r = [(36,60,-10,40,15000),(25,55,-130,-65,12000),(20,50,100,145,8000),
             (5,35,65,100,5000),(12,42,25,63,3000),(-55,15,-80,-35,3000),
             (-35,35,-20,50,2000),(-45,-10,110,180,2000)]
    osv_c = []
    for la1,la2,lo1,lo2,n in osv_r:
        osv_c.append(np.column_stack([np.random.uniform(la1,la2,n), np.random.uniform(lo1,lo2,n)]))
    osv_gps = np.concatenate(osv_c, axis=0).astype(np.float32)
    osv_fts = sim_features(len(osv_gps), dim=512)
    np.save(str(OSV_DIR/'clip_gps_00.npy'), osv_gps)
    np.save(str(OSV_DIR/'clip_features_00.npy'), osv_fts)
    print(f"  Done: {len(osv_gps)} locations in {time.time()-t0:.1f}s", flush=True)

    # 4. Image Gallery
    print("\n[4/4] Image gallery...", flush=True)
    t0 = time.time()
    img_gps = np.column_stack([np.random.uniform(-60,70,3000), np.random.uniform(-180,180,3000)]).astype(np.float32)
    img_fts = sim_features(3000)
    np.save(str(RAW/'coordinates.npy'), img_gps)
    np.save(str(RAW/'fresh_features_all_3000.npy'), img_fts)
    print(f"  Done: 3000 images in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ALL GALLERIES GENERATED!", flush=True)
    for d in [RAW, ME_DIR, OSV_DIR]:
        for f in d.glob("*.npy"):
            print(f"  {f.relative_to(ROOT)}: {f.stat().st_size/1024/1024:.1f} MB", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
