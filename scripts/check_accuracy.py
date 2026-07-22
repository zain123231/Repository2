"""
========================================================
  فحص شامل لدقة جميع الموديلات والـ Galleries
  يفحص كل شي متوفر بالمشروع بدون الحاجة لصور خارجية
========================================================
"""
import sys, os, time, json, traceback
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ── Setup paths ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Colors for terminal ─────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    END    = "\033[0m"

def ok(msg):   print(f"  {C.GREEN}✓{C.END} {msg}")
def fail(msg): print(f"  {C.RED}✗{C.END} {msg}")
def warn(msg): print(f"  {C.YELLOW}⚠{C.END} {msg}")
def info(msg): print(f"  {C.CYAN}ℹ{C.END} {msg}")
def header(msg):
    print(f"\n{C.BOLD}{'='*65}{C.END}")
    print(f"  {C.BOLD}{C.CYAN}{msg}{C.END}")
    print(f"{C.BOLD}{'='*65}{C.END}")

# Track results
ALL_RESULTS = OrderedDict()
ERRORS = []

def record(section, key, value, unit=""):
    if section not in ALL_RESULTS:
        ALL_RESULTS[section] = OrderedDict()
    ALL_RESULTS[section][key] = f"{value}{unit}" if unit else value

# ═══════════════════════════════════════════════════════════
# 1. CHECK PROJECT FILES
# ═══════════════════════════════════════════════════════════
def check_project_files():
    header("1. فحص ملفات المشروع")

    RAW = PROJECT_ROOT / "data" / "raw"

    galleries = {
        "GeoCLIP Wikidata Features":  RAW / "im2gps3k" / "wikidata_features_expanded.npy",
        "GeoCLIP Wikidata GPS":       RAW / "im2gps3k" / "wikidata_gps_expanded.npy",
        "GeoCLIP OSM Features":       RAW / "im2gps3k" / "osm_features.npy",
        "GeoCLIP OSM GPS":            RAW / "im2gps3k" / "osm_gps.npy",
        "GeoCLIP Image Features":     RAW / "im2gps3k" / "fresh_features_all_3000.npy",
        "GeoCLIP Image GPS":          RAW / "im2gps3k" / "coordinates.npy",
        "ME Dense Features":          RAW / "me_landmarks" / "me_features_dense.npy",
        "ME Dense GPS":               RAW / "me_landmarks" / "me_gps_dense.npy",
        "OSV-5M CLIP Features":       RAW / "osv5m_clean" / "clip_features_00.npy",
        "OSV-5M CLIP GPS":            RAW / "osv5m_clean" / "clip_gps_00.npy",
    }

    checkpoints = {
        "Quadtree JSON": PROJECT_ROOT / "checkpoints" / "quadtree.json",
    }

    source_files = {
        "config.py":            PROJECT_ROOT / "src" / "config.py",
        "metrics.py":           PROJECT_ROOT / "src" / "evaluation" / "metrics.py",
        "faiss_index.py":       PROJECT_ROOT / "src" / "models" / "faiss_index.py",
        "quadtree.py":          PROJECT_ROOT / "src" / "models" / "quadtree.py",
        "geoclip.py":           PROJECT_ROOT / "src" / "models" / "geoclip.py",
        "hybrid.py":            PROJECT_ROOT / "src" / "models" / "hybrid.py",
        "streetclip.py":        PROJECT_ROOT / "src" / "models" / "streetclip.py",
        "feature_extractor.py": PROJECT_ROOT / "src" / "models" / "feature_extractor.py",
        "app.py":               PROJECT_ROOT / "app.py",
    }

    print(f"\n  {'─'*60}")
    print(f"  {C.BOLD}Gallery Files (بيانات البحث){C.END}")
    print(f"  {'─'*60}")
    found_count = 0
    for name, path in galleries.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            ok(f"{name:40s} {size_mb:>8.2f} MB")
            found_count += 1
        else:
            fail(f"{name:40s} {'مفقود':>8s}")
    record("Files", "Gallery Files Found", f"{found_count}/{len(galleries)}")

    print(f"\n  {'─'*60}")
    print(f"  {C.BOLD}Checkpoints (نقاط حفظ الموديل){C.END}")
    print(f"  {'─'*60}")
    for name, path in checkpoints.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            ok(f"{name:40s} {size_kb:>8.1f} KB")
        else:
            fail(f"{name:40s} {'مفقود':>8s}")

    print(f"\n  {'─'*60}")
    print(f"  {C.BOLD}Source Code (الكود المصدري){C.END}")
    print(f"  {'─'*60}")
    src_found = 0
    for name, path in source_files.items():
        if path.exists():
            ok(f"{name}")
            src_found += 1
        else:
            fail(f"{name}")
    record("Files", "Source Files Found", f"{src_found}/{len(source_files)}")


# ═══════════════════════════════════════════════════════════
# 2. LOAD & VALIDATE GALLERIES
# ═══════════════════════════════════════════════════════════
def check_galleries():
    header("2. فحص الـ Galleries (بيانات البحث)")

    RAW = PROJECT_ROOT / "data" / "raw"

    gallery_data = {}

    # --- GeoCLIP Location Gallery ---
    print(f"\n  {C.BOLD}[A] GeoCLIP Location Gallery (Wikidata + OSM){C.END}")
    try:
        wd_gps = np.load(str(RAW / "im2gps3k" / "wikidata_gps_expanded.npy"))
        wd_fts = np.load(str(RAW / "im2gps3k" / "wikidata_features_expanded.npy"))
        osm_gps = np.load(str(RAW / "im2gps3k" / "osm_gps.npy"))
        osm_fts = np.load(str(RAW / "im2gps3k" / "osm_features.npy"))

        gc_gps = np.concatenate([wd_gps, osm_gps], axis=0).astype(np.float32)
        gc_fts = np.concatenate([wd_fts, osm_fts], axis=0).astype(np.float32)

        ok(f"Wikidata:  {wd_gps.shape[0]:>8,} نقطة  |  Features dim: {wd_fts.shape[1]}")
        ok(f"OSM:       {osm_gps.shape[0]:>8,} نقطة  |  Features dim: {osm_fts.shape[1]}")
        ok(f"المجموع:   {gc_gps.shape[0]:>8,} نقطة")

        # Validate GPS ranges
        lat_range = (gc_gps[:,0].min(), gc_gps[:,0].max())
        lon_range = (gc_gps[:,1].min(), gc_gps[:,1].max())
        ok(f"Lat range: [{lat_range[0]:.2f}, {lat_range[1]:.2f}]")
        ok(f"Lon range: [{lon_range[0]:.2f}, {lon_range[1]:.2f}]")

        # Check for NaN/Inf
        nan_count = np.isnan(gc_fts).sum() + np.isnan(gc_gps).sum()
        inf_count = np.isinf(gc_fts).sum() + np.isinf(gc_gps).sum()
        if nan_count == 0 and inf_count == 0:
            ok(f"لا يوجد NaN أو Inf ✓")
        else:
            fail(f"NaN: {nan_count}, Inf: {inf_count}")

        # Feature norms
        norms = np.linalg.norm(gc_fts, axis=1)
        ok(f"Feature norms: mean={norms.mean():.4f}, std={norms.std():.4f}")

        gallery_data['gc'] = (gc_gps, gc_fts)
        record("Galleries", "GeoCLIP Location", f"{gc_gps.shape[0]:,} نقطة, dim={gc_fts.shape[1]}")
    except Exception as e:
        fail(f"فشل تحميل GeoCLIP gallery: {e}")
        ERRORS.append(f"GeoCLIP gallery: {e}")

    # --- ME Dense Gallery ---
    print(f"\n  {C.BOLD}[B] Middle East Dense Gallery{C.END}")
    try:
        me_gps = np.load(str(RAW / "me_landmarks" / "me_gps_dense.npy")).astype(np.float32)
        me_fts = np.load(str(RAW / "me_landmarks" / "me_features_dense.npy")).astype(np.float32)

        ok(f"النقاط:    {me_gps.shape[0]:>8,} نقطة  |  Features dim: {me_fts.shape[1]}")

        # Count by region
        iraq = ((me_gps[:,1]>39)&(me_gps[:,1]<49)&(me_gps[:,0]>29)&(me_gps[:,0]<37)).sum()
        iran = ((me_gps[:,1]>44)&(me_gps[:,1]<64)&(me_gps[:,0]>25)&(me_gps[:,0]<40)).sum()
        info(f"Iraq points:  ~{iraq}")
        info(f"Iran overlap: ~{iran}")

        lat_range = (me_gps[:,0].min(), me_gps[:,0].max())
        lon_range = (me_gps[:,1].min(), me_gps[:,1].max())
        ok(f"Lat range: [{lat_range[0]:.2f}, {lat_range[1]:.2f}]")
        ok(f"Lon range: [{lon_range[0]:.2f}, {lon_range[1]:.2f}]")

        gallery_data['me'] = (me_gps, me_fts)
        record("Galleries", "ME Dense", f"{me_gps.shape[0]:,} نقطة, dim={me_fts.shape[1]}")
    except Exception as e:
        fail(f"فشل تحميل ME gallery: {e}")
        ERRORS.append(f"ME gallery: {e}")

    # --- OSV-5M Gallery ---
    print(f"\n  {C.BOLD}[C] OSV-5M CLIP Gallery{C.END}")
    try:
        osv_gps = np.load(str(RAW / "osv5m_clean" / "clip_gps_00.npy")).astype(np.float32)
        osv_fts = np.load(str(RAW / "osv5m_clean" / "clip_features_00.npy")).astype(np.float32)

        ok(f"النقاط:    {osv_gps.shape[0]:>8,} نقطة  |  Features dim: {osv_fts.shape[1]}")

        lat_range = (osv_gps[:,0].min(), osv_gps[:,0].max())
        lon_range = (osv_gps[:,1].min(), osv_gps[:,1].max())
        ok(f"Lat range: [{lat_range[0]:.2f}, {lat_range[1]:.2f}]")
        ok(f"Lon range: [{lon_range[0]:.2f}, {lon_range[1]:.2f}]")

        gallery_data['osv'] = (osv_gps, osv_fts)
        record("Galleries", "OSV-5M CLIP", f"{osv_gps.shape[0]:,} نقطة, dim={osv_fts.shape[1]}")
    except Exception as e:
        fail(f"فشل تحميل OSV-5M gallery: {e}")
        ERRORS.append(f"OSV-5M gallery: {e}")

    # --- GeoCLIP Image Gallery ---
    print(f"\n  {C.BOLD}[D] GeoCLIP Image Gallery (3K){C.END}")
    try:
        img_gps = np.load(str(RAW / "im2gps3k" / "coordinates.npy")).astype(np.float32)
        img_fts = np.load(str(RAW / "im2gps3k" / "fresh_features_all_3000.npy")).astype(np.float32)

        ok(f"النقاط:    {img_gps.shape[0]:>8,} نقطة  |  Features dim: {img_fts.shape[1]}")

        gallery_data['img'] = (img_gps, img_fts)
        record("Galleries", "GeoCLIP Image", f"{img_gps.shape[0]:,} نقطة, dim={img_fts.shape[1]}")
    except Exception as e:
        fail(f"فشل تحميل Image gallery: {e}")
        ERRORS.append(f"Image gallery: {e}")

    return gallery_data


# ═══════════════════════════════════════════════════════════
# 3. EVALUATION METRICS TEST
# ═══════════════════════════════════════════════════════════
def check_metrics():
    header("3. فحص دوال التقييم (Evaluation Metrics)")

    try:
        from src.evaluation.metrics import (
            haversine, haversine_single, acc_at_tau,
            median_error, mean_error, evaluate_system, smooth_labels
        )
        ok("تم استيراد جميع دوال التقييم")
    except ImportError as e:
        fail(f"فشل استيراد: {e}")
        ERRORS.append(f"Metrics import: {e}")
        return

    tests_passed = 0
    total_tests = 0

    # Test 1: Haversine zero distance
    total_tests += 1
    d = haversine_single(40.7128, -74.0060, 40.7128, -74.0060)
    if d < 0.001:
        ok(f"Haversine (نفس النقطة) = {d:.6f} km")
        tests_passed += 1
    else:
        fail(f"Haversine (نفس النقطة) = {d} (المتوقع ~0)")

    # Test 2: Known distances
    total_tests += 1
    d_ny_london = haversine_single(40.7128, -74.0060, 51.5074, -0.1278)
    if 5500 < d_ny_london < 5700:
        ok(f"NY → London = {d_ny_london:.0f} km (المتوقع ~5,570)")
        tests_passed += 1
    else:
        fail(f"NY → London = {d_ny_london:.0f} km (المتوقع ~5,570)")

    total_tests += 1
    d_london_paris = haversine_single(51.5074, -0.1278, 48.8566, 2.3522)
    if 330 < d_london_paris < 360:
        ok(f"London → Paris = {d_london_paris:.0f} km (المتوقع ~343)")
        tests_passed += 1
    else:
        fail(f"London → Paris = {d_london_paris:.0f} km (المتوقع ~343)")

    total_tests += 1
    d_karbala_baghdad = haversine_single(32.6164, 44.0323, 33.3152, 44.3661)
    if 70 < d_karbala_baghdad < 90:
        ok(f"كربلاء → بغداد = {d_karbala_baghdad:.0f} km (المتوقع ~80)")
        tests_passed += 1
    else:
        fail(f"كربلاء → بغداد = {d_karbala_baghdad:.0f} km (المتوقع ~80)")

    # Test 3: Symmetry
    total_tests += 1
    d1 = haversine_single(40.7128, -74.0060, 51.5074, -0.1278)
    d2 = haversine_single(51.5074, -0.1278, 40.7128, -74.0060)
    if abs(d1 - d2) < 0.001:
        ok(f"Haversine symmetry: |{d1:.2f} - {d2:.2f}| < 0.001 ✓")
        tests_passed += 1
    else:
        fail(f"Haversine NOT symmetric: {d1} vs {d2}")

    # Test 4: Batch haversine
    total_tests += 1
    pred = np.array([[40.7128, -74.0060], [48.8566, 2.3522], [32.6164, 44.0323]])
    gt   = np.array([[40.7128, -74.0060], [48.8566, 2.3522], [32.6164, 44.0323]])
    dists = haversine(pred[:,0], pred[:,1], gt[:,0], gt[:,1])
    if all(d < 0.001 for d in dists):
        ok(f"Batch haversine (3 identical pairs) → all ~0 ✓")
        tests_passed += 1
    else:
        fail(f"Batch haversine: {dists}")

    # Test 5: Acc@tau
    total_tests += 1
    distances = np.array([0.5, 10, 50, 100, 500, 1000])
    accs = acc_at_tau(distances, [1, 25, 200, 750, 2500])
    expected = {1.0: 1/6, 25.0: 2/6, 200.0: 4/6, 750.0: 5/6, 2500.0: 1.0}
    all_ok = True
    for tau, exp_val in expected.items():
        if abs(accs[tau] - exp_val) > 0.001:
            fail(f"Acc@{tau}km = {accs[tau]:.4f} (المتوقع {exp_val:.4f})")
            all_ok = False
    if all_ok:
        ok(f"Acc@τ: @1km={accs[1.0]*100:.1f}%  @25km={accs[25.0]*100:.1f}%  @200km={accs[200.0]*100:.1f}%  @750km={accs[750.0]*100:.1f}%  @2500km={accs[2500.0]*100:.1f}%")
        tests_passed += 1

    # Test 6: Median/Mean error
    total_tests += 1
    med = median_error(np.array([10, 20, 30, 40, 50]))
    mn = mean_error(np.array([10, 20, 30, 40, 50]))
    if abs(med - 30.0) < 0.001 and abs(mn - 30.0) < 0.001:
        ok(f"Median error = {med}, Mean error = {mn} ✓")
        tests_passed += 1
    else:
        fail(f"Median={med}, Mean={mn}")

    # Test 7: Label smoothing
    total_tests += 1
    centers = np.random.RandomState(42).uniform(-90, 90, (20, 2))
    labels = smooth_labels(centers, np.array([40.0, -74.0]), temperature=0.1)
    if abs(labels.sum() - 1.0) < 1e-6 and np.all(labels >= 0):
        ok(f"Label smoothing: sum={labels.sum():.8f}, all>=0 ✓")
        tests_passed += 1
    else:
        fail(f"Label smoothing invalid: sum={labels.sum()}")

    record("Metrics", "Tests Passed", f"{tests_passed}/{total_tests}")
    return tests_passed == total_tests


# ═══════════════════════════════════════════════════════════
# 4. FAISS INDEX TEST
# ═══════════════════════════════════════════════════════════
def check_faiss(gallery_data):
    header("4. فحص FAISS Index (محرك البحث)")

    try:
        import faiss
        ok(f"FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'installed'}")
    except ImportError:
        fail("FAISS غير مثبت!")
        ERRORS.append("FAISS not installed")
        return

    from src.evaluation.metrics import haversine

    def norm_f(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-8)

    for gname, (gps, fts) in gallery_data.items():
        label = {"gc": "GeoCLIP Location", "me": "ME Dense", "osv": "OSV-5M", "img": "GeoCLIP Image"}.get(gname, gname)
        print(f"\n  {C.BOLD}Building FAISS for: {label}{C.END}")

        t0 = time.time()
        fts_norm = norm_f(fts.astype(np.float32))
        dim = fts_norm.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(fts_norm)
        build_time = time.time() - t0
        ok(f"Build: {idx.ntotal:,} vectors, dim={dim}, time={build_time:.2f}s")

        # Self-search: query with gallery items → top-1 should be itself
        n_test = min(100, len(gps))
        test_indices = np.random.RandomState(42).choice(len(gps), n_test, replace=False)
        test_fts = fts_norm[test_indices]

        t0 = time.time()
        D, I = idx.search(test_fts, 1)
        search_time = (time.time() - t0) / n_test * 1000

        self_match = (I[:,0] == test_indices).sum()
        ok(f"Self-retrieval: {self_match}/{n_test} exact matches ({self_match/n_test*100:.0f}%)")
        ok(f"Search time: {search_time:.2f} ms/query")

        # kNN accuracy test: search with k=5, measure GPS error
        D5, I5 = idx.search(test_fts, 5)
        errors = []
        for i in range(n_test):
            true_gps = gps[test_indices[i]]
            nn_gps = gps[I5[i]]
            pred = np.median(nn_gps, axis=0)  # median of 5-NN
            err = float(haversine(
                np.array([pred[0]]), np.array([pred[1]]),
                np.array([true_gps[0]]), np.array([true_gps[1]])
            ))
            errors.append(err)

        errors = np.array(errors)
        med_err = np.median(errors)
        mean_err = np.mean(errors)
        p90_err = np.percentile(errors, 90)

        ok(f"kNN (k=5) self-test:")
        info(f"  Median error:  {med_err:>10.1f} km")
        info(f"  Mean error:    {mean_err:>10.1f} km")
        info(f"  90th pct:      {p90_err:>10.1f} km")
        info(f"  < 1 km:        {(errors < 1).mean()*100:>10.1f}%")
        info(f"  < 25 km:       {(errors < 25).mean()*100:>10.1f}%")
        info(f"  < 200 km:      {(errors < 200).mean()*100:>10.1f}%")

        record(f"FAISS-{label}", "Vectors", f"{idx.ntotal:,}")
        record(f"FAISS-{label}", "Self-match", f"{self_match/n_test*100:.0f}%")
        record(f"FAISS-{label}", "Median kNN err", f"{med_err:.1f} km")
        record(f"FAISS-{label}", "Search time", f"{search_time:.2f} ms")


# ═══════════════════════════════════════════════════════════
# 5. QUADTREE TEST
# ═══════════════════════════════════════════════════════════
def check_quadtree(gallery_data):
    header("5. فحص Quadtree (التقسيم الجغرافي)")

    qt_path = PROJECT_ROOT / "checkpoints" / "quadtree.json"

    from src.models.quadtree import GeographicQuadtree
    from src.evaluation.metrics import haversine

    # Test with saved quadtree
    if qt_path.exists():
        print(f"\n  {C.BOLD}[A] Quadtree المحفوظ{C.END}")
        qt = GeographicQuadtree()
        qt.load(str(qt_path))
        n_cells = len(qt.cells)
        ok(f"Loaded {n_cells} cells")

        centers = qt.get_cell_centers()
        sizes = qt.get_cell_sizes()
        ok(f"Total samples across cells: {sizes.sum():,}")
        ok(f"Cell sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.0f}")

        # Test nearest cell lookup
        test_points = [
            ("بغداد", 33.3152, 44.3661),
            ("كربلاء", 32.6164, 44.0323),
            ("القاهرة", 30.0444, 31.2357),
            ("نيويورك", 40.7128, -74.0060),
            ("طوكيو", 35.6762, 139.6503),
        ]

        print(f"\n  {C.BOLD}Cell Assignment Test:{C.END}")
        for name, lat, lon in test_points:
            nearest, dists = qt.get_nearest_cells(lat, lon, k=1)
            cell = nearest[0]
            info(f"  {name:10s} → Cell #{cell.cell_id:3d}  dist={dists[0]:>8.0f} km  "
                 f"center=({cell.center_lat:.2f}, {cell.center_lon:.2f})")

        record("Quadtree", "Cells", n_cells)
        record("Quadtree", "Total Samples", f"{sizes.sum():,}")
    else:
        warn("quadtree.json غير موجود - سيتم بناء واحد جديد")

    # Test building new quadtree from gallery data
    if 'gc' in gallery_data:
        print(f"\n  {C.BOLD}[B] بناء Quadtree جديد من Gallery{C.END}")
        gc_gps = gallery_data['gc'][0]
        qt_new = GeographicQuadtree(max_cells=64, min_samples_per_cell=20, max_depth=6)
        t0 = time.time()
        qt_new.build(gc_gps)
        build_time = time.time() - t0

        centers_new = qt_new.get_cell_centers()
        sizes_new = qt_new.get_cell_sizes()
        ok(f"Built {len(qt_new.cells)} cells in {build_time:.2f}s")
        ok(f"Coverage: {sizes_new.sum():,} / {len(gc_gps):,} points")

        # Centroid baseline: predict cell center for each point
        n_test = min(500, len(gc_gps))
        test_idx = np.random.RandomState(42).choice(len(gc_gps), n_test, replace=False)
        errors = []
        for i in test_idx:
            lat, lon = gc_gps[i]
            nearest, _ = qt_new.get_nearest_cells(lat, lon, k=1)
            pred = nearest[0].predict()
            err = float(haversine(
                np.array([pred[0]]), np.array([pred[1]]),
                np.array([lat]), np.array([lon])
            ))
            errors.append(err)
        errors = np.array(errors)
        ok(f"Cell-centroid baseline ({n_test} samples):")
        info(f"  Median error: {np.median(errors):.1f} km")
        info(f"  Acc@25km:     {(errors<25).mean()*100:.1f}%")
        info(f"  Acc@200km:    {(errors<200).mean()*100:.1f}%")

        record("Quadtree-New", "Cells", len(qt_new.cells))
        record("Quadtree-New", "Centroid Median Error", f"{np.median(errors):.1f} km")


# ═══════════════════════════════════════════════════════════
# 6. BASELINE ACCURACY BENCHMARKS
# ═══════════════════════════════════════════════════════════
def check_baselines(gallery_data):
    header("6. فحص دقة الـ Baselines")

    from src.evaluation.metrics import haversine, evaluate_system

    np.random.seed(42)

    # Use GeoCLIP image gallery as test set (we know the GPS for these)
    if 'img' not in gallery_data:
        warn("Image gallery غير متوفر - لا يمكن تشغيل الـ baselines")
        return

    img_gps = gallery_data['img'][0]
    n = len(img_gps)

    print(f"\n  Test set: {n} images from GeoCLIP Image Gallery")
    print(f"  {'─'*55}")

    # Baseline 1: Random predictions
    print(f"\n  {C.BOLD}[A] Random Baseline{C.END}")
    pred_random = np.column_stack([
        np.random.uniform(-90, 90, n),
        np.random.uniform(-180, 180, n)
    ])
    result = evaluate_system(pred_random, img_gps)
    info(f"  Median Error: {result['median_error_km']:>10.0f} km")
    info(f"  Acc@1km:      {result.get('acc@1.0km', result.get('acc@1km', 0))*100:>10.2f}%")
    info(f"  Acc@25km:     {result.get('acc@25.0km', result.get('acc@25km', 0))*100:>10.2f}%")
    info(f"  Acc@200km:    {result.get('acc@200.0km', result.get('acc@200km', 0))*100:>10.2f}%")
    info(f"  Acc@2500km:   {result.get('acc@2500.0km', result.get('acc@2500km', 0))*100:>10.2f}%")
    record("Baselines", "Random Median Error", f"{result['median_error_km']:.0f} km")

    # Baseline 2: Global centroid (0, 0)
    print(f"\n  {C.BOLD}[B] Global Centroid (0°, 0°){C.END}")
    pred_centroid = np.zeros((n, 2))
    result = evaluate_system(pred_centroid, img_gps)
    info(f"  Median Error: {result['median_error_km']:>10.0f} km")
    record("Baselines", "Centroid Median Error", f"{result['median_error_km']:.0f} km")

    # Baseline 3: Data-mean centroid
    print(f"\n  {C.BOLD}[C] Data Mean Centroid{C.END}")
    data_mean = img_gps.mean(axis=0)
    pred_mean = np.tile(data_mean, (n, 1))
    result = evaluate_system(pred_mean, img_gps)
    info(f"  Mean point: ({data_mean[0]:.2f}, {data_mean[1]:.2f})")
    info(f"  Median Error: {result['median_error_km']:>10.0f} km")
    record("Baselines", "Data-Mean Median Error", f"{result['median_error_km']:.0f} km")

    # Baseline 4: kNN self-retrieval (perfect retrieval upper bound)
    if 'gc' in gallery_data:
        print(f"\n  {C.BOLD}[D] kNN Cross-Gallery Retrieval{C.END}")
        print(f"      Query: Image Gallery → Search: Location Gallery")
        gc_gps, gc_fts = gallery_data['gc']
        img_fts = gallery_data['img'][1]

        import faiss
        def norm_f(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.maximum(n, 1e-8)

        gc_fts_n = norm_f(gc_fts.astype(np.float32))
        img_fts_n = norm_f(img_fts.astype(np.float32))

        # Match dimensions
        if gc_fts_n.shape[1] != img_fts_n.shape[1]:
            warn(f"Dimension mismatch: gallery={gc_fts_n.shape[1]}, query={img_fts_n.shape[1]}")
            warn("Cannot run cross-gallery kNN test")
        else:
            dim = gc_fts_n.shape[1]
            idx = faiss.IndexFlatIP(dim)
            idx.add(gc_fts_n)

            for K in [1, 5, 10, 20]:
                D, I = idx.search(img_fts_n, K)
                preds = []
                for i in range(n):
                    nn_gps = gc_gps[I[i]]
                    # Weighted by similarity
                    w = np.maximum(D[i], 0) + 1e-8
                    pred = (w[:, None] * nn_gps).sum(0) / w.sum()
                    preds.append(pred)
                preds = np.array(preds)

                dists = haversine(preds[:,0], preds[:,1], img_gps[:,0], img_gps[:,1])
                med = np.median(dists)
                acc25 = (dists < 25).mean() * 100
                acc200 = (dists < 200).mean() * 100
                acc2500 = (dists < 2500).mean() * 100

                info(f"  k={K:>2d}: Median={med:>8.0f}km  @25km={acc25:>5.1f}%  @200km={acc200:>5.1f}%  @2500km={acc2500:>5.1f}%")

            record("Baselines", "kNN-5 Cross-Gallery Median", f"{np.median(dists):.0f} km")

    # Baseline 5: ME gallery self-test
    if 'me' in gallery_data:
        print(f"\n  {C.BOLD}[E] ME Gallery Self-Test (Leave-one-out){C.END}")
        me_gps, me_fts = gallery_data['me']

        import faiss
        def norm_f(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.maximum(n, 1e-8)

        me_fts_n = norm_f(me_fts.astype(np.float32))
        dim = me_fts_n.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(me_fts_n)

        # k=2 because first match will be itself
        D, I = idx.search(me_fts_n, 2)
        preds = me_gps[I[:,1]]  # second nearest (skip self)

        dists = haversine(preds[:,0], preds[:,1], me_gps[:,0], me_gps[:,1])
        med = np.median(dists)
        info(f"  Nearest-neighbor (leave-one-out):")
        info(f"  Median Error: {med:>8.1f} km")
        info(f"  Acc@25km:     {(dists<25).mean()*100:>8.1f}%")
        info(f"  Acc@200km:    {(dists<200).mean()*100:>8.1f}%")
        record("Baselines", "ME Leave-1-out Median", f"{med:.1f} km")


# ═══════════════════════════════════════════════════════════
# 7. DENSITY VOTE ALGORITHM TEST
# ═══════════════════════════════════════════════════════════
def check_density_vote():
    header("7. فحص خوارزمية Density Voting")

    from src.evaluation.metrics import haversine_single

    def density_vote(pts, w, R_km):
        if len(pts) < 2:
            return pts[0] if len(pts) > 0 else np.array([0., 0.])
        pts_km = pts * 111.32
        dists = np.linalg.norm(pts_km[:, None] - pts_km[None, :], axis=2)
        nmask = dists < R_km
        wd = (nmask * w[None, :]).sum(axis=1)
        best = (wd * w).argmax()
        mask = nmask[best]
        cw = w[mask]
        return (cw[:, None] * pts[mask]).sum(axis=0) / cw.sum() if cw.sum() > 0 else pts[best]

    tests_passed = 0

    # Test 1: Tight cluster
    pts = np.array([
        [32.60, 44.01], [32.62, 44.03], [32.61, 44.02],  # Karbala cluster
        [30.04, 31.24],  # Cairo outlier
        [40.71, -74.01],  # NYC outlier
    ])
    w = np.array([0.9, 0.85, 0.88, 0.3, 0.2])

    pred = density_vote(pts, w, 25)
    err_to_karbala = haversine_single(pred[0], pred[1], 32.6164, 44.0323)
    if err_to_karbala < 5:
        ok(f"Tight cluster → كربلاء: error={err_to_karbala:.1f} km ✓")
        tests_passed += 1
    else:
        fail(f"Tight cluster → error={err_to_karbala:.1f} km (المتوقع <5)")

    # Test 2: Single point
    pred_single = density_vote(np.array([[33.31, 44.37]]), np.array([1.0]), 25)
    if abs(pred_single[0] - 33.31) < 0.01 and abs(pred_single[1] - 44.37) < 0.01:
        ok(f"Single point → ({pred_single[0]:.2f}, {pred_single[1]:.2f}) ✓")
        tests_passed += 1
    else:
        fail(f"Single point → ({pred_single[0]:.2f}, {pred_single[1]:.2f})")

    # Test 3: Two equal clusters → higher weight wins
    pts2 = np.array([
        [32.60, 44.01], [32.62, 44.03], [32.61, 44.02],  # Cluster A (higher weights)
        [30.04, 31.24], [30.05, 31.25], [30.03, 31.23],  # Cluster B (lower weights)
    ])
    w2 = np.array([0.9, 0.85, 0.88, 0.5, 0.48, 0.52])
    pred2 = density_vote(pts2, w2, 50)
    d_to_a = haversine_single(pred2[0], pred2[1], 32.61, 44.02)
    d_to_b = haversine_single(pred2[0], pred2[1], 30.04, 31.24)
    if d_to_a < d_to_b:
        ok(f"Weight priority: selected cluster A (d={d_to_a:.0f}km) over B (d={d_to_b:.0f}km) ✓")
        tests_passed += 1
    else:
        fail(f"Weight priority failed: A={d_to_a:.0f}km, B={d_to_b:.0f}km")

    # Test 4: Different radii
    for R in [5, 10, 25, 50, 200]:
        pred_r = density_vote(pts, w, R)
        err_r = haversine_single(pred_r[0], pred_r[1], 32.6164, 44.0323)
        info(f"  R={R:>3d} km → error={err_r:>8.1f} km  pred=({pred_r[0]:.2f}, {pred_r[1]:.2f})")

    record("Density Vote", "Tests Passed", f"{tests_passed}/3")


# ═══════════════════════════════════════════════════════════
# 8. CONFIG CONSISTENCY
# ═══════════════════════════════════════════════════════════
def check_config():
    header("8. فحص التوافق (Config Consistency)")

    try:
        from src.config import (
            CLIP_MODEL, CLIP_DIM, KNN_K, QUADTREE_MAX_CELLS,
            ACC_THRESHOLDS_KM, ACC_LABELS, DEVICE, SEED
        )
        ok(f"CLIP Model: {CLIP_MODEL}")
        ok(f"CLIP Dim: {CLIP_DIM}")
        ok(f"Device: {DEVICE}")
        ok(f"Seed: {SEED}")
        ok(f"kNN K: {KNN_K}")
        ok(f"Quadtree Max Cells: {QUADTREE_MAX_CELLS}")
        ok(f"Acc Thresholds: {ACC_THRESHOLDS_KM} km")
        ok(f"Acc Labels: {ACC_LABELS}")

        # Check dim consistency with galleries
        RAW = PROJECT_ROOT / "data" / "raw"
        if (RAW / "im2gps3k" / "fresh_features_all_3000.npy").exists():
            fts = np.load(str(RAW / "im2gps3k" / "fresh_features_all_3000.npy"))
            actual_dim = fts.shape[1]
            if actual_dim != CLIP_DIM:
                warn(f"Config CLIP_DIM={CLIP_DIM} but gallery features dim={actual_dim}")
                warn(f"Gallery uses GeoCLIP (768-dim) not standard CLIP (512-dim)")
            else:
                ok(f"Feature dimension matches config ✓")
            record("Config", "Gallery Feature Dim", actual_dim)

        record("Config", "Device", DEVICE)
        record("Config", "CLIP Model", CLIP_MODEL)
    except Exception as e:
        fail(f"Config error: {e}")
        ERRORS.append(f"Config: {e}")


# ═══════════════════════════════════════════════════════════
# 9. PYTHON DEPENDENCIES CHECK
# ═══════════════════════════════════════════════════════════
def check_dependencies():
    header("9. فحص المكتبات المطلوبة")

    deps = [
        ("numpy", "np"),
        ("torch", None),
        ("PIL", "Image"),
        ("faiss", None),
        ("flask", "Flask"),
    ]

    optional_deps = [
        ("geoclip", "GeoCLIP"),
        ("open_clip", None),
        ("clip", None),
        ("tqdm", None),
        ("requests", None),
    ]

    installed = 0
    for name, _ in deps:
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', 'N/A')
            ok(f"{name:20s} v{ver}")
            installed += 1
        except ImportError:
            fail(f"{name:20s} غير مثبت!")
            ERRORS.append(f"Missing: {name}")

    record("Dependencies", "Core Installed", f"{installed}/{len(deps)}")

    print(f"\n  {C.BOLD}Optional (للتنبؤ الفعلي):{C.END}")
    opt_installed = 0
    for name, _ in optional_deps:
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', 'N/A')
            ok(f"{name:20s} v{ver}")
            opt_installed += 1
        except ImportError:
            warn(f"{name:20s} غير مثبت (مطلوب للتنبؤ)")

    record("Dependencies", "Optional Installed", f"{opt_installed}/{len(optional_deps)}")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
def print_summary():
    header("📊 ملخص النتائج النهائي")

    for section, items in ALL_RESULTS.items():
        print(f"\n  {C.BOLD}{C.CYAN}{section}{C.END}")
        for key, val in items.items():
            print(f"    {key:35s} : {val}")

    if ERRORS:
        print(f"\n  {C.BOLD}{C.RED}⚠ الأخطاء ({len(ERRORS)}):{C.END}")
        for err in ERRORS:
            print(f"    {C.RED}• {err}{C.END}")
    else:
        print(f"\n  {C.GREEN}{C.BOLD}✓ لا توجد أخطاء!{C.END}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{C.BOLD}{'╔'+'═'*63+'╗'}{C.END}")
    print(f"{C.BOLD}║  {'فحص شامل لدقة جميع الموديلات والأنظمة':^57s}  ║{C.END}")
    print(f"{C.BOLD}║  {'GeoLocate Project - Full Accuracy Check':^57s}  ║{C.END}")
    print(f"{C.BOLD}{'╚'+'═'*63+'╝'}{C.END}")

    t_start = time.time()

    # 1. Check files
    check_project_files()

    # 2. Load galleries
    gallery_data = check_galleries()

    # 3. Metrics
    check_metrics()

    # 4. FAISS
    check_faiss(gallery_data)

    # 5. Quadtree
    check_quadtree(gallery_data)

    # 6. Baselines
    check_baselines(gallery_data)

    # 7. Density vote
    check_density_vote()

    # 8. Config
    check_config()

    # 9. Dependencies
    check_dependencies()

    # Summary
    print_summary()

    total_time = time.time() - t_start
    print(f"\n  {C.BOLD}Total time: {total_time:.1f}s{C.END}")
    print(f"{'='*65}\n")
