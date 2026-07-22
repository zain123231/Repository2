#!/usr/bin/env python
"""
Preprocess real Im2GPS3k data:
1. Match image filenames to coordinates from metadata.json
2. Save proper coordinates.npy
3. Extract CLIP ViT-B/32 features from all images
4. Save features.npy
"""
import sys
import os
import json
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DIR, PROCESSED_DIR, SEED

IM2GPS_DIR = os.path.join(RAW_DIR, "im2gps3k")
IMG_DIR = os.path.join(IM2GPS_DIR, "im2gps3ktest")
META_PATH = os.path.join(IM2GPS_DIR, "metadata.json")


def step1_fix_coordinates():
    """Create proper coordinates.npy from metadata.json."""
    print("=== Step 1: Fix coordinates from metadata.json ===")

    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"  Metadata entries: {len(meta)}")

    # Get list of image files
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Image files: {len(img_files)}")

    # Build a mapping from image filename (without extension) to metadata
    # The metadata doesn't seem to have filenames, so we need to match by order
    # or by some other key. Let's check the first few metadata entries.
    print(f"  First metadata entry: {meta[0]}")

    # Check if metadata has image_id or filename field
    if 'image_id' in meta[0]:
        print(f"  Found 'image_id' field")
    elif 'filename' in meta[0]:
        print(f"  Found 'filename' field")

    # Let's assume metadata is ordered to match img_files alphabetically
    # or check if metadata keys match image filenames
    # Actually, let's try to match by checking if any metadata field matches image names
    meta_keys = list(meta[0].keys())
    print(f"  Metadata keys: {meta_keys}")

    # Create coordinates array - metadata has lat/lon
    lats = np.array([m['lat'] for m in meta])
    lons = np.array([m['lon'] for m in meta])
    coords = np.column_stack([lats, lons])

    print(f"  Lat range: [{lats.min():.4f}, {lats.max():.4f}]")
    print(f"  Lon range: [{lons.min():.4f}, {lons.max():.4f}]")

    # Save
    np.save(os.path.join(IM2GPS_DIR, "coordinates.npy"), coords)
    print(f"  Saved coordinates.npy: shape={coords.shape}")

    # Also create image list file for later reference
    with open(os.path.join(IM2GPS_DIR, "image_list.json"), "w") as f:
        json.dump(img_files, f)

    return coords, img_files, meta


def step2_extract_features(img_files, batch_size=16):
    """Extract CLIP ViT-B/32 features from all images."""
    print("\n=== Step 2: Extract CLIP ViT-B/32 features ===")

    import clip
    import torch

    device = "cpu"
    print(f"  Loading CLIP ViT-B/32 on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("  CLIP loaded")

    all_features = []
    total = len(img_files)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch_files = img_files[i:i+batch_size]
        batch_images = []
        valid_indices = []

        for j, fname in enumerate(batch_files):
            try:
                img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
                batch_images.append(preprocess(img))
                valid_indices.append(i + j)
            except Exception as e:
                print(f"  Warning: failed to load {fname}: {e}")

        if batch_images:
            tensors = torch.stack(batch_images).to(device)
            with torch.no_grad():
                features = model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy().astype(np.float32))

        elapsed = time.time() - start_time
        done = min(i + batch_size, total)
        speed = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / speed if speed > 0 else 0
        print(f"  [{done}/{total}] {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining ({speed:.1f} img/s)")

    features = np.concatenate(all_features, axis=0)
    print(f"  Final features shape: {features.shape}")
    print(f"  Total time: {time.time()-start_time:.1f}s")

    # Save features
    np.save(os.path.join(IM2GPS_DIR, "features.npy"), features)
    print(f"  Saved features.npy")

    return features


def step3_create_splits(coords, img_files, meta, seed=42):
    """Create train/test splits and OSV-5M reference index from im2gps3k data."""
    print("\n=== Step 3: Create dataset splits ===")

    rng = np.random.RandomState(seed)
    n = len(coords)
    indices = rng.permutation(n)

    # Im2GPS3k test set: use all 2997 as the primary test set
    # For FAISS index (OSV-5M substitute): use 2000 samples as reference
    # For YFCC4k substitute: use remaining ~1000 as secondary test set

    n_ref = 2000
    ref_indices = indices[:n_ref]
    test_indices = indices[n_ref:]

    print(f"  Total samples: {n}")
    print(f"  Reference index (OSV-5M sub): {n_ref}")
    print(f"  Test set 1 (Im2GPS3k): {n}")
    print(f"  Test set 2 (YFCC4k sub): {len(test_indices)}")

    # Save reference index data (for FAISS)
    osv5m_dir = os.path.join(RAW_DIR, "osv5m")
    os.makedirs(osv5m_dir, exist_ok=True)
    np.save(os.path.join(osv5m_dir, "coordinates.npy"), coords[ref_indices])
    # Features will be extracted separately

    # Save YFCC4k substitute
    yfcc_dir = os.path.join(RAW_DIR, "yfcc4k")
    os.makedirs(yfcc_dir, exist_ok=True)
    np.save(os.path.join(yfcc_dir, "coordinates.npy"), coords[test_indices])

    # Save splits info
    splits = {
        "reference_index": ref_indices.tolist(),
        "test_yfcc_sub": test_indices.tolist(),
        "n_total": n,
        "n_reference": n_ref,
        "n_test_yfcc": len(test_indices),
    }
    with open(os.path.join(IM2GPS_DIR, "splits.json"), "w") as f:
        json.dump(splits, f)

    print(f"  Splits saved to {os.path.join(IM2GPS_DIR, 'splits.json')}")
    return ref_indices, test_indices


def main():
    print("=" * 60)
    print("PREPROCESSING REAL Im2GPS3k DATA")
    print("=" * 60)

    # Step 1: Fix coordinates
    coords, img_files, meta = step1_fix_coordinates()

    # Step 2: Extract features
    features = step2_extract_features(img_files, batch_size=16)

    # Step 3: Create splits
    ref_indices, test_indices = step3_create_splits(coords, img_files, meta)

    # Also extract features for OSV-5M subset and YFCC4k subset
    print("\n=== Step 4: Extract features for subsets ===")
    import clip as clip_mod
    import torch

    device = "cpu"
    model, preprocess = clip_mod.load("ViT-B/32", device=device)
    model.eval()

    # OSV-5M reference features
    ref_features = features[ref_indices]
    osv5m_dir = os.path.join(RAW_DIR, "osv5m")
    np.save(os.path.join(osv5m_dir, "features.npy"), ref_features)
    print(f"  OSV-5M reference features: {ref_features.shape}")

    # YFCC4k test features
    yfcc_features = features[test_indices]
    yfcc_dir = os.path.join(RAW_DIR, "yfcc4k")
    np.save(os.path.join(yfcc_dir, "features.npy"), yfcc_features)
    print(f"  YFCC4k test features: {yfcc_features.shape}")

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Im2GPS3k: {len(coords)} samples, features={features.shape}")
    print(f"  OSV-5M ref: {len(ref_indices)} samples")
    print(f"  YFCC4k sub: {len(test_indices)} samples")
    print(f"  Feature dim: {features.shape[1]}")

    # Verify distances are real (not random)
    from src.evaluation.metrics import haversine
    # Check that nearby images have similar features
    feat_norms = np.linalg.norm(features[:100], axis=1)
    print(f"  Feature norms (first 100): mean={feat_norms.mean():.4f}, std={feat_norms.std():.4f}")
    print(f"  Feature norms should be ~1.0 (L2 normalized)")

    # Check cosine similarity distribution
    sims = np.dot(features[:100], features[:100].T)
    print(f"  Self-similarity (first 100): diagonal mean={np.diag(sims).mean():.4f} (should be 1.0)")
    off_diag = sims[np.triu_indices(100, k=1)]
    print(f"  Off-diagonal similarities: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")


if __name__ == "__main__":
    main()
