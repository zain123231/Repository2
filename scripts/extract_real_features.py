#!/usr/bin/env python
"""
Extract CLIP ViT-B/32 features from real Im2GPS3k images.
Also creates OSV-5M reference index and YFCC4k test split.
"""
import sys
import os
import json
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DIR, SEED

IM2GPS_DIR = os.path.join(RAW_DIR, "im2gps3k")
IMG_DIR = os.path.join(IM2GPS_DIR, "im2gps3ktest")


def main():
    print("=" * 60)
    print("EXTRACTING REAL CLIP FEATURES")
    print("=" * 60)

    # Load coordinates (already fixed from metadata.json)
    coords = np.load(os.path.join(IM2GPS_DIR, "coordinates.npy"))
    print(f"Coordinates: {coords.shape}")

    # Get image list
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Image files: {len(img_files)}")

    # Filter to only images with valid coordinates (first 2997)
    valid_mask = ~np.isnan(coords[:, 0])
    valid_indices = np.where(valid_mask)[0]
    print(f"Valid samples (with coordinates): {len(valid_indices)}")

    # Use only valid samples
    valid_coords = coords[valid_indices]
    valid_files = [img_files[i] for i in valid_indices]

    # Save valid coordinates
    np.save(os.path.join(IM2GPS_DIR, "coordinates.npy"), valid_coords)
    with open(os.path.join(IM2GPS_DIR, "image_list.json"), "w") as f:
        json.dump(valid_files, f)
    print(f"Saved {len(valid_files)} valid coordinates and image list")

    # Load CLIP
    import clip as clip_mod
    import torch

    device = "cpu"
    print(f"\nLoading CLIP ViT-B/32 on {device}...")
    model, preprocess = clip_mod.load("ViT-B/32", device=device)
    model.eval()
    print("CLIP loaded")

    # Extract features in batches
    batch_size = 16
    all_features = []
    total = len(valid_files)
    start_time = time.time()

    print(f"\nExtracting features from {total} images (batch_size={batch_size})...")
    for i in range(0, total, batch_size):
        batch_files = valid_files[i:i+batch_size]
        batch_tensors = []

        for fname in batch_files:
            try:
                img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
                batch_tensors.append(preprocess(img))
            except Exception as e:
                print(f"  Warning: failed to load {fname}: {e}")
                # Use a blank image as fallback
                batch_tensors.append(preprocess(Image.new("RGB", (224, 224))))

        tensors = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy().astype(np.float32))

        elapsed = time.time() - start_time
        done = min(i + batch_size, total)
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        if done % (batch_size * 10) == 0 or done == total:
            print(f"  [{done}/{total}] {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining ({speed:.1f} img/s)")

    features = np.concatenate(all_features, axis=0)
    elapsed = time.time() - start_time
    print(f"\nFeatures extracted: {features.shape} in {elapsed:.1f}s ({elapsed/total*1000:.1f} ms/image)")

    # Save im2gps3k features
    np.save(os.path.join(IM2GPS_DIR, "features.npy"), features)
    print(f"Saved im2gps3k features: {features.shape}")

    # Create splits
    print("\n--- Creating dataset splits ---")
    rng = np.random.RandomState(SEED)
    n = len(valid_coords)
    indices = rng.permutation(n)

    # 2000 for reference index (OSV-5M substitute)
    n_ref = min(2000, n)
    ref_indices = indices[:n_ref]
    test_yfcc_indices = indices[n_ref:]

    # OSV-5M reference index
    osv5m_dir = os.path.join(RAW_DIR, "osv5m")
    os.makedirs(osv5m_dir, exist_ok=True)
    np.save(os.path.join(osv5m_dir, "coordinates.npy"), valid_coords[ref_indices])
    np.save(os.path.join(osv5m_dir, "features.npy"), features[ref_indices])
    with open(os.path.join(osv5m_dir, "image_list.json"), "w") as f:
        json.dump([valid_files[i] for i in ref_indices], f)
    print(f"OSV-5M reference: {n_ref} samples")

    # YFCC4k test split
    yfcc_dir = os.path.join(RAW_DIR, "yfcc4k")
    os.makedirs(yfcc_dir, exist_ok=True)
    np.save(os.path.join(yfcc_dir, "coordinates.npy"), valid_coords[test_yfcc_indices])
    np.save(os.path.join(yfcc_dir, "features.npy"), features[test_yfcc_indices])
    with open(os.path.join(yfcc_dir, "image_list.json"), "w") as f:
        json.dump([valid_files[i] for i in test_yfcc_indices], f)
    print(f"YFCC4k test: {len(test_yfcc_indices)} samples")

    # Save splits info
    splits = {
        "reference_index": ref_indices.tolist(),
        "test_yfcc": test_yfcc_indices.tolist(),
        "n_total": n,
        "n_reference": n_ref,
        "n_test_yfcc": len(test_yfcc_indices),
    }
    with open(os.path.join(IM2GPS_DIR, "splits.json"), "w") as f:
        json.dump(splits, f)

    # Verification
    print("\n--- Verification ---")
    from src.evaluation.metrics import haversine
    # Check feature quality
    feat_norms = np.linalg.norm(features, axis=1)
    print(f"Feature norms: mean={feat_norms.mean():.4f}, std={feat_norms.std():.6f}")

    # Self-similarity
    sample_sims = np.dot(features[:100], features[:100].T)
    diag_mean = np.diag(sample_sims).mean()
    off_diag = sample_sims[np.triu_indices(100, k=1)]
    print(f"Self-similarity diagonal: {diag_mean:.4f} (should be 1.0)")
    print(f"Off-diagonal similarities: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")

    # Geographic distribution
    print(f"\nGeographic distribution:")
    print(f"  Lat range: [{valid_coords[:, 0].min():.2f}, {valid_coords[:, 0].max():.2f}]")
    print(f"  Lon range: [{valid_coords[:, 1].min():.2f}, {valid_coords[:, 1].max():.2f}]")

    # Country distribution
    meta_path = os.path.join(IM2GPS_DIR, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    countries = {}
    for m in meta:
        c = m['country']
        countries[c] = countries.get(c, 0) + 1
    top_countries = sorted(countries.items(), key=lambda x: -x[1])[:10]
    print(f"  Top 10 countries:")
    for c, n in top_countries:
        print(f"    {c}: {n}")

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Im2GPS3k: {features.shape}")
    print(f"  OSV-5M: {features[ref_indices].shape}")
    print(f"  YFCC4k: {features[test_yfcc_indices].shape}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
