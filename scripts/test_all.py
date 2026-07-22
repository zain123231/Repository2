"""
Comprehensive Test Suite
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n[TEST] Evaluation Metrics")
    print("-" * 40)
    
    import numpy as np
    from src.evaluation.metrics import haversine, haversine_batch, accuracy_at_threshold, median_error
    
    # Test single haversine
    dist = haversine(40.7128, -74.0060, 40.7128, -74.0060)
    assert dist == 0.0, f"Expected 0, got {dist}"
    print("  haversine (same point): PASS")
    
    # Test different points
    dist = haversine(40.7128, -74.0060, 51.5074, -0.1278)
    assert dist > 5000, f"Expected > 5000km, got {dist}"
    print(f"  haversine (NY-London): {dist:.0f} km - PASS")
    
    # Test batch
    pred = np.array([[40.7128, -74.0060], [51.5074, -0.1278]])
    true = np.array([[40.7128, -74.0060], [51.5074, -0.1278]])
    batch_dist = haversine_batch(pred, true)
    assert all(d < 0.01 for d in batch_dist), "Batch distances should be ~0"
    print("  haversine_batch: PASS")
    
    # Test accuracy
    distances = np.array([0.5, 10, 50, 100, 500])
    acc_1 = accuracy_at_threshold(distances, 1)
    acc_25 = accuracy_at_threshold(distances, 25)
    acc_200 = accuracy_at_threshold(distances, 200)
    assert acc_1 == 20.0, f"Expected 20%, got {acc_1}"
    assert acc_25 == 40.0, f"Expected 40%, got {acc_25}"
    assert acc_200 == 80.0, f"Expected 80%, got {acc_200}"
    print("  accuracy_at_threshold: PASS")
    
    # Test median error
    med = median_error(distances)
    assert med == 50.0, f"Expected 50, got {med}"
    print("  median_error: PASS")
    
    print("All evaluation metric tests PASSED!")


def test_faiss_index():
    """Test FAISS index."""
    print("\n[TEST] FAISS Index")
    print("-" * 40)
    
    import numpy as np
    from src.models.faiss_index import FAISSIndex
    
    # Create index
    dimension = 512
    index = FAISSIndex(dimension=dimension)
    
    # Add features
    n_features = 1000
    features = np.random.randn(n_features, dimension).astype(np.float32)
    coordinates = np.random.uniform(-90, 90, (n_features, 2)).astype(np.float32)
    
    index.add(features, coordinates=coordinates)
    assert len(index) == n_features, f"Expected {n_features}, got {len(index)}"
    print(f"  Added {n_features} features: PASS")
    
    # Search
    query = np.random.randn(1, dimension).astype(np.float32)
    distances, indices = index.search(query, k=5)
    assert distances.shape == (1, 5), f"Expected (1, 5), got {distances.shape}"
    print("  Search query: PASS")
    
    # Save/Load
    index.save("test_index")
    new_index = FAISSIndex(dimension=dimension)
    new_index.load("test_index")
    assert len(new_index) == n_features, "Loaded index size mismatch"
    print("  Save/Load: PASS")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_index")
    
    print("All FAISS index tests PASSED!")


def test_quadtree():
    """Test quadtree geocells."""
    print("\n[TEST] Quadtree Geocells")
    print("-" * 40)
    
    import numpy as np
    from src.models.quadtree import QuadtreeGeocells
    
    # Generate data
    n_samples = 1000
    coordinates = np.random.uniform(-90, 90, (n_samples, 2))
    
    # Build quadtree
    qt = QuadtreeGeocells(min_samples_per_cell=50, max_depth=8)
    cells = qt.build(coordinates)
    
    assert len(cells) > 0, "Should create cells"
    print(f"  Created {len(cells)} cells: PASS")
    
    # Assign cells
    cell_ids = qt.assign_cells(coordinates)
    assert len(cell_ids) == n_samples, "Cell IDs length mismatch"
    print("  Assign cells: PASS")
    
    # Get centers
    centers = qt.get_cell_centers()
    assert centers.shape[1] == 2, "Centers should be (N, 2)"
    print("  Get centers: PASS")
    
    print("All quadtree tests PASSED!")


def test_feature_extractor():
    """Test feature extractor."""
    print("\n[TEST] Feature Extractor")
    print("-" * 40)
    
    from PIL import Image
    from src.models.feature_extractor import CLIPFeatureExtractor
    
    # Create extractor
    extractor = CLIPFeatureExtractor(model_name="ViT-B-16", pretrained="openai")
    
    # Get feature dimension
    dim = extractor.get_feature_dim()
    assert dim > 0, f"Feature dim should be > 0, got {dim}"
    print(f"  Feature dimension: {dim}: PASS")
    
    # Test extraction
    dummy_image = Image.new('RGB', (224, 224))
    features = extractor.extract_features([dummy_image])
    assert features.shape == (1, dim), f"Expected (1, {dim}), got {features.shape}"
    print("  Extract features: PASS")
    
    # Test batch extraction
    features_batch = extractor.extract_features_batch([dummy_image], batch_size=1)
    assert features_batch.shape == (1, dim), "Batch extraction failed"
    print("  Batch extraction: PASS")
    
    print("All feature extractor tests PASSED!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" Running All Tests")
    print("=" * 60)
    
    test_evaluation_metrics()
    test_faiss_index()
    test_quadtree()
    test_feature_extractor()
    
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
