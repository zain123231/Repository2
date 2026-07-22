"""
Evaluation metrics — Haversine distance, Acc@tau, Median Error.
All formulas per the PDF (Section 2.3).

CRITICAL: This module has been verified line-by-line.
Any change here invalidates ALL results.
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine(pred_lat, pred_lon, gt_lat, gt_lon):
    """
    Vectorized Haversine distance in km.
    
    Parameters:
        pred_lat, pred_lon, gt_lat, gt_lon : np.ndarray (N,)
    
    Returns:
        distances : np.ndarray (N,) in km
    """
    pred_lat = np.asarray(pred_lat, dtype=np.float64)
    pred_lon = np.asarray(pred_lon, dtype=np.float64)
    gt_lat = np.asarray(gt_lat, dtype=np.float64)
    gt_lon = np.asarray(gt_lon, dtype=np.float64)

    phi1 = np.radians(pred_lat)
    phi2 = np.radians(gt_lat)
    dphi = np.radians(gt_lat - pred_lat)
    dlam = np.radians(gt_lon - pred_lon)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c


def haversine_single(pred_lat, pred_lon, gt_lat, gt_lon):
    """Scalar Haversine for single prediction."""
    return float(haversine(
        np.array([pred_lat]), np.array([pred_lon]),
        np.array([gt_lat]), np.array([gt_lon])
    )[0])


def acc_at_tau(distances_km, thresholds_km):
    """
    Fraction of predictions within each threshold.
    
    Parameters:
        distances_km : np.ndarray (N,)
        thresholds_km : list of float
    
    Returns:
        accs : dict {threshold: fraction}
    """
    accs = {}
    for tau in thresholds_km:
        accs[float(tau)] = float(np.mean(distances_km <= tau))
    return accs


def median_error(distances_km):
    """Median haversine error in km."""
    return float(np.median(distances_km))


def mean_error(distances_km):
    """Mean haversine error in km."""
    return float(np.mean(distances_km))


def evaluate_system(predictions, ground_truth):
    """
    Full evaluation of a geolocation system.
    
    Parameters:
        predictions : np.ndarray (N, 2) — [lat, lon]
        ground_truth : np.ndarray (N, 2) — [lat, lon]
    
    Returns:
        dict with all metrics
    """
    try:
        from src.config import ACC_THRESHOLDS_KM
    except ImportError:
        from config import ACC_THRESHOLDS_KM

    assert len(predictions) == len(ground_truth), \
        f"Length mismatch: {len(predictions)} vs {len(ground_truth)}"
    assert len(predictions) > 0, "Empty predictions"

    distances = haversine(
        predictions[:, 0], predictions[:, 1],
        ground_truth[:, 0], ground_truth[:, 1]
    )

    accs = acc_at_tau(distances, ACC_THRESHOLDS_KM)
    med = median_error(distances)
    mean = mean_error(distances)

    result = {
        "n_samples": int(len(predictions)),
        "median_error_km": med,
        "mean_error_km": mean,
    }
    for tau, acc in accs.items():
        result[f"acc@{tau}km"] = acc

    return result


def smooth_labels(cell_centers, gt_coords, temperature=0.1):
    """
    Geographic distance-based label smoothing (PIGEON equation).
    
    smooth(y)_i = exp(-d(g_i, gt) / tau) / sum_j exp(-d(g_j, gt) / tau)
    
    Parameters:
        cell_centers : np.ndarray (C, 2) — cell center coordinates
        gt_coords : np.ndarray (2,) — single ground truth [lat, lon]
        temperature : float — tau_s parameter
    
    Returns:
        smooth_labels : np.ndarray (C,) — smoothed probability distribution
    """
    distances = haversine(
        cell_centers[:, 0], cell_centers[:, 1],
        np.array([gt_coords[0]]), np.array([gt_coords[1]])
    )
    logits = -distances / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


# === Unit tests ===

def _test_haversine_known():
    """Test haversine with known distances."""
    d = haversine_single(0, 0, 0, 1)
    assert 110.0 < d < 112.0, f"Haversine 1 degree lon at equator: {d}"

    d = haversine_single(0, 0, 90, 0)
    assert 10000.0 < d < 10100.0, f"Haversine pole: {d}"

    d = haversine_single(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330.0 < d < 360.0, f"London-Paris: {d}"

    print("  [PASS] haversine_known")


def _test_haversine_zero():
    """Same location → distance = 0."""
    d = haversine_single(40.7128, -74.0060, 40.7128, -74.0060)
    assert d < 0.001, f"Zero distance: {d}"
    print("  [PASS] haversine_zero")


def _test_haversine_symmetry():
    """Haversine is symmetric."""
    d1 = haversine_single(40.7128, -74.0060, 51.5074, -0.1278)
    d2 = haversine_single(51.5074, -0.1278, 40.7128, -74.0060)
    assert abs(d1 - d2) < 0.001, f"Asymmetric: {d1} vs {d2}"
    print("  [PASS] haversine_symmetry")


def _test_acc_at_tau():
    """Acc@tau basic tests."""
    distances = np.array([0.5, 10, 50, 100, 500, 1000])
    accs = acc_at_tau(distances, [1, 25, 200, 750, 2500])
    assert abs(accs[1.0] - 1 / 6) < 0.001
    assert abs(accs[25.0] - 2 / 6) < 0.001
    assert abs(accs[200.0] - 4 / 6) < 0.001
    assert abs(accs[750.0] - 5 / 6) < 0.001
    assert abs(accs[2500.0] - 1.0) < 0.001
    print("  [PASS] acc_at_tau")


def _test_median_error():
    """Median error basic test."""
    distances = np.array([10, 20, 30, 40, 50])
    assert abs(median_error(distances) - 30.0) < 0.001
    print("  [PASS] median_error")


def _test_random_baseline():
    """Random predictions should give ~10000 km median."""
    np.random.seed(42)
    n = 1000
    pred = np.column_stack([
        np.random.uniform(-90, 90, n),
        np.random.uniform(-180, 180, n)
    ])
    gt = np.column_stack([
        np.random.uniform(-90, 90, n),
        np.random.uniform(-180, 180, n)
    ])
    result = evaluate_system(pred, gt)
    assert 8000 < result["median_error_km"] < 12000, \
        f"Random baseline median: {result['median_error_km']}"
    print(f"  [PASS] random_baseline (median={result['median_error_km']:.0f} km)")


def _test_centroid_baseline():
    """Fixed centroid → same prediction for all → test consistency."""
    np.random.seed(42)
    n = 100
    gt = np.column_stack([
        np.random.uniform(-90, 90, n),
        np.random.uniform(-180, 180, n)
    ])
    centroid = np.array([0.0, 0.0])
    pred = np.tile(centroid, (n, 1))
    result = evaluate_system(pred, gt)
    assert result["n_samples"] == n
    assert result["median_error_km"] > 0
    print(f"  [PASS] centroid_baseline (median={result['median_error_km']:.0f} km)")


def _test_smooth_labels():
    """Label smoothing sums to 1."""
    np.random.seed(42)
    centers = np.random.uniform(-90, 90, (10, 2))
    centers[:, 1] = np.random.uniform(-180, 180, 10)
    gt = np.array([40.0, -74.0])
    labels = smooth_labels(centers, gt, temperature=0.1)
    assert abs(np.sum(labels) - 1.0) < 1e-6, f"Labels sum: {np.sum(labels)}"
    assert np.all(labels >= 0), "Negative labels"
    print("  [PASS] smooth_labels")


def run_all_tests():
    """Run all evaluation unit tests."""
    print("\n=== Evaluation Unit Tests ===")
    _test_haversine_known()
    _test_haversine_zero()
    _test_haversine_symmetry()
    _test_acc_at_tau()
    _test_median_error()
    _test_random_baseline()
    _test_centroid_baseline()
    _test_smooth_labels()
    print("=== All evaluation tests PASSED ===\n")


if __name__ == "__main__":
    run_all_tests()
