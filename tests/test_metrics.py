import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from src.metrics import haversine_distance, calculate_metrics

def test_haversine_known_locations():
    # London: (51.5074, -0.1278)
    # Paris: (48.8566, 2.3522)
    # Approximate distance: ~344 km
    london = torch.tensor([[51.5074, -0.1278]])
    paris = torch.tensor([[48.8566, 2.3522]])
    
    dist = haversine_distance(london, paris)
    
    # Check if distance is approximately 343-345 km
    assert torch.isclose(dist[0], torch.tensor(343.5), atol=2.0), f"Incorrect measurement: {dist[0]} km"
    print(f"\n[LOG] Calculated distance between London and Paris: {dist[0]:.2f} km")

def test_haversine_same_point():
    point = torch.tensor([[25.2048, 55.2708]]) # Dubai
    dist = haversine_distance(point, point)
    
    assert dist[0].item() == 0.0, "Distance between the same point must be zero"
    print(f"\n[LOG] Passed identical point test.")

def test_random_and_constant_predictions():
    # Sanity Check 
    N = 1000
    torch.manual_seed(42)
    # Random true coordinates: lat [-90, 90], lon [-180, 180]
    targets = torch.rand(N, 2) * torch.tensor([180.0, 360.0]) - torch.tensor([90.0, 180.0])
    
    # Random predictions
    preds_random = torch.rand(N, 2) * torch.tensor([180.0, 360.0]) - torch.tensor([90.0, 180.0])
    
    # Constant predictions (e.g., center of earth map - equator/prime meridian intersection)
    preds_constant = torch.zeros(N, 2)
    
    dist_random = haversine_distance(preds_random, targets)
    dist_constant = haversine_distance(preds_constant, targets)
    
    metrics_rand = calculate_metrics(dist_random)
    metrics_const = calculate_metrics(dist_constant)
    
    assert 0.0 <= metrics_rand['acc@2500km'] <= 1.0, "Error in percentage calculation"
    print(f"\n[LOG] Random Median Error: {metrics_rand['median_error_km']:.2f} km")
    print(f"\n[LOG] Constant Median Error: {metrics_const['median_error_km']:.2f} km")
    
    # Assert specific shape
    assert dist_random.shape == (N,)
    assert dist_constant.shape == (N,)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
