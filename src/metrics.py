import torch
import numpy as np

def haversine_distance(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Haversine distance between predictions and targets.
    Args:
        preds: Tensor of shape (N, 2) [latitude, longitude] in degrees.
        targets: Tensor of shape (N, 2) [latitude, longitude] in degrees.
    Returns:
        Distances in kilometers Tensor of shape (N,)
    """
    # 1. Convert degrees to radians
    preds_rad = torch.deg2rad(preds)
    targets_rad = torch.deg2rad(targets)
    
    # 2. Calculate differences
    lat1, lon1 = preds_rad[:, 0], preds_rad[:, 1]
    lat2, lon2 = targets_rad[:, 0], targets_rad[:, 1]
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # 3. Apply Haversine formula
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    # clamp a to [0, 1] to avoid rounding errors leading to negative values inside sqrt
    a = torch.clamp(a, 0.0, 1.0) 
    
    c = 2 * torch.asin(torch.sqrt(a))
    
    # 4. Multiply by Earth radius (6371 km)
    R = 6371.0
    distances = R * c
    
    return distances

def calculate_metrics(distances: torch.Tensor) -> dict:
    """
    Calculate median error and accuracy at specific thresholds.
    """
    assert distances.dim() == 1, "Distance tensor must be 1D"
    
    thresholds = [1, 25, 200, 750, 2500]
    metrics = {}
    
    if len(distances) == 0:
        metrics['median_error_km'] = float('nan')
        for tau in thresholds:
            metrics[f'acc@{tau}km'] = 0.0
        return metrics
        
    # Calculate median error
    metrics['median_error_km'] = torch.median(distances).item()
    
    # Calculate Acc@tau
    N = distances.shape[0]
    for tau in thresholds:
        correct = (distances <= tau).sum().item()
        acc = correct / N
        metrics[f'acc@{tau}km'] = acc
        
    return metrics
