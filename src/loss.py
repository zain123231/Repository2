import torch
import torch.nn.functional as F
import numpy as np

def pairwise_haversine_distance(loc1: torch.Tensor, loc2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Haversine distance matrix between two sets of locations.
    Args:
        loc1: Tensor of shape (N, 2) [latitude, longitude] in degrees.
        loc2: Tensor of shape (M, 2) [latitude, longitude] in degrees.
    Returns:
        Tensor of shape (N, M) containing distances in kilometers.
    """
    # Convert degrees to radians
    loc1_rad = torch.deg2rad(loc1)
    loc2_rad = torch.deg2rad(loc2)
    
    lat1, lon1 = loc1_rad[:, 0].unsqueeze(1), loc1_rad[:, 1].unsqueeze(1) # (N, 1)
    lat2, lon2 = loc2_rad[:, 0].unsqueeze(0), loc2_rad[:, 1].unsqueeze(0) # (1, M)
    
    dlat = lat2 - lat1 # (N, M)
    dlon = lon2 - lon1 # (N, M)
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    # Clamp a to [0, 1] for numerical stability
    a = torch.clamp(a, 0.0, 1.0)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    R = 6371.0 # Earth radius in kilometers
    return R * c

def contrastive_loss(image_features: torch.Tensor, location_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Standard InfoNCE loss for image and location features.
    Args:
        image_features: L2-normalized image embeddings (N, D).
        location_features: L2-normalized location embeddings (N, D).
        logit_scale: Learnable temperature parameter.
    Returns:
        Scalar loss.
    """
    logits_per_image = logit_scale * image_features @ location_features.t()
    logits_per_location = logits_per_image.t()
    
    # Ground truth labels are the diagonal indices
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_l = F.cross_entropy(logits_per_location, labels)
    
    return (loss_i + loss_l) / 2.0

def geographic_loss(logits: torch.Tensor, batch_locations: torch.Tensor) -> torch.Tensor:
    """
    Geographic penalty loss based on expected Haversine distance.
    Args:
        logits: Similarity logits of shape (N, M).
        batch_locations: Coordinates of shape (M, 2).
    Returns:
        Scalar loss representing the expected distance in kilometers.
    """
    # Softmax probabilities over candidates (N, M)
    probs = F.softmax(logits, dim=-1)
    
    # For standard batch contrastive learning, N == M and target is diagonal
    # Pairwise distances between true locations (N) and candidate locations (M)
    dist_matrix = pairwise_haversine_distance(batch_locations, batch_locations) # (N, N) if N==M
    
    # Expected distance = sum(prob_i * dist_i)
    expected_distances = torch.sum(probs * dist_matrix, dim=-1) # (N,)
    
    return expected_distances.mean()

def geoclip_total_loss(image_features: torch.Tensor, location_features: torch.Tensor, 
                       logit_scale: torch.Tensor, batch_locations: torch.Tensor, 
                       alpha: float = 0.01) -> torch.Tensor:
    """
    Combines Contrastive and Geographic loss.
    alpha: Weight for the geographic penalty.
    """
    c_loss = contrastive_loss(image_features, location_features, logit_scale)
    
    logits = logit_scale * image_features @ location_features.t()
    g_loss = geographic_loss(logits, batch_locations)
    
    total_loss = c_loss + alpha * g_loss
    return total_loss, c_loss, g_loss

if __name__ == "__main__":
    print("[LOG] Running Sanity Check for Loss Functions...")
    # Mock data
    N = 4 # Batch size
    D = 512 # Feature dimension
    
    img_feats = torch.randn(N, D, requires_grad=True)
    img_feats_norm = F.normalize(img_feats, dim=-1)
    img_feats_norm.retain_grad()
    
    loc_feats = torch.randn(N, D, requires_grad=True)
    loc_feats_norm = F.normalize(loc_feats, dim=-1)
    loc_feats_norm.retain_grad()
    
    scale = torch.tensor(np.log(1 / 0.07), requires_grad=True)
    
    # Mock coordinates (lat, lon)
    lats = torch.rand(N, 1) * 180 - 90
    lons = torch.rand(N, 1) * 360 - 180
    coords = torch.cat([lats, lons], dim=-1)
    
    print("Coordinates:\n", coords)
    
    total, c_loss, g_loss = geoclip_total_loss(img_feats_norm, loc_feats_norm, scale, coords, alpha=0.01)
    
    print(f"Contrastive Loss: {c_loss.item():.4f}")
    print(f"Geographic Loss: {g_loss.item():.4f} km (Expected Distance)")
    print(f"Total Loss: {total.item():.4f}")
    
    # Test gradients
    total.backward()
    print("Image Features Grad Shape:", img_feats_norm.grad.shape)
    print("Location Features Grad Shape:", loc_feats_norm.grad.shape)
    print("Logit Scale Grad:", scale.grad.item())
    
    if img_feats_norm.grad is not None and not torch.isnan(img_feats_norm.grad).any():
        print("[LOG] Sanity check PASSED. Gradients computed successfully.")
    else:
        print("[ERROR] Gradient computation failed or produced NaNs.")
