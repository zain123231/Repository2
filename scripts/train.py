"""
Main Training Script
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import *
from src.data.dataloader import get_image_transform
from src.models.feature_extractor import CLIPFeatureExtractor
from src.evaluation.metrics import evaluate_geolocation, print_results


def main():
    """Main training function."""
    print("=" * 60)
    print(" Single-Image Visual Geolocation System")
    print(" Training Script")
    print("=" * 60)
    
    # Step 1: Initialize feature extractor
    print("\n[1/5] Initializing CLIP Feature Extractor...")
    feature_extractor = CLIPFeatureExtractor(
        model_name="ViT-B-16",
        pretrained="openai",
        device=DEVICE
    )
    
    # Get feature dimension
    feature_dim = feature_extractor.get_feature_dim()
    print(f"Feature dimension: {feature_dim}")
    
    # Step 2: Load datasets
    print("\n[2/5] Loading datasets...")
    print("Note: Datasets need to be downloaded first.")
    print("Please download:")
    print("  - OSV-5M from HuggingFace")
    print("  - Im2GPS3k")
    print("  - YFCC4k")
    
    # Step 3: Build baselines
    print("\n[3/5] Building baseline systems...")
    print("  - GeoCLIP baseline")
    print("  - Zero-shot classification")
    print("  - kNN retrieval")
    
    # Step 4: Train hybrid system
    print("\n[4/5] Training hybrid system...")
    print("  - Geographic cell classifier")
    print("  - Multi-scale prediction")
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating systems...")
    print("  - Calculate metrics")
    print("  - Generate visualizations")
    
    print("\n" + "=" * 60)
    print(" Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
