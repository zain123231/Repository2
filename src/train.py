import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "geo-clip"))
from geoclip.model.GeoCLIP import GeoCLIP
from src.dataset import GeoDataset
from src.loss import geoclip_total_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Using device: {device}")

    if not args.mock and not os.path.exists(args.csv):
        print(f"\n[ERROR] Dataset not found at {args.csv}")
        print("Please ensure you have downloaded the dataset. See src/README.md for instructions.")
        print("If you want to run a quick test with fake data, use the --mock flag.")
        sys.exit(1)

    # 1. Dataset & DataLoader
    dataset = GeoDataset(csv_file=args.csv, img_dir="data/images", mock=args.mock, size=200 if args.mock else None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[LOG] Dataset size: {len(dataset)}")

    # 2. Model
    model = GeoCLIP(from_pretrained=True).to(device)
    
    # Freeze CLIP backbone, train only MLP and Location Encoder
    for param in model.image_encoder.CLIP.parameters():
        param.requires_grad = False
    
    # Enable gradients for parts we want to train
    for param in model.image_encoder.mlp.parameters():
        param.requires_grad = True
    for param in model.location_encoder.parameters():
        param.requires_grad = True
    model.logit_scale.requires_grad = True

    model.train()

    # 3. Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.mlp.parameters(), 'lr': args.lr},
        {'params': model.location_encoder.parameters(), 'lr': args.lr},
        {'params': [model.logit_scale], 'lr': args.lr}
    ], weight_decay=1e-4)

    # 4. Training Loop
    os.makedirs(args.out_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n[LOG] Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_c_loss = 0.0
        epoch_g_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training Batch")
        for images, coords in progress_bar:
            images = images.to(device)
            coords = coords.to(device)
            
            optimizer.zero_grad()
            
            # Forward Features
            img_feats = model.image_encoder(images)
            img_feats = F.normalize(img_feats, dim=-1)
            
            loc_feats = model.location_encoder(coords)
            loc_feats = F.normalize(loc_feats, dim=-1)
            
            logit_scale = model.logit_scale.exp()
            
            # Compute Loss
            total_loss, c_loss, g_loss = geoclip_total_loss(
                img_feats, loc_feats, logit_scale, coords, alpha=args.alpha
            )
            
            # Backward & Step
            total_loss.backward()
            optimizer.step()
            
            # Tracking
            epoch_loss += total_loss.item()
            epoch_c_loss += c_loss.item()
            epoch_g_loss += g_loss.item()
            
            progress_bar.set_postfix({
                'Total': f"{total_loss.item():.4f}", 
                'Contrastive': f"{c_loss.item():.4f}",
                'Geographic': f"{g_loss.item():.2f}"
            })
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"[LOG] Epoch {epoch+1} completed. Avg Total Loss: {avg_loss:.4f}")
        
    # 5. Save Checkpoint
    checkpoint_path = os.path.join(args.out_dir, "geoclip_checkpoint.pth")
    torch.save({
        'image_encoder_mlp': model.image_encoder.mlp.state_dict(),
        'location_encoder': model.location_encoder.state_dict(),
        'logit_scale': model.logit_scale
    }, checkpoint_path)
    print(f"[LOG] Model saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train.csv", help="Path to training CSV metadata")
    parser.add_argument("--mock", action="store_true", help="Use mock data without loading real images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.01, help="Geographic loss weight")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory for weights")
    args = parser.parse_args()
    
    train(args)
