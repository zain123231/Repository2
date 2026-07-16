import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np

class GeoDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str = None, mock: bool = False, size: int = 100):
        """
        Dataset for GeoCLIP training.
        Args:
            csv_file (str): Path to the csv file with annotations (IMG_PATH, LAT, LON).
            img_dir (str, optional): Directory with all the images.
            mock (bool): If True, generates random tensors and coordinates instead of reading disk.
            size (int): Size of the mock dataset.
        """
        self.mock = mock
        self.size = size
        self.img_dir = img_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        if not self.mock:
            self.data_frame = pd.read_csv(csv_file)
            
    def __len__(self):
        if self.mock:
            return self.size
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.mock:
            # Generate random image tensor
            img = torch.randn(3, 224, 224)
            # Generate random GPS coordinates
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            coords = torch.tensor([lat, lon], dtype=torch.float32)
            return img, coords

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx]['IMG_PATH']
        if self.img_dir:
            img_path = os.path.join(self.img_dir, img_path)
            
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Fallback to random if image is missing/corrupted
            image = torch.zeros(3, 224, 224)
            
        lat = self.data_frame.iloc[idx]['LAT']
        lon = self.data_frame.iloc[idx]['LON']
        coords = torch.tensor([lat, lon], dtype=torch.float32)

        return image, coords

if __name__ == "__main__":
    print("[LOG] Testing GeoDataset (Mock)...")
    ds = GeoDataset(csv_file=None, mock=True, size=10)
    img, coord = ds[0]
    print(f"Image shape: {img.shape}")
    print(f"Coord shape: {coord.shape} -> {coord}")
