"""
CLIP Feature Extractor — uses open_clip (clip-anytorch).
Supports ViT-B/32 (512-d) and ViT-B/16 (512-d).
"""
import torch
import numpy as np
import clip
from PIL import Image
import time


class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading CLIP {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.dim = 512
        print(f"CLIP {model_name} loaded. Feature dim={self.dim}")

    @torch.no_grad()
    def extract(self, images, batch_size=32):
        """
        Extract features from list of PIL images.
        Returns np.ndarray (N, dim) L2-normalized.
        """
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0).astype(np.float32)

    @torch.no_grad()
    def extract_single(self, image):
        """Extract features from a single PIL image."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feature = self.model.encode_image(tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        return feature.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts):
        """Encode text strings to feature vectors."""
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)


def benchmark_extractor(extractor, n_images=50):
    """Measure inference time per image."""
    dummy_images = [Image.new("RGB", (224, 224)) for _ in range(n_images)]
    start = time.time()
    features = extractor.extract(dummy_images)
    elapsed = time.time() - start
    per_image = elapsed / n_images * 1000
    print(f"Benchmark: {n_images} images in {elapsed:.2f}s ({per_image:.1f} ms/image)")
    return per_image


if __name__ == "__main__":
    extractor = CLIPFeatureExtractor("ViT-B/32")
    print(f"Feature dim: {extractor.dim}")
    dummy = Image.new("RGB", (224, 224))
    feat = extractor.extract_single(dummy)
    print(f"Single feature shape: {feat.shape}")
    benchmark_extractor(extractor, n_images=20)
