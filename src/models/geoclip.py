"""
GeoCLIP-style model — CLIP encoder + linear cell classifier.
Uses frozen CLIP ViT features with a trained classification head.
"""
import torch
import torch.nn as nn
import numpy as np


class GeoCLIPClassifier(nn.Module):
    """
    Linear classifier on frozen CLIP features for geographic cell prediction.
    Cross-entropy loss with geographic distance-based label smoothing.
    """
    def __init__(self, clip_dim=512, n_cells=256, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_cells)
        )

    def forward(self, features):
        return self.classifier(features)

    def predict_cell(self, features):
        logits = self.forward(features)
        return torch.argmax(logits, dim=-1)

    def predict_probs(self, features):
        logits = self.forward(features)
        return torch.softmax(logits, dim=-1)


class GeoCLIPLocationPredictor:
    """
    Full GeoCLIP-style predictor:
    1. Extract CLIP features (frozen)
    2. Classify to cell
    3. Return cell center as prediction (or refine with kNN inside cell)
    """
    def __init__(self, clip_extractor, quadtree, classifier=None, device="cpu"):
        self.clip = clip_extractor
        self.quadtree = quadtree
        self.classifier = classifier
        self.device = device

    def predict(self, images):
        """
        Predict coordinates for a list of PIL images.
        Returns np.ndarray (N, 2) [lat, lon]
        """
        features = self.clip.extract(images)
        features_t = torch.tensor(features).to(self.device)

        if self.classifier is not None:
            self.classifier.eval()
            with torch.no_grad():
                probs = self.classifier.predict_probs(features_t).cpu().numpy()
            predictions = self._aggregate_probs(probs)
        else:
            predictions = self._fallback_predict(features)

        return predictions

    def _aggregate_probs(self, probs):
        """Weighted average of cell centers by predicted probabilities."""
        centers = self.quadtree.get_cell_centers()
        predictions = np.dot(probs, centers)
        return predictions

    def _fallback_predict(self, features):
        """If no classifier, use nearest cell center."""
        centers = self.quadtree.get_cell_centers()
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(features, centers)
        nearest_idx = np.argmin(dists, axis=1)
        return centers[nearest_idx]

    def predict_with_cell_id(self, images):
        """Returns (predictions, cell_ids)."""
        features = self.clip.extract(images)
        features_t = torch.tensor(features).to(self.device)

        if self.classifier is not None:
            self.classifier.eval()
            with torch.no_grad():
                cell_ids = self.classifier.predict_cell(features_t).cpu().numpy()
                probs = self.classifier.predict_probs(features_t).cpu().numpy()
            predictions = self._aggregate_probs(probs)
        else:
            cell_ids = np.zeros(len(features), dtype=int)
            predictions = self._fallback_predict(features)

        return predictions, cell_ids
