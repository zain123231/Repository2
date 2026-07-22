"""
FAISS Index for geolocation retrieval.
Supports add, search, save, load.
"""
import numpy as np
import faiss
import os
import time


class FAISSGeolocIndex:
    def __init__(self, dim=512, use_gpu=False):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.coordinates = None
        self.ids = None

    def build(self, features, coordinates, ids=None):
        """
        Build index from features.
        
        Parameters:
            features : np.ndarray (N, dim) float32
            coordinates : np.ndarray (N, 2) — [lat, lon]
            ids : optional array of IDs
        """
        assert features.shape[1] == self.dim
        assert len(features) == len(coordinates)
        features = np.ascontiguousarray(features.astype(np.float32))
        self.index.add(features)
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.ids = ids if ids is not None else np.arange(len(features))
        print(f"FAISS index built: {self.index.ntotal} vectors")

    def search(self, query_features, k=5):
        """
        Search for k nearest neighbors.
        
        Returns:
            distances : np.ndarray (Q, k)
            indices : np.ndarray (Q, k) — indices into self.coordinates
            matched_coords : np.ndarray (Q, k, 2) — [lat, lon] of matches
        """
        query_features = np.ascontiguousarray(query_features.astype(np.float32))
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_features, k)
        matched_coords = self.coordinates[indices]
        return distances, indices, matched_coords

    def predict_knn(self, query_features, k=5):
        """kNN prediction: median of k nearest neighbors."""
        _, _, matched_coords = self.search(query_features, k=k)
        predictions = np.median(matched_coords, axis=1)
        return predictions

    def predict_knn_weighted(self, query_features, k=5):
        """Weighted kNN: inverse-distance weighted average."""
        distances, _, matched_coords = self.search(query_features, k=k)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        predictions = np.sum(matched_coords * weights[:, :, np.newaxis], axis=1)
        return predictions

    def benchmark(self, query_features, k=5):
        """Measure search time per query."""
        start = time.time()
        self.search(query_features, k=k)
        elapsed = time.time() - start
        per_query = elapsed / len(query_features) * 1000
        print(f"FAISS benchmark: {len(query_features)} queries in {elapsed:.3f}s ({per_query:.1f} ms/query)")
        return per_query

    def save(self, path):
        """Save index and coordinates."""
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        np.save(os.path.join(path, "coordinates.npy"), self.coordinates)
        np.save(os.path.join(path, "ids.npy"), self.ids)
        print(f"FAISS index saved to {path}")

    def load(self, path):
        """Load index and coordinates."""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        self.coordinates = np.load(os.path.join(path, "coordinates.npy"))
        self.ids = np.load(os.path.join(path, "ids.npy"))
        print(f"FAISS index loaded: {self.index.ntotal} vectors")

    @property
    def size(self):
        return self.index.ntotal
