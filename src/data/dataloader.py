"""
Data loader — handles Im2GPS3k, YFCC4k, OSV-5M, GLDv2 datasets.
All data is loaded from pre-downloaded .npy files.
"""
import numpy as np
import os
import json
from PIL import Image


class GeoDataset:
    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir
        self.coordinates = None
        self.features = None
        self.images = None
        self.metadata = None
        self._loaded = False

    def load(self):
        """Load dataset from disk."""
        coord_path = os.path.join(self.data_dir, "coordinates.npy")
        if os.path.exists(coord_path):
            self.coordinates = np.load(coord_path)
        else:
            raise FileNotFoundError(f"Coordinates not found: {coord_path}")

        feat_path = os.path.join(self.data_dir, "features.npy")
        if os.path.exists(feat_path):
            self.features = np.load(feat_path)

        meta_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        self._loaded = True
        print(f"Loaded {self.name}: {len(self.coordinates)} samples")
        return self

    @property
    def n_samples(self):
        return len(self.coordinates) if self.coordinates is not None else 0

    @property
    def has_features(self):
        return self.features is not None

    def get_coords(self):
        return self.coordinates.copy()

    def get_features(self):
        if self.features is None:
            raise ValueError(f"{self.name}: features not loaded")
        return self.features.copy()

    def get_subset(self, indices):
        """Return a subset of the dataset."""
        subset = GeoDataset(f"{self.name}_subset", self.data_dir)
        subset.coordinates = self.coordinates[indices]
        if self.features is not None:
            subset.features = self.features[indices]
        return subset


class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.datasets = {}

    def load_im2gps3k(self):
        ds = GeoDataset("im2gps3k", os.path.join(self.base_dir, "im2gps3k"))
        ds.load()
        self.datasets["im2gps3k"] = ds
        return ds

    def load_yfcc4k(self):
        ds = GeoDataset("yfcc4k", os.path.join(self.base_dir, "yfcc4k"))
        ds.load()
        self.datasets["yfcc4k"] = ds
        return ds

    def load_osv5m(self, max_samples=None):
        ds = GeoDataset("osv5m", os.path.join(self.base_dir, "osv5m"))
        ds.load()
        if max_samples and ds.n_samples > max_samples:
            indices = np.random.RandomState(42).choice(ds.n_samples, max_samples, replace=False)
            ds = ds.get_subset(indices)
            ds.name = "osv5m_subset"
        self.datasets["osv5m"] = ds
        return ds

    def load_gldv2(self):
        ds = GeoDataset("gldv2", os.path.join(self.base_dir, "gldv2"))
        ds.load()
        self.datasets["gldv2"] = ds
        return ds

    def load_all(self, osv5m_max=10000):
        self.load_im2gps3k()
        self.load_yfcc4k()
        self.load_osv5m(max_samples=osv5m_max)
        return self.datasets


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.config import RAW_DIR
    loader = DataLoader(RAW_DIR)
    datasets = loader.load_all()
    for name, ds in datasets.items():
        print(f"  {name}: {ds.n_samples} samples, has_features={ds.has_features}")
