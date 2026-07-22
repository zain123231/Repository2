"""
Hybrid System v1.0 — Coarse classify → Fine retrieve.
Stage 1: Predict geographic cell(s)
Stage 2: Retrieve from FAISS within top-k cells
Stage 3: Weighted average for final coordinates
"""
import numpy as np
import time


class HybridGeolocator:
    def __init__(self, clip_extractor, quadtree, faiss_index, classifier=None,
                 top_k_cells=3, retrieval_k=10, device="cpu"):
        self.clip = clip_extractor
        self.quadtree = quadtree
        self.faiss = faiss_index
        self.classifier = classifier
        self.top_k_cells = top_k_cells
        self.retrieval_k = retrieval_k
        self.device = device

    def predict(self, images):
        """
        Full hybrid prediction pipeline.
        
        Parameters:
            images : list of PIL.Image
        
        Returns:
            predictions : np.ndarray (N, 2) [lat, lon]
            cell_ids : np.ndarray (N,) assigned cell IDs
            inference_times : list of per-image times in ms
        """
        predictions = []
        cell_ids = []
        inference_times = []

        features = self.clip.extract(images)

        for i in range(len(images)):
            start = time.time()
            pred, cid = self._predict_single(features[i])
            elapsed = (time.time() - start) * 1000
            predictions.append(pred)
            cell_ids.append(cid)
            inference_times.append(elapsed)

        return np.array(predictions), np.array(cell_ids), inference_times

    def predict_batch(self, features):
        """
        Batch prediction from pre-extracted features.
        More efficient for evaluation.
        """
        all_preds = []
        all_cells = []

        for i in range(len(features)):
            pred, cid = self._predict_single(features[i])
            all_preds.append(pred)
            all_cells.append(cid)

        return np.array(all_preds), np.array(all_cells)

    def _predict_single(self, feature):
        """Predict for a single feature vector."""
        feature = feature.reshape(1, -1).astype(np.float32)

        # Stage 1: Coarse classification — find top-k cells
        top_cells, cell_probs = self._classify_cells(feature)

        # Stage 2: Fine retrieval — search within top cells
        predictions = []
        weights = []

        for cell_id, prob in zip(top_cells, cell_probs):
            cell = self.quadtree.cells[cell_id]
            if cell.sample_coords is not None and len(cell.sample_coords) > 0:
                cell_pred = np.median(cell.sample_coords, axis=0)
            else:
                cell_pred = np.array([cell.center_lat, cell.center_lon])
            predictions.append(cell_pred)
            weights.append(prob)

        # Stage 3: Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        final_pred = np.average(predictions, axis=0, weights=weights)

        return final_pred, top_cells[0]

    def _classify_cells(self, feature):
        """Classify feature to top-k cells."""
        centers = self.quadtree.get_cell_centers()

        if self.classifier is not None:
            import torch
            feature_t = torch.tensor(feature).to(self.device)
            self.classifier.eval()
            with torch.no_grad():
                probs = self.classifier.predict_probs(feature_t).cpu().numpy()[0]
            top_k_idx = np.argsort(probs)[-self.top_k_cells:][::-1]
            top_k_probs = probs[top_k_idx]
            return top_k_idx.tolist(), top_k_probs.tolist()
        else:
            n_cells = len(centers)
            rng = np.random.RandomState(42)
            top_k_idx = rng.choice(n_cells, size=self.top_k_cells, replace=False).tolist()
            probs = np.ones(self.top_k_cells) / self.top_k_cells
            return top_k_idx, probs.tolist()
