"""
Geographic Quadtree — hierarchical spatial subdivision for geolocation.
Generates cells on training data; each cell has a center, bounds, and sample coordinates.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GeoCell:
    cell_id: int
    center_lat: float
    center_lon: float
    bounds: Tuple[float, float, float, float]  # (min_lat, max_lat, min_lon, max_lon)
    sample_indices: List[int] = field(default_factory=list)
    sample_coords: Optional[np.ndarray] = None
    depth: int = 0

    @property
    def area_deg2(self):
        return (self.bounds[1] - self.bounds[0]) * (self.bounds[3] - self.bounds[2])

    def contains(self, lat, lon):
        return (self.bounds[0] <= lat <= self.bounds[1] and
                self.bounds[2] <= lon <= self.bounds[3])

    def predict(self):
        """Predict coordinates from cell center."""
        return np.array([self.center_lat, self.center_lon])

    def predict_weighted(self, query_features=None, index=None, k=5):
        """Predict using weighted average of sample coordinates."""
        if self.sample_coords is not None and len(self.sample_coords) > 0:
            return np.median(self.sample_coords, axis=0)
        return self.predict()


class GeographicQuadtree:
    def __init__(self, max_cells=256, min_samples_per_cell=10, max_depth=8):
        self.max_cells = max_cells
        self.min_samples = min_samples_per_cell
        self.max_depth = max_depth
        self.cells = []
        self._cell_counter = 0

    def build(self, coordinates):
        """
        Build quadtree from coordinate array.
        
        Parameters:
            coordinates : np.ndarray (N, 2) — [lat, lon]
        """
        self.cells = []
        self._cell_counter = 0
        global_indices = np.arange(len(coordinates))
        self._split(coordinates, global_indices,
                    min_lat=-90, max_lat=90,
                    min_lon=-180, max_lon=180,
                    depth=0)
        print(f"Quadtree built: {len(self.cells)} cells from {len(coordinates)} points")
        return self.cells

    def _split(self, coords, indices, min_lat, max_lat, min_lon, max_lon, depth):
        if len(indices) < self.min_samples or depth >= self.max_depth or len(self.cells) >= self.max_cells:
            self._create_cell(coords, indices, min_lat, max_lat, min_lon, max_lon, depth)
            return

        sub_coords = coords[indices]
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2

        masks = [
            (sub_coords[:, 0] <= mid_lat) & (sub_coords[:, 1] <= mid_lon),
            (sub_coords[:, 0] <= mid_lat) & (sub_coords[:, 1] > mid_lon),
            (sub_coords[:, 0] > mid_lat) & (sub_coords[:, 1] <= mid_lon),
            (sub_coords[:, 0] > mid_lat) & (sub_coords[:, 1] > mid_lon),
        ]
        bounds_list = [
            (min_lat, mid_lat, min_lon, mid_lon),
            (min_lat, mid_lat, mid_lon, max_lon),
            (mid_lat, max_lat, min_lon, mid_lon),
            (mid_lat, max_lat, mid_lon, max_lon),
        ]

        created_any = False
        for mask, bounds in zip(masks, bounds_list):
            sub_indices = indices[mask]
            if len(sub_indices) >= self.min_samples:
                self._split(coords, sub_indices, *bounds, depth + 1)
                created_any = True

        if not created_any:
            self._create_cell(coords, indices, min_lat, max_lat, min_lon, max_lon, depth)

    def _create_cell(self, coords, indices, min_lat, max_lat, min_lon, max_lon, depth):
        cell_coords = coords[indices]
        center = np.median(cell_coords, axis=0)
        cell = GeoCell(
            cell_id=self._cell_counter,
            center_lat=float(center[0]),
            center_lon=float(center[1]),
            bounds=(float(min_lat), float(max_lat), float(min_lon), float(max_lon)),
            sample_indices=indices.tolist(),
            sample_coords=cell_coords,
            depth=depth
        )
        self.cells.append(cell)
        self._cell_counter += 1

    def get_cell_centers(self):
        """Returns (C, 2) array of cell centers."""
        return np.array([[c.center_lat, c.center_lon] for c in self.cells])

    def get_cell_sizes(self):
        """Returns array of cell sample counts."""
        return np.array([len(c.sample_indices) for c in self.cells])

    def assign_to_cell(self, lat, lon):
        """Find which cell contains a point. Returns cell_id or -1."""
        for cell in self.cells:
            if cell.contains(lat, lon):
                return cell.cell_id
        return -1

    def get_nearest_cells(self, lat, lon, k=3):
        """Find k nearest cells by center distance."""
        centers = self.get_cell_centers()
        from src.evaluation.metrics import haversine
        dists = haversine(
            centers[:, 0], centers[:, 1],
            np.array([lat]), np.array([lon])
        )
        nearest_idx = np.argsort(dists)[:k]
        return [self.cells[i] for i in nearest_idx], dists[nearest_idx]

    def save(self, path):
        """Save quadtree data."""
        import json
        data = []
        for cell in self.cells:
            data.append({
                "cell_id": cell.cell_id,
                "center_lat": cell.center_lat,
                "center_lon": cell.center_lon,
                "bounds": cell.bounds,
                "sample_indices": cell.sample_indices,
                "sample_coords": cell.sample_coords.tolist() if cell.sample_coords is not None else None,
                "depth": cell.depth,
            })
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Quadtree saved: {len(self.cells)} cells to {path}")

    def load(self, path):
        """Load quadtree data."""
        import json
        with open(path) as f:
            data = json.load(f)
        self.cells = []
        for d in data:
            cell = GeoCell(
                cell_id=d["cell_id"],
                center_lat=d["center_lat"],
                center_lon=d["center_lon"],
                bounds=tuple(d["bounds"]),
                sample_indices=d["sample_indices"],
                sample_coords=np.array(d["sample_coords"]) if d["sample_coords"] else None,
                depth=d["depth"],
            )
            self.cells.append(cell)
        print(f"Quadtree loaded: {len(self.cells)} cells")


if __name__ == "__main__":
    np.random.seed(42)
    coords = np.column_stack([
        np.random.uniform(-90, 90, 1000),
        np.random.uniform(-180, 180, 1000)
    ])
    qt = GeographicQuadtree(max_cells=50, min_samples_per_cell=20)
    cells = qt.build(coords)
    print(f"Cells: {len(cells)}")
    print(f"Cell sizes: {qt.get_cell_sizes()}")
    nearest, dists = qt.get_nearest_cells(40.7128, -74.006)
    print(f"Nearest cell to NYC: id={nearest[0].cell_id}, dist={dists[0]:.1f}km")
