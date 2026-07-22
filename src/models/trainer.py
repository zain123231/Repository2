"""
Trainer — trains the geographic cell classifier head.
Uses cross-entropy with geographic distance-based label smoothing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm


class HaversineLabelSmoothing(nn.Module):
    """
    Cross-entropy with geographic distance-based label smoothing.
    Equation from PIGEON: smooth(y)_i = exp(-d(g_i, gt) / tau) / Z
    """
    def __init__(self, cell_centers, temperature=0.1, earth_radius=6371.0):
        super().__init__()
        self.cell_centers = torch.tensor(cell_centers, dtype=torch.float32)
        self.temperature = temperature
        self.earth_radius = earth_radius

    def forward(self, logits, gt_indices, gt_coords):
        """
        Parameters:
            logits : (B, C) raw classifier output
            gt_indices : (B,) ground truth cell indices
            gt_coords : (B, 2) ground truth [lat, lon]
        """
        B, C = logits.shape

        phi1 = torch.deg2rad(self.cell_centers[:, 0:1]).T
        phi2 = torch.deg2rad(gt_coords[:, 0:1])
        dphi = torch.deg2rad(self.cell_centers[:, 0].unsqueeze(0) - gt_coords[:, 0:1])
        dlam = torch.deg2rad(self.cell_centers[:, 1].unsqueeze(0) - gt_coords[:, 1:2])

        a = torch.sin(dphi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dlam / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        dists = self.earth_radius * c  # (B, C)

        smooth_labels = F.softmax(-dists / self.temperature, dim=1)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        return loss


class GeoCellTrainer:
    def __init__(self, model, cell_centers, temperature=0.1, lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.cell_centers = cell_centers
        self.criterion = HaversineLabelSmoothing(cell_centers, temperature).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train(self, train_features, train_labels, train_coords,
              val_features=None, val_labels=None, val_coords=None,
              epochs=50, batch_size=32, save_path=None):
        """
        Train the classifier.
        
        Parameters:
            train_features : np.ndarray (N, D) frozen CLIP features
            train_labels : np.ndarray (N,) cell indices
            train_coords : np.ndarray (N, 2) [lat, lon]
            val_features, val_labels, val_coords : validation data
            epochs : int
            batch_size : int
            save_path : path to save best model
        """
        train_features = torch.tensor(train_features, dtype=torch.float32).to(self.device)
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(self.device)
        train_coords = torch.tensor(train_coords, dtype=torch.float32).to(self.device)
        n_train = len(train_features)

        if val_features is not None:
            val_features = torch.tensor(val_features, dtype=torch.float32).to(self.device)
            val_labels = torch.tensor(val_labels, dtype=torch.long).to(self.device)
            val_coords = torch.tensor(val_coords, dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n_train)
            total_loss = 0
            correct = 0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                features = train_features[idx]
                labels = train_labels[idx]
                coords = train_coords[idx]

                logits = self.model(features)
                loss = self.criterion(logits, labels, coords)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                n_batches += 1

            self.scheduler.step()
            avg_train_loss = total_loss / n_batches
            train_acc = correct / n_train

            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)

            if val_features is not None:
                val_loss, val_acc = self._validate(val_features, val_labels, val_coords, batch_size)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}")

        if save_path and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

        return self.history

    @torch.no_grad()
    def _validate(self, val_features, val_labels, val_coords, batch_size):
        self.model.eval()
        total_loss = 0
        correct = 0
        n = len(val_features)
        n_batches = 0

        for i in range(0, n, batch_size):
            features = val_features[i:i+batch_size]
            labels = val_labels[i:i+batch_size]
            coords = val_coords[i:i+batch_size]

            logits = self.model(features)
            loss = self.criterion(logits, labels, coords)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            n_batches += 1

        return total_loss / n_batches, correct / n

    def save_history(self, path):
        """Save training history to JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_history(self, save_path=None):
        """Plot training curves."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(self.history["train_loss"]) + 1)
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Train")
        if self.history["val_loss"]:
            ax1.plot(epochs, self.history["val_loss"], "r-", label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history["train_acc"], "b-", label="Train")
        if self.history["val_acc"]:
            ax2.plot(epochs, self.history["val_acc"], "r-", label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Cell Classification Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
