"""
Visualization — publication-quality figures (300 DPI).
CDF, comparison bars, median error, geographic maps, qualitative examples.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import os
import json


COLORS = {
    "Random": "#e74c3c",
    "Nearest Centroid": "#e67e22",
    "kNN (FAISS)": "#f39c12",
    "GeoCLIP (Cells)": "#2ecc71",
    "StreetCLIP ZS": "#3498db",
    "Popularity-weighted": "#9b59b6",
    "Hybrid v1.0": "#1abc9c",
}


def plot_cdf(results_dict, save_path, title="Cumulative Error Distribution"):
    """CDF plot for all systems on one dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result in results_dict.items():
        distances = result.get("distances")
        if distances is None:
            continue
        sorted_d = np.sort(distances)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
        color = COLORS.get(name, "#333333")
        ax.plot(sorted_d, cdf, label=name, color=color, linewidth=2)

    ax.axvline(x=25, color="gray", linestyle="--", alpha=0.5, label="25 km (city)")
    ax.axvline(x=200, color="gray", linestyle=":", alpha=0.5, label="200 km (region)")
    ax.axvline(x=750, color="gray", linestyle="-.", alpha=0.5, label="750 km (country)")
    ax.set_xlabel("Haversine Distance (km)", fontsize=12)
    ax.set_ylabel("Percentage of Predictions (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 20000)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_comparison(results_per_dataset, save_path):
    """Grouped bar chart: Acc@tau for all systems across datasets."""
    from src.config import ACC_THRESHOLDS_KM
    dataset_names = list(results_per_dataset.keys())
    system_names = list(results_per_dataset[dataset_names[0]].keys())
    n_datasets = len(dataset_names)
    n_systems = len(system_names)
    n_thresholds = len(ACC_THRESHOLDS_KM)

    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        x = np.arange(n_thresholds)
        width = 0.8 / n_systems

        for i, sys_name in enumerate(system_names):
            result = results_per_dataset[ds_name][sys_name]
            accs = [result.get(f"acc@{tau}km", 0) * 100 for tau in ACC_THRESHOLDS_KM]
            color = COLORS.get(sys_name, "#333333")
            ax.bar(x + i * width, accs, width, label=sys_name, color=color, alpha=0.85)

        ax.set_xlabel("Threshold", fontsize=11)
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "", fontsize=11)
        ax.set_title(ds_name, fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * n_systems / 2)
        ax.set_xticklabels([f"{t}km" for t in ACC_THRESHOLDS_KM], fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_median_error_comparison(results_per_dataset, save_path):
    """Horizontal bar chart: median error comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dataset_names = list(results_per_dataset.keys())
    system_names = list(results_per_dataset[dataset_names[0]].keys())

    y_positions = np.arange(len(system_names))
    bar_height = 0.35

    for i, ds_name in enumerate(dataset_names):
        medians = [results_per_dataset[ds_name][sys].get("median_error_km", 0) for sys in system_names]
        offset = (i - len(dataset_names) / 2 + 0.5) * bar_height
        color = ["#3498db", "#e74c3c"][i % 2]
        bars = ax.barh(y_positions + offset, medians, bar_height, label=ds_name, color=color, alpha=0.8)
        for bar, val in zip(bars, medians):
            ax.text(val + 50, bar.get_y() + bar.get_height() / 2, f"{val:.0f}",
                    va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(system_names, fontsize=10)
    ax.set_xlabel("Median Haversine Error (km)", fontsize=12)
    ax.set_title("Median Error Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_geographic_distribution(preds, gt, save_path, title="Geographic Distribution"):
    """Scatter plot of predictions vs ground truth on world map."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.scatter(gt[:, 1], gt[:, 0], c="#3498db", alpha=0.3, s=5, label="Ground Truth")
    ax.scatter(preds[:, 1], preds[:, 0], c="#e74c3c", alpha=0.3, s=5, label="Predictions")

    errors = np.sqrt(np.sum((preds - gt) ** 2, axis=1))
    worst_idx = np.argsort(errors)[-20:]
    for idx in worst_idx:
        ax.plot([gt[idx, 1], preds[idx, 1]], [gt[idx, 0], preds[idx, 0]],
                "r-", alpha=0.5, linewidth=0.5)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distribution(distances, save_path, title="Error Distribution"):
    """Histogram of prediction errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(distances, bins=100, color="#3498db", alpha=0.7, edgecolor="white")
    ax1.axvline(x=np.median(distances), color="red", linestyle="--",
                label=f"Median: {np.median(distances):.0f} km")
    ax1.set_xlabel("Haversine Distance (km)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Linear Scale", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    ax2.hist(distances, bins=100, color="#3498db", alpha=0.7, edgecolor="white", log=True)
    ax2.axvline(x=np.median(distances), color="red", linestyle="--",
                label=f"Median: {np.median(distances):.0f} km")
    ax2.set_xlabel("Haversine Distance (km)", fontsize=11)
    ax2.set_ylabel("Count (log)", fontsize=11)
    ax2.set_title("Log Scale", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_qualitative_examples(images, predictions, ground_truth, distances,
                               save_path, n_success=4, n_failure=4):
    """
    Qualitative examples: top-n success and top-n failure cases.
    Each example shows the image + predicted vs ground truth location.
    """
    n = n_success + n_failure
    sorted_idx = np.argsort(distances)

    success_idx = sorted_idx[:n_success]
    failure_idx = sorted_idx[-n_failure:][::-1]
    selected_idx = np.concatenate([success_idx, failure_idx])

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    titles = ["Success"] * n_success + ["Failure"] * n_failure

    for col, (idx, title) in enumerate(zip(selected_idx, titles)):
        ax_img = axes[0, col]
        ax_map = axes[1, col]

        if images[idx] is not None:
            ax_img.imshow(images[idx])
        ax_img.set_title(f"{title}\n{distances[idx]:.0f} km", fontsize=9,
                         color="green" if title == "Success" else "red")
        ax_img.axis("off")

        gt_lat, gt_lon = ground_truth[idx]
        pred_lat, pred_lon = predictions[idx]
        ax_map.scatter([gt_lon], [gt_lat], c="blue", s=50, zorder=5, label="GT")
        ax_map.scatter([pred_lon], [pred_lat], c="red", s=50, zorder=5, label="Pred")
        ax_map.plot([gt_lon, pred_lon], [gt_lat, pred_lat], "k--", alpha=0.5)
        ax_map.set_xlim(gt_lon - 5, gt_lon + 5)
        ax_map.set_ylim(gt_lat - 5, gt_lat + 5)
        ax_map.set_aspect("equal")
        if col == 0:
            ax_map.legend(fontsize=8)

    plt.suptitle("Qualitative Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_geographical_breakdown(per_region_results, save_path):
    """Bar chart of performance per continent/region."""
    regions = list(per_region_results.keys())
    system_names = list(per_region_results[regions[0]].keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(regions))
    width = 0.8 / len(system_names)

    for i, sys_name in enumerate(system_names):
        medians = [per_region_results[r][sys_name].get("median_error_km", 0) for r in regions]
        color = COLORS.get(sys_name, "#333333")
        ax.bar(x + i * width, medians, width, label=sys_name, color=color, alpha=0.85)

    ax.set_xticks(x + width * len(system_names) / 2)
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel("Median Error (km)", fontsize=12)
    ax.set_title("Performance by Geographic Region", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, save_path):
    """Training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], "r-", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    if history.get("val_acc"):
        ax2.plot(epochs, history["val_acc"], "r-", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Cell Classification Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
