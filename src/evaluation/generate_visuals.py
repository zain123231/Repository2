import sys
import os
import json
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories
for d in [RESULTS_DIR, FIGURES_DIR,
          FIGURES_DIR / "cdf", FIGURES_DIR / "comparison",
          FIGURES_DIR / "error_analysis", FIGURES_DIR / "geographic",
          FIGURES_DIR / "qualitative"]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Map our A1-A4 variants to visually appealing names
VARIANT_NAMES = {
    "A1": "Baseline (GeoCLIP 100K)",
    "A2": "A2: +TTA",
    "A3": "A3: +TTA + Micro-Grid",
    "A4": "A4: +TTA + Micro-Grid + OCR"
}

COLORS = {
    "Baseline (GeoCLIP 100K)": "#e74c3c",  # Red
    "A2: +TTA": "#e67e22",                 # Orange
    "A3: +TTA + Micro-Grid": "#f1c40f",    # Yellow
    "A4: +TTA + Micro-Grid + OCR": "#2ecc71" # Green
}

ACC_THRESHOLDS = [1, 25, 200, 750, 2500]
ACC_LABELS = ["Street\n1km", "City\n25km", "Region\n200km", "Country\n750km", "Continent\n2500km"]

def acc_at_tau(distances, thresholds):
    return {t: float(np.mean(distances <= t) * 100) for t in thresholds}

def generate_figures(csv_path):
    print(f"\n[LOG] Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    ds_name = "Im2GPS3k"
    all_distances = {ds_name: {}}
    all_results = {ds_name: {}}
    
    # Process each variant
    for var_code, var_name in VARIANT_NAMES.items():
        var_df = df[df['VARIANT'] == var_code]
        if len(var_df) == 0:
            continue
            
        dists = var_df['ERROR_KM'].values
        all_distances[ds_name][var_name] = dists
        
        accs = acc_at_tau(dists, ACC_THRESHOLDS)
        med = float(np.median(dists))
        mn = float(np.mean(dists))
        
        res = {
            "n_samples": len(dists),
            "median_error_km": round(med, 1),
            "mean_error_km": round(mn, 1),
        }
        for t, a in accs.items():
            res[f"acc@{t}km"] = round(a, 1)
            
        all_results[ds_name][var_name] = res

    if not all_distances[ds_name]:
        print("[ERROR] No data found in CSV for A1-A4 variants.")
        return

    print("\n" + "=" * 70)
    print("  GENERATING UNIFIED FIGURES (300 DPI)")
    print("=" * 70)

    # ---- Figure 1: CDF Plot ----
    print("\n  [1/6] CDF Plot...")
    for ds in all_distances:
        fig, ax = plt.subplots(figsize=(10, 6))
        for sys_name, distances in all_distances[ds].items():
            sorted_d = np.sort(distances)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
            ax.plot(sorted_d, cdf, label=sys_name, color=COLORS.get(sys_name, "#333"),
                    linewidth=2.5, alpha=0.9)

        for thresh, label, ls in [(25, "City (25km)", "--"), (200, "Region (200km)", ":"), (750, "Country (750km)", "-.")]:
            ax.axvline(x=thresh, color="gray", linestyle=ls, alpha=0.4, linewidth=1)
            ax.text(thresh * 1.1, 5, label, fontsize=8, color="gray", rotation=90, va='bottom')

        ax.set_xscale("log")
        ax.set_xlim(0.1, 20000)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Haversine Distance (km)", fontsize=13)
        ax.set_ylabel("Percentage of Predictions (%)", fontsize=13)
        ax.set_title(f"Cumulative Error Distribution — {ds}", fontsize=15, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#fafafa")
        plt.tight_layout()
        path = FIGURES_DIR / "cdf" / f"cdf_unified_{ds.lower()}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    [OK] {path.name}")

    # ---- Figure 2: Accuracy Comparison Bar Chart ----
    print("  [2/6] Accuracy Comparison Bar Chart...")
    for ds in all_results:
        fig, ax = plt.subplots(figsize=(14, 7))
        sys_names = list(all_results[ds].keys())
        n_sys = len(sys_names)
        x = np.arange(len(ACC_THRESHOLDS))
        width = 0.15

        for i, sys_name in enumerate(sys_names):
            r = all_results[ds][sys_name]
            vals = [r[f"acc@{t}km"] for t in ACC_THRESHOLDS]
            bars = ax.bar(x + i * width, vals, width, label=sys_name,
                         color=COLORS.get(sys_name, "#333"), alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 3:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f"{val:.1f}", ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xlabel("Distance Threshold", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title(f"Accuracy at Distance Thresholds — {ds}", fontsize=15, fontweight="bold")
        ax.set_xticks(x + width * n_sys / 2)
        ax.set_xticklabels(ACC_LABELS, fontsize=10)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.15, axis="y")
        ax.set_facecolor("#fafafa")
        plt.tight_layout()
        path = FIGURES_DIR / "comparison" / f"accuracy_unified_{ds.lower()}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✅ {path.name}")

    # ---- Figure 3: Error Distribution Histogram ----
    print("  [3/6] Error Distribution Histogram...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pick the best variant (A4) if it exists, else use whatever we have
    best_sys = "A4: +TTA + Micro-Grid + OCR"
    if best_sys not in all_distances[ds_name]:
        best_sys = list(all_distances[ds_name].keys())[-1]
        
    dists = all_distances[ds_name][best_sys]

    ax1.hist(dists, bins=80, color="#3498db", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax1.axvline(x=np.median(dists), color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Median: {np.median(dists):.0f} km")
    ax1.set_xlabel("Haversine Distance (km)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Error Distribution (Linear Scale)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.15)

    ax2.hist(dists[dists > 0], bins=80, color="#3498db", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.axvline(x=np.median(dists), color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Median: {np.median(dists):.0f} km")
    ax2.set_xlabel("Haversine Distance (km, log scale)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Error Distribution (Log Scale)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.15)

    plt.suptitle(f"Error Distribution — {best_sys}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "error_analysis" / "error_distribution_unified.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✅ {path.name}")

    # ---- Figure 4: Regional Performance Heatmap ----
    print("  [4/6] Regional Performance Heatmap...")
    regions = {
        "Europe": (36, 60, -10, 40),
        "N. America": (25, 55, -130, -65),
        "E. Asia": (20, 50, 100, 145),
        "Middle East": (12, 42, 25, 63),
        "S. America": (-55, 15, -80, -35),
        "Africa": (-35, 37, -20, 50),
        "S. Asia": (5, 35, 65, 100),
        "Oceania": (-45, -10, 110, 180),
    }

    # Use A4 predictions for map
    best_df = df[df['VARIANT'] == list(VARIANT_NAMES.keys())[-1]]
    if len(best_df) > 0:
        gt_lats = best_df['TRUE_LAT'].values
        gt_lons = best_df['TRUE_LON'].values
        dists_full = best_df['ERROR_KM'].values

        region_medians = {}
        region_counts = {}
        for rname, (la1, la2, lo1, lo2) in regions.items():
            mask = (gt_lats >= la1) & (gt_lats <= la2) & (gt_lons >= lo1) & (gt_lons <= lo2)
            if mask.sum() > 0:
                region_medians[rname] = float(np.median(dists_full[mask]))
                region_counts[rname] = int(mask.sum())
            else:
                region_medians[rname] = 0
                region_counts[rname] = 0

        fig, ax = plt.subplots(figsize=(12, 6))
        rnames = list(region_medians.keys())
        medians = [region_medians[r] for r in rnames]
        counts = [region_counts[r] for r in rnames]
        colors_bar = plt.cm.RdYlGn_r(np.array(medians) / max(max(medians), 1))

        bars = ax.bar(rnames, medians, color=colors_bar, alpha=0.88, edgecolor="white", linewidth=1.5)
        for bar, val, cnt in zip(bars, medians, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(medians)*0.02,
                    f"{val:.0f} km\n(n={cnt})", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel("Median Error (km)", fontsize=13)
        ax.set_title("Unified Performance by Geographic Region (A4 Variant)", fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.15, axis="y")
        ax.set_facecolor("#fafafa")
        plt.xticks(rotation=15, fontsize=11)
        plt.tight_layout()
        path = FIGURES_DIR / "error_analysis" / "region_performance_unified.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✅ {path.name}")
        
    print(f"\n  All figures successfully generated in {FIGURES_DIR}/")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the most recent run in results
        results_dir = PROJECT_ROOT / "results"
        import glob
        csv_files = glob.glob(str(results_dir / "202*" / "csv" / "detailed_predictions.csv"))
        if not csv_files:
            print("[ERROR] Could not find any detailed_predictions.csv files.")
            sys.exit(1)
        # Get latest
        csv_file = sorted(csv_files)[-1]
        
    generate_figures(csv_file)
