"""
Visualization Script
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import *
from src.visualization.plots import (
    plot_cdf,
    plot_comparison_bar,
    plot_geographic_distribution,
    create_interactive_map,
    plot_error_by_region
)


def load_predictions(results_dir):
    """Load prediction results from files."""
    predictions = {}
    results_path = Path(results_dir)
    
    for result_file in results_path.glob("*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            if 'predictions' in data:
                predictions[result_file.stem] = data['predictions']
    
    return predictions


def main():
    """Main visualization function."""
    print("=" * 60)
    print(" Single-Image Visual Geolocation System")
    print(" Visualization Script")
    print("=" * 60)
    
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("\n[1/5] Loading predictions...")
    predictions = load_predictions(RESULTS_DIR)
    
    if not predictions:
        print("No predictions found. Please run evaluation first.")
        return
    
    # Generate CDF plots
    print("\n[2/5] Generating CDF plots...")
    for system_name, preds in predictions.items():
        if 'distances' in preds:
            distances = np.array(preds['distances'])
            plot_cdf(
                distances,
                title=f"CDF - {system_name}",
                save_path=FIGURES_DIR / f"cdf_{system_name}.png"
            )
    
    # Generate comparison plots
    print("\n[3/5] Generating comparison plots...")
    # Load summary if available
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Load comparison table
        comparison_path = RESULTS_DIR / "comparison_table.csv"
        if comparison_path.exists():
            import pandas as pd
            df = pd.read_csv(comparison_path, index_col=0)
            
            for metric in ['acc@25km', 'acc@200km', 'median_error']:
                if metric in df.columns:
                    plot_comparison_bar(
                        df[metric].to_dict(),
                        metric=metric,
                        title=f"System Comparison - {metric}",
                        save_path=FIGURES_DIR / f"comparison_{metric}.png"
                    )
    
    # Generate geographic distribution
    print("\n[4/5] Generating geographic maps...")
    for system_name, preds in predictions.items():
        if 'coordinates' in preds:
            coords = np.array(preds['coordinates'])
            plot_geographic_distribution(
                coords,
                title=f"Geographic Distribution - {system_name}",
                save_path=FIGURES_DIR / f"geo_dist_{system_name}.png"
            )
    
    # Generate interactive map
    print("\n[5/5] Generating interactive map...")
    # This would create an interactive map with all predictions
    
    print("\n" + "=" * 60)
    print(" Visualization completed!")
    print(f" Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
