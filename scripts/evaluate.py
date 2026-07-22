"""
Main Evaluation Script
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import *
from src.evaluation.metrics import evaluate_geolocation, print_results


def load_results(results_dir):
    """Load evaluation results from files."""
    results = {}
    results_path = Path(results_dir)
    
    for result_file in results_path.glob("*.json"):
        with open(result_file, 'r') as f:
            results[result_file.stem] = json.load(f)
    
    return results


def compare_systems(results):
    """Compare different systems."""
    print("\n" + "=" * 60)
    print(" System Comparison")
    print("=" * 60)
    
    # Create comparison table
    metrics = ['median_error', 'acc@1km', 'acc@25km', 'acc@200km', 'acc@750km', 'acc@2500km']
    
    header = f"{'System':<20}"
    for metric in metrics:
        header += f"{metric:>15}"
    print(header)
    print("-" * (20 + 15 * len(metrics)))
    
    for system_name, system_results in results.items():
        row = f"{system_name:<20}"
        for metric in metrics:
            value = system_results.get(metric, 0)
            if metric == 'median_error':
                row += f"{value:>14.1f}km"
            else:
                row += f"{value:>14.1f}%"
        print(row)
    
    print("=" * 60)


def generate_report(results, output_dir="results"):
    """Generate evaluation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    import pandas as pd
    df = pd.DataFrame(results).T
    df.to_csv(output_path / "comparison_table.csv")
    
    # Generate summary
    summary = {
        'best_system': max(results.keys(), key=lambda k: results[k].get('acc@25km', 0)),
        'best_acc@25km': max(r.get('acc@25km', 0) for r in results.values()),
        'best_median_error': min(r.get('median_error', float('inf')) for r in results.values())
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nReport generated in {output_path}")
    return summary


def main():
    """Main evaluation function."""
    print("=" * 60)
    print(" Single-Image Visual Geolocation System")
    print(" Evaluation Script")
    print("=" * 60)
    
    # Load results
    print("\n[1/3] Loading results...")
    results = load_results(RESULTS_DIR)
    
    if not results:
        print("No results found. Please run training first.")
        return
    
    # Compare systems
    print("\n[2/3] Comparing systems...")
    compare_systems(results)
    
    # Generate report
    print("\n[3/3] Generating report...")
    summary = generate_report(results)
    
    print("\n" + "=" * 60)
    print(" Evaluation completed!")
    print(f" Best system: {summary['best_system']}")
    print(f" Best Acc@25km: {summary['best_acc@25km']:.2f}%")
    print(f" Best Median Error: {summary['best_median_error']:.2f} km")
    print("=" * 60)


if __name__ == "__main__":
    main()
