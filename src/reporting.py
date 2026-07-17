import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_reporting_directories():
    """Creates timestamped directories for the evaluation results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", timestamp)
    
    dirs = {
        "base": base_dir,
        "tables": os.path.join(base_dir, "tables"),
        "figures": os.path.join(base_dir, "figures"),
        "logs": os.path.join(base_dir, "logs"),
        "csv": os.path.join(base_dir, "csv"),
        "latex": os.path.join(base_dir, "latex")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    return dirs

def save_figure(fig, filename, fig_dir):
    """Saves a matplotlib figure in PNG, PDF, and SVG formats at 300 DPI."""
    for ext in ['png', 'pdf', 'svg']:
        path = os.path.join(fig_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', format=ext)

def plot_cdf(errors_km, fig_dir):
    """Generates and saves the CDF curve."""
    if len(errors_km) == 0: return
    
    sorted_errors = np.sort(errors_km)
    p = 1. * np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax.plot(sorted_errors, p, marker='', color='b', linewidth=2.5, label='GeoCLIP Zero-Shot')
    
    thresholds = [1, 25, 200, 750, 2500]
    for t in thresholds:
        ax.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        
    ax.set_xscale('symlog')
    ax.set_xlim(left=0, right=20000)
    ax.set_ylim(0, 1.05)
    
    ax.set_xlabel("Localization Error (km)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Fraction of Images", fontsize=14, fontweight='bold')
    ax.set_title("Cumulative Distribution of Localization Error", fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    save_figure(fig, "cdf_curve", fig_dir)
    plt.close(fig)

def plot_histogram(errors_km, fig_dir):
    """Generates and saves a histogram of localization errors."""
    if len(errors_km) == 0: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors_km, bins=50, log_scale=(True, False), color='purple', kde=True, ax=ax)
    
    ax.set_xlabel("Localization Error (km) [Log Scale]", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.set_title("Localization Error Histogram", fontsize=16, fontweight='bold')
    
    save_figure(fig, "error_histogram", fig_dir)
    plt.close(fig)

def plot_boxplot(errors_km, fig_dir):
    """Generates and saves a boxplot of localization errors."""
    if len(errors_km) == 0: return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=errors_km, color='cyan', ax=ax)
    ax.set_yscale('symlog')
    
    ax.set_ylabel("Localization Error (km)", fontsize=14, fontweight='bold')
    ax.set_title("Error Distribution (Box Plot)", fontsize=16, fontweight='bold')
    
    save_figure(fig, "error_boxplot", fig_dir)
    plt.close(fig)

def plot_accuracy_bar(metrics_dict, fig_dir):
    """Generates and saves a bar chart of threshold accuracies."""
    thresholds = [1, 25, 200, 750, 2500]
    accs = [metrics_dict[f'Acc@{t}km'] for t in thresholds]
    labels = [f'{t} km' for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=labels, y=accs, palette="viridis", ax=ax)
    
    ax.set_xlabel("Radius Threshold", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
    ax.set_title("Localization Accuracy by Threshold", fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        
    save_figure(fig, "accuracy_chart", fig_dir)
    plt.close(fig)

def save_tables(df, filename_base, dirs):
    """Saves a dataframe to CSV, Markdown, and LaTeX formats."""
    # Save CSV
    df.to_csv(os.path.join(dirs['csv'], f"{filename_base}.csv"), index=False)
    
    # Save Markdown
    with open(os.path.join(dirs['tables'], f"{filename_base}.md"), 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
        
    # Save LaTeX
    with open(os.path.join(dirs['latex'], f"{filename_base}.tex"), 'w', encoding='utf-8') as f:
        f.write(df.style.to_latex())

def generate_report(errors_km, results_list, total_runtime_sec, dataset_name="Im2GPS3k"):
    """
    Main entry point for generating the scientific report and all associated artifacts.
    """
    print("[LOG] Initiating automated scientific reporting...")
    dirs = create_reporting_directories()
    
    errors_km = np.array(errors_km)
    num_images = len(errors_km)
    
    # Calculate Metrics
    metrics = {}
    if num_images > 0:
        metrics['Mean Error (km)'] = np.mean(errors_km)
        metrics['Median Error (km)'] = np.median(errors_km)
        thresholds = [1, 25, 200, 750, 2500]
        for t in thresholds:
            metrics[f'Acc@{t}km'] = np.mean(errors_km <= t) * 100
            
    metrics['Evaluated Images'] = num_images
    metrics['Total Runtime (s)'] = total_runtime_sec
    metrics['Avg Inference Time (s/img)'] = total_runtime_sec / num_images if num_images > 0 else 0
    
    # 1. Generate Figures
    print("[LOG] Generating publication-quality figures...")
    plot_cdf(errors_km, dirs['figures'])
    plot_histogram(errors_km, dirs['figures'])
    plot_boxplot(errors_km, dirs['figures'])
    if num_images > 0:
        plot_accuracy_bar(metrics, dirs['figures'])
        
    # 2. Generate Tables
    print("[LOG] Generating publication-ready tables...")
    summary_df = pd.DataFrame([metrics])
    save_tables(summary_df, "evaluation_summary", dirs)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(dirs['csv'], "detailed_predictions.csv"), index=False)
    
    # 3. Generate Scientific Markdown Report
    print("[LOG] Writing scientific report...")
    report_path = os.path.join(dirs['base'], "report.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"# Automated Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Evaluation Setup\n")
        f.write(f"- **Dataset:** {dataset_name}\n")
        f.write(f"- **Images Evaluated:** {num_images}\n")
        f.write(f"- **Model:** GeoCLIP (Official Pretrained)\n")
        f.write(f"- **Runtime:** {total_runtime_sec:.2f} seconds\n")
        f.write(f"- **Avg Inference Time:** {metrics['Avg Inference Time (s/img)']:.2f} seconds/image\n")
        f.write(f"- **Hardware:** GPU (if available) / CPU\n\n")
        
        f.write("## 2. Benchmark Metrics\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        
        f.write("## 3. Artifact Locations\n")
        f.write("- **Figures (PNG/PDF/SVG):** `figures/`\n")
        f.write("  - `cdf_curve.*`: CDF of localization errors.\n")
        f.write("  - `error_histogram.*`: Histogram distribution of errors.\n")
        f.write("  - `error_boxplot.*`: Box plot highlighting error variance.\n")
        f.write("  - `accuracy_chart.*`: Bar chart of Accuracy @ Thresholds.\n")
        f.write("- **Tables (CSV/Markdown/LaTeX):** `tables/`, `csv/`, `latex/`\n")
        f.write("  - `evaluation_summary.*`: Core metrics for publication insertion.\n")
        f.write("- **Detailed Logs & Data:** `csv/detailed_predictions.csv`\n")
        
    print(f"[LOG] Reporting complete. All artifacts saved in: {dirs['base']}")
    return dirs['base']
