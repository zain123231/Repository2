import os
import sys
import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except:
        return "Unknown"

def get_pip_freeze():
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
    except:
        return "Unknown"

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

def write_captions(fig_dir, captions_dict):
    """Writes academic captions for figures into a markdown file."""
    path = os.path.join(fig_dir, "captions.md")
    with open(path, "w", encoding='utf-8') as f:
        f.write("# Figure Captions\n\n")
        for k, v in captions_dict.items():
            f.write(f"**{k}**: {v}\n\n")

def plot_cdf_multi(systems_errors, fig_dir):
    """Generates and saves the CDF curve for multiple systems."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    colors = ['b', 'r', 'g', 'm', 'c']
    for idx, (system_name, errors) in enumerate(systems_errors.items()):
        if len(errors) == 0: continue
        sorted_errors = np.sort(errors)
        p = 1. * np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)
        c = colors[idx % len(colors)]
        ax.plot(sorted_errors, p, marker='', color=c, linewidth=2.5, label=system_name)
    
    thresholds = [1, 25, 200, 750, 2500]
    for t in thresholds:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        
    ax.set_xscale('symlog')
    ax.set_xlim(left=0, right=20000)
    ax.set_ylim(0, 1.05)
    
    ax.set_xlabel("Localization Error (km)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Fraction of Images", fontsize=14, fontweight='bold')
    # Removed inline title to adhere to cartographic/academic standards
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
    
    save_figure(fig, "error_histogram", fig_dir)
    plt.close(fig)

def paired_bootstrap(errors_base, errors_target, B=10000):
    """Computes bootstrap confidence intervals for Acc@τ."""
    thresholds = [1, 25, 200, 750, 2500]
    n = len(errors_base)
    
    results = {}
    for t in thresholds:
        base_accs = []
        target_accs = []
        
        # We bootstrap the indices to keep the pair structure
        for _ in range(B):
            idx = np.random.choice(n, n, replace=True)
            b_sample = errors_base[idx]
            t_sample = errors_target[idx]
            
            base_accs.append(np.mean(b_sample <= t) * 100)
            target_accs.append(np.mean(t_sample <= t) * 100)
            
        target_accs = np.array(target_accs)
        ci_lower = np.percentile(target_accs, 2.5)
        ci_upper = np.percentile(target_accs, 97.5)
        results[f'Acc@{t}km_CI'] = (ci_lower, ci_upper)
        
    return results

def statistical_tests(errors_base, errors_target):
    """Wilcoxon signed-rank test between base and target errors."""
    # Holm correction can be applied later if doing multiple comparisons.
    # Here we do a single paired comparison between A1 (base) and A4 (target) usually.
    diff = errors_base - errors_target
    if np.all(diff == 0):
        return 1.0, 0.0
    res = wilcoxon(errors_base, errors_target)
    # Effect size r = Z / sqrt(N). Scipy doesn't give Z directly for wilcoxon easily.
    # We will just return p-value.
    return res.pvalue

def save_tables(df, filename_base, dirs):
    """Saves a dataframe to CSV, Markdown, and LaTeX formats."""
    # Save CSV
    df.to_csv(os.path.join(dirs['csv'], f"{filename_base}.csv"), index=False)
    
    # Save Markdown
    with open(os.path.join(dirs['tables'], f"{filename_base}.md"), 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
        
    # Save LaTeX (booktabs format)
    latex_str = df.to_latex(index=False, escape=False)
    # Apply basic booktabs replacements if pandas didn't do it natively depending on version
    latex_str = latex_str.replace('\\toprule', '\\toprule\n').replace('\\midrule', '\\midrule\n').replace('\\bottomrule', '\\bottomrule\n')
    
    with open(os.path.join(dirs['latex'], f"{filename_base}.tex"), 'w', encoding='utf-8') as f:
        f.write(latex_str)

def generate_report(systems_results, total_expected_images, dataset_name="Im2GPS3k"):
    """
    Main entry point for generating the scientific report.
    systems_results: dict mapping system name to list of error dicts.
    """
    print("[LOG] Initiating automated scientific reporting...")
    dirs = create_reporting_directories()
    
    # Process Metrics
    metrics_list = []
    systems_errors = {}
    
    base_errors = None
    if "A1" in systems_results:
        base_errors = np.array([res['error'] for res in systems_results["A1"]])
        
    for sys_name, res_list in systems_results.items():
        errors = np.array([r['error'] for r in res_list])
        systems_errors[sys_name] = errors
        
        num_evaluated = len(errors)
        coverage_pct = (num_evaluated / total_expected_images) * 100 if total_expected_images > 0 else 0
        
        row = {
            'System': sys_name,
            'Median (km)': np.median(errors) if num_evaluated > 0 else 0,
        }
        
        thresholds = [1, 25, 200, 750, 2500]
        for t in thresholds:
            acc = np.mean(errors <= t) * 100 if num_evaluated > 0 else 0
            row[f'Acc@{t}km'] = f"{acc:.1f}"
            
        # Bootstrap CI if A1 exists and this is not A1
        if base_errors is not None and sys_name != "A1" and len(errors) == len(base_errors):
            ci_res = paired_bootstrap(base_errors, errors)
            for t in thresholds:
                lower, upper = ci_res[f'Acc@{t}km_CI']
                row[f'Acc@{t}km'] = f"{row[f'Acc@{t}km']} ({lower:.1f}-{upper:.1f})"
                
        row['Evaluated N'] = num_evaluated
        row['Coverage'] = f"{coverage_pct:.1f}%"
        metrics_list.append(row)
        
    metrics_df = pd.DataFrame(metrics_list)
    save_tables(metrics_df, "evaluation_summary", dirs)
    
    # Generate Figures
    print("[LOG] Generating publication-quality figures...")
    plot_cdf_multi(systems_errors, dirs['figures'])
    
    # Just plot histogram for the best system (or last)
    best_sys = list(systems_errors.keys())[-1]
    plot_histogram(systems_errors[best_sys], dirs['figures'])
    
    captions = {
        "cdf_curve": "Cumulative distribution of localization errors (km) across different systems. Solid lines represent the fraction of images localized within a given error threshold.",
        "error_histogram": f"Histogram of localization errors (km) in logarithmic scale for {best_sys}."
    }
    write_captions(dirs['figures'], captions)
    
    # Write metadata log
    with open(os.path.join(dirs['logs'], "run_metadata.txt"), "w") as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Git Hash: {get_git_hash()}\n")
        f.write(f"Seed: 42 (Fixed)\n")
        f.write(f"Evaluated N: {total_expected_images}\n\n")
        f.write(f"Packages:\n{get_pip_freeze()}\n")
        
    print(f"[LOG] Reporting complete. All artifacts saved in: {dirs['base']}")
    return dirs['base']
