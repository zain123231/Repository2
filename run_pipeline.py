import os
import sys
import time
import subprocess
import psutil
import gc

# Stages mirror the original make_all.ps1
STAGES = [
    {
        "name": "Evaluate A1-A4 Variants",
        "cmd": [sys.executable, "src/evaluate_all.py", "--csv", "data/im2gps3k.csv", "--img-dir", "data/im2gps3k/im2gps3ktest/im2gps3ktest", "--cities", "data/global_cities.csv"]
    },
    # Temporarily disable Ablation Study as per advisor review
    # {
    #     "name": "Ablation Study",
    #     "cmd": [sys.executable, "src/ablation.py"]
    # },
    {
        "name": "Build Galleries",
        "cmd": [sys.executable, "src/build_galleries.py", "--cities", "data/global_cities.csv"]
    },
    {
        "name": "Evaluate Galleries",
        "cmd": [sys.executable, "src/evaluate_galleries.py", "--csv", "data/im2gps3k.csv", "--img-dir", "data/im2gps3k/im2gps3ktest/im2gps3ktest"]
    },
    {
        "name": "Confidence-Gated Refinement Sweep",
        "cmd": [sys.executable, "src/evaluate_gated_refinement.py", "--val-csv", "data/val.csv", "--img-dir", "data/im2gps3k/im2gps3ktest/im2gps3ktest", "--cities", "data/global_cities.csv"]
    },
    {
        "name": "StreetCLIP Zero-Shot Baseline",
        "cmd": [sys.executable, "src/evaluate_streetclip.py", "--csv", "data/im2gps3k.csv", "--data-dir", "data/im2gps3k/im2gps3ktest/im2gps3ktest", "--cities", "data/global_cities.csv"]
    },
    {
        "name": "Image-to-Image Retrieval Baseline",
        "cmd": [sys.executable, "src/evaluate_retrieval.py", "--query-csv", "data/im2gps3k.csv", "--query-dir", "data/im2gps3k/im2gps3ktest/im2gps3ktest", "--ref-csv", "data/mp16_train.csv", "--ref-dir", "data/mp16/images"]
    },
    {
        "name": "Processing Iraqi Dataset",
        "cmd": [sys.executable, "src/research_wikidata.py"]
    }
]

MAX_RAM_PERCENT = 95.0
MAX_DISK_PERCENT = 98.0

def check_system_resources():
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    
    if ram > MAX_RAM_PERCENT:
        print(f"[FATAL] RAM usage is critically high: {ram}%. Halting pipeline to prevent system freeze.")
        sys.exit(1)
        
    if disk > MAX_DISK_PERCENT:
        print(f"[FATAL] Disk space usage is critically high: {disk}%. Halting pipeline.")
        sys.exit(1)
        
    return ram, disk

def cleanup_memory():
    # Force python GC
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # Sleep briefly to let OS reclaim resources
    time.sleep(3)

def run_pipeline():
    print("==========================================================")
    print("Robust GeoCLIP Evaluation Pipeline (Memory-Safe Mode)")
    print("==========================================================")
    
    for i, stage in enumerate(STAGES):
        print(f"\n[{i+1}/{len(STAGES)}] Starting Stage: {stage['name']}")
        print(f"Command: {' '.join(stage['cmd'])}")
        
        # Pre-flight check
        check_system_resources()
        
        start_time = time.time()
        
        # Start subprocess
        process = subprocess.Popen(stage['cmd'])
        
        peak_ram = 0
        peak_disk = 0
        
        # Poll while running
        while process.poll() is None:
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            peak_ram = max(peak_ram, ram)
            peak_disk = max(peak_disk, disk)
            
            # Real-time kill switch
            if ram > MAX_RAM_PERCENT:
                print(f"\n[FATAL] RAM exceeded {MAX_RAM_PERCENT}% during stage execution (Current: {ram}%). Killing process!")
                process.kill()
                sys.exit(1)
                
            time.sleep(2)
            
        duration = time.time() - start_time
        exit_code = process.returncode
        
        # Post-flight cleanup
        cleanup_memory()
        
        print(f"\n--- Stage '{stage['name']}' Summary ---")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Peak RAM: {peak_ram}%")
        print(f"Peak Disk: {peak_disk}%")
        
        if exit_code != 0:
            print(f"[ERROR] Stage failed with exit code {exit_code}. Halting pipeline.")
            sys.exit(exit_code)
        else:
            print("[SUCCESS] Stage completed successfully.")
            
    print("\n==========================================================")
    print("Pipeline Complete! All artifacts are saved in the results/ directory.")
    print("==========================================================")

if __name__ == "__main__":
    run_pipeline()
