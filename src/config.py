"""
Global configuration — All settings in one place.
Every number here must be reproducible via a single script.
"""
import os
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "..", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "..", "figures")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "..", "checkpoints")
LOGS_DIR = os.path.join(PROJECT_ROOT, "..", "logs")

IM2GPS3K_DIR = os.path.join(RAW_DIR, "im2gps3k")
YFCC4K_DIR = os.path.join(RAW_DIR, "yfcc4k")
OSV5M_DIR = os.path.join(RAW_DIR, "osv5m")
GLDV2_DIR = os.path.join(RAW_DIR, "gldv2")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL = "ViT-B/32"
CLIP_DIM = 512

KNN_K = 5
KNN_TOP_K_CELLS = 3
RETRIEVAL_LIST_SIZE = 20

QUADTREE_MAX_CELLS = 256
QUADTREE_MIN_SAMPLES = 10

TRAIN_BATCH_SIZE = 32
TRAIN_LR = 1e-3
TRAIN_EPOCHS = 50
TRAIN_VAL_SPLIT = 0.1

HIERARCHICAL_TEMPERATURE = 0.1

ACC_THRESHOLDS_KM = [1, 25, 200, 750, 2500]
ACC_LABELS = ["street", "city", "region", "country", "continent"]

RANDOM_PREDICTIONS = 100
GEO_CENTROID_COUNT = 200


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


for _dir in [RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, LOGS_DIR,
             os.path.join(FIGURES_DIR, "cdf"), os.path.join(FIGURES_DIR, "comparison"),
             os.path.join(FIGURES_DIR, "error_analysis"), os.path.join(FIGURES_DIR, "geographic"),
             os.path.join(FIGURES_DIR, "qualitative"), os.path.join(FIGURES_DIR, "interactive")]:
    os.makedirs(_dir, exist_ok=True)
