# GeoCLIP Fine-tuning Pipeline

This directory contains the entire source code to process datasets, train the GeoCLIP model, and evaluate its performance.

## Prerequisites
Ensure your environment is set up and all requirements are installed:
```bash
pip install -r requirements.txt
pip install folium pandas
```

## Dataset Structure
For real-world training, you need an images directory and a CSV file.
The default expected structure is:
```text
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── train.csv
```
`train.csv` must contain the following columns:
- `IMG_PATH`: The filename of the image (e.g. `image1.jpg`)
- `LAT`: Latitude of the image location
- `LON`: Longitude of the image location

**Note:** If you want to download a tiny sample dataset of famous landmarks to test your pipeline quickly, run:
```bash
python src/download_dataset.py
```

## Running the Pipeline

### 1. Training
To train the model on the real dataset:
```bash
python src/train.py --csv data/train.csv --epochs 5
```
*(If `--csv` is omitted, it defaults to `data/train.csv`)*

To run a quick test using auto-generated mock data without needing any real images:
```bash
python src/train.py --mock
```

### 2. Ablation Studies (Evaluation)
To compare your trained checkpoint (`checkpoints/geoclip_checkpoint.pth`) against the baseline untrained model:
```bash
python src/ablation.py
```
*(Use `--mock` if you are testing without real data)*

### 3. Visualization
To generate an interactive HTML map (`error_map_real.html`) that plots the model's predictions vs. the ground truth locations:
```bash
python src/visualize.py --samples 10
```
*(Use `--mock` to visualize mock data, which will output to `error_map_mock.html`)*
