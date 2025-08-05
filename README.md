# Artificial Intelligence for Figure Skating – Scoring

This project is an implementation based on [Audio-Visual-Figure-Skating](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) for scoring figure skating videos on the FS800/FS1000 datasets.

## Table of Contents

```text
.
├── dataset/                      # FS800/FS1000 feature datasets
├── fs1000_result/                 # Training & evaluation outputs
├── model.py                      # Model definition (PositionalEncoding + ScoringHead)
├── evaluate.py                   # Batch evaluation & Excel report generation
├── inference.py                  # Single-sample or new-video inference
├── grid_search.py                # Hyperparameter grid search
├── main.py                       # Single-task training script
├── main_auto.py                  # Multi-task + noise-robust automated training
├── show_npy_feature.py           # Inspect .npy feature files
└── .gitignore
```

## Features

- **End-to-End Pipeline**: From feature loading and model training to hyperparameter search, batch evaluation, and inference  
- **Multi-Task & Noise-Robust**: Supports multiple scoring dimensions and various noise levels  
- **Utility Scripts**: Quick commands for single-sample inference, batch evaluation, grid search, and feature inspection  

## Scripts & Usage

### `show_npy_feature.py`

Inspect the shape and basic stats of a single `.npy` feature file.

```bash
python show_npy_feature.py --npy /path/to/sample.npy
```

| Argument   | Type | Required | Description                    |
|------------|------|----------|--------------------------------|
| `--npy`    | str  | Yes      | Path to the `.npy` feature file |


### `evaluate.py`

Batch-evaluate on the validation set and export both summary and per-sample metrics to an Excel file.

```bash
python evaluate.py   --dataset_root /path/to/FS1000_dataset   --checkpoint ./fs1000_result/checkpoint_pe.pth   --score_item_index 0 --limit 100 --output_dir ./results
```

| Argument               | Type | Required | Default                                 | Description                                            |
|------------------------|------|----------|-----------------------------------------|--------------------------------------------------------|
| `--dataset_root`       | str  | Yes      | —                                       | Root of the validation dataset         |
| `--checkpoint`         | str  | No       | `./fs1000_result/checkpoint_pe.pth`     | Path to trained model weights                          |
| `--score_item_index`   | int  | No       | `0`                                     | Index of the scoring dimension (0-based)               |
| `--limit`              | int  | No       | All samples                             | Only evaluate first N samples                          |
| `--output_dir`         | str  | No       | `./results`                             | Directory to save generated `.xlsx` report             |

After running, you’ll find `score_item_index_<idx>.xlsx` under `--output_dir`, containing two sheets:
- **Summary**: MSE / MAE / Spearman correlations  
- **PerSample**: Individual sample scores and errors  

### `inference.py`

Run inference on one validation sample or any external video file.

```bash
python inference.py --infer_video_index 5 --dataset_root /path/to/FS1000_dataset --checkpoint ./fs1000_result/checkpoint_pe.pth
```

| Argument                | Type | Required       | Description                                                                 |
|-------------------------|------|----------------|-----------------------------------------------------------------------------|
| `--infer_video_index`   | int  | Conditional    | Validation-sample index (mutually exclusive with `--video_path`)           |
| `--video_path`          | str  | Conditional    | Path to any video file (requires `extract_features_from_video` to be implemented) |
| `--dataset_root`        | str  | If using index | Root of the validation dataset                                              |
| `--checkpoint`          | str  | No             | Model weights path (default: `./fs1000_result/checkpoint_pe.pth`)           |


### `grid_search.py`

Perform grid search over learning rates, weight decays, and batch sizes. Logs are appended to `grid_search.log`.

```bash
python grid_search.py
```

### `main.py`

Train a single scoring dimension and save the best model automatically.

```bash
python main.py
```

Best weights are saved to `./fs1000_result/checkpoint_pe.pth`.


### `main_auto.py`

Automated multi-task training across scoring dimensions and noise levels.

```bash
python main_auto.py
```

Results for each combination are stored under `./fs1000_result/auto/`.