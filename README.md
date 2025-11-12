# CaptureNet-Deep
Deep learning model and dashboard for automated detection of capture phases in nanopore protein sequencing data.

> Please cite: Martin A., Kontogiorgos-Heintz D., Nivala J., “CaptureNet-Deep: Deep learning identification of capture phases in nanopore protein sequencing data,” *Bioinformatics*, 2025.  
> Code: GitHub. Archive: Zenodo DOI: `10.5281/zenodo.XXXXXXX`.

## Quickstart

### 1) Install
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

### 2) Data format
`train.csv` / `val.csv` / `test.csv` with columns:
- `raw_signal`: path to a CSV containing a column `current` **or** a JSON array of floats
- `label`: JSON list of objects `{"start": int, "end": int}` in **downsampled** indices
- `run_id` (optional): used for grouping/splitting upstream

### 3) Train
python -m src.train \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --out_dir checkpoints \
  --window_size 2000 --step_size 2200 \
  --batch_size 128 --gpu

- Models output **logits**; training uses `BCEWithLogitsLoss`.
- Class imbalance handled by ~1:1 sampling and `pos_weight` computed from train set.

### 4) Inference
python -m src.inference \
  --ckpt checkpoints/best-XX.ckpt \
  --signal data/sample_signal.csv \
  --window_size 2000 --step_size 2200 \
  --out_json outputs/predicted_captures.json --gpu

Output JSON:
{
  "intervals": [[start_idx, end_idx], ...],
  "cutoff": 0.524,
  "window_size": 2000,
  "step_size": 2200
}

### Pretrained Model
A trained CaptureNet-Deep checkpoint is included for reproducibility:
`models/best-model.ckpt`

To test inference:
python -m src.inference --ckpt models/best-model.ckpt --signal data/demo_signal.csv --out_json outputs/predicted.json

## Notes
- Window labels are positive if ≥50% overlap with a capture interval.
- A simple smoothing pass removes isolated flips (`[0,1,0]→[0,0,0]`, `[1,0,1]→[1,1,1]`).
- For the paper’s demo, signals were downsampled by 100× before windowing.
- Raw nanopore sequencing traces used for model training were generated in the Molecular Information Systems Laboratory (MISL) at the University of Washington and are available upon reasonable request due to laboratory data agreements.

## License
MIT