# CaptureNet-Deep
Deep learning model for automated detection of capture phases in nanopore protein sequencing data.

> **Citation**: Martin A., Kontogiorgos-Heintz D., Nivala J., "CaptureNet-Deep: Deep learning identification of capture phases in nanopore protein sequencing data," *Bioinformatics*, 2025.

## Quick Demo (2 minutes)

**Want to see it in action? Start here:**

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows (.venv/bin/activate on Mac/Linux)
   pip install -r requirements.txt
   ```

2. **Run the interactive demo:**
   Open and run `examples/demo_capture_detection.ipynb`
   
   This notebook demonstrates end-to-end inference on synthetic nanopore data using the pretrained model.

3. **Try the GUI Dashboard (Optional):**
   ```bash
   python launch_dashboard.py
   ```
   
   Real-time 512-channel visualization for analyzing FAST5 nanopore files with capture detection.

---

## Dashboard (GUI Interface)

For users who prefer a graphical interface, CaptureNet-Deep includes a PyQt5-based dashboard for real-time analysis of nanopore FAST5 files.

### Dashboard Features
- **Multi-channel visualization**: Real-time display of all 512 nanopore channels
- **Interactive analysis**: Click channels for detailed view and capture confidence scores  
- **Batch processing**: Analyze complete FAST5 files with automatic capture detection
- **Remote file access**: SSH connectivity for lab server integration
- **Export capabilities**: Save analysis results as JSON with capture intervals

### Dashboard Usage

1. **Launch the dashboard:**
   ```bash
   # Recommended: Robust launcher (auto-finds Python)
   python launch_dashboard.py
   
   # Windows batch file (if Python in PATH)
   .\launch_dashboard.bat
   
   # Direct execution
   python dashboard/start_screen.py
   ```

2. **Load FAST5 files:**
   - **Local files**: Click "Import local fast5" and select your file
   - **Remote files**: Click "Import remote fast5" for SSH server access

3. **Analysis results:**
   - **Red channels**: Dead/inactive pores  
   - **Blue channels**: Active pores with detected captures
   - **Green channels**: Channels with translocations (if available)
   - Click any channel for detailed signal view and confidence scores

### Generate Demo FAST5
Create sample data for testing the dashboard:
```bash
python data/generate_demo_data.py --fast5-only --channels 16
```

### Troubleshooting Dashboard
If you encounter issues:

1. **"Python was not found"**:
   - Use: `python launch_dashboard.py` (auto-finds Python)
   - Or install Python and add to PATH

2. **Import errors (PyQt5, etc.)**:
   ```bash
   pip install -r requirements.txt
   ```

3. **"No module named 'dashboard'"**:
   - Use direct launch: `python dashboard/start_screen.py`
   - Or the batch file: `.\launch_dashboard.bat`

---

## Detailed Usage

### Data Format
Training/validation/test CSV files should contain:
- `raw_signal`: Path to CSV with `current` column **or** JSON array of floats
- `label`: JSON list of capture intervals `[{"start": int, "end": int}, ...]`
- `run_id` (optional): For grouping/splitting

### Training
```bash
python -m src.train \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --out_dir checkpoints \
  --window_size 2000 --step_size 2200 \
  --batch_size 128 --gpu
```

**Technical details:**
- Model outputs logits; training uses `BCEWithLogitsLoss`
- Class imbalance handled by ~1:1 sampling and computed `pos_weight`
- Window labels are positive if ≥50% overlap with capture intervals

### Inference
```bash
python -m src.inference \
  --ckpt checkpoints/best-model.ckpt \
  --signal data/demo_signal.csv \
  --window_size 2000 --step_size 2200 \
  --out_json outputs/predicted.json --gpu
```

**Output format:**
```json
{
  "intervals": [[start_idx, end_idx], ...],
  "cutoff": 0.524,
  "window_size": 2000,
  "step_size": 2200
}
```

### Pretrained Model
A trained checkpoint is included: `models/best-model.ckpt`

## Model Architecture

**CaptureNet-Deep** consists of:

### CaptureNetDeep (Core CNN)
- **Input**: 1D nanopore current signal (2000 samples)
- **Architecture**: 5 Conv1D layers with BatchNorm and ReLU
  - 1→64→128→256→256→256 channels
  - Kernel sizes: 5,5,5,3,3
- **Output**: Logits (pre-sigmoid predictions)

### CaptureLightningModule (Training Framework)
- **PyTorch Lightning** wrapper for automated training
- **Loss**: BCEWithLogitsLoss for binary classification
- **Optimizer**: AdamW with learning rate scheduling
- **Features**: Automatic checkpointing, logging, and validation

## Technical Notes
- Signals are processed in sliding windows (default: 2000 samples, 2200 step)
- Simple smoothing removes isolated predictions: `[0,1,0]→[0,0,0]`, `[1,0,1]→[1,1,1]`
- For paper experiments, raw signals were downsampled 100× before windowing
- Training data from MISL (University of Washington) available on reasonable request

## License
MIT