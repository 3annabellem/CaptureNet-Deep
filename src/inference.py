"""
Inference utilities:
- load a trained checkpoint
- run windowed predictions on a 1D current trace
- smooth and merge into capture intervals
- CLI: predict from a CSV/JSON signal and save intervals as JSON
"""
import argparse, json, os
import numpy as np
import torch

from .model import CaptureNetDeep, CaptureLightningModule

def window_predict(signal: np.ndarray, model: torch.nn.Module, window_size=2000, step_size=2200, cutoff=0.524, device="cpu"):
    model.eval()
    xs, spans = [], []
    for start in range(0, len(signal)-window_size+1, step_size):
        end = start + window_size
        spans.append((start, end))
        xs.append(signal[start:end].astype(np.float32))
    if not xs:
        return [], []

    x = torch.from_numpy(np.stack(xs)).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

    # threshold + simple smoothing of isolated flips
    labels = (probs > cutoff).astype(np.uint8)
    # [0,1,0] -> [0,0,0]; [1,0,1] -> [1,1,1]
    for i in range(1, len(labels)-1):
        if labels[i-1] == labels[i+1] != labels[i]:
            labels[i] = labels[i-1]
    return labels, spans

def merge_windows(labels, spans):
    """Merge consecutive positive windows into [start, end] capture intervals in sample indices."""
    intervals = []
    cur = None
    for lab, (s, e) in zip(labels, spans):
        if lab == 1 and cur is None:
            cur = [s, e]
        elif lab == 1:
            cur[1] = e
        elif lab == 0 and cur is not None:
            intervals.append(cur); cur = None
    if cur is not None:
        intervals.append(cur)
    return intervals

def load_checkpoint(ckpt_path, dropout=0.739, device="cpu"):
    base = CaptureNetDeep(dropout=dropout)
    lit = CaptureLightningModule.load_from_checkpoint(ckpt_path, model=base, map_location=device)
    return lit.model.to(device).eval(), getattr(lit.hparams, "cutoff", 0.524)

def load_signal(ref):
    # CSV with 'current' column, or JSON array string
    if isinstance(ref, str) and os.path.isfile(ref):
        import pandas as pd
        df = pd.read_csv(ref)
        if "current" in df:
            return df["current"].to_numpy(dtype=np.float32)
    try:
        arr = json.loads(ref) if isinstance(ref, str) else ref
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        raise ValueError("Could not load signal. Provide path to CSV with column 'current' or a JSON array of floats.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    ap.add_argument("--signal", required=True, help="CSV path with 'current' OR JSON array of floats")
    ap.add_argument("--window_size", type=int, default=2000)
    ap.add_argument("--step_size", type=int, default=2200)
    ap.add_argument("--out_json", default="predicted_captures.json")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    model, cutoff = load_checkpoint(args.ckpt, device=device)
    sig = load_signal(args.signal)

    labels, spans = window_predict(sig, model, args.window_size, args.step_size, cutoff, device)
    intervals = merge_windows(labels, spans)

    with open(args.out_json, "w") as f:
        json.dump({"intervals": intervals, "cutoff": float(cutoff), "window_size": args.window_size, "step_size": args.step_size}, f)
    print(f"Wrote {len(intervals)} intervals -> {args.out_json}")

if __name__ == "__main__":
    main()
