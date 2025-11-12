import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import CaptureBeefyCNNNet, CaptureLightningModule

# ---------- Dataset that windows full traces on-the-fly ----------
class CaptureWindowDataset(Dataset):
    """
    Expects a CSV with columns:
      - raw_signal: path to a CSV file containing a column 'current' OR a JSON array of floats
      - label: JSON list of dicts with 'start','end' (capture intervals in sample indices of the *downsampled* signal)
      - run_id: (optional) used for grouping/splitting upstream
    """
    def __init__(self, df, window_size=2000, step_size=None, balance=True):
        self.ws = int(window_size)
        self.step = int(step_size or round(1.1 * self.ws))
        self.balance = balance
        self.samples = []  # list of (np.ndarray, label)
        for _, row in df.iterrows():
            sig = self._load_signal(row["raw_signal"])
            if sig is None or len(sig) < self.ws: 
                continue
            mask = np.zeros(len(sig), dtype=np.uint8)
            caps = json.loads(row["label"]) if isinstance(row["label"], str) else row["label"]
            for c in caps:
                s = max(0, int(c["start"])); e = min(len(sig)-1, int(c["end"]))
                if e >= s: mask[s:e+1] = 1
            pos = neg = 0
            for start in range(0, len(sig)-self.ws+1, self.step):
                end = start + self.ws
                y = int(mask[start:end].mean() >= 0.5)
                if not self.balance or y == 1 or neg < pos:  # keep ~1:1
                    self.samples.append((sig[start:end].astype(np.float32), y))
                    pos += (y==1); neg += (y==0)

    def _load_signal(self, ref):
        # Allow path to CSV with 'current' or a JSON array string
        if isinstance(ref, str) and os.path.isfile(ref):
            try:
                col = pd.read_csv(ref)
                if "current" in col: 
                    return col["current"].to_numpy(dtype=np.float32)
            except Exception:
                return None
        # Try JSON array
        try:
            arr = json.loads(ref) if isinstance(ref, str) else ref
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            return None

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# ---------- Simple DataModule ----------
class CaptureDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv=None, test_csv=None, batch_size=128, window_size=2000, step_size=None, num_workers=2):
        super().__init__()
        self.train_csv, self.val_csv, self.test_csv = train_csv, val_csv, test_csv
        self.bs, self.ws, self.step = batch_size, window_size, step_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_csv)
        self.train_ds = CaptureWindowDataset(train_df, self.ws, self.step, balance=True)
        if self.val_csv:
            val_df = pd.read_csv(self.val_csv)
            self.val_ds = CaptureWindowDataset(val_df, self.ws, self.step, balance=False)
        else:
            self.val_ds = self.train_ds
        if self.test_csv:
            test_df = pd.read_csv(self.test_csv)
            self.test_ds = CaptureWindowDataset(test_df, self.ws, self.step, balance=False)
        else:
            self.test_ds = self.val_ds

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True,  num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):   return DataLoader(self.val_ds,   batch_size=self.bs, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def test_dataloader(self):  return DataLoader(self.test_ds,  batch_size=self.bs, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# ---------- Entry point ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="CSV with columns raw_signal,label[,run_id]")
    ap.add_argument("--val_csv", help="Optional CSV for validation")
    ap.add_argument("--test_csv", help="Optional CSV for test")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--window_size", type=int, default=2000)
    ap.add_argument("--step_size", type=int, default=2200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.83e-4)
    ap.add_argument("--wd", type=float, default=3.32e-3)
    ap.add_argument("--dropout", type=float, default=0.739)
    ap.add_argument("--cutoff", type=float, default=0.524)
    ap.add_argument("--max_epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    dm = CaptureDataModule(args.train_csv, args.val_csv, args.test_csv,
                           batch_size=args.batch_size, window_size=args.window_size,
                           step_size=args.step_size)
    dm.setup("fit")

    # Compute pos_weight from training set
    ys = torch.stack([y for _, y in dm.train_ds])
    counts = torch.bincount(ys.long(), minlength=2)
    neg, pos = int(max(counts[0].item(), 1)), int(max(counts[1].item(), 1))
    pos_weight = neg / pos

    model = CaptureBeefyCNNNet(dropout=args.dropout)
    lit = CaptureLightningModule(model, lr=args.lr, weight_decay=args.wd, cutoff=args.cutoff, pos_weight=pos_weight)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out_dir, filename="best-{epoch:02d}-{val_loss:.4f}",
                           monitor="val_loss", mode="min", save_top_k=1)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)

    accelerator = "gpu" if args.gpu and torch.cuda.is_available() else "cpu"
    devices = 1 if accelerator == "gpu" else None

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=accelerator, devices=devices,
                         precision="16-mixed" if accelerator=="gpu" else 32,
                         callbacks=[ckpt, es], log_every_n_steps=50)
    trainer.fit(lit, dm)
    trainer.test(lit, dm)
    print("Best checkpoint:", ckpt.best_model_path)

if __name__ == "__main__":
    main()
