import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

# -------------------- Data --------------------
class CaptureWindowDataset(Dataset):
    def __init__(self, df, window_size=1000, step_size=None):
        self.window_size = window_size
        self.step = step_size or int(1.1 * window_size)
        X, y = [], []
        for _, row in df.iterrows():
            seq = np.asarray(row["current"], dtype=np.float32)
            if len(seq) < window_size: continue
            labmask = np.zeros(len(seq), dtype=np.int32)
            for s, e in row["captures"]:
                s = max(0, int(s)); e = min(len(seq)-1, int(e))
                labmask[s:e+1] = 1
            caps = non = 0
            for start in range(0, len(seq)-window_size+1, self.step):
                end = start + window_size
                lbl = int(labmask[start:end].mean() >= 0.5)
                if lbl == 1 or non < caps:   # balance 1:1
                    X.append(seq[start:end]); y.append(lbl)
                    caps += (lbl == 1); non += (lbl == 0)
        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class CaptureDataModule(pl.LightningDataModule):
    def __init__(self, data_csv, batch_size=128, window_size=1000, test_size=0.2, val_size=0.1, base_path=None, num_workers=2):
        super().__init__()
        self.data_csv, self.bs, self.ws = data_csv, batch_size, window_size
        self.test_size, self.val_size = test_size, val_size
        self.base_path, self.num_workers = base_path, num_workers

    def setup(self, stage=None):
        df = pd.read_csv(self.data_csv)
        # expected columns: raw_signal (path or filename), label (json)
        df["captures"] = df["label"].apply(json.loads).apply(lambda L: [[d["start"], d["end"]] for d in L])
        df["current"]  = df["raw_signal"].apply(self._load_current)
        # group split by run id robustly
        df["run_id"] = df.get("run_id", df["raw_signal"].apply(lambda s: os.path.basename(str(s)).split("_")[2] if "_" in str(s) else "run"))
        gss = GroupShuffleSplit(test_size=self.test_size, random_state=42)
        tr, te = next(gss.split(df, groups=df["run_id"]))
        trainv, test = df.iloc[tr], df.iloc[te]
        gss2 = GroupShuffleSplit(test_size=self.val_size/(1-self.test_size), random_state=42)
        tr2, va2 = next(gss2.split(trainv, groups=trainv["run_id"]))
        train, val = trainv.iloc[tr2], trainv.iloc[va2]
        self.train_ds = CaptureWindowDataset(train, self.ws)
        self.val_ds   = CaptureWindowDataset(val,   self.ws)
        self.test_ds  = CaptureWindowDataset(test,  self.ws)

    def _load_current(self, fn):
        path = fn if (self.base_path is None or os.path.isabs(str(fn))) else os.path.join(self.base_path, str(fn))
        try:
            return pd.read_csv(path)["current"].astype(np.float32).tolist()
        except Exception:
            return []

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True,  num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):   return DataLoader(self.val_ds,   batch_size=self.bs, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def test_dataloader(self):  return DataLoader(self.test_ds,  batch_size=self.bs, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# -------------------- Models (logits out!) --------------------
class CaptureBeefyCNNNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,128,5, padding=2); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,256,5, padding=2); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256,256,3, padding=1); self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256,256,3, padding=1); self.bn5 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(dropout)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(256, 1)   # logits (no sigmoid)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop(x)
        x = self.gap(x).squeeze(2)
        return self.fc(x).squeeze(1)

# -------------------- Lightning --------------------
class CaptureLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-3, cutoff=0.5, pos_weight=1.0):
        super().__init__()
        self.model, self.lr, self.weight_decay, self.cutoff = model, lr, weight_decay, cutoff
        self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, x): return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)                  # logits
        loss = self.crit(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > self.cutoff).float()
        acc   = (preds == y).float().mean()
        self.log(f'{stage}_loss', loss, prog_bar=(stage!='train'), on_epoch=True, on_step=False)
        if stage != 'train': self.log(f'{stage}_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _):  return self._step(batch, 'train')
    def validation_step(self, batch, _):return self._step(batch, 'val')
    def test_step(self, batch, _):      return self._step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv', required=True)
    ap.add_argument('--base_path', default=None)
    ap.add_argument('--out_dir', default='checkpoints')
    ap.add_argument('--window_size', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1.83e-4)
    ap.add_argument('--wd', type=float, default=3.32e-3)
    ap.add_argument('--dropout', type=float, default=0.74)
    ap.add_argument('--cutoff', type=float, default=0.524)
    ap.add_argument('--max_epochs', type=int, default=500)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--gpu', action='store_true')
    args = ap.parse_args()

    dm = CaptureDataModule(args.data_csv, batch_size=args.batch_size, window_size=args.window_size, base_path=args.base_path)
    dm.setup('fit')

    # class weights
    counts = Counter(dm.train_ds.y.cpu().numpy().tolist()); neg = max(int(counts.get(0,0)),1); pos = max(int(counts.get(1,0)),1)
    pos_weight = neg / pos

    model = CaptureBeefyCNNNet(dropout=args.dropout)  # swap here for other architectures
    lit   = CaptureLightningModule(model, lr=args.lr, weight_decay=args.wd, cutoff=args.cutoff, pos_weight=pos_weight)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out_dir, filename='best-{epoch:02d}-{val_loss:.4f}', monitor='val_loss', mode='min', save_top_k=1)
    es   = EarlyStopping(monitor='val_loss', mode='min', patience=args.patience)

    accelerator = 'gpu' if args.gpu and torch.cuda.is_available() else 'cpu'
    devices = 1 if accelerator == 'gpu' else None

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator=accelerator, devices=devices, precision='16-mixed' if accelerator=='gpu' else 32, callbacks=[ckpt, es], log_every_n_steps=50)
    trainer.fit(lit, dm)
    trainer.test(lit, dm)
    print('Best checkpoint:', ckpt.best_model_path)

if __name__ == '__main__':
    main()
