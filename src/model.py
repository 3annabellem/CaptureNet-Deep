import torch
from torch import nn
import torch.nn.functional as F

# -------- Core 1D CNN (logits out; no sigmoid) --------
class CaptureBeefyCNNNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2);  self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,128,5, padding=2);  self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,256,5, padding=2); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256,256,3, padding=1); self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256,256,3, padding=1); self.bn5 = nn.BatchNorm1d(256)
        self.drop  = nn.Dropout(dropout)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(256, 1)  # logits

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop(x)
        x = self.gap(x).squeeze(2)
        return self.fc(x).squeeze(1)

# -------- Lightning module wrapper --------
import pytorch_lightning as pl
from torch import optim

class CaptureLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1.83e-4, weight_decay=3.32e-3, cutoff=0.524, pos_weight=1.0):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cutoff = cutoff
        self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, x):  # returns logits
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.crit(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > self.cutoff).float()
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage!="train"), on_epoch=True)
        if stage != "train":
            self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _):  return self._shared_step(batch, "train")
    def validation_step(self, batch, _):return self._shared_step(batch, "val")
    def test_step(self, batch, _):      return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
