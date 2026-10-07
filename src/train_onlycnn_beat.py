# -*- coding: utf-8 -*-
"""
CNN-only for PPG → MAP regression
Train only on (train + val), NO test leakage ✅
"""

import h5py
import numpy as np
import random, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- CONFIG ----------------
BATCH = 256
EPOCHS = 60
LR = 1e-3
EARLY_STOP = 10
WDECAY = 1e-5
AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ---------------- DATASET ----------------
class PPGDataset(Dataset):
    def __init__(self, X, Y, aug=False):
        self.X = X
        self.Y = Y
        self.aug = aug

    def normalize(self, x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-6:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)

        x = self.normalize(x)

        if self.aug:
            x = x * np.random.uniform(0.95, 1.05)
            x += np.random.normal(0, 0.01, size=x.shape)

        x = torch.from_numpy(x).unsqueeze(-1)  # [T,1]
        return x, torch.tensor([y], dtype=torch.float32)

# ---------------- MODEL ----------------
class AttentionPool(nn.Module):
    """시간축 어텐션 풀링: x∈[B,T,D] → [B,D]"""
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)
    def forward(self, x):
        # x: [B, T, D]
        w = torch.softmax(self.att(x), dim=1)   # [B, T, 1]
        return (x * w).sum(dim=1)               # [B, D]

class CNN_ONLY_BP(nn.Module):
    """
    CNN 특징 추출 → (Attention) 시간 풀링 → 회귀 헤드
    LSTM 제거. 시간 정보는 CNN의 receptive field로 모델링.
    """
    def __init__(self, use_attention=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,  64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),                        # T → T/4
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),                        # T → T/16
            # 필요시 더 얕은 Conv를 추가해도 됨
        )

        self.use_attention = use_attention
        if use_attention:
            self.pool = AttentionPool(128)          # [B,T',128] → [B,128]
        else:
            self.pool = None                        # 대신 global average 사용

        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):            # x: [B,T,1]
        z = x.permute(0, 2, 1)       # [B,1,T]
        z = self.cnn(z)              # [B,128,T']
        z = z.permute(0, 2, 1)       # [B,T',128]
        if self.use_attention:
            z = self.pool(z)         # [B,128]
        else:
            z = z.mean(dim=1)        # Global Average Pool over time
        y = self.head(z)             # [B,1]
        return y

# ---------------- METRICS ----------------
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# ---------------- TRAIN ----------------
print(f"🚀 Device: {DEVICE}")

# Load only train & val
with h5py.File("ppg_train.h5", "r") as f:
    X_train = np.array(f["X"]); Y_train = np.array(f["Y"])

with h5py.File("ppg_val.h5", "r") as f:
    X_val = np.array(f["X"]); Y_val = np.array(f["Y"])

train_ds = PPGDataset(X_train, Y_train, aug=True)
val_ds   = PPGDataset(X_val,   Y_val,   aug=False)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

model = CNN_ONLY_BP(use_attention=True).to(DEVICE)   # attention=False로 전역평균 사용 가능
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WDECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=AMP)

best = 1e9; patience = 0

for ep in range(1, EPOCHS+1):
    # ---- Train ----
    model.train()
    tr_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
            pr = model(xb)
            loss = ((pr - yb) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tr_loss += loss.item() * len(xb)
    tr_loss /= len(train_dl.dataset)

    # ---- Validation ----
    model.eval()
    vl_loss = 0.0; ps=[]; rs=[]
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pr = model(xb)
            loss = ((pr - yb) ** 2).mean()
            vl_loss += loss.item() * len(xb)
            ps.append(pr.cpu().numpy()); rs.append(yb.cpu().numpy())
    vl_loss /= len(val_dl.dataset)
    ps = np.concatenate(ps); rs = np.concatenate(rs)
    r = corr(rs.flatten(), ps.flatten())

    print(f"[{ep:02d}] train={tr_loss:.4f}  val={vl_loss:.4f}  corr={r:.3f}")

    if vl_loss < best:
        best = vl_loss; patience = 0
        torch.save(model.state_dict(), "cnn_only_best.pt")
    else:
        patience += 1
        if patience >= EARLY_STOP:
            print("⛔ Early stop triggered")
            break

    sched.step()

print("✅ Best model saved → cnn_only_best.pt")
