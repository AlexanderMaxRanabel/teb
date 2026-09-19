import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import urllib.request
import time

# ==========================================
# TEB on ModelNet10 — 3D Point Cloud Classification
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# -------- Download ModelNet10 --------
MODELNET10_URL = "https://cloud.tsinghua.edu.cn/f/5414376f6afd41ce9b6d/?dl=1"
H5_PATH = "modelnet10.h5"

if not os.path.exists(H5_PATH):
    print("Downloading ModelNet10 (72.5 MB)...")
    urllib.request.urlretrieve(MODELNET10_URL, H5_PATH)
    print("Done.")

with h5py.File(H5_PATH, 'r') as f:
    print("Keys:", list(f.keys()))

# -------- Hyperparameters --------
NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 80
DIM = 128
NUM_LAYERS = 3
TEB_STEPS = 3
K_NEIGHBORS = 16
SIGMA = 1.0
ALPHA = 1.5
LR = 1e-3
NUM_CLASSES = 10

# -------- Dataset --------
class ModelNet10Dataset(Dataset):
    def __init__(self, h5_path, split='train', num_points=1024, augment=True):
        with h5py.File(h5_path, 'r') as f:
            if split == 'train':
                data = f['train_data'][:]
                labels = f['train_label'][:]
            else:
                data = f['test_data'][:]
                labels = f['test_label'][:]

        data = data - data.mean(axis=1, keepdims=True)
        scale = np.max(np.linalg.norm(data, axis=2), axis=1, keepdims=True)
        data = data / np.maximum(scale, 1e-8)

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_points = num_points
        self.augment = augment and (split == 'train')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx]
        label = self.labels[idx]
        if pts.shape[0] != self.num_points:
            perm = torch.randperm(pts.shape[0])[:self.num_points]
            pts = pts[perm]
        if self.augment:
            theta = torch.rand(1).item() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            rot = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            pts = pts @ rot.T + torch.randn_like(pts) * 0.01
        return pts, label


train_ds = ModelNet10Dataset(H5_PATH, 'train', NUM_POINTS)
test_ds  = ModelNet10Dataset(H5_PATH, 'test',  NUM_POINTS)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
print("-" * 50)

# -------- TEB Layer (adapted for point clouds) --------
# Changes from language version:
#   - NO causal mask (fully connected graph)
#   - NO positional embeddings (permutation invariant)
#   - Input dim = 3 (XYZ), projected to DIM

class SparseIterativeEmergenceLayer(nn.Module):
    def __init__(self, dim, steps=3, k=16, sigma=1.0, alpha=1.5):
        super().__init__()
        self.steps = steps
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.update_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        k = min(self.k, N - 1)
        diag = torch.arange(N, device=x.device)

        for _ in range(self.steps):
            dist_sq = torch.cdist(x, x) ** 2
            dist_sq[:, diag, diag] = 0.0

            knn_dist, knn_idx = torch.topk(dist_sq, k + 1, largest=False, dim=-1)
            knn_dist = knn_dist[:, :, 1:]
            knn_idx = knn_idx[:, :, 1:]

            W = torch.exp(-knn_dist / (2 * self.sigma ** 2))
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
            x_neighbors = x[batch_idx, knn_idx]
            idx_j = knn_idx[batch_idx, knn_idx]

            knn_exp = knn_idx.unsqueeze(2).unsqueeze(-1)
            idx_exp = idx_j.unsqueeze(-2)
            intersect = (knn_exp == idx_exp).any(dim=-1).float()

            weighted = intersect * W.unsqueeze(-1)
            mask_sum = weighted.sum(dim=3, keepdim=True) + 1e-8
            c_ij = torch.einsum('bijk,bikd->bijd', weighted, x_neighbors) / mask_sum

            overlap = weighted.sum(dim=3)
            centroid = (overlap.unsqueeze(-1) * c_ij).sum(dim=2) / (overlap.sum(dim=2, keepdim=True) + 1e-8)

            direction = centroid - x
            extrapolated = x + self.alpha * direction

            dist_to_extrap = torch.cdist(extrapolated, x) ** 2
            dist_to_extrap[:, diag, diag] = float('inf')
            _, nearest_idx = torch.topk(dist_to_extrap, 1, largest=False, dim=-1)
            nearest_idx = nearest_idx.squeeze(-1)

            batch_idx_2 = torch.arange(B, device=x.device).view(B, 1).expand(B, N)
            update = x[batch_idx_2, nearest_idx]

            update = self.update_proj(update)
            gate = torch.sigmoid(self.gate_proj(torch.cat([x, update], dim=-1)))
            x = self.ln(x + gate * update)

        return x


class TEBPointCloud(nn.Module):
    def __init__(self, num_classes=10, dim=128, num_layers=3,
                 steps=3, k=16, sigma=1.0, alpha=1.5):
        super().__init__()
        self.input_proj = nn.Linear(3, dim)
        self.layers = nn.ModuleList([
            SparseIterativeEmergenceLayer(dim, steps=steps, k=k, sigma=sigma, alpha=alpha)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h_max = h.max(dim=1).values
        h_mean = h.mean(dim=1)
        return self.head(torch.cat([h_max, h_mean], dim=-1))


# -------- Training --------
def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(pts)
        loss = crit(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * pts.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += pts.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        logits = model(pts)
        loss = crit(logits, labels)
        total_loss += loss.item() * pts.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += pts.size(0)
    return total_loss / total, correct / total


# -------- Run TEB --------
model = TEBPointCloud(
    num_classes=NUM_CLASSES, dim=DIM, num_layers=NUM_LAYERS,
    steps=TEB_STEPS, k=K_NEIGHBORS, sigma=SIGMA, alpha=ALPHA
).to(device)

print(f"TEB params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)

print("\nTraining TEB on ModelNet10...")
best = 0.0
start = time.time()

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_epoch(model, train_loader, opt, crit)
    te_loss, te_acc = evaluate(model, test_loader, crit)
    sched.step()
    best = max(best, te_acc)
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d} | Train {tr_loss:.3f}/{tr_acc:.3f} | "
              f"Test {te_loss:.3f}/{te_acc:.3f} | {time.time()-start:.1f}s")

print(f"\nBest TEB test accuracy: {best:.4f}")

# -------- PointNet Baseline --------
class PointNetBaseline(nn.Module):
    def __init__(self, num_classes=10, dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, dim), nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_classes))

    def forward(self, x):
        h = self.mlp(x).max(dim=1).values
        return self.head(h)


pn = PointNetBaseline(NUM_CLASSES, DIM).to(device)
print(f"\nPointNet params: {sum(p.numel() for p in pn.parameters()):,}")

pn_opt = torch.optim.AdamW(pn.parameters(), lr=LR, weight_decay=1e-4)
pn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pn_opt, T_max=EPOCHS)

print("\nTraining PointNet baseline...")
best_pn = 0.0
for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_epoch(pn, train_loader, pn_opt, crit)
    te_loss, te_acc = evaluate(pn, test_loader, crit)
    pn_sched.step()
    best_pn = max(best_pn, te_acc)
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d} | Train {tr_loss:.3f}/{tr_acc:.3f} | Test {te_loss:.3f}/{te_acc:.3f}")

print(f"\nBest PointNet test accuracy: {best_pn:.4f}")
print(f"Best TEB test accuracy:      {best:.4f}")
