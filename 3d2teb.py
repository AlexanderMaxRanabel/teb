import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import urllib.request
import zipfile
import glob
import time

# ==========================================
# TEB on ModelNet10 — 3D Point Cloud Classification
# Source: Princeton 3DShapeNets (original dataset, .off files)
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

ZIP_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
ZIP_PATH = "ModelNet10.zip"
EXTRACT_DIR = "ModelNet10"
CACHE = "modelnet10_cache.npz"
NUM_POINTS = 1024

# ==========================================
# Parse OFF file -> sample points from vertices
# ==========================================
def parse_off(path, num_points):
    with open(path, 'r') as f:
        first = f.readline().strip()
        if first.startswith('OFF'):
            rest = first[3:].strip().split()
            counts = rest if rest else f.readline().strip().split()
        else:
            line = f.readline()
            while line.startswith('#') or line.strip() == '':
                line = f.readline()
            counts = line.strip().split()

        nv = int(counts[0])
        verts = np.zeros((nv, 3), dtype=np.float32)
        for i in range(nv):
            verts[i] = [float(x) for x in f.readline().strip().split()[:3]]

    if nv >= num_points:
        idx = np.random.choice(nv, num_points, replace=False)
    else:
        idx = np.random.choice(nv, num_points, replace=True)
    return verts[idx]


# ==========================================
# Build cache (once) from Princeton zip
# ==========================================
if not os.path.exists(CACHE):
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(ZIP_PATH):
            print(f"Downloading ModelNet10 from Princeton (~460 MB)...")
            urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(".")

    CLASSES = sorted([d for d in os.listdir(EXTRACT_DIR)
                      if os.path.isdir(os.path.join(EXTRACT_DIR, d))])
    print(f"Classes ({len(CLASSES)}): {CLASSES}")

    np.random.seed(0)
    train_pts, train_lbls, test_pts, test_lbls = [], [], [], []

    for ci, cls in enumerate(CLASSES):
        for split, pts_list, lbl_list in [
            ('train', train_pts, train_lbls),
            ('test',  test_pts,  test_lbls),
        ]:
            folder = os.path.join(EXTRACT_DIR, cls, split)
            if not os.path.isdir(folder):
                continue
            offs = glob.glob(os.path.join(folder, '*.off'))
            for off in offs:
                try:
                    pts_list.append(parse_off(off, NUM_POINTS))
                    lbl_list.append(ci)
                except Exception as e:
                    print(f"  skip {os.path.basename(off)}: {e}")

    train_pts = np.stack(train_pts); train_lbls = np.array(train_lbls)
    test_pts  = np.stack(test_pts);  test_lbls  = np.array(test_lbls)

    np.savez_compressed(CACHE,
        train_data=train_pts, train_label=train_lbls,
        test_data=test_pts,   test_label=test_lbls)
    print(f"Cached {CACHE}")
else:
    print(f"Loading cache {CACHE}")

d = np.load(CACHE)
train_pts, train_lbls = d['train_data'], d['train_label']
test_pts,  test_lbls  = d['test_data'],  d['test_label']

print(f"Train: {len(train_lbls)} | Test: {len(test_lbls)}")
print(f"Shape: {train_pts.shape}")
print("-" * 50)

# ==========================================
# Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 80
DIM = 128
NUM_LAYERS = 3
TEB_STEPS = 3
K_NEIGHBORS = 16
SIGMA = 1.0
ALPHA = 1.5
LR = 1e-3
NUM_CLASSES = len(set(train_lbls.tolist()))

# ==========================================
# Dataset
# ==========================================
class PCData(Dataset):
    def __init__(self, pts, lbls, augment=False):
        pts = pts - pts.mean(axis=1, keepdims=True)
        s = np.max(np.linalg.norm(pts, axis=2), axis=1, keepdims=True)
        pts = pts / np.maximum(s[:, :, None], 1e-8)
        self.pts = torch.tensor(pts, dtype=torch.float32)
        self.lbls = torch.tensor(lbls, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, i):
        p = self.pts[i]
        if self.augment:
            theta = torch.rand(1).item() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            rot = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            p = p @ rot.T + torch.randn_like(p) * 0.01
        return p, self.lbls[i]


train_ds = PCData(train_pts, train_lbls, augment=True)
test_ds  = PCData(test_pts,  test_lbls,  augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# TEB Layer (point cloud version)
# ==========================================
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
    def __init__(self, num_classes, dim=128, num_layers=3,
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


# ==========================================
# Training helpers
# ==========================================
def train_epoch(model, loader, opt, crit):
    model.train()
    tl, c, t = 0., 0, 0
    for pts, lbls in loader:
        pts, lbls = pts.to(device), lbls.to(device)
        opt.zero_grad()
        logits = model(pts)
        loss = crit(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item() * pts.size(0)
        c += (logits.argmax(-1) == lbls).sum().item()
        t += pts.size(0)
    return tl / t, c / t


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tl, c, t = 0., 0, 0
    for pts, lbls in loader:
        pts, lbls = pts.to(device), lbls.to(device)
        logits = model(pts)
        loss = crit(logits, lbls)
        tl += loss.item() * pts.size(0)
        c += (logits.argmax(-1) == lbls).sum().item()
        t += pts.size(0)
    return tl / t, c / t


# ==========================================
# Train TEB
# ==========================================
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


# ==========================================
# PointNet baseline
# ==========================================
class PointNetBaseline(nn.Module):
    def __init__(self, num_classes, dim=128):
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
