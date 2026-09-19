import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

# ==========================================
# 🚀 TEB-PATH-LM: Path Extrapolation + Prefix-LM
# ==========================================
# NEW MECHANISM: Instead of pulling toward the centroid of intersections,
# we EXTRAPOLATE past the centroid along the consensus direction, then
# SNAP to the nearest real token. This creates directional motion instead
# of regressive mean-collapse.

# 1. Download Tiny Shakespeare
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "tinyshakespeare.txt"
if not os.path.exists(file_path):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(url, file_path)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# ==========================================
# Hyperparameters
# ==========================================
DATA_SIZE = len(text)
SEQ_LEN = 128
PREFIX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 3000
DIM = 128
TEB_STEPS = 3
K_NEIGHBORS = 16
SIGMA = 24.0
LR = 1e-3
ALPHA = 1.5            # Path extrapolation factor (>1 = walk past centroid)
NUM_LAYERS = 3

PAD_IDX = vocab_size   # Extra embedding slot for generation padding

data = torch.tensor([char_to_idx[ch] for ch in text[:DATA_SIZE]], dtype=torch.long)
print(f"Dataset loaded: {len(data)} characters, {vocab_size} unique chars.")
print(f"Path Extrapolation ALPHA = {ALPHA}")
print("-" * 50)

# ==========================================
# Prefix Mask: True = BLOCKED
# ==========================================
def make_prefix_mask(N, prefix_len, device):
    mask = torch.ones(N, N, dtype=torch.bool, device=device)
    # Prefix <-> Prefix: bidirectional
    mask[:prefix_len, :prefix_len] = False
    # Suffix -> Prefix: visible
    mask[prefix_len:, :prefix_len] = False
    # Suffix <-> Suffix: causal
    causal = torch.triu(
        torch.ones(N - prefix_len, N - prefix_len, device=device),
        diagonal=1
    ).bool()
    mask[prefix_len:, prefix_len:] = causal
    return mask

# ==========================================
# Sparse Iterative Emergence Layer with Path Extrapolation
# ==========================================
class SparseIterativeEmergenceLayer(nn.Module):
    def __init__(self, dim, steps=1, k=16, sigma=24.0, alpha=1.5):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.update_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, block_mask):
        B, N, D = x.shape
        k = min(self.k, N - 1)
        diag = torch.arange(N, device=x.device)

        for _ in range(self.steps):
            # ---- 1. kNN ----
            dist_sq = torch.cdist(x, x) ** 2
            dist_sq = dist_sq.masked_fill(block_mask.unsqueeze(0), float('inf'))
            dist_sq[:, diag, diag] = 0.0

            knn_dist, knn_indices = torch.topk(dist_sq, k + 1, largest=False, dim=-1)
            knn_dist = knn_dist[:, :, 1:]
            knn_indices = knn_indices[:, :, 1:]

            W_ij = torch.exp(-knn_dist / (2 * self.sigma ** 2))
            W_ij = torch.nan_to_num(W_ij, nan=0.0, posinf=0.0, neginf=0.0)

            # ---- 2. Intersection / Centroid ----
            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
            x_neighbors = x[batch_idx, knn_indices]
            idx_j = knn_indices[batch_idx, knn_indices]

            knn_exp = knn_indices.unsqueeze(2).unsqueeze(-1)
            idx_exp = idx_j.unsqueeze(-2)
            intersect = (knn_exp == idx_exp).any(dim=-1).float()

            weighted_mask = intersect * W_ij.unsqueeze(-1)
            mask_sum = weighted_mask.sum(dim=3, keepdim=True) + 1e-8
            c_ij = torch.einsum('bijk,bikd->bijd', weighted_mask, x_neighbors) / mask_sum

            overlap_sizes = weighted_mask.sum(dim=3)
            weighted_c = overlap_sizes.unsqueeze(-1) * c_ij
            centroid = weighted_c.sum(dim=2) / (overlap_sizes.sum(dim=2, keepdim=True) + 1e-8)

            # ---- 3. PATH EXTRAPOLATION (NEW) ----
            # Direction the consensus is pulling us
            direction = centroid - x

            # Walk PAST the centroid by alpha
            extrapolated = x + self.alpha * direction

            # Find the nearest REAL token to the extrapolated point
            dist_to_extrap = torch.cdist(extrapolated, x) ** 2
            dist_to_extrap = dist_to_extrap.masked_fill(
                block_mask.unsqueeze(0), float('inf')
            )
            # Self excluded (we don't want to snap back to ourselves)
            dist_to_extrap[:, diag, diag] = float('inf')

            _, nearest_idx = torch.topk(dist_to_extrap, 1, largest=False, dim=-1)
            nearest_idx = nearest_idx.squeeze(-1)

            batch_idx_2 = torch.arange(B, device=x.device).view(B, 1).expand(B, N)
            update_candidate = x[batch_idx_2, nearest_idx]

            # ---- 4. Gated update ----
            update_candidate = self.update_proj(update_candidate)
            gate_input = torch.cat([x, update_candidate], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))

            x = x + gate * update_candidate
            x = self.ln(x)

        return x

# ==========================================
# TEB Path-LM
# ==========================================
class TEBPathLM(nn.Module):
    def __init__(self, vocab_size, dim=128, max_len=128, prefix_len=64,
                 num_layers=3, steps=3, k=16, sigma=24.0, alpha=1.5):
        super().__init__()
        self.prefix_len = prefix_len
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        self.layers = nn.ModuleList([
            SparseIterativeEmergenceLayer(dim, steps=steps, k=k, sigma=sigma, alpha=alpha)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(dim, vocab_size)

        self.register_buffer(
            'block_mask',
            make_prefix_mask(max_len, prefix_len, 'cpu'),
            persistent=False
        )

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        h = self.embed(x) + self.pos_embed(pos)

        mask = self.block_mask[:N, :N].to(x.device)
        for layer in self.layers:
            h = layer(h, mask)

        return self.out(h)

# ==========================================
# Data Loader
# ==========================================
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y

# ==========================================
# Training
# ==========================================
def train_model(model, data, epochs, batch_size, seq_len, device, lr, prefix_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        x, y = get_batch(data, batch_size, seq_len, device)
        optimizer.zero_grad()
        logits = model(x)

        suffix_logits = logits[:, prefix_len:, :]
        suffix_targets = y[:, prefix_len:]
        loss = criterion(
            suffix_logits.reshape(-1, suffix_logits.size(-1)),
            suffix_targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % max(1, epochs // 10) == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

    return model

# ==========================================
# Generation
# ==========================================
@torch.no_grad()
def generate(model, start_str="ROMEO: ", length=300, device='cpu', temperature=0.8):
    model.eval()
    prefix_len = model.prefix_len
    max_len = model.max_len

    prompt_ids = [char_to_idx.get(c, 0) for c in start_str]
    if len(prompt_ids) > prefix_len:
        prompt_ids = prompt_ids[-prefix_len:]

    pad_count = prefix_len - len(prompt_ids)
    full = [PAD_IDX] * pad_count + prompt_ids

    for _ in range(length):
        if len(full) > max_len:
            full = full[-max_len:]

        ctx = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ctx)
        next_logits = logits[0, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        full.append(next_id)

    return ''.join([idx_to_char.get(i, '') for i in full[pad_count:]])

# ==========================================
# Run
# ==========================================
print(f"\nTraining TEB Path-LM...")
print(f"  Prefix: {PREFIX_LEN} (bidirectional) | Suffix: {SEQ_LEN - PREFIX_LEN} (causal)")
print(f"  Extrapolation ALPHA: {ALPHA}")
print(f"  Layers: {NUM_LAYERS} | Dim: {DIM} | Steps: {TEB_STEPS} | k: {K_NEIGHBORS}")
print(f"  Loss computed ONLY on suffix positions")
print("-" * 50)

model = TEBPathLM(
    vocab_size=vocab_size,
    dim=DIM,
    max_len=SEQ_LEN,
    prefix_len=PREFIX_LEN,
    num_layers=NUM_LAYERS,
    steps=TEB_STEPS,
    k=K_NEIGHBORS,
    sigma=SIGMA,
    alpha=ALPHA,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print("-" * 50)

model = train_model(
    model, data, EPOCHS, BATCH_SIZE, SEQ_LEN, device, lr=LR, prefix_len=PREFIX_LEN
)

print("\n--- GENERATED TEXT (TEB Path-LM) ---")
print(generate(model, start_str="ROMEO: ", length=300, device=device))
print("\n--- GENERATED TEXT (different prompt) ---")
print(generate(model, start_str="JULIET: ", length=300, device=device))
