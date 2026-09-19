import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

# ==========================================
# 🚀 TEB-PREFIX-LM: See the Future, Feed Forwards
# ==========================================
# Architecture: Prefix-LM (PaLM/GLM style)
# - First PREFIX_LEN tokens are BIDIRECTIONAL (see each other)
# - Remaining tokens are CAUSAL (see only past + full prefix)
# - Loss computed ONLY on the causal suffix (autoregressive)
# - Generation: prompt goes in prefix region, tokens generate causally

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

# Hyperparameters
DATA_SIZE = len(text)
SEQ_LEN = 128
PREFIX_LEN = 64           # First 64 tokens bidirectional, rest causal
BATCH_SIZE = 64
EPOCHS = 3000
DIM = 128
TEB_STEPS = 3
K_NEIGHBORS = 16
SIGMA = 24.0
LR = 1e-3

# Special PAD token index
PAD_IDX = vocab_size

data = torch.tensor([char_to_idx[ch] for ch in text[:DATA_SIZE]], dtype=torch.long)
print(f"Dataset loaded: {len(data)} characters, {vocab_size} unique chars.")
print("-" * 50)

# ==========================================
# Prefix Mask: True = BLOCKED
# ==========================================
def make_prefix_mask(N, prefix_len, device):
    """
    Prefix region [0, prefix_len): bidirectional (all see all)
    Suffix region [prefix_len, N): causal (each sees past + full prefix)
    Prefix cannot see suffix (no future leakage from generation region).
    """
    mask = torch.ones(N, N, dtype=torch.bool, device=device)

    # Prefix <-> Prefix: all visible
    mask[:prefix_len, :prefix_len] = False

    # Suffix -> Prefix: visible (context)
    mask[prefix_len:, :prefix_len] = False

    # Suffix <-> Suffix: causal (lower triangular + diagonal)
    causal = torch.triu(
        torch.ones(N - prefix_len, N - prefix_len, device=device),
        diagonal=1
    ).bool()
    mask[prefix_len:, prefix_len:] = causal

    return mask

# ==========================================
# Sparse Iterative Emergence Layer (with prefix mask + self-loops)
# ==========================================
class SparseIterativeEmergenceLayer(nn.Module):
    def __init__(self, dim, steps=1, k=16, sigma=24.0):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.k = k
        self.sigma = sigma
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.update_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, block_mask):
        B, N, D = x.shape
        k = min(self.k, N - 1)

        for _ in range(self.steps):
            dist_sq = torch.cdist(x, x) ** 2

            # Blocked positions -> inf distance
            dist_sq = dist_sq.masked_fill(block_mask.unsqueeze(0), float('inf'))

            # Self-loop: diagonal = 0 ensures self is always nearest neighbor
            diag = torch.arange(N, device=x.device)
            dist_sq[:, diag, diag] = 0.0

            # k+1 neighbors (self is first, then drop it)
            knn_dist, knn_indices = torch.topk(dist_sq, k + 1, largest=False, dim=-1)
            knn_dist = knn_dist[:, :, 1:]
            knn_indices = knn_indices[:, :, 1:]

            # Gaussian weights. Blocked neighbors (inf) -> exp(-inf) = 0
            W_ij = torch.exp(-knn_dist / (2 * self.sigma ** 2))
            W_ij = torch.nan_to_num(W_ij, nan=0.0, posinf=0.0, neginf=0.0)

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
            update_candidate = weighted_c.sum(dim=2) / (
                overlap_sizes.sum(dim=2, keepdim=True) + 1e-8
            )
            update_candidate = self.update_proj(update_candidate)

            gate_input = torch.cat([x, update_candidate], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))

            x = x + gate * update_candidate
            x = self.ln(x)

        return x

# ==========================================
# TEB Prefix-LM Model
# ==========================================
class TEBPrefixLM(nn.Module):
    def __init__(self, vocab_size, dim=128, max_len=128, prefix_len=64,
                 num_layers=3, steps=3, k=16, sigma=24.0):
        super().__init__()
        self.prefix_len = prefix_len
        self.max_len = max_len

        # Embedding has one extra slot for PAD token
        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        self.layers = nn.ModuleList([
            SparseIterativeEmergenceLayer(dim, steps=steps, k=k, sigma=sigma)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(dim, vocab_size)

        # Precompute mask for max length
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
# Data Loader (standard AR shift)
# ==========================================
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y

# ==========================================
# Training: AR loss on suffix only
# ==========================================
def train_model(model, data, epochs, batch_size, seq_len, device, lr, prefix_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        x, y = get_batch(data, batch_size, seq_len, device)
        optimizer.zero_grad()
        logits = model(x)

        # Only the causal suffix contributes loss
        suffix_logits = logits[:, prefix_len:, :]
        suffix_targets = y[:, prefix_len:]
        loss = criterion(
            suffix_logits.reshape(-1, suffix_logits.size(-1)),
            suffix_targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    return model

# ==========================================
# Generation: pad prompt on LEFT into prefix region
# ==========================================
@torch.no_grad()
def generate(model, start_str="ROMEO: ", length=200, device='cpu', temperature=0.8):
    model.eval()
    prefix_len = model.prefix_len
    max_len = model.max_len

    # Encode prompt, truncate to prefix_len if too long
    prompt_ids = [char_to_idx.get(c, 0) for c in start_str]
    if len(prompt_ids) > prefix_len:
        prompt_ids = prompt_ids[-prefix_len:]

    # Left-pad with PAD token so prompt sits at end of prefix region
    pad_count = prefix_len - len(prompt_ids)
    full = [PAD_IDX] * pad_count + prompt_ids

    for _ in range(length):
        # Slide window if we exceed max_len
        if len(full) > max_len:
            full = full[-max_len:]

        ctx = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ctx)

        # Last position output predicts the next token
        next_logits = logits[0, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        full.append(next_id)

    # Strip pad region from output
    return ''.join([idx_to_char.get(i, '') for i in full[pad_count:]])

# ==========================================
# Run
# ==========================================
print(f"\nTraining TEB Prefix-LM (prefix={PREFIX_LEN} bidirectional, suffix causal)...")
print(f"Loss computed on suffix positions [{PREFIX_LEN}, {SEQ_LEN})")
print("-" * 50)

model = TEBPrefixLM(
    vocab_size=vocab_size,
    dim=DIM,
    max_len=SEQ_LEN,
    prefix_len=PREFIX_LEN,
    num_layers=3,
    steps=TEB_STEPS,
    k=K_NEIGHBORS,
    sigma=SIGMA,
).to(device)

model = train_model(
    model, data, EPOCHS, BATCH_SIZE, SEQ_LEN, device, lr=LR, prefix_len=PREFIX_LEN
)

print("\n--- GENERATED TEXT (TEB Prefix-LM) ---")
print(generate(model, start_str="ROMEO: ", length=300, device=device))
