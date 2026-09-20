import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

# ==========================================
# TEB-SW: Small-World TEB
#
# Hypothesis: the 2.3 loss plateau is caused
# by limited graph diameter. With k=16 neighbors
# and steps=3, information decays too fast to
# cross the sequence. Increasing seq_len doesn't
# help because TEB doesn't actually use far tokens.
#
# Fix: each token is assigned a random integer ID
# in [0, NK_GROUPS). Tokens sharing an ID get a
# direct gated update regardless of feature distance.
# This punches long-range tunnels into the kNN
# graph, reducing effective diameter from O(N) to
# O(log N) without changing the local geometry.
#
# WHY A SEPARATE PATHWAY:
# Zeroing kNN distance for same-ID tokens sounds
# right but silently fails. Intersection-based
# centroid = shared-neighbor triangles. A random
# distant token shares zero neighbors with your
# local cluster -> intersect=0 -> zero contribution
# to centroid, even if it's in the kNN. The tunnel
# is invisible. Fix: second gated update that
# directly averages same-ID tokens, bypassing the
# intersection mechanism entirely.
# ==========================================

URL       = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FILE_PATH = "tinyshakespeare.txt"

if not os.path.exists(FILE_PATH):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(URL, FILE_PATH)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars       = sorted(set(text))
vocab_size  = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# Hyperparameters
# ==========================================
DATA_SIZE   = len(text)
SEQ_LEN     = 128
BATCH_SIZE  = 64
EPOCHS      = 3000
DIM         = 128
NUM_LAYERS  = 3
TEB_STEPS   = 3
K_NEIGHBORS = 16
SIGMA       = 24.0
LR          = 1e-3

# NK_GROUPS: number of random ID buckets.
#   Expected tokens per group = SEQ_LEN / NK_GROUPS.
#   NK_GROUPS = SEQ_LEN // 2 = 64 -> ~2 tokens/group
#   -> ~1 random past-partner per token.
#   Try: 32 (4 partners), 64 (1-2), 128 (0-1).
NK_GROUPS = SEQ_LEN // 2   # default: ~1 tunnel partner/token

data = torch.tensor(
    [char_to_idx[ch] for ch in text[:DATA_SIZE]], dtype=torch.long
)
print(f"Dataset: {len(data)} chars | vocab: {vocab_size}")
print(f"NK_GROUPS: {NK_GROUPS} -> ~{SEQ_LEN // NK_GROUPS} tokens/group")
print("-" * 60)


# ==========================================
# Layer
# ==========================================
class SparseIterativeEmergenceLayer(nn.Module):
    """
    Two updates per step, both gated:

    1. LOCAL (intersection centroid):
       - Build kNN graph in feature space (causal).
       - For each token i and neighbor j, compute the
         centroid of tokens in N(i) ∩ N(j), weighted
         by Gaussian W_ij.
       - Gate and residual-add.

    2. TUNNEL (small-world):
       - Assign random IDs in [0, nk_groups).
       - Average all same-ID tokens in the PAST
         (causal: only past tokens can be connected).
       - Gate and residual-add.
       - If nk_groups=0, tunnel is disabled (baseline).

    LayerNorm after both updates.
    """
    def __init__(self, dim, steps=3, k=16, sigma=24.0, nk_groups=0):
        super().__init__()
        self.steps     = steps
        self.k         = k
        self.sigma     = sigma
        self.nk_groups = nk_groups

        # Local: intersection-weighted centroid
        self.update_proj = nn.Linear(dim, dim)
        self.gate_local  = nn.Linear(dim * 2, dim)

        # Tunnel: direct average of same-ID past tokens
        if nk_groups > 0:
            self.tunnel_proj = nn.Linear(dim, dim)
            self.gate_tunnel = nn.Linear(dim * 2, dim)

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        k    = min(self.k, N - 1)
        diag = torch.arange(N, device=x.device)

        # Causal mask: True = blocked (future token)
        causal = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )

        for _ in range(self.steps):

            # ----------------------------------------
            # LOCAL: kNN intersection centroid
            # ----------------------------------------
            dist_sq = torch.cdist(x, x) ** 2
            dist_sq = dist_sq.masked_fill(causal.unsqueeze(0), float("inf"))
            dist_sq[:, diag, diag] = 0.0   # self always nearest

            knn_dist, knn_idx = torch.topk(dist_sq, k + 1, largest=False, dim=-1)
            knn_dist = knn_dist[:, :, 1:]   # drop self
            knn_idx  = knn_idx[:, :, 1:]

            # Gaussian weights; blocked (inf) -> 0
            W = torch.exp(-knn_dist / (2 * self.sigma ** 2))
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

            # Neighbor features: x[b, knn_idx[b,i,j], :]
            bi     = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
            x_nbrs = x[bi, knn_idx]                  # [B, N, k, D]

            # For each neighbor j, fetch j's neighbors
            # nbr_knn[b,i,j_pos,l] = knn_idx[b, j, l]
            nbr_knn = knn_idx[bi, knn_idx]            # [B, N, k, k]

            # Intersection: is each of i's neighbors also in j's kNN?
            # intersect[b,i,j_pos,i_pos] = True if knn_idx[b,i,i_pos] in N(j)
            knn_exp  = knn_idx.unsqueeze(2).unsqueeze(-1)   # [B,N,1,k,1]
            nbr_exp  = nbr_knn.unsqueeze(-2)                # [B,N,k,1,k]
            intersect = (knn_exp == nbr_exp).any(dim=-1).float()  # [B,N,k,k]

            # Weighted centroid per (i,j) pair
            wm     = intersect * W.unsqueeze(-1)             # [B,N,k,k]
            wm_sum = wm.sum(dim=3, keepdim=True) + 1e-8
            c_ij   = torch.einsum("bijk,bikd->bijd", wm, x_nbrs) / wm_sum  # [B,N,k,D]

            # Aggregate across neighbors, weighted by intersection size
            ov      = wm.sum(dim=3)                          # [B,N,k]
            local_u = (ov.unsqueeze(-1) * c_ij).sum(dim=2) / (
                ov.sum(dim=2, keepdim=True) + 1e-8
            )                                                # [B,N,D]
            local_u = self.update_proj(local_u)

            g_local = torch.sigmoid(self.gate_local(torch.cat([x, local_u], dim=-1)))
            x       = x + g_local * local_u

            # ----------------------------------------
            # TUNNEL: direct average of same-ID past tokens
            # ----------------------------------------
            if self.nk_groups > 0:
                ids = torch.randint(0, self.nk_groups, (B, N), device=x.device)

                # id_match[b,i,j] = 1 if token j has same ID as token i
                id_match = (ids.unsqueeze(2) == ids.unsqueeze(1)).float()  # [B,N,N]

                # Causal: block future tokens (j > i)
                id_match = id_match.masked_fill(causal.unsqueeze(0), 0.0)

                # No self-connection
                id_match[:, diag, diag] = 0.0

                # Weighted average of past same-ID tokens
                id_sum    = id_match.sum(dim=2, keepdim=True) + 1e-8
                tunnel_u  = (id_match.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=2) / id_sum
                tunnel_u  = self.tunnel_proj(tunnel_u)

                g_tunnel = torch.sigmoid(self.gate_tunnel(torch.cat([x, tunnel_u], dim=-1)))
                x        = x + g_tunnel * tunnel_u

            x = self.ln(x)

        return x


# ==========================================
# Model
# ==========================================
class TEBSW(nn.Module):
    def __init__(self, vocab_size, dim=128, max_len=128,
                 num_layers=3, steps=3, k=16, sigma=24.0, nk_groups=0):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.layers    = nn.ModuleList([
            SparseIterativeEmergenceLayer(
                dim, steps=steps, k=k, sigma=sigma, nk_groups=nk_groups
            )
            for _ in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
        )

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0)
        h   = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


# ==========================================
# Training
# ==========================================
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([data[i:i + seq_len]       for i in ix]).to(device)
    y  = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


def run(model, label, epochs, batch_size, seq_len, lr):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best  = float("inf")
    t0    = time.time()
    n_par = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  params: {n_par:,}")
    print(f"{'=' * 60}")

    log_every = max(1, epochs // 20)

    for epoch in range(epochs):
        model.train()
        x, y = get_batch(data, batch_size, seq_len, device)
        opt.zero_grad()
        loss = crit(model(x).reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if loss.item() < best:
            best = loss.item()

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(
                f"  ep {epoch:4d} | "
                f"loss {loss.item():.4f} | "
                f"best {best:.4f} | "
                f"{time.time() - t0:.0f}s"
            )

    return best


@torch.no_grad()
def generate(model, start="ROMEO: ", length=300, temp=0.8):
    model.eval()
    ids     = [char_to_idx[c] for c in start]
    x       = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    max_len = model.pos_embed.num_embeddings

    for _ in range(length):
        logits = model(x[:, -max_len:])
        probs  = F.softmax(logits[0, -1] / temp, dim=-1)
        x      = torch.cat([x, torch.multinomial(probs, 1).unsqueeze(0)], dim=1)

    return "".join(idx_to_char[i] for i in x[0].tolist())


def make(nk):
    return TEBSW(
        vocab_size=vocab_size,
        dim=DIM,
        max_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        steps=TEB_STEPS,
        k=K_NEIGHBORS,
        sigma=SIGMA,
        nk_groups=nk,
    ).to(device)


# ==========================================
# Experiment: baseline vs small-world
# ==========================================
baseline = make(nk=0)
sw_model = make(nk=NK_GROUPS)

best_base = run(baseline, "TEB  (baseline, local only)",  EPOCHS, BATCH_SIZE, SEQ_LEN, LR)
best_sw   = run(sw_model, f"TEB-SW (NK_GROUPS={NK_GROUPS})", EPOCHS, BATCH_SIZE, SEQ_LEN, LR)

delta = best_base - best_sw
print(f"\n{'=' * 60}")
print(f"  TEB baseline : {best_base:.4f}")
print(f"  TEB-SW       : {best_sw:.4f}")
print(f"  Δ            : {delta:+.4f}  ({'better' if delta > 0 else 'worse or same'})")
print(f"{'=' * 60}")

print("\n--- Baseline sample ---")
print(generate(baseline))
print(f"\n--- TEB-SW (NK_GROUPS={NK_GROUPS}) sample ---")
print(generate(sw_model))

# ==========================================
# Optional: ablate different NK_GROUPS values.
# Uncomment to sweep after the main run.
# ==========================================
# for nk in [32, 64, 128]:
#     m    = make(nk=nk)
#     best = run(m, f"TEB-SW nk={nk}", EPOCHS, BATCH_SIZE, SEQ_LEN, LR)
#     print(f"  nk={nk:3d} -> {best:.4f}")
