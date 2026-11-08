"""
gn.py  --  Geometric Neurons (GN)

TEB, but instead of one big layer with one k / one sigma / one set of weights,
each layer is a bank of G small "geometric neurons". Every GN is a complete
TEB unit with its OWN weights (and its own learned view of the embedding space).
Design choices (from the plan):
    * specialisation is left to EMERGE   (no diversity loss, no hard routing)
    * outputs are CONCATENATED then PROJECTED
    * SOFT aggregation: every GN sees every token

Usage:
    python gn.py --mode lite                 # phone / CPU
    python gn.py --mode full                 # Colab T4
    python gn.py --mode full --num_gn 8 --k 8 16 32 --sigma_rel 0.7 1.0 1.5
    python gn.py --selftest                  # shape / causality / grad checks only

Expects Tiny Shakespeare at ./input.txt (downloaded automatically if network is up).
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  TEB core: one sparse iterative emergence unit
# --------------------------------------------------------------------------- #
class TEBCore(nn.Module):
    """
    x: [B, N, D] -> [B, N, D]

    per step:
      1. causal kNN by squared euclidean distance
      2. gaussian weights  W_ij = exp(-d_ij^2 / (2 sigma^2))   (+ optional hard radius)
      3. for every neighbour pair (i, j): intersection = N(i) & N(j)
      4. centroid of each intersection (gaussian-weighted, from i's point of view)
      5. weight centroids by intersection size, sum, linear projection
      6. gated residual update + LayerNorm

    sigma / radius are RELATIVE to sqrt(D) because the GN input space is LayerNorm'd
    (typical squared distance between two points ~ 2*D). sigma_rel ~= 1.0 matches your
    old FULL-mode setting (sigma 12 at D=128).
    """

    def __init__(self, dim, k=8, sigma_rel=1.0, steps=2, radius_rel=0.0):
        super().__init__()
        self.dim, self.k, self.steps = dim, k, steps
        self.sigma = sigma_rel * math.sqrt(dim)
        self.radius = radius_rel * math.sqrt(dim)  # 0 = off
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # start the gate mostly closed-ish so early training is stable
        nn.init.constant_(self.gate.bias, -1.0)

    def one_step(self, x):
        B, N, D = x.shape
        k = min(self.k, N)

        # ---- causal squared distances ------------------------------------ #
        sq = (x * x).sum(-1)                                            # [B,N]
        dist = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * x @ x.transpose(1, 2)).clamp_min(0)
        future = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
        blocked = future
        if self.radius > 0:
            blocked = blocked | (dist > self.radius ** 2)
        dist = dist.masked_fill(blocked, float("inf"))

        # ---- kNN ----------------------------------------------------------- #
        vals, idx = torch.topk(dist, k, dim=-1, largest=False)         # [B,N,k]
        valid = torch.isfinite(vals)                                    # early tokens have < k nbrs
        validf = valid.float()
        w = torch.exp(-vals.masked_fill(~valid, 0.0) / (2 * self.sigma ** 2)) * validf

        # dense adjacency / weight matrices (cheap: N is small)
        A = torch.zeros(B, N, N, device=x.device, dtype=x.dtype).scatter(2, idx, validf)
        Wd = torch.zeros(B, N, N, device=x.device, dtype=x.dtype).scatter(2, idx, w)

        # ---- intersections N(i) & N(j) for every neighbour j of i ---------- #
        # A_j[b,i,a,m] = 1 if token m is a neighbour of i's a-th neighbour
        A_j = torch.gather(A, 1, idx.reshape(B, N * k, 1).expand(-1, -1, N)).view(B, N, k, N)

        size = torch.einsum("bnam,bnm->bna", A_j, A) * validf          # |N(i) & N(j_a)|
        den = torch.einsum("bnam,bnm->bna", A_j, Wd)                    # total gaussian mass in each intersection

        # centroid_a = sum_m Wd[i,m] A[i,m] A_j[a,m] x_m / den_a
        # update     = sum_a (size_a / sum size) * centroid_a
        # -> collapse a into one [N,N] mixing matrix so we never build [B,N,k,D]
        coef = size / (size.sum(-1, keepdim=True) + 1e-8) / (den + 1e-8)  # [B,N,k]
        M = Wd * torch.einsum("bna,bnam->bnm", coef, A_j)                 # [B,N,N], rows sum to <= 1
        candidate = self.proj(M @ x)

        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * candidate)

    def forward(self, x):
        for _ in range(self.steps):
            x = self.one_step(x)
        return x


# --------------------------------------------------------------------------- #
#  One geometric neuron = own view of the space + own TEB rules + own readout
# --------------------------------------------------------------------------- #
class GeometricNeuron(nn.Module):
    def __init__(self, dim, gn_dim, k, sigma_rel, steps, radius_rel):
        super().__init__()
        # the learned input projection is what lets identical-hyperparameter GNs
        # differentiate: kNN geometry is computed in each GN's OWN space, so the
        # neighbourhoods themselves differ, not just the readout weights.
        self.in_proj = nn.Sequential(nn.Linear(dim, gn_dim), nn.LayerNorm(gn_dim))
        self.teb = TEBCore(gn_dim, k, sigma_rel, steps, radius_rel)
        self.out_proj = nn.Linear(gn_dim, dim)

    def forward(self, x):
        return self.out_proj(self.teb(self.in_proj(x)))


def _per_gn(value, num_gn, name):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return list(value) * num_gn
        assert len(value) == num_gn, f"{name} needs 1 or {num_gn} values, got {len(value)}"
        return list(value)
    return [value] * num_gn


class GNLayer(nn.Module):
    """G geometric neurons in parallel -> concat -> project -> residual."""

    def __init__(self, dim, num_gn=4, gn_dim=None, k=8, sigma_rel=1.0, steps=2, radius_rel=0.0):
        super().__init__()
        gn_dim = gn_dim or dim
        ks = _per_gn(k, num_gn, "k")
        sigmas = _per_gn(sigma_rel, num_gn, "sigma_rel")
        stepl = _per_gn(steps, num_gn, "steps")
        radii = _per_gn(radius_rel, num_gn, "radius_rel")
        self.norm_in = nn.LayerNorm(dim)
        self.gns = nn.ModuleList(
            [GeometricNeuron(dim, gn_dim, ks[i], sigmas[i], stepl[i], radii[i]) for i in range(num_gn)]
        )
        self.merge = nn.Linear(dim * num_gn, dim)
        self.last_similarity = None  # diagnostic, see below

    def forward(self, x):
        h = self.norm_in(x)
        outs = [gn(h) for gn in self.gns]                               # G x [B,N,D]
        if not self.training:  # cheap collapse monitor: mean pairwise cosine sim between GN outputs
            with torch.no_grad():
                o = F.normalize(torch.stack(outs, 0).flatten(1, 2), dim=-1)   # [G, B*N, D]
                sim = torch.einsum("gtd,htd->gh", o, o) / o.shape[1]
                G = sim.shape[0]
                self.last_similarity = ((sim.sum() - sim.diag().sum()) / max(G * (G - 1), 1)).item()
        return x + self.merge(torch.cat(outs, dim=-1))


# --------------------------------------------------------------------------- #
#  Language model
# --------------------------------------------------------------------------- #
class GNLM(nn.Module):
    def __init__(self, vocab, dim, seq_len, depth=2, num_gn=4, gn_dim=None, k=8,
                 sigma_rel=1.0, steps=2, radius_rel=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList(
            [GNLayer(dim, num_gn, gn_dim, k, sigma_rel, steps, radius_rel) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, idx, targets=None):
        B, N = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(N, device=idx.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=300, temperature=0.8):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.seq_len:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# --------------------------------------------------------------------------- #
#  Config presets (mirrors your LITE / FULL tables)
# --------------------------------------------------------------------------- #
PRESETS = {
    "lite": dict(data_size=180_000, seq_len=32, batch=8, steps_train=2000, dim=40, depth=1,
                 num_gn=4, gn_dim=40, k=[4, 6], sigma_rel=1.0, teb_steps=2, lr=5e-4),
    "full": dict(data_size=1_115_394, seq_len=128, batch=64, steps_train=5000, dim=128, depth=2,
                 num_gn=4, gn_dim=64, k=[8, 16], sigma_rel=1.0, teb_steps=2, lr=1e-3),
}


def load_text(path="input.txt"):
    if not os.path.exists(path):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    return open(path, encoding="utf-8").read()


def get_batch(data, B, N, device):
    ix = torch.randint(len(data) - N - 1, (B,))
    x = torch.stack([data[i:i + N] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + N + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, val, cfg, device, iters=20):
    model.eval()
    losses = [model(*get_batch(val, cfg["batch"], cfg["seq_len"], device))[1].item() for _ in range(iters)]
    model.train()
    sims = [l.last_similarity for l in model.layers if l.last_similarity is not None]
    return sum(losses) / len(losses), sims


def train(cfg, device):
    text = load_text()[: cfg["data_size"]]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    tr, va = data[:n], data[n:]

    model = GNLM(len(chars), cfg["dim"], cfg["seq_len"], cfg["depth"], cfg["num_gn"], cfg["gn_dim"],
                 cfg["k"], cfg["sigma_rel"], cfg["teb_steps"]).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M  device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    best, t0 = 9e9, time.time()
    for step in range(1, cfg["steps_train"] + 1):
        x, y = get_batch(tr, cfg["batch"], cfg["seq_len"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            vl, sims = evaluate(model, va, cfg, device)
            best = min(best, vl)
            s = " ".join(f"{v:.2f}" for v in sims)
            print(f"step {step:5d} | train {loss.item():.3f} | val {vl:.3f} | best {best:.3f} "
                  f"| GN-sim/layer [{s}] | {time.time()-t0:.0f}s")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- sample ---")
    print("".join(itos[i] for i in model.generate(ctx)[0].tolist()))


# --------------------------------------------------------------------------- #
#  Self test
# --------------------------------------------------------------------------- #
def selftest():
    torch.manual_seed(0)
    m = GNLM(vocab=50, dim=32, seq_len=24, depth=2, num_gn=3, gn_dim=24, k=[4, 6, 8], sigma_rel=1.0, steps=2)
    x = torch.randint(0, 50, (2, 24))
    logits, loss = m(x, x)
    assert logits.shape == (2, 24, 50), logits.shape
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    # causality: changing the future must not change the past
    m.eval()
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 50, (2, 12))
    a, b = m(x)[0], m(x2)[0]
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5), "causality leak!"
    print("selftest OK  | loss", round(loss.item(), 3), "| GN similarity", m.layers[0].last_similarity)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lite", choices=PRESETS)
    ap.add_argument("--num_gn", type=int)
    ap.add_argument("--gn_dim", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--k", type=int, nargs="+", help="1 value (all GNs) or one per GN")
    ap.add_argument("--sigma_rel", type=float, nargs="+")
    ap.add_argument("--teb_steps", type=int)
    ap.add_argument("--steps_train", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        cfg = dict(PRESETS[a.mode])
        cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
        train(cfg, "cuda" if torch.cuda.is_available() else "cpu")
