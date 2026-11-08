"""
gn2.py  --  Geometric Neurons v2

What changed vs gn.py (GN v1 got best val 2.439, worse than TEB's 2.32):

 1. RELATIVE-POSITION DISTANCE BIAS (the main change).
    In v1 the kNN geometry lived purely in embedding space, so a token's nearest
    neighbours were probably other tokens with the same / similar CHARACTER, not the
    tokens next to it. v2 adds a learnable per-GN penalty on sequence distance:
        dist_total(i,j) = dist_content(i,j) + softplus(beta_g) * (2*D / pos_ref) * (i - j)
    beta is initialised log-spaced across the GNs (e.g. 0.05 ... 1.0), so GNs START in
    different range regimes (very local ... fairly global) and can still learn their own.
    beta=1 means "pos_ref tokens back costs as much as a typical full content distance".
    Pass --no_pos_bias to switch it off (this is then v1 geometry).

 2. DIAGNOSTICS THAT MEAN SOMETHING (the v1 cosine "GN-sim" did not, because each GN
    writes through its own random output basis). Printed at init and at the end:
      * per GN: % of neighbours that are the SAME character, mean look-back distance,
        % of neighbours within 4 tokens
      * per layer: mean Jaccard overlap of the neighbour SETS between GNs
        (basis-free: 1.0 = GNs find identical neighbourhoods, ~0 = different)
      * per GN ablation: val-loss increase when that GN's output is zeroed
        (the real test of "is this neuron doing something / are they different")

 3. TRAINING: warmup + cosine LR decay (v1 was constant LR and still improving),
    fixed validation batches (smoother, comparable curve), seed flag, best checkpoint
    saved to gn2_best.pt, no weight decay on beta.

Reproduce roughly the v1 setup:   python gn2.py --mode full --no_pos_bias --sched const
Main v2 run:                      python -u gn2.py --mode full 2>&1 | tee gn2_run.log
Check everything is wired:        python gn2.py --selftest
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y):
    return math.log(math.expm1(y))


# --------------------------------------------------------------------------- #
#  TEB core with optional learnable relative-position distance bias
# --------------------------------------------------------------------------- #
class TEBCore(nn.Module):
    """
    x: [B, N, D] -> [B, N, D]   (see gn.py for the derivation of the M-matrix trick)

    sigma / radius are relative to sqrt(D) because the GN input space is LayerNorm'd
    (typical squared distance between two points ~ 2*D).
    """

    def __init__(self, dim, k=8, sigma_rel=1.0, steps=2, radius_rel=0.0, pos_beta=None, pos_ref=16.0):
        super().__init__()
        self.dim, self.k, self.steps = dim, k, steps
        self.sigma = sigma_rel * math.sqrt(dim)
        self.radius = radius_rel * math.sqrt(dim)  # 0 = off
        self.pos_ref = float(pos_ref)
        if pos_beta is None:
            self.beta_raw = None
        else:
            assert pos_beta > 0, "pos_beta must be > 0 (use None to disable)"
            self.beta_raw = nn.Parameter(torch.tensor(inv_softplus(float(pos_beta))))
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.constant_(self.gate.bias, -1.0)
        # diagnostics: when record=True the neighbour indices of the FIRST step are kept
        self.record = False
        self._rec = None

    def beta(self):
        return None if self.beta_raw is None else F.softplus(self.beta_raw)

    def one_step(self, x):
        B, N, D = x.shape
        k = min(self.k, N)

        # ---- causal squared distances (+ relative-position penalty) ---------- #
        sq = (x * x).sum(-1)
        dist = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * x @ x.transpose(1, 2)).clamp_min(0)
        if self.beta_raw is not None:
            ar = torch.arange(N, device=x.device)
            rel = (ar[:, None] - ar[None, :]).clamp_min(0).to(x.dtype)       # i - j for j <= i
            dist = dist + F.softplus(self.beta_raw) * (2.0 * D / self.pos_ref) * rel
        future = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
        blocked = future
        if self.radius > 0:
            blocked = blocked | (dist > self.radius ** 2)
        dist = dist.masked_fill(blocked, float("inf"))

        # ---- kNN ----------------------------------------------------------- #
        vals, idx = torch.topk(dist, k, dim=-1, largest=False)               # [B,N,k]
        valid = torch.isfinite(vals)
        validf = valid.float()
        if self.record and self._rec is None:
            self._rec = (idx.detach(), valid.detach())
        w = torch.exp(-vals.masked_fill(~valid, 0.0) / (2 * self.sigma ** 2)) * validf

        A = torch.zeros(B, N, N, device=x.device, dtype=x.dtype).scatter(2, idx, validf)
        Wd = torch.zeros(B, N, N, device=x.device, dtype=x.dtype).scatter(2, idx, w)

        # ---- intersections ------------------------------------------------- #
        A_j = torch.gather(A, 1, idx.reshape(B, N * k, 1).expand(-1, -1, N)).view(B, N, k, N)
        size = torch.einsum("bnam,bnm->bna", A_j, A) * validf
        den = torch.einsum("bnam,bnm->bna", A_j, Wd)
        coef = size / (size.sum(-1, keepdim=True) + 1e-8) / (den + 1e-8)
        M = Wd * torch.einsum("bna,bnam->bnm", coef, A_j)
        candidate = self.proj(M @ x)

        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * candidate)

    def forward(self, x):
        for _ in range(self.steps):
            x = self.one_step(x)
        return x


class GeometricNeuron(nn.Module):
    def __init__(self, dim, gn_dim, k, sigma_rel, steps, radius_rel, pos_beta, pos_ref):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(dim, gn_dim), nn.LayerNorm(gn_dim))
        self.teb = TEBCore(gn_dim, k, sigma_rel, steps, radius_rel, pos_beta, pos_ref)
        self.out_proj = nn.Linear(gn_dim, dim)

    def forward(self, x):
        return self.out_proj(self.teb(self.in_proj(x)))


def _per_gn(value, num_gn):
    """scalar -> repeated; list of any length -> cycled across GNs ([8,16] x4 -> 8,16,8,16)."""
    if isinstance(value, (list, tuple)):
        return [value[i % len(value)] for i in range(num_gn)]
    return [value] * num_gn


def default_betas(num_gn, lo=0.05, hi=1.0):
    if num_gn == 1:
        return [0.2]
    return [lo * (hi / lo) ** (g / (num_gn - 1)) for g in range(num_gn)]


class GNLayer(nn.Module):
    """G geometric neurons in parallel -> concat -> project -> residual."""

    def __init__(self, dim, num_gn=4, gn_dim=None, k=8, sigma_rel=1.0, steps=2, radius_rel=0.0,
                 pos_beta="auto", pos_ref=16.0):
        super().__init__()
        gn_dim = gn_dim or dim
        ks, sigmas = _per_gn(k, num_gn), _per_gn(sigma_rel, num_gn)
        stepl, radii = _per_gn(steps, num_gn), _per_gn(radius_rel, num_gn)
        betas = default_betas(num_gn) if pos_beta == "auto" else _per_gn(pos_beta, num_gn)
        self.norm_in = nn.LayerNorm(dim)
        self.gns = nn.ModuleList([
            GeometricNeuron(dim, gn_dim, ks[i], sigmas[i], stepl[i], radii[i], betas[i], pos_ref)
            for i in range(num_gn)
        ])
        self.merge = nn.Linear(dim * num_gn, dim)
        self.ablate = None  # index of a GN whose output is zeroed (for ablation studies)

    def forward(self, x):
        h = self.norm_in(x)
        outs = [gn(h) for gn in self.gns]
        if self.ablate is not None:
            outs[self.ablate] = torch.zeros_like(outs[self.ablate])
        return x + self.merge(torch.cat(outs, dim=-1))


class GNLM(nn.Module):
    def __init__(self, vocab, dim, seq_len, depth=2, num_gn=4, gn_dim=None, k=8, sigma_rel=1.0,
                 steps=2, radius_rel=0.0, pos_beta="auto", pos_ref=16.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([
            GNLayer(dim, num_gn, gn_dim, k, sigma_rel, steps, radius_rel, pos_beta, pos_ref)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight

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
#  Diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diagnose(model, ids, title=""):
    """Neighbour statistics per GN + neighbour-set Jaccard between GNs in each layer."""
    was_training = model.training
    model.eval()
    gns = [(li, gi, gn) for li, l in enumerate(model.layers) for gi, gn in enumerate(l.gns)]
    for _, _, gn in gns:
        gn.teb.record, gn.teb._rec = True, None
    model(ids)

    B, N = ids.shape
    dev = ids.device
    pos = torch.arange(N, device=dev)
    tok_exp = ids.unsqueeze(1).expand(B, N, N)
    eye = torch.eye(N, device=dev).unsqueeze(0)
    adj = {}
    print(f"\n=== neighbour diagnostics {title} ===")
    print("layer GN |   k  beta |  same-char%  mean-lookback  within-4%")
    for li, gi, gn in gns:
        idx, valid = gn.teb._rec
        offset = pos.view(1, N, 1) - idx                           # how far back each neighbour is
        keep = valid & (offset > 0)                                # drop self
        nb_tok = torch.gather(tok_exp, 2, idx)
        same = (nb_tok == ids.unsqueeze(-1)) & keep
        n = keep.sum().clamp_min(1).float()
        same_pct = 100 * same.sum().item() / n.item()
        mean_off = (offset * keep).sum().item() / n.item()
        near_pct = 100 * ((offset <= 4) & keep).sum().item() / n.item()
        b = gn.teb.beta()
        bs = "  off" if b is None else f"{b.item():5.2f}"
        print(f"  {li}    {gi}  | {gn.teb.k:3d} {bs} | {same_pct:9.1f}  {mean_off:13.1f}  {near_pct:9.1f}")
        A = torch.zeros(B, N, N, device=dev).scatter(2, idx, valid.float()) * (1 - eye)
        adj[(li, gi)] = A
    for li, layer in enumerate(model.layers):
        G = len(layer.gns)
        js = []
        for a in range(G):
            for b2 in range(a + 1, G):
                Aa, Ab = adj[(li, a)], adj[(li, b2)]
                inter = (Aa * Ab).sum()
                union = ((Aa + Ab) > 0).float().sum().clamp_min(1)
                js.append((inter / union).item())
        if js:
            print(f"layer {li}: mean neighbour-set Jaccard between GNs = {sum(js)/len(js):.3f} "
                  f"(1 = identical neighbourhoods, 0 = disjoint)")
    for _, _, gn in gns:
        gn.teb.record, gn.teb._rec = False, None
    model.train(was_training)
    print()


@torch.no_grad()
def ablation(model, batches):
    """Val-loss increase when each GN's output is zeroed. Near 0 for all = GNs not contributing."""
    was_training = model.training
    model.eval()

    def val_loss():
        return sum(model(x, y)[1].item() for x, y in batches) / len(batches)

    base = val_loss()
    print(f"=== GN ablation (base val loss {base:.4f}; value = loss increase when that GN is zeroed) ===")
    for li, layer in enumerate(model.layers):
        row = []
        for gi in range(len(layer.gns)):
            layer.ablate = gi
            row.append(val_loss() - base)
        layer.ablate = None
        print(f"layer {li}: " + "  ".join(f"GN{gi}:{d:+.4f}" for gi, d in enumerate(row)))
    model.train(was_training)
    print()


# --------------------------------------------------------------------------- #
#  Data / config / training
# --------------------------------------------------------------------------- #
PRESETS = {
    "lite": dict(data_size=180_000, seq_len=32, batch=8, steps_train=2000, dim=40, depth=1,
                 num_gn=4, gn_dim=40, k=[4, 6], sigma_rel=1.0, teb_steps=2, lr=5e-4,
                 pos_beta="auto", pos_ref=16.0, sched="cosine", seed=0),
    "full": dict(data_size=1_115_394, seq_len=128, batch=64, steps_train=5000, dim=128, depth=2,
                 num_gn=4, gn_dim=64, k=[8, 16], sigma_rel=1.0, teb_steps=2, lr=1e-3,
                 pos_beta="auto", pos_ref=16.0, sched="cosine", seed=0),
}


def load_text(path="input.txt"):
    if not os.path.exists(path):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    return open(path, encoding="utf-8").read()


def get_batch(data, B, N, device, gen=None):
    ix = torch.randint(len(data) - N - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + N] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + N + 1] for i in ix]).to(device)
    return x, y


def fixed_batches(data, B, N, device, n=20, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return [get_batch(data, B, N, device, g) for _ in range(n)]


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in batches) / len(batches)
    model.train()
    return loss


def lr_at(step, total, base, sched, warmup=100):
    if step < warmup:
        return base * step / warmup
    if sched == "const":
        return base
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def build_model(vocab, cfg):
    return GNLM(vocab, cfg["dim"], cfg["seq_len"], cfg["depth"], cfg["num_gn"], cfg["gn_dim"], cfg["k"],
                cfg["sigma_rel"], cfg["teb_steps"], 0.0, cfg["pos_beta"], cfg["pos_ref"])


def beta_str(model):
    out = []
    for l in model.layers:
        bs = [gn.teb.beta() for gn in l.gns]
        out.append("/".join("off" if b is None else f"{b.item():.2f}" for b in bs))
    return " | ".join(out)


def train(cfg, device):
    torch.manual_seed(cfg["seed"])
    text = load_text()[: cfg["data_size"]]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    tr, va = data[:n], data[n:]
    vb = fixed_batches(va, cfg["batch"], cfg["seq_len"], device, n=20)

    model = build_model(len(chars), cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M  device: {device}  "
          f"pos_bias: {'off' if cfg['pos_beta'] is None else 'on'}  sched: {cfg['sched']}  seed: {cfg['seed']}")
    decay = [p for nme, p in model.named_parameters() if "beta_raw" not in nme]
    nodecay = [p for nme, p in model.named_parameters() if "beta_raw" in nme]
    groups = [{"params": decay, "weight_decay": 0.01}]
    if nodecay:
        groups.append({"params": nodecay, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])

    diagnose(model, vb[0][0][:16], title="at init")

    best, t0, total = 9e9, time.time(), cfg["steps_train"]
    for step in range(1, total + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, cfg["lr"], cfg["sched"])
        x, y = get_batch(tr, cfg["batch"], cfg["seq_len"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            vl = evaluate(model, vb)
            if vl < best:
                best = vl
                torch.save(model.state_dict(), "gn2_best.pt")
            print(f"step {step:5d} | train {loss.item():.3f} | val {vl:.3f} | best {best:.3f} "
                  f"| beta {beta_str(model)} | {time.time()-t0:.0f}s")

    diagnose(model, vb[0][0][:16], title="after training")
    ablation(model, vb[:8])
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("--- sample ---")
    print("".join(itos[i] for i in model.generate(ctx)[0].tolist()))


# --------------------------------------------------------------------------- #
#  Self test
# --------------------------------------------------------------------------- #
def selftest():
    torch.manual_seed(0)
    for pb in ("auto", None):
        m = GNLM(vocab=50, dim=32, seq_len=24, depth=2, num_gn=3, gn_dim=24, k=[4, 6, 8],
                 sigma_rel=1.0, steps=2, pos_beta=pb)
        x = torch.randint(0, 50, (2, 24))
        logits, loss = m(x, x)
        assert logits.shape == (2, 24, 50), logits.shape
        loss.backward()
        for nme, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {nme}"
        if pb == "auto":
            assert m.layers[0].gns[0].teb.beta_raw.grad is not None, "beta got no gradient"
        m.eval()
        x2 = x.clone()
        x2[:, 12:] = torch.randint(0, 50, (2, 12))
        assert torch.allclose(m(x)[0][:, :12], m(x2)[0][:, :12], atol=1e-5), f"causality leak (pos_beta={pb})"
        print(f"model OK (pos_beta={pb}): shapes, grads, causality")

    m = GNLM(vocab=50, dim=32, seq_len=24, depth=2, num_gn=3, gn_dim=24, k=[4, 6, 8], pos_beta="auto")
    x = torch.randint(0, 50, (4, 24))
    diagnose(m, x, title="selftest")
    ablation(m, [(x, x)])

    for name, c in PRESETS.items():
        cc = dict(c, seq_len=16)
        mm = build_model(65, cc)
        xx = torch.randint(0, 65, (2, 16))
        _, ll = mm(xx, xx)
        ll.backward()
        assert torch.isfinite(ll), name
        print(f"preset '{name}' OK")
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lite", choices=PRESETS)
    ap.add_argument("--num_gn", type=int)
    ap.add_argument("--gn_dim", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--k", type=int, nargs="+", help="1 value (all GNs) or a list that cycles across GNs")
    ap.add_argument("--sigma_rel", type=float, nargs="+")
    ap.add_argument("--teb_steps", type=int)
    ap.add_argument("--steps_train", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sched", choices=["cosine", "const"])
    ap.add_argument("--pos_ref", type=float)
    ap.add_argument("--pos_beta", type=float, nargs="+", help="initial beta per GN (cycles); default log-spaced")
    ap.add_argument("--no_pos_bias", action="store_true", help="disable the position penalty (v1 geometry)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        cfg = dict(PRESETS[a.mode])
        cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
        if a.no_pos_bias:
            cfg["pos_beta"] = None
        train(cfg, "cuda" if torch.cuda.is_available() else "cpu")
