"""
causalconv.py -- Baseline to prove GN2 isn't just a local window.
Run: python -u causalconv.py --k 5 2>&1 | tee log_causalconv.log
"""
import argparse, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from gn2 import load_text, get_batch, fixed_batches, evaluate, PRESETS

class CausalConvBlock(nn.Module):
    def __init__(self, dim, k=5, mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Depthwise convolution, causal (left-padded)
        self.conv = nn.Conv1d(dim, dim, k, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*mult), nn.GELU(), nn.Linear(dim*mult, dim))
        self.k = k
        
    def forward(self, x):
        h = self.norm1(x).transpose(1, 2)
        h = F.pad(h, (self.k - 1, 0))
        h = self.pw(self.conv(h)).transpose(1, 2)
        x = x + h
        return x + self.ff(self.norm2(x))

class CausalConvLM(nn.Module):
    def __init__(self, vocab, dim, seq_len, depth=2, k=5):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([CausalConvBlock(dim, k) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        
    def forward(self, idx, targets=None):
        B, N = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(N, device=idx.device))
        for b in self.blocks: 
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    
    cfg = dict(PRESETS["full"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    
    text = load_text()[:cfg["data_size"]]
    chars = sorted(set(text))
    stoi = {c:i for i,c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.9*len(data))
    tr, va = data[:n], data[n:]
    
    vb = fixed_batches(va, cfg["batch"], cfg["seq_len"], dev, n=20)
    m = CausalConvLM(len(chars), cfg["dim"], cfg["seq_len"], cfg["depth"], a.k).to(dev)
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.3f}M  device: {dev}  k={a.k}")
    
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=0.01)
    best, t0 = 9e9, time.time()
    
    for step in range(1, cfg["steps_train"]+1):
        lr = cfg["lr"] * (0.1 + 0.9*0.5*(1+math.cos(math.pi*step/cfg["steps_train"])))
        for g in opt.param_groups: g["lr"] = lr
        
        x, y = get_batch(tr, cfg["batch"], cfg["seq_len"], dev)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        
        if step % 500 == 0:
            vl = evaluate(m, vb)
            best = min(best, vl)
            print(f"step {step:5d} | train {loss.item():.3f} | val {vl:.3f} | best {best:.3f} | {time.time()-t0:.0f}s")
