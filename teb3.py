import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

# ==========================================
# 🚀 TEB-BIDIR-LM: Bidirectional TEB with Recursive Smaller Dimension
# ==========================================

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
BATCH_SIZE = 64
EPOCHS = 3000
DIM = 128
DIM_SMALL = 64  # The "selected smaller dimension" for recursion
TEB_STEPS = 3
K_NEIGHBORS = 16
SIGMA = 24.0
RADIUS = 0.0
LR = 1e-3
MASK_PROB = 0.15

data = torch.tensor([char_to_idx[ch] for ch in text[:DATA_SIZE]], dtype=torch.long)
print(f"Dataset loaded: {len(data)} characters, {vocab_size} unique chars.")
print("-" * 50)

# ==========================================
# 2. Sparse Iterative Emergence Layer (No Causal Mask)
# ==========================================
class SparseIterativeEmergenceLayer(nn.Module):
    def __init__(self, dim, steps=1, k=16, sigma=24.0, radius=0.0):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.k = k
        self.sigma = sigma
        self.radius = radius
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.update_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        k = min(self.k, N - 1)
        
        # NO CAUSAL MASK. Every token sees every other token.
        
        for step in range(self.steps):
            dist_sq = torch.cdist(x, x) ** 2
            # Removed causal mask line here
            
            knn_dist, knn_indices = torch.topk(dist_sq, k + 1, largest=False, dim=-1)
            knn_dist = knn_dist[:, :, 1:]
            knn_indices = knn_indices[:, :, 1:]
            
            W_ij = torch.exp(-knn_dist / (2 * self.sigma**2))
            if self.radius > 0:
                W_ij = W_ij * (knn_dist <= self.radius**2).float()
            
            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
            x_neighbors = x[batch_idx, knn_indices]
            
            idx_j = knn_indices[batch_idx, knn_indices]
            
            knn_exp = knn_indices.unsqueeze(2).unsqueeze(-1)
            idx_exp = idx_j.unsqueeze(-2)
            mask = (knn_exp == idx_exp).any(dim=-1)
            mask_float = mask.float()
            
            weighted_mask = mask_float * W_ij.unsqueeze(-1)
            
            mask_sum = weighted_mask.sum(dim=3, keepdim=True) + 1e-8
            c_ij = torch.einsum('bijk,bikd->bijd', weighted_mask, x_neighbors) / mask_sum
            
            overlap_sizes = weighted_mask.sum(dim=3)
            weighted_c = overlap_sizes.unsqueeze(-1) * c_ij
            update_candidate = weighted_c.sum(dim=2) / (overlap_sizes.sum(dim=2, keepdim=True) + 1e-8)
            update_candidate = self.update_proj(update_candidate)
            
            gate_input = torch.cat([x, update_candidate], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            
            x = x + gate * update_candidate
            x = self.ln(x)
            
        return x

# ==========================================
# 3. Recursive Smaller Dimension TEB Block
# ==========================================
class RecursiveTEBBlock(nn.Module):
    def __init__(self, dim, dim_small, steps=3, k=16, sigma=24.0, radius=0.0):
        super().__init__()
        self.down_project = nn.Linear(dim, dim_small)
        self.teb = SparseIterativeEmergenceLayer(dim_small, steps=steps, k=k, sigma=sigma, radius=radius)
        self.up_project = nn.Linear(dim_small, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # Project down to smaller dimension
        x_small = self.down_project(x)
        
        # Apply TEB recursively in the smaller dimension
        x_small = self.teb(x_small)
        
        # Project back up
        x_out = self.up_project(x_small)
        
        # Residual connection + LayerNorm
        return self.ln(x + x_out)

# ==========================================
# 4. TEB Bidirectional Language Model
# ==========================================
class TEBBidirLM(nn.Module):
    def __init__(self, vocab_size, dim=128, dim_small=64, max_len=128, num_layers=3, steps=3, k=16, sigma=24.0, radius=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        
        self.layers = nn.ModuleList([
            RecursiveTEBBlock(dim, dim_small, steps, k, sigma, radius)
            for _ in range(num_layers)
        ])
        
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x = self.embed(x) + self.pos_embed(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.out(x)

# ==========================================
# 5. MLM Data Loader and Training Loop
# ==========================================
def get_mlm_batch(data, batch_size, seq_len, device, mask_prob=0.15):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    labels = x.clone()
    
    # Create mask
    prob_matrix = torch.full(x.shape, mask_prob, device=device)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    
    # 80% of the time, replace with a random token. (We omit the [MASK] token for simplicity, just using random noise)
    # 10% of the time, keep the same. 10% random.
    random_tokens = torch.randint(vocab_size, x.shape, device=device)
    x[masked_indices] = random_tokens[masked_indices]
    
    return x, labels, masked_indices

def train_model(model, data, epochs, batch_size, seq_len, device, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        x, labels, masked_indices = get_mlm_batch(data, batch_size, seq_len, device, MASK_PROB)
        optimizer.zero_grad()
        logits = model(x)
        
        # Only compute loss on masked tokens
        loss = criterion(logits[masked_indices], labels[masked_indices])
        
        loss.backward()
        optimizer.step()
        
        if epoch % (epochs // 10) == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
    return model

# ==========================================
# 6. MLM Text Generation (Iterative Filling)
# ==========================================
def generate_mlm(model, start_str="ROMEO: ", length=200, device='cpu', steps=10):
    model.eval()
    chars = [char_to_idx[c] for c in start_str]
    x = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    
    # Start with a fully masked sequence
    seq = torch.full((1, length), 0, dtype=torch.long, device=device)
    seq[0, :len(chars)] = x[0]
    
    with torch.no_grad():
        for step in range(steps):
            logits = model(seq)
            probs = F.softmax(logits, dim=-1)
            # Randomly sample predictions for all positions, then average over steps (Gibbs sampling style)
            # For a quick demo, we'll just replace the most confident predictions
            preds = torch.argmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1).values
            
            # Replace the least confident tokens with predictions
            # (This is a crude iterative decoding, but works for MLM)
            seq = preds # For simplicity, just take the argmax at each step
            break # Break after one step for now
            
    return ''.join([idx_to_char[i] for i in seq[0].tolist()])

# ==========================================
# 7. Run the Experiment
# ==========================================
print(f"\nTraining TEB-Bidir-LM (No Causal Mask, Recursive Dims)...")
teb_model = train_model(
    TEBBidirLM(
        vocab_size=vocab_size, dim=DIM, dim_small=DIM_SMALL, max_len=SEQ_LEN,
        num_layers=3, steps=TEB_STEPS, k=K_NEIGHBORS, sigma=SIGMA, radius=RADIUS
    ).to(device),
    data, EPOCHS, BATCH_SIZE, SEQ_LEN, device, lr=LR
)

print("\n--- GENERATED TEXT (TEB-Bidir-LM) ---")
# Note: Because we removed the causal mask, generation is now an iterative filling process, not left-to-right.
print(generate_mlm(teb_model, device=device))
