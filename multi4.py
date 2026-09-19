import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

# ==========================================
# 🚀 TEB: MULTI4 (3 Layers, GPU-Optimized, Fixed)
# ==========================================
LITE_MODE = False  # GPU mode

# ==========================================
# 1. Download and Prepare Tiny Shakespeare
# ==========================================
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

# Hyperparameters (Fixed for GPU / 128D)
if LITE_MODE:
    DATA_SIZE = 180000
    SEQ_LEN = 32
    BATCH_SIZE = 8
    EPOCHS = 1500
    DIM = 32
    STEPS = 2
    K_NEIGHBORS = 4
    SIGMA = 0.5
    RADIUS = 0.1
    LR = 5e-4
    print("Running in LITE MODE (Termux optimized)")
else:
    DATA_SIZE = len(text)       # Full 1.1M characters
    SEQ_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 5000
    DIM = 128
    STEPS = 2
    K_NEIGHBORS = 8             # Scaled for 128D
    SIGMA = 12.0                # Scaled for 128D
    RADIUS = 0.0                # No hard cutoff
    LR = 1e-3                   # Higher LR for GPU
    print("Running in FULL MODE (GPU optimized) - DIM=128, SIGMA=12.0, RADIUS=0.0")

data = torch.tensor([char_to_idx[ch] for ch in text[:DATA_SIZE]], dtype=torch.long)
print(f"Dataset loaded: {len(data)} characters, {vocab_size} unique chars.")
print("-" * 50)

# ==========================================
# 2. The Sparse Iterative Emergence Layer (TEB)
# ==========================================
class SparseIterativeEmergenceLayer(nn.Module):
    def __init__(self, dim, steps=2, k=8, sigma=12.0, radius=0.0):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.k = k
        self.sigma = sigma
        self.radius = radius
        self.V = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.update_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        k = min(self.k, N - 1)
        
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        
        for step in range(self.steps):
            dist_sq = torch.cdist(x, x) ** 2
            dist_sq = dist_sq.masked_fill(causal_mask, float('inf'))
            
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
            
            # Unrestricted update (no dampening)
            x = x + gate * update_candidate
            x = self.ln(x)
            
        return x

# ==========================================
# 3. The Models (3 Stacked Layers)
# ==========================================
class SparseEmergenceLM(nn.Module):
    def __init__(self, vocab_size, dim=128, max_len=128, steps=2, k=8, sigma=12.0, radius=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        
        # 🧠 STACK THREE LAYERS
        self.layer1 = SparseIterativeEmergenceLayer(dim, steps, k, sigma, radius)
        self.layer2 = SparseIterativeEmergenceLayer(dim, steps, k, sigma, radius)
        self.layer3 = SparseIterativeEmergenceLayer(dim, steps, k, sigma, radius)
        
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x = self.embed(x) + self.pos_embed(pos)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return self.out(x)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=128, nhead=4, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        # Stack 3 Transformer layers for fair comparison
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.out = nn.Linear(dim, vocab_size)
        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(max_len)

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x = self.embed(x) + self.pos_embed(pos)
        mask = self.causal_mask[:N, :N].to(x.device)
        out = self.transformer(x, mask=mask)
        return self.out(out)

# ==========================================
# 4. Data Loader and Training Loop
# ==========================================
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

def train_model(model, data, epochs, batch_size, seq_len, device, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        x, y = get_batch(data, batch_size, seq_len, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        
        # No gradient clipping, unrestricted updates
        optimizer.step()
        
        if epoch % (epochs // 10) == 0: # Print 10 times
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
    return model

# ==========================================
# 5. Text Generation
# ==========================================
def generate(model, start_str="ROMEO: ", length=200, device='cpu'):
    model.eval()
    chars = [char_to_idx[c] for c in start_str]
    x = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    max_len = model.pos_embed.num_embeddings
    
    with torch.no_grad():
        for _ in range(length):
            x_cond = x[:, -max_len:]
            logits = model(x_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            x = torch.cat([x, torch.tensor([[next_char]], device=device)], dim=1)
    return ''.join([idx_to_char[i] for i in x[0].tolist()])

# ==========================================
# 6. Run the Experiment
# ==========================================
print("\nTraining Sparse Iterative Emergence LM (TEB) - 3 Layers...")
emergence_model = train_model(
    SparseEmergenceLM(vocab_size=vocab_size, dim=DIM, max_len=SEQ_LEN, 
        steps=STEPS, k=K_NEIGHBORS, sigma=SIGMA, radius=RADIUS).to(device),
    data, EPOCHS, BATCH_SIZE, SEQ_LEN, device, lr=LR
)

print("\nTraining Transformer LM - 3 Layers...")
transformer_model = train_model(
    TransformerLM(vocab_size=vocab_size, dim=DIM, nhead=4, max_len=SEQ_LEN).to(device),
    data, EPOCHS, BATCH_SIZE, SEQ_LEN, device, lr=LR
)

# ==========================================
# 7. Generate Text (No Quantization on GPU)
# ==========================================
print("\n--- GENERATED TEXT (TEB - 3 Layers) ---")
print(generate(emergence_model, device=device))
print("\n--- GENERATED TEXT (Transformer - 3 Layers) ---")
print(generate(transformer_model, device=device))
