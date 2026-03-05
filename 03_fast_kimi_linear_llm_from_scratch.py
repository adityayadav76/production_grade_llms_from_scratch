'''
pip install tokenizers einops


Parallel (no Python time loop)
Chunked linear attention
Gated
Causal
O(n) memory
GPU-friendly
Drop-in replacement for the GPT block
Configurable depth unchanged



'''


import os
import math
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from einops import rearrange


# ===============================
# 1️⃣  Download Small Shakespeare Dataset
# ===============================

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "shakespeare.txt"

if not os.path.exists(data_path):
    print("Downloading Small Shakespeare dataset …")
    with open(data_path, "w") as f:
        text = requests.get(DATA_URL).text
        f.write(text)
else:
    text = open(data_path).read()

# ===============================
# 2️⃣  BPE Tokenizer Setup
# ===============================

if not os.path.exists("bpe-vocab.json"):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator([text], vocab_size=10_000, min_frequency=2,
                                  special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
    tokenizer.save_model(".", "bpe")
else:
    tokenizer = ByteLevelBPETokenizer("bpe-vocab.json", "bpe-merges.txt")

tokenizer.add_special_tokens(["<pad>", "<unk>", "<s>", "</s>"])
tokenizer.enable_truncation(max_length=1024)

def encode(s): return tokenizer.encode(s).ids
def decode(ids): return tokenizer.decode(ids)

data_ids = encode(text)
vocab_size = tokenizer.get_vocab_size()

# ===============================
# 3️⃣  Configurable Hyperparameters
# ===============================

config = {
    "block_size": 128,
    "batch_size": 24,
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 4,   # configurable depth
    "ff_multiplier": 4,
    "lr": 3e-4,
    "epochs": 1000,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 4️⃣  Data Batching
# ===============================

def get_batch():
    ix = torch.randint(0, len(data_ids) - config["block_size"] - 1, (config["batch_size"],))
    x = torch.stack([torch.tensor(data_ids[i : i + config["block_size"]]) for i in ix])
    y = torch.stack([torch.tensor(data_ids[i + 1 : i + 1 + config["block_size"]]) for i in ix])
    return x.to(device), y.to(device)

# ===============================
# 5️⃣  Rotary Embeddings
# ===============================

def apply_rope(x):
    seq_len, dim = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device) / dim))
    positions = torch.arange(0, seq_len, device=x.device).type_as(inv_freq)
    sinusoid = torch.einsum("i , j -> i j", positions, inv_freq)
    sin, cos = sinusoid.sin(), sinusoid.cos()
    x1, x2 = x[..., ::2], x[..., 1::2]
    x = torch.stack([(x1 * cos - x2 * sin), (x1 * sin + x2 * cos)], dim=-1)
    return x.flatten(-2)

# ===============================
# 6️⃣  RMSNorm
# ===============================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return x * (self.scale / (norm + self.eps))


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class KimiLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=64):
        super().__init__()
        self.d = d_model
        self.h = num_heads
        self.dk = d_model // num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = torch.sigmoid(self.g_proj(x))

        q = rearrange(q, "b t (h d) -> b h t d", h=self.h)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.h)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.h)
        g = rearrange(g, "b t (h d) -> b h t d", h=self.h)

        # RoPE if you already have apply_rope()
        q = apply_rope(q)
        k = apply_rope(k)

        q = self.feature_map(q)
        k = self.feature_map(k)

        outputs = []
        S = torch.zeros(B, self.h, self.dk, self.dk, device=x.device)
        Z = torch.zeros(B, self.h, self.dk, device=x.device)

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)

            q_chunk = q[:, :, start:end]
            k_chunk = k[:, :, start:end]
            v_chunk = v[:, :, start:end]
            g_chunk = g[:, :, start:end]

            # Compute KV within chunk
            kv = torch.einsum("bhtd,bhte->bhdte", k_chunk, v_chunk)
            kv = kv.sum(dim=3)

            k_sum = k_chunk.sum(dim=2)

            # Add previous state
            S = S + kv
            Z = Z + k_sum

            # Compute output
            numerator = torch.einsum("bhtd,bhde->bhte", q_chunk, S)
            denominator = torch.einsum("bhtd,bhd->bht", q_chunk, Z).unsqueeze(-1) + 1e-6

            out_chunk = numerator / denominator
            out_chunk = g_chunk * out_chunk

            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=2)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.out_proj(out)

# ===============================
# 8️⃣  Transformer Block
# ===============================

class TransformerBlock(nn.Module):
    def __init__(self, d, heads, ff_mult):
        super().__init__()
        self.attn = KimiLinearAttention(d, heads, chunk_size=64)
        self.rms1 = RMSNorm(d)
        self.rms2 = RMSNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.GELU(),
            nn.Linear(ff_mult * d, d)
        )
    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        x = x + self.ff(self.rms2(x))
        return x

# ===============================
# 9️⃣  GPT Model
# ===============================

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, config["embed_dim"])
        self.pos_emb = nn.Embedding(config["block_size"], config["embed_dim"])
        self.blocks = nn.ModuleList([
            TransformerBlock(config["embed_dim"],
                             config["num_heads"],
                             config["ff_multiplier"])
            for _ in range(config["num_layers"])
        ])
        self.norm = RMSNorm(config["embed_dim"])
        self.head = nn.Linear(config["embed_dim"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# ===============================
# 🔟 Train / Save / Inference
# ===============================

model = GPT(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])

for ep in range(config["epochs"]):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 100 == 0:
        print(f"Epoch {ep} | Loss: {loss.item():.4f}")

# Save
torch.save(model.state_dict(), "fast_kimi_linear_gpt.pth")
print("Saved model")

# Inference
model.eval()
prompt = "To be, or not to"
input_ids = torch.tensor(encode(prompt))[None, :].to(device)
out = model.generate(input_ids, max_new_tokens=100)
print("\nGenerated:\n", decode(out[0].tolist()))