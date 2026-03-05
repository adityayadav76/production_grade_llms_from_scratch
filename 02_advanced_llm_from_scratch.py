'''
pip install torch tokenizers einops flash-attn
flash-attn only builds on linux not on windows.

Uses the Small Shakespeare dataset
Replaced character tokenizer with BPE (via tokenizers library)
Adds FlashAttention for fast attention
Uses Rotary Positional Embeddings (RoPE)
Replaces LayerNorm with RMSNorm
Fully configurable depth, with sensible defaults
Complete train / eval / save / generate script



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

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    FLASH_OK = True
except ImportError:
    FLASH_OK = False
    print("FlashAttention not available — falling back to PyTorch attention.")

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
    "epochs": 5000, # train it a lot, lets see how it performs
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

# ===============================
# 7️⃣  Multi-Head Attention (Flash / Fallback)
# ===============================

class MultiHeadAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.d, self.h = d, heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # apply rotary
        q, k = apply_rope(q), apply_rope(k)

        if FLASH_OK:
            # FlashAttention expects packed QKV
            qkv_packed = torch.cat([q, k, v], dim=-1)
            attn_out = flash_attn_unpadded_qkvpacked_func(qkv_packed, None, dropout_p=0.1)
        else:
            scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.dk)
            mask = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, -1e9)
            probs = scores.softmax(dim=-1)
            attn_out = probs @ v

        out = rearrange(attn_out, "b h t d -> b t (h d)")
        return self.out(out)

# ===============================
# 8️⃣  Transformer Block
# ===============================

class TransformerBlock(nn.Module):
    def __init__(self, d, heads, ff_mult):
        super().__init__()
        self.attn = MultiHeadAttention(d, heads)
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
torch.save(model.state_dict(), "advanced_gpt.pth")
print("Saved model")

# Inference
model.eval()
prompt = "To be, or not to"
input_ids = torch.tensor(encode(prompt))[None, :].to(device)
out = model.generate(input_ids, max_new_tokens=100)
print("\nGenerated:\n", decode(out[0].tolist()))