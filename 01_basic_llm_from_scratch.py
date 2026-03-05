import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# ===============================
# 1. Small Dataset
# ===============================

text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""

# ===============================
# 2. Character-Level Tokenizer
# ===============================

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return ''.join([itos[i] for i in t])

data = encode(text)

# ===============================
# 3. Hyperparameters
# ===============================

block_size = 64
batch_size = 16
embed_dim = 128
num_heads = 4
num_layers = 2
dropout = 0.1
lr = 3e-4
epochs = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 4. Data Loader
# ===============================

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ===============================
# 5. Transformer Components
# ===============================

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# ===============================
# 6. Train Model
# ===============================

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ===============================
# 7. Save Model
# ===============================

torch.save({
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos
}, "mini_gpt.pth")

print("Model saved to mini_gpt.pth")

# ===============================
# 8. Inference
# ===============================

checkpoint = torch.load("mini_gpt.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "To be"
context = torch.tensor([stoi[c] for c in prompt], dtype=torch.long)[None, :].to(device)

generated = model.generate(context, max_new_tokens=100)
print("\nGenerated Text:\n")
print(decode(generated[0].tolist()))