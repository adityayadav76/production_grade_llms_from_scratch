'''

BPE tokenizer (trained on Shakespeare)
RoPE positional embeddings
RMSNorm
Kimi-style linear attention
Mixtral-style MoE feed-forward
    Top-2 routing
    Configurable experts
    Capacity factor
    Load balancing loss
Configurable depth
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

config = {
    "block_size":128,
    "batch_size":24,
    "embed_dim":512,
    "num_heads":8,
    "num_layers":4,
    "ff_multiplier":4,

    # MoE
    "num_experts":4,
    "top_k":2,
    "capacity_factor":1.25,

    "lr":3e-4,
    "train_steps":2000
}

# ------------------------------------------------------------
# Download Shakespeare
# ------------------------------------------------------------

if not os.path.exists("shakespeare.txt"):

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    with open("shakespeare.txt","w",encoding="utf8") as f:
        f.write(text)

with open("shakespeare.txt","r",encoding="utf8") as f:
    text = f.read()

# ------------------------------------------------------------
# Train BPE tokenizer
# ------------------------------------------------------------

if not os.path.exists("tokenizer.json"):

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=8000)

    tokenizer.train(["shakespeare.txt"], trainer)

    tokenizer.save("tokenizer.json")

tokenizer = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenizer.get_vocab_size()

# ------------------------------------------------------------
# Encode dataset
# ------------------------------------------------------------

data = tokenizer.encode(text).ids
data = torch.tensor(data)

split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

# ------------------------------------------------------------
# Data loader
# ------------------------------------------------------------

def get_batch(split):

    data = train_data if split=="train" else val_data

    ix = torch.randint(len(data)-config["block_size"],(config["batch_size"],))

    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])

    return x.to(device),y.to(device)

# ------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self,d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self,x):

        norm = x.norm(dim=-1,keepdim=True)*(1.0/(x.size(-1)**0.5))

        return x/norm * self.scale

# ------------------------------------------------------------
# RoPE
# ------------------------------------------------------------

def apply_rope(x):

    B,H,T,D = x.shape

    half = D//2

    freqs = torch.arange(half,device=x.device)
    freqs = 1.0/(10000**(freqs/half))

    pos = torch.arange(T,device=x.device)

    freqs = torch.outer(pos,freqs)

    sin = freqs.sin()[None,None,:,:]
    cos = freqs.cos()[None,None,:,:]

    x1 = x[...,:half]
    x2 = x[...,half:]

    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos],dim=-1)

# ------------------------------------------------------------
# Kimi Linear Attention
# ------------------------------------------------------------

class KimiLinearAttention(nn.Module):

    def __init__(self,d_model,num_heads,chunk_size=64):

        super().__init__()

        self.h = num_heads
        self.dk = d_model//num_heads
        self.chunk = chunk_size

        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.g = nn.Linear(d_model,d_model)

        self.out = nn.Linear(d_model,d_model)

    def phi(self,x):
        return F.elu(x)+1

    def forward(self,x):

        B,T,C = x.shape

        q = rearrange(self.q(x),"b t (h d)-> b h t d",h=self.h)
        k = rearrange(self.k(x),"b t (h d)-> b h t d",h=self.h)
        v = rearrange(self.v(x),"b t (h d)-> b h t d",h=self.h)
        g = rearrange(torch.sigmoid(self.g(x)),"b t (h d)-> b h t d",h=self.h)

        q = apply_rope(q)
        k = apply_rope(k)

        q = self.phi(q)
        k = self.phi(k)

        S = torch.zeros(B,self.h,self.dk,self.dk,device=x.device)
        Z = torch.zeros(B,self.h,self.dk,device=x.device)

        outs=[]

        for start in range(0,T,self.chunk):

            end = min(start+self.chunk,T)

            qc = q[:,:,start:end]
            kc = k[:,:,start:end]
            vc = v[:,:,start:end]
            gc = g[:,:,start:end]

            kv = torch.einsum("bhtd,bhte->bhde",kc,vc)
            ks = kc.sum(dim=2)

            S = S + kv
            Z = Z + ks

            num = torch.einsum("bhtd,bhde->bhte",qc,S)
            den = torch.einsum("bhtd,bhd->bht",qc,Z).unsqueeze(-1)+1e-6

            out = gc * (num/den)

            outs.append(out)

        out = torch.cat(outs,dim=2)

        out = rearrange(out,"b h t d -> b t (h d)")

        return self.out(out)

# ------------------------------------------------------------
# Expert MLP
# ------------------------------------------------------------

class Expert(nn.Module):

    def __init__(self,d_model,mult):
        super().__init__()

        hidden = d_model*mult

        self.net = nn.Sequential(
            nn.Linear(d_model,hidden),
            nn.GELU(),
            nn.Linear(hidden,d_model)
        )

    def forward(self,x):
        return self.net(x)

# ------------------------------------------------------------
# Mixtral MoE
# ------------------------------------------------------------

class MixtralMoE(nn.Module):

    def __init__(self,d_model,num_experts,mult,top_k,capacity_factor):

        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(d_model,num_experts)

        self.experts = nn.ModuleList([
            Expert(d_model,mult) for _ in range(num_experts)
        ])

    def forward(self,x):

        B,T,C = x.shape
        N = B*T

        x_flat = x.view(N,C)

        logits = self.router(x_flat)
        probs = torch.softmax(logits,dim=-1)

        topk_probs,topk_idx = torch.topk(probs,self.top_k,dim=-1)

        topk_probs = topk_probs/topk_probs.sum(dim=-1,keepdim=True)

        capacity = int(self.capacity_factor*(N/self.num_experts))

        output = torch.zeros_like(x_flat)

        importance = probs.sum(0)
        load = torch.zeros(self.num_experts,device=x.device)

        for eid,expert in enumerate(self.experts):

            mask = topk_idx==eid
            tok,choice = torch.where(mask)

            if tok.numel()==0:
                continue

            if tok.numel()>capacity:
                tok = tok[:capacity]
                choice = choice[:capacity]

            inp = x_flat[tok]

            out = expert(inp)

            gates = topk_probs[tok,choice].unsqueeze(-1)

            output[tok] += gates*out

            load[eid] = tok.numel()

        importance = importance/importance.sum()
        load = load/load.sum()

        aux_loss = (importance*load).sum()*self.num_experts**2

        return output.view(B,T,C),aux_loss

# ------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        d = config["embed_dim"]

        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

        self.attn = KimiLinearAttention(
            d,
            config["num_heads"]
        )

        self.moe = MixtralMoE(
            d,
            config["num_experts"],
            config["ff_multiplier"],
            config["top_k"],
            config["capacity_factor"]
        )

    def forward(self,x):

        x = x + self.attn(self.norm1(x))

        moe_out,aux = self.moe(self.norm2(x))

        x = x + moe_out

        return x,aux

# ------------------------------------------------------------
# GPT Model
# ------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self):

        super().__init__()

        d = config["embed_dim"]

        self.tok = nn.Embedding(vocab_size,d)

        self.blocks = nn.ModuleList([
            TransformerBlock() for _ in range(config["num_layers"])
        ])

        self.norm = RMSNorm(d)

        self.head = nn.Linear(d,vocab_size)

    def forward(self,x,targets=None):

        B,T = x.shape

        x = self.tok(x)

        aux_total = 0

        for block in self.blocks:

            x,aux = block(x)
            aux_total += aux

        x = self.norm(x)

        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1,vocab_size),
            targets.view(-1)
        )

        loss = loss + 0.01*aux_total

        return loss,logits

# ------------------------------------------------------------
# Initialize Model
# ------------------------------------------------------------

model = GPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["lr"]
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

for step in range(config["train_steps"]):

    x,y = get_batch("train")

    loss,_ = model(x,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%100==0:
        print(step,loss.item())

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

torch.save(model.state_dict(),"moe_llm.pt")

# ------------------------------------------------------------
# Text Generation
# ------------------------------------------------------------

def generate(prompt,max_tokens=100):

    tokens = tokenizer.encode(prompt).ids
    tokens = torch.tensor(tokens,device=device)[None,:]

    for _ in range(max_tokens):

        x = tokens[:,-config["block_size"]:]

        logits = model(x)

        next = torch.argmax(logits[:,-1],dim=-1)

        tokens = torch.cat([tokens,next[:,None]],dim=1)

    return tokenizer.decode(tokens[0].tolist())

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

model.load_state_dict(torch.load("moe_llm.pt",map_location=device))

print(generate("To be or not to be"))