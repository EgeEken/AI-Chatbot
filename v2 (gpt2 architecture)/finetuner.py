# =============== EASY PARAMETERS ===================
training_speed = "fast"         # fast, medium, slow
training_data = "input.txt"
# ===================================================






if training_speed == "fast":
    iteration_count = 20
    learning_rate = 1e-3

elif training_speed == "medium":
    iteration_count = 50
    learning_rate = 3e-4

elif training_speed == "slow":
    iteration_count = 200
    learning_rate = 1e-4

# =============== ADVANCED PARAMETERS ==============
# iteration_count = 100
# learning_rate = 3e-4          # 3e-4 was Karpathy's default
# ==================================================


from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math
import matplotlib.pyplot as plt
import time

import tiktoken

cuda_available = torch.cuda.is_available()
print("CUDA available! :)" if cuda_available else "CUDA not available! :(")
if cuda_available:
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

enc = tiktoken.get_encoding("gpt2")

@dataclass
class GPTConfig:
    block_size: int = 1024      # context length
    vocab_size: int = 50257     # number of tokens in the vocabulary, 
                                # 50k BPE, 256 byte (char), 1 for <EOT>
    n_layer: int = 12           # number of layers
    n_head: int = 12            # number of heads
    n_embd: int = 768           # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # ensure embedding dimension is divisible by number of heads 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3* for key, query, value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # KARPATHY ADDITION "Idk if this is pytorch sanctioned but it works for me" -Karpathy
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Karpathy Comment: not really a 'bias', more of a mask, but following the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads"
        # hs is "head size"
        # C (number of channels) = nh * hs
        # Example: In GPT2(124M), n_head = 12, hs = 64 so C = 768
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # split into query, key, value along channel dim
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)         # 4* factor is used to increase the capacity of
                                                                        # the MLP without increasing the embedding size
        
        self.gelu = nn.GELU()                                           # (approximate="tanh"), historically used in GPT-2 due to
                                                                        # performance reasons at the time, no longer necessary
        
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)       # project back to original embedding size
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class Block(nn.Module):                             # an individual Transformer block (made of add & norm, followed by feed-forward)
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)     # layer norm 1 (normalizes across the feature dimension)
        self.attn = CausalSelfAttention(config)     # multi headed attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd)     # layer norm 2 (normalizes across the feature dimension)
        self.mlp = MLP(config)                      # feed-forward multi-layer perceptron (MLP) layer

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))             # residual connection around the self-attention layer
        x = x + self.mlp(self.ln_2(x))              # residual connection around the feed-forward layer
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(                                           # transformer architecture
            dict(                                      
                wte = nn.Embedding(config.vocab_size, config.n_embd),                   # word token embeddings
                wpe = nn.Embedding(config.block_size, config.n_embd),                   # word position embeddings
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),      # hidden layers
                ln_f = nn.LayerNorm(config.n_embd),                                     # final layer norm (layernorm_final)
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)      # language model head, this is the final layer
                                                                                    # that outputs logits (probabilities) for each
                                                                                    # token in the vocabulary

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std = (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size() # batch dimension, time dimension
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """CONSTRUCTOR
        Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

class DataLoaderLite:
    def __init__(self, B, T, text):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y                      


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text




torch.set_float32_matmul_precision("high")

#model = GPT(GPTConfig())
model = GPT.from_pretrained('gpt2')
model.to(device)
#model = torch.compile(model) # does not work on Windows due to lack of triton support

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoaderLite(B=4, T=1024, text=read_data(training_data))


start = time.time()
x, y = train_loader.next_batch()
x, y = x.to(device), y.to(device)
logits, loss = model(x, y)
print(f"Initial loss: {loss.item():.4f}")
losses = [loss.item()]
tpses = []

for i in range(1, iteration_count+1):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    tpses.append(tokens_per_sec)
    if (iteration_count <= 10) or (i % 10 == 0):
        print(f"Iteration {i}/{iteration_count}\t| Loss: {loss.item():.2f} \t| Average Tok/Sec: {np.mean(tpses):.2f} \t| Time: {time.time() - start:.2f} seconds")
        # save a checkpoint model
        torch.save(model.state_dict(), f"gpt2_pretrained_{training_data}_model_checkpoint_iter{i}.pth")

print(f"Iteration {iteration_count}\t| Loss: {loss.item():.2f} \t| Average Tok/Sec: {np.mean(tpses):.2f} \t| Time: {time.time() - start:.2f} seconds")
# final model save
torch.save(model.state_dict(), f"gpt2_pretrained_{training_data}_model.pth")

plt.plot(losses)
plt.title("Training Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()   