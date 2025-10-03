
# |-------------------------------------------------------------------------------------|
# |                             LLM Chatbot Model                                       |
# |                                                                                     |                                           
# |     Trained with data created automatically using locally run LLAMA-3-8B model      |
# |-------------------------------------------------------------------------------------|

# imports

import torch
import torch.nn as nn
from torch.nn import functional as F

import time
import datetime


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("Warning! GPU Acceleration is not available. Running on CPU. Model will be very slow.")
    print("If you have a cuda-compatible GPU, please install a cuda version of pytorch.")
    
DATA_TXT = "data.txt"
print(f"Using data file: {DATA_TXT}")

try:
    with open(DATA_TXT, "r", encoding="utf-8") as f:
	    text = f.read()
except UnicodeDecodeError:
    print("Error reading file with utf-8 encoding. Trying cp1252 encoding.")
    with open(DATA_TXT, "r", encoding="cp1252") as f:
        text = f.read()
 
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("-----------------------------")
print(f"Lines: {len(text.splitlines())}")
print(f"Characters: {len(text)}")
print(f"Unique Characters: {len(chars)}")
print("-----------------------------")

# hyperparameters                                  # important ones:
batch_size = 64                                         # parallel batch size
block_size = 256                                        # context window size
max_iters = 2000                                        # number of training iterations
eval_interval = 200
learning_rate = 1e-3                                    # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 256                                            # embedding dimension
n_head = 8                                              # number of attention heads
n_layer = 13                                            # number of transformer layers
dropout = 0.0   
# ------------


# other parameters
train_split = 0.95                                 # proportion of data to use for training vs validation
trained_model_path = "LLAMA_chatbot_Tensor"        # path to save the trained model
save_interval = 500                                # save model every X iterations
# ------------

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    res = []
    for c in s:
        if c in stoi:
            res.append(stoi[c])
        else:
            res.append(0)
    return res
            
def decode(l):
    res = []
    for i in l:
        if i in itos:
            res.append(itos[i])
    return ''.join(res)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_split*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    
model = BigramLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
trained_model_path += "_" + str(int(sum(p.numel() for p in m.parameters())/1e6)) + "M"
print(f"Will save model to {trained_model_path} once training is complete.")
print("--------------------------------------------------------------")


def parse_time(t):
    return str(datetime.timedelta(seconds=int(t)))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

allstart = time.time()
start = time.time()
for iter in range(max_iters):
    if iter == 1:
        print("\n--------------------------------------\n")
        print("Estimated time to complete:", parse_time((time.time() - estimatestart) * max_iters))
        print("\n--------------------------------------\n")
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        if iter == 0:
            print(f"INITIAL EVALUATION: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, time taken: {parse_time(time.time() - start)}")
        else:
            print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, time taken: {parse_time(time.time() - start)}")
        start = time.time()
    if iter % save_interval == 0 and iter > 0:
        torch.save(m.state_dict(), trained_model_path + "_iter_" + str(iter)) # save the model in case loss goes up again
        print(f"Saved model at iteration {iter} at {trained_model_path + '_iter_' + str(iter)}")

    if iter == 0:
        estimatestart = time.time()
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Training took {parse_time(time.time() - allstart)}")

torch.save(m.state_dict(), trained_model_path)
print(f"Saved final model at {trained_model_path}")

def generate_text(model, context=None, max_tokens=100, device=device):
    if context is None:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    elif type(context) == str:
        context = torch.tensor([encode(context)], dtype=torch.long, device=device)
    elif type(context) == list:
        context = torch.tensor(context, dtype=torch.long, device=device)
        
    return decode(model.generate(context, max_tokens)[0].tolist())

generation = generate_text(m, max_tokens=2000)
print(generation)

print("---------------------------------------------------------\n---------------------------------------------------------\n")
print("Training complete. Model saved to", trained_model_path)
print("Press enter to quit")
print("\n---------------------------------------------------------\n---------------------------------------------------------")

input()