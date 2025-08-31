import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, 256)  # 256 embedding dimensions
        self.position_embedding_table = nn.Embedding(256, 256)      # 256 context length
        self.blocks = nn.Sequential(
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
            Block(256, n_head=8),  # 8 attention heads
        )
        self.ln_f = nn.LayerNorm(256)  # 256 embedding dimensions
        self.lm_head = nn.Linear(256, vocab_size)  # 256 embedding dimensions

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -256:]

            # forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last step logits

            # apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # top-k filtering
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                mask = logits < v[:, [-1]]
                logits[mask] = -float("Inf")

            # top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative_probs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = -float("Inf")
                logits.scatter_(1, sorted_indices, sorted_logits)

            # sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # append to sequence
            idx = torch.cat((idx, next_id), dim=1)

        return idx

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # n_head, head_size
        self.ffwd = FeedForward(n_embd)  # Changed from FeedFoward to FeedForward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(256, head_size, bias=False)  # n_embd = 256
        self.query = nn.Linear(256, head_size, bias=False)
        self.value = nn.Linear(256, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(256, 256)))  # block_size = 256
        self.dropout = nn.Dropout(0.1)  # dropout = 0.1

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(256, 256)  # n_embd = 256
        self.dropout = nn.Dropout(0.1)  # dropout = 0.1
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 256 -> 1024
            nn.GELU(),  # GELU activation as in training
            nn.Linear(4 * n_embd, n_embd),  # 1024 -> 256
            nn.Dropout(0.1)  # dropout = 0.1 (only at the end)
        )
        
    def forward(self, x): 
        return self.net(x)

class ShakespeareModel:
    def __init__(self, checkpoint_path: str, vocab_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load vocabulary from vocabulary file
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path, map_location='cpu')
            self.chars = vocab_data['chars']
            self.vocab_size = vocab_data['vocab_size']
            self.stoi = vocab_data['stoi']
            self.itos = vocab_data['itos']
            print(f"✅ Vocabulary loaded: {self.vocab_size} characters")
        else:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
        
        # Initialize model
        self.model = GPTLanguageModel(self.vocab_size).to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            try:
                # Try loading with different map_location strategies
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                print(f"✅ Model loaded successfully on {self.device}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print(f"🔍 Trying CPU fallback...")
                try:
                    # Fallback to CPU loading
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    print(f"✅ Model loaded successfully on CPU and moved to {self.device}")
                except Exception as e2:
                    print(f"❌ CPU fallback also failed: {e2}")
                    raise e2
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        return "".join([self.itos[i] if i in self.itos else "?" for i in ids])
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> str:
        """Generate text from a prompt."""
        # Encode prompt
        context = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        
        # Limit max_new_tokens for faster response
        max_new_tokens = min(max_new_tokens, 200)  # Cap at 200 tokens for performance
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )[0].tolist()
        
        # Decode and return only the new tokens
        full_text = self.decode(generated_ids)
        return full_text[len(prompt):]  # Return only the generated part
