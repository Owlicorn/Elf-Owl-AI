import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

class ElfOwlModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        
        # Store config values as instance attributes
        self.d_model = config.Config.D_MODEL
        self.n_layers = config.Config.N_LAYERS
        self.n_heads = config.Config.N_HEADS
        self.d_ff = config.Config.D_FF
        self.dropout = config.Config.DROPOUT
        self.max_sequence_length = config.Config.MAX_SEQUENCE_LENGTH
        
        print(f"🧠 Building model: d_model={self.d_model}, layers={self.n_layers}, heads={self.n_heads}")
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_sequence_length, self.d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, targets=None):
        b, t = x.size()
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:, :t, :]
        x = token_emb + pos_emb
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9, eos_token_id=None, repetition_penalty=1.2):
        """Generate text with sampling"""
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        next_token_logits[0, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if sequence gets too long
                if generated.size(1) >= self.max_sequence_length:
                    break
        
        return generated

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(2048, 2048)))
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        
        return y

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)