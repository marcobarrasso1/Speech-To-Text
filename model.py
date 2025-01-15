import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class MLP(nn.Module):
    
    def __init__ (self, n_embd):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.GELU(),
            nn.Linear(n_embd*4, n_embd),
        )
        
    def forward(self, x):
        return self.fc(x)
    

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.head_size = n_embd // n_head
        self.n_head = n_head
        
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out = nn.Linear(n_embd, n_embd)
        
    def forward(self, k, q, v, cross=False):
        B, T, C = k.shape
        
        if cross:
            Tk = T
            Tq = q.shape[1]
        else:
            Tk = Tq = T
            
        k = self.key(k)
        q = self.query(q)
        v = self.value(v)
        
        k = k.view(B, Tk, self.n_head, self.head_size).transpose(1, 2) # (B, n_heads, key_size, head_size)
        q = q.view(B, Tq, self.n_head, self.head_size).transpose(1, 2) # (B, n_heads, query_size, head_size)
        v = v.view(B, Tk, self.n_head, self.head_size).transpose(1, 2) # (B, n_heads, value_size, head_size)
        
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True if cross else False) #(B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C) # (B, T, n_embd)
        
        y = self.out(y)

        return y
        

class AttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head, cross=False):
        super().__init__()
        
        self.cross = cross
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        
        if self.cross:
            self.cross_att = MultiHeadAttention(n_embd, n_head) 
            self.ln2 = nn.LayerNorm(n_embd) 
            
        self.mlp = MLP(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        
    def forward(self, k, q, v):
        x = self.ln1(q)
        x = x + self.attn(q=x, k=x, v=x, cross=self.cross)
        
        if self.cross:
            x = x + self.cross_att(
            q=self.ln2(x), 
            k=self.ln2(k), 
            v=self.ln2(v), 
            cross=True
            )
            
        x = x + self.mlp(self.ln3(x))   
        
        return x
    

class Encoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_layer, n_embd, n_head):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, n_embd, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1, bias=False)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_embd))
        
        self.blocks = nn.ModuleList(AttentionBlock(n_embd, n_head, cross=False) for _ in range (n_layer))
        
        self.ln = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        #x shape: (batch_size, n_mels, n_ctx)
        x = F.gelu(self.conv1(x)) #(batch_size, n_embd, n_ctx)
        x = F.gelu(self.conv2(x)) #(batch_size, n_embd, n_ctx)
        x = x.permute(0, 2, 1) #(batch_size, n_ctx, n_embd)
        
        x = (x + self.positional_embedding).to(x.dtype)
        
        for block in self.blocks:
            x = block(k=x, q=x, v=x)
            
        x = self.ln(x)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_embd, n_head, n_layer):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.postitional_embedding = nn.Parameter(torch.zeros(n_ctx, n_embd))
        
        self.blocks = nn.ModuleList(AttentionBlock(n_embd, n_head, cross=True) for _ in range (n_layer))
        
        self.ln = nn.LayerNorm(n_embd)
        
    def forward(self, x, enc_out):
        n_ctx = x.shape[1]
        
        x = self.token_embedding(x) + self.postitional_embedding[:n_ctx]
        
        for block in self.blocks:
            x = block(q=x, k=enc_out, v=enc_out)
            
        x = self.ln(x)
        
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        
        return logits
    

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = Encoder(
            n_mels=config.n_mels,
            n_ctx=config.n_audio_ctx,
            n_layer=config.n_audio_layer,
            n_embd=config.n_audio_embd,
            n_head=config.n_audio_head,
        )
        
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            n_ctx=config.n_text_ctx,
            n_layer=config.n_text_layer,
            n_embd=config.n_text_embd,
            n_head=config.n_text_head,
        )

        #self.register_buffer("bias", (torch.tril(torch.ones(config.n_text_ctx, config.n_text_ctx)).view(1, 1, config.n_text_ctx, config.n_text_ctx).bool()))
        
        
    def forward(self, enc_input, dec_input):
        enc_out = self.encoder(enc_input)
        
        return self.decoder(x=dec_input, enc_out=enc_out)
        
        
        
        
        
        
        
            
        
        
        
        
        
    
    
        
        
        
        
        
                
        
        
    




