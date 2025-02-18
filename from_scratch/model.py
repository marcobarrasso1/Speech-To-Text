import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tokenizer import idx_2_str


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
        
    def forward(self, k, q, v, cross, mask=False, pad_mask=None):
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
        
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            pad_mask = pad_mask.expand(B, self.n_head, Tq, Tk)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=mask, attn_mask=pad_mask) #(B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C) # (B, query_size, n_embd)
        
        y = self.out(y)

        return y
        

## class AttentionBlock(nn.Module):
##     def __init__(self, n_embd, n_head, cross=False):
##         super().__init__()
        
##         self.cross = cross
##         self.attn = MultiHeadAttention(n_embd, n_head)
##         self.ln1 = nn.LayerNorm(n_embd)
        
##         if self.cross:
##             self.cross_att = MultiHeadAttention(n_embd, n_head) 
##             self.ln2 = nn.LayerNorm(n_embd) 
            
##         self.mlp = MLP(n_embd)
##         self.ln3 = nn.LayerNorm(n_embd)
        
##     def forward(self, k, q, v, mask=False):

##         q_ = self.ln1(q)
##         x = q + self.attn(q=q_, k=q_, v=q_, cross=self.cross, mask=mask)
        
##         if self.cross:
##             x = x + self.cross_att(
##             q=self.ln2(x), 
##             k=k, 
##             v=v, 
##             cross=True,
##             mask=False
##             )
            
##         x = x + self.mlp(self.ln3(x))   
        
##         return x
    


class EncoderAttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        
    def forward(self, k, q, v, pad_mask=False):

        q_ = self.ln1(q)
        x = q + self.attn(q=q_, k=q_, v=q_, cross=False, mask=False, pad_mask=pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    


class DecoderAttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.cross_att = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, k, q, v, mask=True):

        q_ = self.ln1(q)
        x = q + self.attn(q=q_, k=q_, v=q_, cross=False, mask=mask)
        x = x + self.cross_att(
            q=self.ln2(x), 
            k=k, 
            v=v, 
            cross=True,
            mask=False
        )
        x = x + self.mlp(self.ln3(x))
        return x

    

    
class Encoder(nn.Module):
    def __init__(self, n_mels, hidden, n_layer, n_embd, n_head, device):
        super().__init__()

        self.device=device
        self.hidden = hidden
        self.register_buffer("positional_embedding", sinusoids(hidden, n_embd))
        
        self.conv1 = nn.Conv1d(n_mels, n_embd, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1, stride=2)
        # self.adapool = nn.AdaptiveAvgPool1d(audio_len)

        self.blocks = nn.ModuleList(EncoderAttentionBlock(n_embd, n_head) for _ in range (n_layer))
        self.ln = nn.LayerNorm(n_embd)
        
    def forward(self, x):

        conv_out = []
        masks = []
        for spectre in x:
            x = F.gelu(self.conv1(spectre.to(self.device)))
            x = self.conv2(x)

            length = x.shape[-1]
            assert length <= self.hidden
            pad_length = self.hidden - length

            mask = torch.ones(length, dtype=torch.bool, device=self.device)
            if pad_length > 0:
                x = F.pad(x, (0, pad_length), value=0)
                mask = F.pad(mask, (0, pad_length), value=0)

            conv_out.append(x)
            masks.append(mask)
            
        x = torch.stack(conv_out, dim=0)
        x = x.transpose(1, 2)

        mask = torch.stack(masks, dim=0)
        
        B, N, D = x.shape
        
        x = (x + self.positional_embedding).to(x.dtype)
        
        for block in self.blocks:
            x = block(k=x, q=x, v=x, pad_mask=mask)
            
        x = self.ln(x)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_embd, n_head, n_layer):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.postitional_embedding = nn.Parameter(torch.zeros(n_ctx, n_embd))
        
        self.blocks = nn.ModuleList(DecoderAttentionBlock(n_embd, n_head) for _ in range (n_layer))
        self.ln = nn.LayerNorm(n_embd)
        
    def forward(self, x, enc_out, inference=False):
        n_ctx = x.shape[1]
        x = self.token_embedding(x) + self.postitional_embedding[:n_ctx]
        
        for block in self.blocks:
            x = block(q=x, k=enc_out, v=enc_out)
            
        x = self.ln(x)
        
        if inference:
            logits = (
                x[:, -1, :] @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
            ).float()
            
        else:
            logits = (
                x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
            ).float()
        
        return logits
    

class Transformer(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.enc = config.enc
        
        self.encoder = Encoder(
            n_mels=config.n_mels,
            n_layer=config.n_audio_layer,
            n_embd=config.n_audio_embd,
            n_head=config.n_audio_head,
            hidden=config.encoder_hidden,
            device=device
        )
        
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            n_ctx=config.n_text_ctx,
            n_layer=config.n_text_layer,
            n_embd=config.n_text_embd,
            n_head=config.n_text_head
        )
        
        self.initialize_weights(self.encoder)
        self.initialize_weights(self.decoder)
        
        
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):  # Apply only to linear layers
            nn.init.xavier_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # Fan-in with ReLU
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, enc_input, dec_input):
        enc_out = self.encoder(enc_input)
        return self.decoder(x=dec_input, enc_out=enc_out)
    
    def get_logits(self, dec_input, enc_out):
        return self.decoder(x=dec_input, enc_out=enc_out, inference=True)
    
    def get_enc_out(self, enc_input):
        return self.encoder(x=enc_input)


    @torch.no_grad()
    def beam_search(self, max_length, k, enc_in, device):
        self.eval()

        enc_out = self.encoder(enc_in).repeat(k, 1, 1)
        print(enc_out.shape)

        eos = 50258
        sos = 50257
        pad = 50259

        beam = [(torch.tensor([sos], device=device), 0) for _ in range(k)]
        for seq, score in beam:
            print(seq, score)

        for _ in range(max_length):

            all_seqs = [seq for seq, _ in beam]

            batch_seqs = torch.nn.utils.rnn.pad_sequence(
                all_seqs, batch_first=True, padding_value=pad
            ).to(device)

            print(batch_seqs.shape)

            logits = self.get_logits(batch_seqs, enc_out)  # Shape: (num_beams, vocab_size)
            log_p = F.log_softmax(logits, dim=-1)
            print(logits)
            print(log_p)

            candidates = []
            for i, (seq, score) in enumerate(beam):
                seq_logp = log_p[i]
                topk_probs, topk_idx = torch.topk(seq_logp, k, dim=-1)

                for j in range(k):
                    ix = topk_idx[j].unsqueeze(0)
                    logp = topk_probs[j].item()
                    n = seq.size(0)
                    new_score = (score * (n - 1) / n) + (logp / n)
                    candidates.append((torch.cat((seq, ix)), new_score))

            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
            beam = candidates

            for seq, score in beam:
                print(seq, score)

            if all(seq[-1] == eos for seq, _ in beam):
                break

        return beam
    

    @torch.no_grad() 
    def GreedyDecoding(self, max_length, enc_in, temperature=0, clean=False): #enc in must be list of spectrograms

        self.eval()
        
        eos = 50258
        sos = 50257
        
        enc_out = self.encoder(enc_in)
        B, T, C = enc_out.shape

        transcription = torch.tensor([sos for _ in range(B)], dtype=torch.long, device=self.device).reshape(B, 1)
        prods = torch.ones(B, device=self.device)
        
        for i in range(max_length):

            logits = self.get_logits(transcription, enc_out)
            probs = F.softmax(logits, dim=-1)
            p, prediction = torch.max(probs, dim=-1)
            prediction = prediction.reshape(B, 1)
            if i > 0:
                for j in range(B):
                    if last_prediction[j] == eos:
                        prediction[j] = eos
            last_prediction = prediction
            prods *= p 

            transcription = torch.cat((transcription, prediction), dim=1)

            if all(prediction[:, -1] == eos):
                break
        
        return idx_2_str(transcription, self.enc, clean), (1 / prods) ** (1 / transcription.shape[1] - 1)
    


def build_model(config, device, ckpt=0, ddp=0, ddp_local_rank=0):
    model = Transformer(config, device)
    model.to(device)

    if ckpt != 0:
        try:
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            if device == torch.device('cuda') or device == torch.device('cpu'):
                model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}, strict=True)
            else:
                model.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in state_dict.items()}, strict=True)
            print("Model weights loaded succesfully")

        except Exception as e:
            print(f"Unable to load model weights: {e}")
            assert False

        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

    return model