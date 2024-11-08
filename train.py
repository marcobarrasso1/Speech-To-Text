import torch 
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pickle
from utils.data_loader import create_data_loader
from config import config
from utils.tokenizer import custom_encoding
from model import Transformer
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device MPS')
else:
    print('Using device CPU')
    

enc = custom_encoding()

with open('data/audio_transcript_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)
    
data_loader_train, data_loader_val = create_data_loader(pairs, config, enc)

print(len(data_loader_train))
#print(next(iter(data_loader_train)))
#torch.set_float32_matmul_precision('high') 
len(data_loader_train)
model = Transformer(config)
model.to(device)
#model = torch.compile(model)

print(sum(p.numel() for p in model.parameters())/1e6, "M parameters") 

total_steps = config.epochs * len(data_loader_train)
warmup_steps = 2048 

optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        # Linear warmup phase
        return current_step / warmup_steps
    else:
        # Linear decay to zero after warmup
        return max(0.0, (total_steps - current_step) / (total_steps - warmup_steps))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

model.train()
for epoch in range(config.epochs):
    epoch_loss = []
    for i, batch in enumerate(data_loader_train):
        
        enc_input, dec_input, target = batch
        enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
        
        logits = model(enc_input, dec_input)
        B, T, C = logits.shape
        
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        
        loss = F.cross_entropy(logits, target, ignore_index=50259)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss.append(loss)
        
        print(f"epoch: {epoch+1}, batch: {i+1}, loss: {loss}")
        
        
        
   





