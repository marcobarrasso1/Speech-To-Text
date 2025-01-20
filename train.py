import torch 
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pickle
from utils.data_loader import create_data_loader
from config import config
from utils.tokenizer import custom_encoding
from model import Transformer
import torch.nn.functional as F
import time 


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

print(pairs[200])

print(len(pairs))
data_loader_train, data_loader_val = create_data_loader(pairs, config, enc)

print(len(data_loader_train), len(data_loader_val))

torch.set_float32_matmul_precision('high') 
model = Transformer(config)
model.to(device)
model = torch.compile(model)


state_dict = torch.load('weights/model_weights_iter_100.pth', map_location=device)
model.load_state_dict(state_dict)
print("model paramaters loaded correctly")


print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")  


decay_rate = 0.1  

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)

'''
def lr_lambda(epoch):
    return max(1e-5, torch.exp(torch.tensor(-decay_rate * epoch)))

scheduler = LambdaLR(optimizer, lr_lambda)
'''

'''

enc_input, dec_input, target = next(iter(data_loader_train))
enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)

print(dec_input)
print(target)

model.train()


for epoch in range(config.epochs):
    logits = model(enc_input, dec_input)
    B, T, C = logits.shape
    
    logits = logits.view(B * T, C)
    target = target.view(B * T)

    loss = F.cross_entropy(logits, target, ignore_index=50259)
    print(f" epoch {epoch} | batch: {epoch+1} | loss: {loss}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        transcription = model.GreedyDecoding(128, enc_input, device)
        print(dec_input)
        print(transcription)
        print(enc.decode(dec_input.tolist()[0]))
        print(enc.decode(transcription))
        
    #scheduler.step()  
    
'''


model.train()

for epoch in range(config.epochs):
    train_losses = []
    val_losses = []
    train_loss = 0
    
    #print(f"Learning Rate {scheduler.get_last_lr()[0]:.6f}")
    for i, batch in enumerate(data_loader_train):
        
        if i % 50 == 0:
            model.eval()
            val_loss = 0
            
            for val_batch in data_loader_val:
                enc_input, dec_input, target = val_batch
                enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
                
                logits = model(enc_input, dec_input)
                B, T, C = logits.shape
                
                logits = logits.view(B * T, C)
                target = target.view(B * T)

                loss = F.cross_entropy(logits, target, ignore_index=50259)
                val_loss += loss.item()
            
            val_loss = val_loss / len(data_loader_val)
            val_losses.append(val_loss)
            #print(f"batch {i+1} | Validation loss {val_loss}")
            
            with open('val_loss.txt', 'a') as f:
                f.write(f"step {i} | loss {val_loss}\n")
            model.train()
        
        start_time = time.time()

        enc_input, dec_input, target = batch
        enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
    
        if i % 100 == 0:
            transcription, perplexity = model.GreedyDecoding(128, enc_input, device)
            print(transcription, perplexity)
            #print(enc.decode(list(transcription)))
            
        logits = model(enc_input, dec_input)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        
        loss = F.cross_entropy(logits, target, ignore_index=50259)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        batch_time = end_time - start_time

        #print(f" epoch{epoch} | batch: {i+1} | loss: {loss} | time: {batch_time:.4f} seconds")
        
        with open('train_loss.txt', 'a') as f:
            f.write(f"epoch {epoch} | step {i} | loss {loss} | time: {batch_time:.4f} seconds\n")
        
        if i % 100 == 0:
            model_path = f"weights/model_weights_iter_{i}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")
    
    #scheduler.step()   




