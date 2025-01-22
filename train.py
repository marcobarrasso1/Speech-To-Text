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
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from config import config


ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    # with multiple gpus
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_local_rank == 0
    if master_process:
        print(f"Number of threads: {ddp_world_size}")

else:
    # with a single gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device CUDA')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device MPS')
    else:
        print('Using device CPU')
        device = torch.device('cpu')

    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    
enc = custom_encoding()

with open('data/audio_transcript_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)


data_loader_train, data_loader_val = create_data_loader(pairs, config, enc, ddp, ddp_world_size, ddp_rank)

if master_process:
    print(f"Total number of batches in the train data: {len(data_loader_train)}\n", 
          f"Total number of batches in the test data: {len(data_loader_val)}")


torch.set_float32_matmul_precision('high') 
ckpt = 8500

model = Transformer(config)
model.to(device)
model = torch.compile(model)

load_ckpt = True
if load_ckpt:
    try:
        state_dict = torch.load(f"weights/model_big_weights_iter_{ckpt}.pth", map_location=torch.device(device), weights_only=True)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        print("Model weights loaded succesfully")
    except Exception as e:
        print(f"Unable to load model weights: {e}")

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])


#state_dict = torch.load('weights/model_weights_iter_100.pth', map_location=device)
#model.load_state_dict(state_dict)
#print("model paramaters loaded correctly")

if master_process:
    print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")  


decay_rate = 0.1  
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)

'''
def lr_lambda(epoch):
    return max(1e-5, torch.exp(torch.tensor(-decay_rate * epoch)))

scheduler = LambdaLR(optimizer, lr_lambda)
'''

n_train = len(data_loader_train) * (ddp_world_size if ddp else 1)
n_val = len(data_loader_val) 

if master_process:
    from torch.utils.tensorboard import SummaryWriter
    logdir = './results/log'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)



model.train()
global_step = ckpt
for epoch in range(config.epochs):
    train_losses = []
    # val_losses = []
    
    
    #print(f"Learning Rate {scheduler.get_last_lr()[0]:.6f}")
    for i, batch in enumerate(data_loader_train):

        global_step += 1
        
        ##if i % 250 == 0:
        ##    model.eval()
        ##    val_loss = 0
            
        ##    with torch.no_grad():
        ##        for val_batch in data_loader_val:
        ##            enc_input, dec_input, target = val_batch
        ##            enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
                    
        ##            logits = model(enc_input, dec_input)
        ##            B, T, C = logits.shape
                    
        ##            logits = logits.view(B * T, C)
        ##            target = target.view(B * T)

        ##            loss = F.cross_entropy(logits, target, ignore_index=50259)
        ##            val_loss += loss.item()

        ##    if ddp:
        ##        val_loss_tensor = torch.tensor(val_loss, device=device)
        ##        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM) # summing losses calculated by the different processes
        ##        val_loss = val_loss_tensor.item()

        ##    if master_process:
        ##        avg_loss = val_loss / n_val

                
            
        ##    val_loss = val_loss / len(data_loader_val)
        ##    if master_process:
        ##        writer.add_scalar('Loss/Val', val_loss, global_step)

        ##    model.train()
        
        start_time = time.time()

        enc_input, dec_input, target = batch
        enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
    
        #with torch.no_grad():
        #    if i % 50 == 0 and master_process:
        #        transcription, perplexity = model.GreedyDecoding(128, enc_input, device)
        #        print(transcription, perplexity)
        #        # print(enc.decode(list(transcription)))
            
        logits = model(enc_input, dec_input)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        
        loss = F.cross_entropy(logits, target, ignore_index=50259)
        if master_process:
            writer.add_scalar("Loss/Train", loss, global_step)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if ddp:
            train_loss_tensor = torch.tensor(loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()

        torch.cuda.synchronize()
        end_time = time.time()
        batch_time = end_time - start_time    

        if master_process:
            print(f"Iteration {i} | Epoch {epoch} | Global Step {global_step} | Loss {round(train_loss, 6)} | Elapsed {round(batch_time, 5)}")
            writer.add_scalar("Loss / Train", train_loss, global_step)
        
        if global_step % 250 == 0 and master_process:
            model_path = f"weights/model_big_weights_iter_{global_step}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")
    
    #scheduler.step()   


if ddp:
    dist.destroy_process_group()


