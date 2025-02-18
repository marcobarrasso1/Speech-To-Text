import torch 
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pickle
from config import config
from data_loader import create_data_loader
from tokenizer import custom_encoding, idx_2_str
from model import Transformer, build_model
import time 
import os


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
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

torch.set_float32_matmul_precision('high') 

ckpt = 0
path = 0 # 'weights/model_enhanced_16750.pth'

model = build_model(config, device, path, ddp, ddp_local_rank)
if device == torch.device('cuda'):
    model = torch.compile(model)

##logit_dist = model(torch.randn(1, config.n_mels, config.audio_len).to(device), torch.randint(0, 50259, (1, config.n_text_ctx)).to(device)).squeeze(0)

##for temp_pred in logit_dist:
##    print(max(temp_pred), min(temp_pred))

###logit_dist = logit_dist.mean(dim=0)
#### for temp_pred in logit_dist[0]:
####     print(max(temp_pred), min(temp_pred))
#### print(torch.argmax(logit_dist[0], dim=1))
##from matplotlib import pyplot as plt
##plt.hist(logit_dist[0].detach().cpu().numpy(), bins=10000)
##plt.show()

if master_process:
    print("Loaded model with a total of ", sum(p.numel() for p in model.parameters())/1e6, " M parameters")  


with open('data/audio_transcript_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)

data_loader_train, data_loader_val = create_data_loader(pairs, config, ddp, ddp_world_size, ddp_rank)

if master_process:
    print(f"Total number of batches in the train data: {len(data_loader_train)}\n", 
          f"Total number of batches in the test data: {len(data_loader_val)}")



decay_rate = 0.001  
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)


def lr_lambda(epoch):
    return max(0.01, torch.exp(torch.tensor(-decay_rate * epoch)))
scheduler = LambdaLR(optimizer, lr_lambda)



if master_process:
    from torch.utils.tensorboard import SummaryWriter
    logdir = './results/log_enhanced_data'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)


model.train()
global_step = ckpt

for epoch in range(config.epochs):
    train_losses = []
    # val_losses = []
    
    for i, batch in enumerate(data_loader_train):

        global_step += 1
        
        start_time = time.time()

        enc_input, dec_input, target = batch
        dec_input, target = dec_input.to(device), target.to(device)
        _target = target.clone().detach()

        #with torch.no_grad():
        #    if i % 50 == 0 and master_process:
        #        transcription, perplexity = model.GreedyDecoding(128, enc_input, device)
        #        print(transcription, perplexity)
        #        # print(enc.decode(list(transcription)))
            
        logits = model(enc_input, dec_input)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        
        loss = F.cross_entropy(logits, target, ignore_index=50259, reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            loss = loss.item()
        else:
            loss = loss.item()

        if device == torch.device('cuda'):
            torch.cuda.synchronize()
        end_time = time.time()
        batch_time = end_time - start_time    

        if master_process:
            audio_info = model.get_enc_out(enc_input)
            audio_info_var = torch.var(audio_info.view(audio_info.shape[0], -1), dim=0).mean().item()         
            print(f"Iteration {i} | Epoch {epoch} | Global Step {global_step} | Loss {round(loss, 6)} | Elapsed {round(batch_time, 5)}, | Lr {scheduler.get_last_lr()[0]} | Audio Info Var {round(audio_info_var, 5)}")
            writer.add_scalar("Loss / Train", loss, global_step)
            writer.add_scalar("Audio Info Variance", audio_info_var, global_step)
            if global_step % 50 == 1:
                transcription, perplexity = model.GreedyDecoding(config.n_text_ctx, enc_input, device)
                
                for i, (t, p) in enumerate(zip(idx_2_str(_target, config.enc, clean=True), transcription)):
                    if i > 5:
                        break
                    
                    print(model.get_enc_out(enc_input)[i])
                    print(f"Target: {t}\n\nPredicted: {p}\n")
                    print("Perplexity: ", perplexity[i], "\n\n")
        
        if global_step % 250 == 0 and master_process:
            model_path = f"weights/model_enhanced_{global_step}.pth"
            # torch.save(model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")
    
        scheduler.step()   


if ddp:
    dist.destroy_process_group()


