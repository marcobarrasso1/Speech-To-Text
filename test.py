from model import Transformer
import torch 
from config import config
import torchaudio

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device MPS')
else:
    print('Using device CPU')
    
enc_input = torch.randint(30 ,(config.batch_size, config.n_mels, config.n_audio_ctx), dtype=torch.float32).to(device)
# dec_input = torch.randint(30, (config.batch_size, config.n_text_ctx), dtype=torch.long).to(device)

model = Transformer(config=config).to(device)
beam = model.beam_search(5, 5, enc_input, device)
print(beam)


