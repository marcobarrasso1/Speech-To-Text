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
    
enc_input = torch.randint(30 ,(config.batch_size, config.n_mels, config.n_audio_ctx), dtype=torch.float32)
dec_input = torch.randint(30, (config.batch_size, config.n_text_ctx), dtype=torch.long)

dec_input[:,-5:] = 50259

model = Transformer(config=config)

y = model(enc_input, dec_input)

print(y.shape)


