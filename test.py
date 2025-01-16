from model import Transformer
import torch 
from config import config
import torchaudio
from utils.data_loader import create_data_loader
from utils.tokenizer import custom_encoding
import pickle


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

print(len(pairs))
data_loader_train, data_loader_val = create_data_loader(pairs, config, enc)

print(len(data_loader_train), len(data_loader_val))

enc_input, dec_input, target = next(iter(data_loader_val))
enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
print(enc_input.shape, dec_input.shape, target.shape)

model = Transformer(config=config).to(device)
beam = model.beam_search(5, 5, enc_input, device)


# print("\n", enc.decode(dec_input)) non funziona non so
for seq, score in beam:
    print(enc.decode(list(seq)), score)


