from model import Transformer
import torch 
from config import config
import torchaudio
from utils.data_loader import create_data_loader
from utils.tokenizer import custom_encoding
import pickle
from collections import OrderedDict


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
state_dict = torch.load('weights/model_weights_iter_400.pth', map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("module._orig_mod."):
        name = k.replace("module._orig_mod.", "")
    elif k.startswith("module."):
        name = k.replace("module.", "")
    elif k.startswith("_orig_mod."):
        name = k.replace("_orig_mod.", "")
    else:
        name = k
    new_state_dict[name] = v
    
model.load_state_dict(new_state_dict, strict=False)

random_index = torch.randint(0, enc_input.size(0), (1,)).item()
enc_input = enc_input[random_index].unsqueeze(0)
dec_input = dec_input[random_index]
print(enc_input.shape)

out = model.GreedyDecoding(10, enc_input, device)

print(enc.decode(list(dec_input)))
print(enc.decode(list(out)))
assert False
beam = model.beam_search(5, 5, enc_input, device)

# print("\n", enc.decode(dec_input)) non funziona non so
print(enc.decode(list(dec_input)))
for seq, score in beam:
    print(enc.decode(list(seq)), score)


