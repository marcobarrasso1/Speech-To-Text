from model import Transformer, build_model
import torch 
from config import config
from data_loader import create_data_loader
import pickle
from collections import OrderedDict
import jiwer
import string

def wer(reference, hypothesis):
    reference = reference.translate(str.maketrans("", "", string.punctuation)).lower()
    hypothesis = hypothesis.translate(str.maketrans("", "", string.punctuation)).lower()
    return jiwer.wer(reference, hypothesis)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device MPS')
else:
    print('Using device CPU')

with open('data/audio_transcript_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)
    
device = torch.device('cpu')
print(len(pairs))
data_loader_train, data_loader_val = create_data_loader(pairs, config)

print(len(data_loader_train), len(data_loader_val))
enc_input, dec_input, target = next(iter(data_loader_val))
dec_input, target = dec_input.to(device), target.to(device)
print(dec_input.shape, target.shape)


model = build_model(config, device)
out = model(enc_input, dec_input)
print(out.shape)





