from model import Transformer
import torch 
from config import config
import pickle
import jiwer
import string
from tokenizer import custom_encoding
from data_loader import create_data_loader

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
    

with open('data/test_audio_transcript_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)
    
data_loader_test = create_data_loader(pairs, config, test=True)
print(len(data_loader_test))

torch.set_float32_matmul_precision('high') 
model = Transformer(config)
model.to(device)
if device == torch.device('cuda'):
    model = torch.compile(model)

try:
    state_dict = torch.load('weights/model_new_weights_iter_43750.pth', map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=True)
    print("Model weights loaded succesfully")
except Exception as e:
    print(f"Unable to load model weights: {e}")
    assert False


WER = []
perplexities = []
for i, batch in enumerate(data_loader_test):
    enc_input, dec_input, target = batch
    enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
    
    prediction, perplexity = model.GreedyDecoding(128, enc_input, device)
    perplexities.append(perplexity)
    WER.append(wer(target, prediction))

print("Avarage WER: ", sum(WER) / len(WER))
print("Avarage Perplexity: ", sum(perplexities) / len(perplexities))
    