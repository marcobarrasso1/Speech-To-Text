from model import Transformer, build_model
import torch 
from config import config
import pickle
import jiwer
import string
from tokenizer import custom_encoding, idx_2_str
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
    device = torch.device('cpu')
    print('Using device CPU')
device = torch.device('cpu')

with open('data/audio_transcript_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)
    
data_loader_test = create_data_loader(pairs, config, test=True)
print(len(data_loader_test))

torch.set_float32_matmul_precision('high') 
model = build_model(config, device, 'weights/model_enhanced_6750.pth')
model.to(device)

if device == torch.device('cuda'):
    model = torch.compile(model)


WER = []
perplexities = []
for i, batch in enumerate(data_loader_test):

    enc_input, dec_input, target = batch
    dec_input, target = dec_input.to(device), target.to(device)
    
    prediction, perplexity = model.GreedyDecoding(config.n_text_ctx, enc_input, device)

    print(f"Target Sentence: {config.enc.decode(list(target[0]))}\n")
    print(f"Predicted Sentence: {config.enc.decode(prediction)}")
    # print(f"Perplexity: {perplexity}, WER: {wer(config.enc.decode(list(target[0])), config.enc.decode(prediction))}\n\n")

    perplexities.append(perplexity)
    #WER.append(wer(target, prediction))

print("Avarage WER: ", sum(WER) / len(WER))
print("Avarage Perplexity: ", sum(perplexities) / len(perplexities))
    