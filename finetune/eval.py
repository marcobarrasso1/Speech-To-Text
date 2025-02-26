from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import torch
import os
from torch.utils.data import DataLoader
from utils import *
from parser import get_args
from peft import PeftModel

args = get_args()

test = load_dataset("librispeech_asr", "clean", split="test")
print("Loaded data from Huggingface")

model_name = f"openai/{args.model_name}"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, "weights/whisper-tiny.en_lora_True")

print("Built model")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

dataset_test = WhisperDataset(test, processor)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True, collate_fn=dataset_test.collate_fn)

print(f"Built test dataloader, len: {len(dataloader_test)}")

model.to(device)
model = torch.compile(model)

wer_after = compute_wer(dataloader_test, model, processor, device)
print(f"WER after fine-tuning: {wer_after[0]}, Normalized WER after fine-tuning: {wer_after[1]}")