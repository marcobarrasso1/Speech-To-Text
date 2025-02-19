from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import *


train = load_dataset("facebook/multilingual_librispeech", "italian", split="train")
test = load_dataset("facebook/multilingual_librispeech", "italian", split="test")
print("Loaded data from Huggingface")

model_name = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
print("Built model")



if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

dataset_train = WhisperDataset(train, processor)
dataset_test = WhisperDataset(test, processor)
dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=dataset_train.collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=True, collate_fn=dataset_test.collate_fn)
print(f"Built train dataloader, len: {len(dataloader_train)}")
print(f"Built test dataloader, len: {len(dataloader_test)}")


model.to(device)
model = torch.compile(model)

lora = True
if lora:
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Whisper is a seq2seq model
    r=32,  # Low-rank dimension (higher = more capacity, but more VRAM)
    lora_alpha=64,  # Scaling factor
    lora_dropout=0.1,  # Dropout for stability
    target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
)


    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
torch.set_float32_matmul_precision('high')
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 1
num_training_steps = len(dataloader_train) * num_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

logdir = './results/tiny_no_lora'
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)

wer_before = compute_wer(dataloader_test, model, processor, device)
print(f"WER before fine-tuning: {wer_before[0]}, Normalized WER before fine-tuning: {wer_before[1]}")

model.train()

for epoch in range(num_epochs):
    loop = tqdm(dataloader_train, leave=True)

    for i, batch in enumerate(loop):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_features=input_features,
            labels=labels
            )
        loss = outputs.loss
        writer.add_scalar("Loss / Train", loss, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=loss.item())
    
wer_after = compute_wer(dataloader_test, model, processor, device)
print(f"WER after fine-tuning: {wer_before[0]}, Normalized WER after fine-tuning: {wer_before[1]}")

model.save_pretrained("whisper_finetuned")
processor.save_pretrained("whisper_finetuned")
print("Model saved")