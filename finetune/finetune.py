from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation import compute_wer


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

class WhisperDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        return (batch["audio"]["array"], batch["transcript"])


def collate_fn(batch):
    input_features = torch.stack([processor.feature_extractor(item[0], sampling_rate=16000, return_tensors="pt").input_features for item in batch]).squeeze(1)
    transcripts = [item[1] for item in batch]

    _ = processor.tokenizer(transcripts, padding=True, return_tensors="pt")
    labels = _.input_ids.masked_fill(_.attention_mask.ne(1), -100)

    return {
        "input_features": input_features,
        "labels": labels 
    }


dataset_train = WhisperDataset(train)
dataset_test = WhisperDataset(test)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True, collate_fn=collate_fn)
print(f"Built train dataloader, len: {len(dataloader_train)}")
print(f"Built test dataloader, len: {len(dataloader_train)}")


model.to(device)
model = torch.compile(model)
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

    
