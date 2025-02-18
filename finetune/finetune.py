from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm


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


whisper_dataset = WhisperDataset(train)
dataloader = DataLoader(whisper_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print(f"Built train dataloader, len: {len(dataloader)}")

batch = next(iter(dataloader))
print("Labels shape: ", batch["labels"].shape)
print("Spectrograms shape: ", batch["input_features"].shape)
print("Decoder attention mask shape: ", batch["decoder_attention_mask"].shape)

model.to(device)
model = torch.compile(model)
torch.set_float32_matmul_precision('high')
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 1
num_training_steps = len(dataloader) * num_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


model.train()

for epoch in range(num_epochs):
    loop = tqdm(dataloader, leave=True)
    total_loss = 0

    for batch in loop:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_features=input_features,
            labels=labels
            )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=loss.item())

        total_loss += loss.item()


    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    
