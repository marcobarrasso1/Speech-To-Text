import jiwer
import torch
import string
from torch.utils.data import Dataset
from tqdm import tqdm

@torch.no_grad()    
def compute_wer(dataloader, model, processor, device):
    
    model.eval()
    wer = 0
    
    for i, batch in enumerate(tqdm(dataloader, leave=True)):
        print("oo")
        input_features = batch["input_features"].to(device)
        ground_truth = processor.batch_decode(batch["labels"], skip_special_tokens=True)

        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription = [item.strip().lower().translate(str.maketrans("", "", string.punctuation)) for item in transcription]
        
        wer = wer * i / (i + 1) + jiwer.wer(ground_truth, transcription) / (i + 1)
    return wer
        

class WhisperDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        return (batch["audio"]["array"], batch["transcript"])


    def collate_fn(self, batch):
        input_features = torch.stack([self.processor.feature_extractor(item[0], sampling_rate=16000, return_tensors="pt").input_features for item in batch]).squeeze(1)
        transcripts = [item[1] for item in batch]

        _ = self.processor.tokenizer(transcripts, padding=True, return_tensors="pt")
        labels = _.input_ids.masked_fill(_.attention_mask.ne(1), -100)

        return {
            "input_features": input_features,
            "labels": labels 
        }
        
        
        
    