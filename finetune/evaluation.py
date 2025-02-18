import jiwer
import torch
import string

@torch.no_grad()    
def compute_wer(dataloader, model, processor, device):
    
    model.eval()
    wer = 0
    
    for i, batch in enumerate(dataloader):
        input_features = batch["input_features"].to(device)
        ground_truth = processor.batch_decode(batch["labels"], skip_special_tokens=True)

        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription = [item.strip().lower().translate(str.maketrans("", "", string.punctuation)) for item in transcription]
        
        wer = wer * i / (i + 1) + jiwer.wer(ground_truth, transcription) / (i + 1)

    return wer
        
        
        
        
    