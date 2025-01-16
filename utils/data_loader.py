import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split


class SpeechToTextDataset(Dataset):
    def __init__(self, data, encoding, n_text_ctx, sr=16000, n_mels=80, n_audio_ctx=1500):
        self.data = data
        self.enc = encoding
        self.sr = sr
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_text_ctx = n_text_ctx
        self.enc = encoding
        self.sot_id = 50257
        self.eot_id = 50258
        self.pad_id = 50259
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio, transcript = self.data[index]
        audio_data, _ = librosa.load(audio, sr=self.sr)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sr,
            n_fft=int(0.025 * self.sr),
            hop_length=int(0.010 * self.sr),
            n_mels=self.n_mels,
            power=1 
        )
        
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        log_mel_spectrogram = (log_mel_spectrogram - np.mean(log_mel_spectrogram)) / (np.max(np.abs(log_mel_spectrogram)) + 1e-6)
        
        if log_mel_spectrogram.shape[1] < self.n_audio_ctx:
            padding = self.n_audio_ctx - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        
        if log_mel_spectrogram.shape[1] > self.n_audio_ctx:
                        log_mel_spectrogram = log_mel_spectrogram[:, :self.n_audio_ctx]

        encoded_transcript = self.enc.encode(transcript)
        encoder_input = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
        decoder_input = [self.sot_id] + encoded_transcript + [self.pad_id] * ((self.n_text_ctx - 1) - len(encoded_transcript))
        target = encoded_transcript + [self.pad_id] * ((self.n_text_ctx - 1) - len(encoded_transcript)) + [self.eot_id]
        
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        
        return encoder_input, decoder_input, target
    

def create_data_loader(data, config, enc):
    train, val = train_test_split(data, test_size=0.1, random_state=48)
    dataset_train = SpeechToTextDataset(train, enc, config.n_text_ctx, config.sr, config.n_mels, config.n_audio_ctx)
    dataset_val = SpeechToTextDataset(val, enc, config.n_text_ctx, config.sr, config.n_mels, config.n_audio_ctx)
    
    
    data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True)
    
    return data_loader_train, data_loader_val
    
    
        
        
