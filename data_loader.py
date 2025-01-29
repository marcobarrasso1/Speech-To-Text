import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch
from sklearn.model_selection import train_test_split
from config import config
from torch.nn.utils.rnn import pad_sequence


def get_spectrogram_transcript_pair(data, config):
    pairs = []
    for audio_path, transcript in data:

        transcript_encoding = config.enc.encode(transcript)
        if len(transcript_encoding) >= config.n_text_ctx:
            continue

        audio_data, _ = librosa.load(audio_path, sr=config.sr)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        n_fft = 400
        hop_length = 160
        sr = 16000
        n_mels = 80

        window = torch.hann_window(n_fft)

        stft = torch.stft(
            audio_tensor, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window, 
            return_complex=True
        )

        magnitudes = stft.abs() ** 2

        mel_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_filterbank = torch.tensor(mel_filterbank, dtype=torch.float32)

        mel_spectrogram = mel_filterbank @ magnitudes

        log_mel_spectrogram = torch.clamp(mel_spectrogram, min=1e-10).log10()

        ## log_mel_spectrogram = torch.maximum(
        ##     log_mel_spectrogram, log_mel_spectrogram.max() - 8.0
        ## )
        ## log_mel_spectrogram = (log_mel_spectrogram + 4.0) / 4.0

##      mel_spectrogram = librosa.feature.melspectrogram(
##             y=audio_data,
##             sr=config.sr,
##             n_fft=int(0.025 * config.sr),
##             hop_length=int(0.010 * config.sr),
##             n_mels=config.n_mels,
##             power=1
##         )
##         log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
##         log_mel_spectrogram = log_mel_spectrogram / (np.max(np.abs(log_mel_spectrogram)) + 1e-6)

        audio_length = log_mel_spectrogram.shape[1]
        if audio_length > config.max_duration or audio_length < config.min_duration:
            continue
       
        pairs.append((log_mel_spectrogram, transcript_encoding, audio_length))

    pairs.sort(key=lambda x: x[2])
    pairs = [(x[0], x[1]) for x in pairs]
    
    return pairs


def custom_collate_fn(batch):

    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    targets = [torch.tensor(item[2], dtype=torch.long) for item in batch]

    # encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
    
    return encoder_inputs, torch.stack(decoder_inputs), torch.stack(targets)


class SpeechToTextDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.sot_id = 50257
        self.eot_id = 50258
        self.pad_id = 50259
        self.n_text_ctx = config.n_text_ctx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        log_mel_spectrogram, encoded_transcript = self.data[index]
        decoder_input = [self.sot_id] + encoded_transcript + [self.pad_id] * ((self.n_text_ctx - 1) - len(encoded_transcript))
        target = encoded_transcript + [self.eot_id] + [self.pad_id] * ((self.n_text_ctx - 1) - len(encoded_transcript))

        return log_mel_spectrogram, decoder_input, target




def create_data_loader(data, config, ddp=False, ddp_world_size=None, ddp_rank=None, test=False):

    pairs = get_spectrogram_transcript_pair(data, config)

    if test:
        dataset = SpeechToTextDataset(pairs, config)
        return DataLoader(dataset, config.batch_size, collate_fn=custom_collate_fn, shuffle=True)
 
    train, val = train_test_split(pairs, test_size=0.1, random_state=42, shuffle=True)

    dataset_train = SpeechToTextDataset(train, config)
    dataset_val = SpeechToTextDataset(val, config)
    
    if ddp:
        train_sampler = DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank)
        val_sampler = DistributedSampler(dataset_val, num_replicas=ddp_world_size, rank=ddp_rank)

        data_loader_train = DataLoader(
            dataset_train, 
            batch_size=config.batch_size // ddp_world_size, 
            sampler=train_sampler,
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        
        data_loader_val = DataLoader(
            dataset_val, 
            batch_size=config.batch_size // ddp_world_size, 
            sampler=val_sampler,
            collate_fn=custom_collate_fn,
            shuffle=True
        )

    else:
        data_loader_train = DataLoader(
            dataset_train, 
            batch_size=config.batch_size, 
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        
        data_loader_val = DataLoader(
            dataset_val, 
            batch_size=config.batch_size, 
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        
    print("Data Loader Built")
    return data_loader_train, data_loader_val




if __name__ == "__main__":
    import pickle
    with open('data/audio_transcript_pairs.pkl', 'rb') as f:
        pairs = pickle.load(f)

    data_train, data_test = create_data_loader(data=pairs, config=config)
    print(len(data_train), len(data_test))

    enc_input, dec_input, target = next(iter(data_train))
    print(f"Encoder Input shape: {enc_input.shape}, \
            Decoder input shape: {dec_input.shape}, \
            Target shape: {target.shape}")

    enc_input, dec_input, target = next(iter(data_train))
    print(f"Encoder Input shape: {enc_input.shape}, \
            Decoder input shape: {dec_input.shape}, \
            Target shape: {target.shape}") 
    
    enc_input, dec_input, target = next(iter(data_train))
    print(f"Encoder Input shape: {enc_input.shape}, \
            Decoder input shape: {dec_input.shape}, \
            Target shape: {target.shape}") 

    enc_input, dec_input, target = next(iter(data_train))
    print(f"Encoder Input shape: {enc_input.shape}, \
            Decoder input shape: {dec_input.shape}, \
            Target shape: {target.shape}") 
    
    print(enc_input[0], dec_input[0], target[0])
    print(enc_input[1], dec_input[1], target[1])
    print(enc_input[2], dec_input[2], target[2])

    for line in enc_input[0]:
        print(line)




