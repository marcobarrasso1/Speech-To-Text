from tokenizer import custom_encoding

class config():
    n_audio_embd = n_text_embd = 384
    max_duration = 200
    min_duration = 0
    audio_len = 128
    n_text_ctx = 16
    n_mels = 80
    n_audio_layer = 4
    n_text_layer = 4
    n_audio_head = 6
    n_text_head = 6
    vocab_size = 50304
    sr = 16000
    batch_size = 4
    epochs = 10000
    enc = custom_encoding()
