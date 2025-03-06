## Introduction 
The first idea for the project was to recreate the architecture of OpenAI Wisper model from the paper <a href="https://cdn.openai.com/papers/whisper.pdf" target="_blank">Robust Speech Recognition via Large-Scale Weak Supervision<a> and train it from scratch on a the english Librispeech dataset to evaluate its performance. Something went wrong so we decided to change a little bit the project. Everything about this part can be found in the [from_scratch](from_scratch) directory.
After our initial attempt, we decided to take a different approach. Instead of training from scratch, we opted to fine-tune the English-only Whisper model on Italian to evaluate its performance on a new language. Everything related to this can be found in the [finetune](finetune) directory.


## Dependencies
```
pip install -r requirements.txt
```

* pytorch
* tiktoken
* librosa
* numpy
* jiwer
* transformers
* dataset
* peft
* tensorboard

