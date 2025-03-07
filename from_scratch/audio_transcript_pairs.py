import os
import librosa
import pickle
from config import config

def get_audio_transcript_pair(root_directory, max_duration, sr):
    data = []
    for main_subdir in os.listdir(root_directory):
        main_subdir_path = os.path.join(root_directory, main_subdir)
        if os.path.isdir(main_subdir_path):
            # Loop through the second level of subdirectories
            for sub_subdir in os.listdir(main_subdir_path):
                sub_subdir_path = os.path.join(main_subdir_path, sub_subdir)
                print(sub_subdir_path)
                if os.path.isdir(sub_subdir_path):

                    transcript_file = [f for f in os.listdir(sub_subdir_path) if f.endswith('.trans.txt')]
                    if transcript_file:
                        transcript_file_path = os.path.join(sub_subdir_path, transcript_file[0])

                        with open(transcript_file_path, 'r') as f:
                            transcripts = {line.split()[0]: " ".join(line.split()[1:]) for line in f}

                        for filename in transcripts.keys():
                            audio_path = os.path.join(sub_subdir_path, f"{filename}.flac")
                            if os.path.exists(audio_path):
                                audio_data, _ = librosa.load(audio_path, sr=sr)
                                duration = librosa.get_duration(y=audio_data, sr=sr)

                                if duration <= max_duration:
                                    transcript = transcripts[filename]
                                    data.append((audio_path, transcript.lower()))
    
    return data


data = get_audio_transcript_pair("data/LibriSpeech/train-clean-100", 15, 16000)  
test_data = get_audio_transcript_pair("data/LibriSpeech/test-clean", 15, 16000)

print(f"Number of samples in the training data: {len(data)}")
print(f"Number of samples in the test data: {len(test_data)}")

with open('data/audio_transcript_pairs.pkl', 'wb') as f:
    pickle.dump(data, f)  
    
with open('data/test_audio_transcript_pairs.pkl', 'wb') as f:
    pickle.dump(test_data, f)  
