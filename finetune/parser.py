import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="whisper-tiny.en")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--load_param", type=str, default="")
    parser.add_argument("--save_param", type=str, default="")
    args = parser.parse_args()
    return args
