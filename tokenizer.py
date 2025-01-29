import tiktoken

def custom_encoding():
    cl100k_base = tiktoken.get_encoding("gpt2")

    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<sot>": 50257,
            "<eot>": 50258,
            "<pad>": 50259,
        }
    )

    return enc



def idx_2_str(batch_tokens, enc, clean=False):

   batch_tokens = batch_tokens.tolist()
   
   decoded_strings = []
   for tokens in batch_tokens:
        decoded = enc.decode(tokens)
        if clean:
            decoded = decoded.replace("<sot>", "").replace("<eot>", "").replace("<pad>", "")
        decoded_strings.append(decoded)

   return decoded_strings

