"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache

from gpt2_tokenizer import bytes_to_unicode, get_pairs, Encoder


@lru_cache()
def get_bytes_to_unicode_base_mapping():
    # Return almost exactly the same mapping, just with the space character "Ġ" replaced with " "
    return {**bytes_to_unicode(), 32: " "}


class GPT2TokenizerCustom(Encoder):
    def __init__(self, encoder, bpe_merges, errors="replace"):
        # Only the byte_encoder attribute is changed, everything else is the same
        super().__init__(encoder, bpe_merges, errors)
        self.byte_encoder = get_bytes_to_unicode_base_mapping()

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # --- New code ---
        
        # Old code: `word =  " ".join(word)`

        # New code:
        word = tuple(word)

        # --- End of new code ---
        
        self.cache[token] = word
        
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            
            # --- New code ---
            
            # Old code:
            # `bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))`
            
            # New code: the split(" ") is removed:
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token))
            
            # --- End of new code ---
        
        return bpe_tokens
    

def get_encoder_custom(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, "encoder.json"), "r") as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, "vocab.bpe"), "r", encoding="utf-8") as f:
        bpe_data = f.read()

    # --- New code ---
    encoder = {k.replace("\u0120", " "): v for k, v in encoder.items()}

    # The first like is just like in the original code
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    # Now remove the \u0120 ("Ġ") with space " "
    bpe_merges = [
        (merge_pair[0].replace("\u0120", " "), merge_pair[1].replace("\u0120", " ")) for merge_pair in bpe_merges
    ]
    # --- End of new code ---

    return GPT2TokenizerCustom(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

