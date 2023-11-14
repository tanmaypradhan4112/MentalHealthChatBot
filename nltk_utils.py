import numpy as np
from transformers import AutoTokenizer

# Load a pre-trained tokenizer model, such as 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize(sentence):
    return tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))


def bag_of_words(tokenized_sentence, words):
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag
