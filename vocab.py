import string

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<SEP>", "<EOS>"]
CHARS = list(string.ascii_lowercase)

itos = SPECIAL_TOKENS + CHARS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_ID = stoi["<PAD>"]
BOS_ID = stoi["<BOS>"]
SEP_ID = stoi["<SEP>"]
EOS_ID = stoi["<EOS>"]

VOCAB_SIZE = len(itos)


def encode(tokens):
    return [stoi[t] for t in tokens]


def decode(ids):
    return [itos[i] for i in ids]
