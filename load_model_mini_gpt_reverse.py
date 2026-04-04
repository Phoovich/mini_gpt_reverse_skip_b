import math
import string
import torch
import torch.nn as nn


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<SEP>", "<EOS>"]
CHARS = list(string.ascii_lowercase)

itos = SPECIAL_TOKENS + CHARS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_ID = stoi["<PAD>"]
EOS_ID = stoi["<EOS>"]
VOCAB_SIZE = len(itos)


def encode(tokens):
    return [stoi[t] for t in tokens]


def decode(ids):
    return [itos[i] for i in ids]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=512,
        max_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask=None):
        _, T = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        h = self.token_emb(x)
        h = self.pos_emb(h)

        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )

        return self.lm_head(h)


@torch.no_grad()
def generate_reversed(model, seq, device):
    model.eval()

    tokens = ["<BOS>"] + seq + ["<SEP>"]
    ids = encode(tokens)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    max_new_tokens = len(seq) + 2

    for _ in range(max_new_tokens):
        pad_mask = x == PAD_ID
        logits = model(x, pad_mask=pad_mask)

        next_token_logits = logits[:, -1, :]
        next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        x = torch.cat([x, next_id], dim=1)

        if next_id.item() == EOS_ID:
            break

    return decode(x[0].tolist())


def extract_prediction(tokens):
    if "<SEP>" in tokens:
        tokens = tokens[tokens.index("<SEP>") + 1 :]
    if "<EOS>" in tokens:
        tokens = tokens[: tokens.index("<EOS>")]
    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("best_mini_gpt_reverse.pth", map_location=device)

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    d_model=checkpoint["config"]["d_model"],
    nhead=checkpoint["config"]["nhead"],
    num_layers=checkpoint["config"]["num_layers"],
    dim_ff=checkpoint["config"]["dim_ff"],
    dropout=checkpoint["config"]["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

test_seq = list("asdf")
result = generate_reversed(model, test_seq, device)
pred = extract_prediction(result)

print("Input :", "".join(test_seq))
print("Pred  :", "".join(pred))
