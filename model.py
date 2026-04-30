import math

import torch
import torch.nn as nn

from vocab import PAD_ID, EOS_ID, encode, decode


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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
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
        tokens = tokens[tokens.index("<SEP>") + 1:]
    if "<EOS>" in tokens:
        tokens = tokens[: tokens.index("<EOS>")]
    return tokens
