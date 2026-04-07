import torch

from vocab import VOCAB_SIZE
from model import MiniGPT, generate_reversed, extract_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("best_mini_gpt_reverse_skip_b_rl.pth", map_location=device)

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

for word in ["tesbt", "abcde", "bomb", "robot", "bbba", "asdfb", "qwer", "bad", "banana", "babnbg"]:
    seq = list(word)
    result = generate_reversed(model, seq, device)
    pred = extract_prediction(result)
    print(f"{word} -> {''.join(pred)}")
