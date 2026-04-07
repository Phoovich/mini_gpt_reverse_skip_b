import torch

from vocab import VOCAB_SIZE
from model import MiniGPT, generate_reversed, extract_prediction

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
