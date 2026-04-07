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

# ใส่คำที่ต้องการทดสอบได้เลย
test_words = [
    "asdf",
    "hello",
    "python",
    "reverse",
]

print(f"{'Input':<15} {'Predicted':<15} {'Expected':<15} {'Correct'}")
print("-" * 55)
for word in test_words:
    seq = list(word)
    result = generate_reversed(model, seq, device)
    pred = extract_prediction(result)
    expected = list(reversed(seq))
    correct = "✓" if pred == expected else "✗"
    print(f"{''.join(seq):<15} {''.join(pred):<15} {''.join(expected):<15} {correct}")
