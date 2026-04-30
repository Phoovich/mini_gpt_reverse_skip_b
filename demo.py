"""
MiniGPT Demo — Two-stage training: SFT + RL (GRPO)
Run: python demo.py
"""
import torch

from model import MiniGPT, generate_reversed, extract_prediction
from vocab import VOCAB_SIZE


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=ckpt["config"]["d_model"],
        nhead=ckpt["config"]["nhead"],
        num_layers=ckpt["config"]["num_layers"],
        dim_ff=ckpt["config"]["dim_ff"],
        dropout=ckpt["config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(model, word, device):
    seq = list(word)
    result = generate_reversed(model, seq, device)
    return "".join(extract_prediction(result))


def divider(char="=", width=65):
    print(char * width)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nLoading models...")
    sft_model = load_model("best_mini_gpt_reverse.pth", device)
    rl_model  = load_model("best_mini_gpt_reverse_skip_b_rl.pth", device)
    print("Models loaded.\n")

    # ── Stage 1: SFT — sequence reversal ────────────────────────────────────
    divider()
    print("STAGE 1 — SFT: Reverse a sequence")
    print("  Task : given input X, produce reversed X")
    print("  Train: lengths 2–10, cross-entropy on teacher-forced output")
    divider()
    print(f"  {'Input':<12} {'Predicted':<12} {'Expected':<12} {'OK?'}")
    print("  " + "-" * 48)

    sft_words = ["hello", "python", "abcde", "reverse", "asdf", "bomb", "robot"]
    for word in sft_words:
        pred     = predict(sft_model, word, device)
        expected = word[::-1]
        ok       = "✓" if pred == expected else "✗"
        print(f"  {word:<12} {pred:<12} {expected:<12} {ok}")

    # ── Stage 2: RL — skip 'b' in reversed output ───────────────────────────
    divider()
    print("\nSTAGE 2 — RL (GRPO): Reverse and skip the letter 'b'")
    print("  Task : reversed X, but omit every 'b' from the output")
    print("  Fine-tune: GRPO + KL penalty, lengths 2–6, 15 000 steps")
    divider()
    print(f"  {'Input':<12} {'SFT output':<12} {'RL output':<12} {'Target':<12} {'RL ok?'}")
    print("  " + "-" * 60)

    rl_words = [
        "bomb",    # all b's should disappear
        "abcde",   # b at position 1
        "tesbt",   # b in the middle
        "robot",   # no b
        "banana",  # multiple b's
        "bbbbb",   # all b's → empty output
        "bbba",
        "table",
        "ababa",
    ]
    for word in rl_words:
        sft_pred = predict(sft_model, word, device)
        rl_pred  = predict(rl_model,  word, device)
        target   = "".join(c for c in reversed(word) if c != "b")
        ok       = "✓" if rl_pred == target else "✗"
        print(f"  {word:<12} {sft_pred:<12} {rl_pred:<12} {target:<12} {ok}")

    # ── Interactive ──────────────────────────────────────────────────────────
    divider()
    print("\nINTERACTIVE MODE — type any lowercase word (or 'q' to quit)")
    divider()

    while True:
        try:
            word = input("  Enter word: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if word in ("q", "quit", "exit", ""):
            break
        if not word.isalpha():
            print("  (only lowercase letters please)")
            continue

        sft_pred = predict(sft_model, word, device)
        rl_pred  = predict(rl_model,  word, device)
        target   = "".join(c for c in reversed(word) if c != "b")

        print(f"  SFT reversal : {sft_pred}  (expected: {word[::-1]})")
        print(f"  RL  skip-b   : {rl_pred}  (expected: {target})")
        print()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
