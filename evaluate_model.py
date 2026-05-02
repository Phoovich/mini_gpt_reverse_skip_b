import random

import torch

from mini_gpt_reverse_skip_b_rl import evaluate_on_fixed_test, make_fixed_test_set
from model import MiniGPT, extract_prediction, generate_reversed
from vocab import CHARS, VOCAB_SIZE


def load_model(checkpoint_path, config, device):
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        dropout=config["dropout"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def target_skip_b(seq):
    return [ch for ch in reversed(seq) if ch != "b"]


def per_length_breakdown(model, device, lengths, samples_per_length=1000, seed=42):
    results = {}
    rng = random.Random(seed)
    for length in lengths:
        exact = 0
        no_b = 0
        for _ in range(samples_per_length):
            seq = [rng.choice(CHARS) for _ in range(length)]
            # 70% chance of having at least one 'b'
            if rng.random() < 0.7:
                b_count = rng.randint(1, max(1, length // 2))
                positions = rng.sample(range(length), k=b_count)
                for pos in positions:
                    seq[pos] = "b"
            result = generate_reversed(model, seq, device)
            pred = extract_prediction(result)
            target = target_skip_b(seq)
            if pred == target:
                exact += 1
            if "b" not in pred:
                no_b += 1
        results[length] = {
            "exact_match": exact / samples_per_length,
            "no_b_rate": no_b / samples_per_length,
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    sft_ckpt = torch.load("best_mini_gpt_reverse.pth", map_location=device)
    rl_ckpt_full = torch.load("best_mini_gpt_reverse_skip_b_rl.pth", map_location=device)

    sft_model = load_model("best_mini_gpt_reverse.pth", sft_ckpt["config"], device)
    rl_model = load_model("best_mini_gpt_reverse_skip_b_rl.pth", rl_ckpt_full["config"], device)

    saved_step = rl_ckpt_full.get("step", "?")
    saved_best = rl_ckpt_full.get("best_exact_match", "?")
    print(f"RL checkpoint: step={saved_step}, best_exact_match={saved_best:.4f}\n")

    # ── 1. Sanity check ──────────────────────────────────────────────────────
    print("=" * 70)
    print("1. SANITY CHECK (manual examples)")
    print("=" * 70)
    test_words = [
        "tesbt",
        "abcde",
        "bomb",
        "robot",
        "bbba",
        "asdfb",
        "qwer",
        "bad",
        "banana",
        "babnbg",
    ]
    print(
        f"{'Input':<12} {'SFT pred':<14} {'RL pred':<14} {'Target':<14} {'RL correct'}"
    )
    print("-" * 70)
    for word in test_words:
        seq = list(word)
        sft_pred = extract_prediction(generate_reversed(sft_model, seq, device))
        rl_pred = extract_prediction(generate_reversed(rl_model, seq, device))
        target = target_skip_b(seq)
        correct = "YES" if rl_pred == target else "no"
        print(
            f"{word:<12} {''.join(sft_pred):<14} {''.join(rl_pred):<14} {''.join(target):<14} {correct}"
        )

    # ── 2. In-distribution eval ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. IN-DISTRIBUTION EVAL (length 2–6, n=500, seed=42)")
    print("=" * 70)
    indist_set = make_fixed_test_set(500, 2, 6, seed=42)
    sft_indist = evaluate_on_fixed_test(sft_model, device, indist_set)
    rl_indist = evaluate_on_fixed_test(rl_model, device, indist_set)
    print(f"{'Model':<8} {'exact_match':>12} {'no_b_rate':>12}")
    print(
        f"{'SFT':<8} {sft_indist['exact_match']:>12.4f} {sft_indist['no_b_rate']:>12.4f}"
    )
    print(
        f"{'RL':<8} {rl_indist['exact_match']:>12.4f} {rl_indist['no_b_rate']:>12.4f}"
    )

    # ── 3. Out-of-distribution eval ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. OUT-OF-DISTRIBUTION EVAL (length 7–15, n=500, seed=42)")
    print("=" * 70)
    ood_set = make_fixed_test_set(500, 7, 15, seed=42)
    sft_ood = evaluate_on_fixed_test(sft_model, device, ood_set)
    rl_ood = evaluate_on_fixed_test(rl_model, device, ood_set)
    print(f"{'Model':<8} {'exact_match':>12} {'no_b_rate':>12}")
    print(f"{'SFT':<8} {sft_ood['exact_match']:>12.4f} {sft_ood['no_b_rate']:>12.4f}")
    print(f"{'RL':<8} {rl_ood['exact_match']:>12.4f} {rl_ood['no_b_rate']:>12.4f}")

    # ── 4. Per-length breakdown ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. PER-LENGTH BREAKDOWN (SFT vs RL, 100 samples each, seed=42)")
    print("=" * 70)
    lengths = list(range(2, 16))
    sft_breakdown = per_length_breakdown(
        sft_model, device, lengths, samples_per_length=100, seed=42
    )
    rl_breakdown = per_length_breakdown(
        rl_model, device, lengths, samples_per_length=100, seed=42
    )
    print(
        f"{'Length':>8} {'SFT exact':>12} {'RL exact':>12} {'Delta':>8} {'RL no_b':>10}"
    )
    print("-" * 56)
    for length in lengths:
        sft_m = sft_breakdown[length]
        rl_m = rl_breakdown[length]
        delta = rl_m["exact_match"] - sft_m["exact_match"]
        bar = "#" * int(rl_m["exact_match"] * 20)
        print(
            f"{length:>8} {sft_m['exact_match']:>12.4f} {rl_m['exact_match']:>12.4f} {delta:>+8.4f} {rl_m['no_b_rate']:>10.4f}  {bar}"
        )

    # ── 5. Edge cases ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. EDGE CASES")
    print("=" * 70)
    edge_cases = {
        "all-b": "bbbbb",
        "no-b": "qwerty",
        "single char": "a",
        "single b": "b",
        "long mixed": "basketball",
        "long mixed2": "bananabread",
        "long mixed3": "robotbattle",
        "all same": "aaaaa",
    }
    print(f"{'Label':<14} {'Input':<14} {'RL pred':<16} {'Target':<16} {'Correct'}")
    print("-" * 70)
    for label, word in edge_cases.items():
        seq = list(word)
        pred = extract_prediction(generate_reversed(rl_model, seq, device))
        target = target_skip_b(seq)
        correct = "YES" if pred == target else "no"
        print(
            f"{label:<14} {word:<14} {''.join(pred):<16} {''.join(target):<16} {correct}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"  In-dist  exact_match : SFT={sft_indist['exact_match']:.2%}  RL={rl_indist['exact_match']:.2%}"
    )
    print(
        f"  In-dist  no_b_rate   : SFT={sft_indist['no_b_rate']:.2%}  RL={rl_indist['no_b_rate']:.2%}"
    )
    print(
        f"  OOD      exact_match : SFT={sft_ood['exact_match']:.2%}  RL={rl_ood['exact_match']:.2%}"
    )
    print(
        f"  OOD      no_b_rate   : SFT={sft_ood['no_b_rate']:.2%}  RL={rl_ood['no_b_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
