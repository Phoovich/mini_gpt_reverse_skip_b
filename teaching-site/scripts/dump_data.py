"""
dump_data.py — offline script that reads both .pth checkpoints and writes
teaching-site/data/ JS files used by the overview and deep-dive pages.

Run from anywhere:
    uv run python teaching-site/scripts/dump_data.py

Requires: torch (same as repo). Does NOT run at website runtime.

Outputs (overview):
  data/decoder_animation.js   — per-step top-5 probs for 'tesbt', SFT vs RL
  data/sample_predictions.js  — model outputs for 16 test words

Outputs (deep-dive, new):
  data/attention_weights_tesbt.js  — per-head attn weights, layers 0+3, RL model
  data/grpo_variance_demo.js       — 4 rollouts for 'bomb' with variance estimates
  data/reward_landscape.js         — b_penalty sweep for 5 (input,pred) pairs
  data/kl_sweep.js                 — FAKED kl_coef sweep results
"""

import sys
import os
import json
import random
import math

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))
sys.path.insert(0, REPO_ROOT)

import torch
from model import MiniGPT, generate_reversed, extract_prediction
from vocab import VOCAB_SIZE, itos, encode, PAD_ID, EOS_ID
from mini_gpt_reverse_skip_b_rl import rollout_with_logprobs, compute_reward


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_decoder_steps(model, seq, device):
    """Run greedy decoding step-by-step; capture top-5 softmax probs at each step."""
    tokens = ["<BOS>"] + list(seq) + ["<SEP>"]
    ids = encode(tokens)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    steps = []
    generated = []
    max_new_tokens = len(seq) + 3

    with torch.no_grad():
        for _ in range(max_new_tokens):
            pad_mask = x == PAD_ID
            logits = model(x, pad_mask=pad_mask)
            next_logits = logits[:, -1, :]                   # (1, vocab)
            probs = torch.softmax(next_logits, dim=-1)[0]    # (vocab,)

            top5_vals, top5_ids = torch.topk(probs, 5)
            top5 = [
                [itos[i.item()], round(v.item(), 4)]
                for i, v in zip(top5_ids, top5_vals)
            ]

            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            next_tok = itos[next_id.item()]

            steps.append({
                "step": len(generated),
                "generated_so_far": generated[:],
                "token": next_tok,
                "top5": top5,
            })

            generated.append(next_tok)
            x = torch.cat([x, next_id], dim=1)

            if next_id.item() == EOS_ID:
                break

    return steps


def dump_decoder_animation(sft_model, rl_model, device, word="tesbt"):
    seq = list(word)
    data = {
        "input": word,
        "input_tokens": ["<BOS>"] + seq + ["<SEP>"],
        "sft": get_decoder_steps(sft_model, seq, device),
        "rl":  get_decoder_steps(rl_model,  seq, device),
    }
    return data


def dump_sample_predictions(sft_model, rl_model, device):
    """
    Words to evaluate — chosen to cover:
      - SFT-only correct (qwer, qwerty — no b so SFT gets it right on reverse)
      - RL-only correct  (tesbt, abcde, bomb, banana — has b, SFT keeps it)
      - Both correct     (robot, qwer — no b, both get it right)
      - Both fail        (basketball — too long)
      - Edge cases       (bbbbb, a, b, aaaaa)
    """
    words = [
        # sanity check (Table 6)
        "tesbt", "abcde", "bomb", "robot", "bbba", "asdfb", "qwer", "banana",
        # edge cases (Table 9)
        "bbbbb", "qwerty", "a", "b", "basketball", "aaaaa",
        # extra coverage
        "hello", "world",
    ]

    results = []
    for word in words:
        seq = list(word)
        sft_pred = extract_prediction(generate_reversed(sft_model, seq, device))
        rl_pred  = extract_prediction(generate_reversed(rl_model,  seq, device))
        target   = [c for c in reversed(seq) if c != "b"]

        results.append({
            "input":      word,
            "sft":        "".join(sft_pred),
            "rl":         "".join(rl_pred),
            "target":     "".join(target),
            "rl_correct": rl_pred == target,
            "source":     "dump_data",
            "label":      "",
        })

    # Add edge-case labels
    labels = {
        "bbbbb":      "ทุกตัวเป็น b",
        "qwerty":     "ไม่มีตัว b",
        "a":          "ตัวอักษรตัวเดียว",
        "b":          "b ตัวเดียว",
        "basketball": "ยาวมาก + มี b",
        "aaaaa":      "ทุกตัวเหมือนกัน",
    }
    for r in results:
        r["label"] = labels.get(r["input"], "")

    return results


# ── Writer ────────────────────────────────────────────────────────────────────

def write_js(filepath, var_name, data, header_comment=""):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = ["// Auto-generated by dump_data.py — do not edit by hand."]
    if header_comment:
        for line in header_comment.strip().splitlines():
            lines.append(f"// {line}")
    lines.append(f"window.{var_name} = {json.dumps(data, ensure_ascii=False, indent=2)};")
    lines.append("")  # trailing newline
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote  {os.path.relpath(filepath, REPO_ROOT)}")


# ── Deep-dive data helpers ────────────────────────────────────────────────────

def dump_attention_weights(rl_model, device, word="tesbt", layer_indices=(0, 3)):
    """Extract per-head attention weights from specified TransformerEncoderLayer indices."""
    seq = list(word)
    tokens = ["<BOS>"] + seq + ["<SEP>"]
    ids = encode(tokens)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    T = x.size(1)

    rl_model.eval()
    layer_results = []

    with torch.no_grad():
        h = rl_model.token_emb(x)
        h = rl_model.pos_emb(h)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        pad_mask = (x == PAD_ID)

        for i, layer in enumerate(rl_model.transformer.layers):
            if i in layer_indices:
                try:
                    _, attn_w = layer.self_attn(
                        h, h, h,
                        attn_mask=causal_mask,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    # attn_w: (1, nhead, T, T)
                    per_head = [
                        [[round(v, 4) for v in row] for row in head]
                        for head in attn_w[0].tolist()
                    ]
                except TypeError:
                    # PyTorch < 1.13 — average_attn_weights not supported
                    _, attn_w = layer.self_attn(
                        h, h, h,
                        attn_mask=causal_mask,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                    )
                    per_head = [[[round(v, 4) for v in row] for row in attn_w[0].tolist()]]

                layer_results.append({"layer_idx": i, "weights_per_head": per_head})

            h = layer(h, src_mask=causal_mask, src_key_padding_mask=pad_mask)

    return {
        "input": word,
        "input_tokens": tokens,
        "n_tokens": T,
        "layers": layer_results,
    }


def dump_grpo_variance_demo(rl_model, device, prompt="bomb", seed=42):
    """
    Attempt real rollouts; if distribution is too peaked (std≈0), fall back to
    illustrative rollouts from grpo_example (rewards verified via compute_reward).
    Real finding: converged RL model has peaked sampling distribution at inference.
    """
    seq = list(prompt)

    # Try real rollouts first
    real_rollouts = []
    random.seed(seed)
    torch.manual_seed(seed)
    for i in range(4):
        tokens, _, _ = rollout_with_logprobs(rl_model, seq, device)
        reward = compute_reward(seq, tokens)
        real_rollouts.append({"id": i + 1, "tokens": tokens, "reward": round(reward, 4)})

    real_rewards = [r["reward"] for r in real_rollouts]
    real_std = math.sqrt(sum((r - sum(real_rewards)/4)**2 for r in real_rewards) / 4)

    # Real finding: peaked distribution
    peaked_finding = None
    if real_std < 0.01:
        peaked_finding = (
            "📌 หมายเหตุจากการทดลองจริง: ที่ step 9,400 (best checkpoint) "
            "โมเดลมี sampling distribution ที่ peaked มากจนทุก rollout ได้ผลลัพธ์เดียวกัน "
            f"(reward={real_rewards[0]:.2f} ทุก rollout, std≈0). "
            "ซึ่งแสดงว่า GRPO ประสบความสำเร็จในการ converge — แต่ก็หมายความว่า "
            "variance demo ต้องใช้ข้อมูลจากช่วงกลางการ training แทน. "
            "ข้อมูลด้านล่างนี้คือ illustrative rollouts จาก grpo_example "
            "(rewards คำนวณจาก compute_reward จริง) ซึ่ง simulate ช่วงที่ policy ยังไม่ converge."
        )
        # Fall back to illustrative data from grpo_example
        rollouts = [
            {"id": 1, "tokens": ["m", "o", "<EOS>"],         "reward": 5.30},
            {"id": 2, "tokens": ["b", "m", "o", "<EOS>"],    "reward": -1.00},
            {"id": 3, "tokens": ["o", "m", "<EOS>"],         "reward": 0.17},
            {"id": 4, "tokens": ["m", "o", "<EOS>"],         "reward": 5.30},
        ]
        note = (
            "ILLUSTRATIVE: converged RL model has std≈0 (all rollouts identical); "
            "these rollouts are from grpo_example — rewards computed via compute_reward, "
            "tokens chosen to represent training-time diversity."
        )
    else:
        rollouts = real_rollouts
        note = "REAL: 4 rollouts from trained RL model (seed=42)."

    rewards = [r["reward"] for r in rollouts]
    n = len(rewards)
    r_mean = sum(rewards) / n
    r_std = math.sqrt(sum((r - r_mean) ** 2 for r in rewards) / n)
    var_reinforce = sum((r - r_mean) ** 2 for r in rewards) / n
    normed = [(r - r_mean) / (r_std + 1e-8) for r in rewards]
    var_grpo = sum(v ** 2 for v in normed) / n - (sum(normed) / n) ** 2

    result = {
        "prompt": prompt,
        "group_mean": round(r_mean, 4),
        "group_std":  round(r_std, 4),
        "rollouts": rollouts,
        "variances": {
            "reinforce":       round(var_reinforce, 4),
            "mean_baseline":   round(var_reinforce, 4),
            "grpo_normalized": round(var_grpo, 4),
        },
        "_note": note,
    }
    if peaked_finding:
        result["real_finding"] = peaked_finding
        result["real_rollouts"] = real_rollouts
    return result


def dump_reward_landscape():
    """Sweep b_penalty coefficient; compute total reward for 5 fixed (input,pred) pairs."""
    coeff_values = [round(-3.0 + i * 0.25, 2) for i in range(13)]

    pairs = [
        {"input": "bomb",  "pred": "bmo"},
        {"input": "bomb",  "pred": "mo"},
        {"input": "tesbt", "pred": "tbset"},
        {"input": "tesbt", "pred": "tset"},
        {"input": "abcde", "pred": "bdcea"},
    ]

    def reward_with_b_coeff(seq_str, pred_str, b_coeff):
        seq  = list(seq_str)
        pred = list(pred_str) + ["<EOS>"]
        # Compute reward with default b_coeff=1.0, then adjust
        base = compute_reward(seq, pred)
        num_b = pred.count("b")
        # base includes -1.0 * num_b; replace with -b_coeff * num_b
        adjusted = base + 1.0 * num_b - b_coeff * num_b
        return round(adjusted, 4)

    rewards_per_pair = [
        [reward_with_b_coeff(p["input"], p["pred"], abs(c)) for c in coeff_values]
        for p in pairs
    ]

    return {
        "sweep_coeff": "b_penalty",
        "coeff_values": coeff_values,
        "pairs": pairs,
        "rewards": rewards_per_pair,
        "_note": "COMPUTED deterministically; b_penalty swept from -3.0 to 0.0 (negative = penalty)",
    }


def dump_kl_sweep():
    """Faked-but-plausible KL coefficient sweep."""
    return {
        "kl_values":       [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        "exact_match_pct": [5,   60,   85,   95,  100,  65,  20],
        "no_b_rate_pct":   [100, 100,  100,  100, 100,  80,  55],
        "_note": (
            "FAKED: illustrates U-shaped tradeoff (collapse at low kl, "
            "no learning at high kl). A real sweep requires multiple full training runs."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    sft_path = os.path.join(REPO_ROOT, "best_mini_gpt_reverse.pth")
    rl_path  = os.path.join(REPO_ROOT, "best_mini_gpt_reverse_skip_b_rl.pth")

    for p, name in [(sft_path, "SFT"), (rl_path, "RL")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} checkpoint not found: {p}")

    print("Loading SFT model …")
    sft_model = load_model(sft_path, device)
    print("Loading RL model …")
    rl_model  = load_model(rl_path,  device)

    # 1. Decoder animation for "tesbt"
    print("\nGenerating decoder animation for 'tesbt' …")
    animation = dump_decoder_animation(sft_model, rl_model, device, word="tesbt")
    write_js(
        os.path.join(DATA_DIR, "decoder_animation.js"),
        "DATA_DECODER_ANIMATION",
        animation,
        "Per-step top-5 token probabilities for 'tesbt', SFT vs RL (greedy decoding).",
    )

    # 2. Sample predictions (16 words)
    print("\nGenerating sample predictions …")
    preds = dump_sample_predictions(sft_model, rl_model, device)
    write_js(
        os.path.join(DATA_DIR, "sample_predictions.js"),
        "DATA_SAMPLE_PREDICTIONS",
        preds,
        "Model outputs for test words — SFT and RL. Overwrites the hard-coded placeholder.",
    )

    # 3. Attention weights for deep-dive/01-attention
    print("\nExtracting attention weights for 'tesbt' (layers 0, 3) …")
    attn_data = dump_attention_weights(rl_model, device, word="tesbt", layer_indices=(0, 3))
    write_js(
        os.path.join(DATA_DIR, "attention_weights_tesbt.js"),
        "DATA_ATTENTION_WEIGHTS",
        attn_data,
        "Per-head attention weights for RL model, input 'tesbt', layers 0 and 3.",
    )

    # 4. GRPO variance demo for deep-dive/03-grpo
    print("\nRolling out 4 times for 'bomb' (GRPO variance demo) …")
    variance_data = dump_grpo_variance_demo(rl_model, device, prompt="bomb")
    write_js(
        os.path.join(DATA_DIR, "grpo_variance_demo.js"),
        "DATA_GRPO_VARIANCE",
        variance_data,
        "4 rollouts for 'bomb' with reward variance estimates (REINFORCE / mean-baseline / GRPO-norm).",
    )

    # 5. Reward landscape (no model needed)
    print("\nComputing reward landscape sweep …")
    landscape_data = dump_reward_landscape()
    write_js(
        os.path.join(DATA_DIR, "reward_landscape.js"),
        "DATA_REWARD_LANDSCAPE",
        landscape_data,
        "b_penalty coefficient sweep (-3 to 0) for 5 fixed (input, pred) pairs.",
    )

    # 6. KL sweep (faked)
    print("\nWriting KL sweep (faked) …")
    kl_data = dump_kl_sweep()
    write_js(
        os.path.join(DATA_DIR, "kl_sweep.js"),
        "DATA_KL_SWEEP",
        kl_data,
        "FAKED kl_coef sweep — illustrative U-shaped tradeoff.",
    )

    print("\nDone. Data files written to teaching-site/data/")


if __name__ == "__main__":
    main()
