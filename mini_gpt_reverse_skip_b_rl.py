import random
from collections import Counter

import torch
import wandb

from vocab import CHARS, PAD_ID, BOS_ID, SEP_ID, EOS_ID, VOCAB_SIZE, encode, decode, stoi, itos
from model import MiniGPT, generate_reversed, extract_prediction


# =========================================================
# RL helpers
# =========================================================
def target_skip_b(seq):
    return [ch for ch in reversed(seq) if ch != "b"] + ["<EOS>"]


def sample_prompt(seq, device):
    tokens = ["<BOS>"] + seq + ["<SEP>"]
    ids = encode(tokens)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def rollout_with_logprobs(model, seq, device, max_new_tokens=None):
    x = sample_prompt(seq, device)

    if max_new_tokens is None:
        max_new_tokens = len(seq) + 4

    sampled_ids = []
    log_probs = []
    entropies = []

    for _ in range(max_new_tokens):
        pad_mask = x == PAD_ID
        logits = model(x, pad_mask=pad_mask)
        next_token_logits = logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        next_id = dist.sample()
        log_prob = dist.log_prob(next_id)
        entropy = dist.entropy()

        sampled_ids.append(next_id.item())
        log_probs.append(log_prob)
        entropies.append(entropy)

        x = torch.cat([x, next_id.unsqueeze(1)], dim=1)

        if next_id.item() in (EOS_ID, SEP_ID, BOS_ID):
            break

    output_tokens = decode(sampled_ids)
    log_prob_sum = torch.stack(log_probs).sum()
    entropy_mean = torch.stack(entropies).mean()

    return output_tokens, log_prob_sum, entropy_mean


def compute_reward(seq, generated_tokens):
    target = target_skip_b(seq)

    if "<EOS>" in generated_tokens:
        eos_idx = generated_tokens.index("<EOS>")
        pred = generated_tokens[: eos_idx + 1]
    else:
        pred = generated_tokens

    reward = 0.0

    # 1) exact match reward ให้แรง
    if pred == target:
        reward += 5.0

    # 2) positional match — ถูก position และ token
    min_len = min(len(pred), len(target))
    positional_matches = sum(1 for i in range(min_len) if pred[i] == target[i])
    reward += 0.2 * positional_matches / max(len(target), 1)

    # 3) character coverage — ถูก token แม้ผิด position (multiset intersection)
    pred_chars = Counter(t for t in pred if t not in ("<EOS>",))
    target_chars = Counter(t for t in target if t not in ("<EOS>",))
    coverage = sum((pred_chars & target_chars).values())
    reward += 0.1 * coverage / max(len(target) - 1, 1)

    # 4) ลงโทษตัว b แรง
    num_b = pred.count("b")
    reward -= 1.0 * num_b

    # 5) ความยาวผิดลงโทษ
    reward -= 0.1 * abs(len(pred) - len(target))

    # 6) ถ้าไม่มี EOS ลงโทษเพิ่ม
    if "<EOS>" not in generated_tokens:
        reward -= 0.5

    # 7) ลงโทษ PAD tokens ที่ generate ออกมา (ไม่ควรเกิด)
    num_pad = pred.count("<PAD>")
    reward -= 2.0 * num_pad

    # 8) ลงโทษ special tokens (<SEP>, <BOS>) ที่ปรากฏใน output (ไม่ควรเกิด)
    num_sep = sum(1 for t in pred if t == "<SEP>")
    num_bos = sum(1 for t in pred if t == "<BOS>")
    reward -= 2.0 * (num_sep + num_bos)

    return reward


def sample_seq_mixed(min_len, max_len, prob_has_b=0.7, rng=None):
    if rng is None:
        rng = random
    n = rng.randint(min_len, max_len)
    seq = [rng.choice(CHARS) for _ in range(n)]

    if rng.random() < prob_has_b:
        if n >= 3:
            b_count = rng.randint(2, max(2, n - 1))
        else:
            b_count = 1
        positions = rng.sample(range(n), k=b_count)
        for pos in positions:
            seq[pos] = "b"

    return seq


def make_fixed_test_set(num_samples, min_len, max_len, seed=42):
    rng = random.Random(seed)
    return [sample_seq_mixed(min_len, max_len, prob_has_b=0.7, rng=rng) for _ in range(num_samples)]


def evaluate_on_fixed_test(model, device, test_set):
    model.eval()
    exact = 0
    no_b = 0

    for seq in test_set:
        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]

        if pred == target:
            exact += 1
        if "b" not in pred:
            no_b += 1

    n = len(test_set)
    return {"exact_match": exact / n, "no_b_rate": no_b / n}


def evaluate_skip_b_behavior(model, device, num_samples=200, min_len=3, max_len=15):
    model.eval()

    exact = 0
    no_b = 0

    for _ in range(num_samples):
        seq = sample_seq_mixed(min_len, max_len, prob_has_b=0.7)

        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]

        if pred == target:
            exact += 1

        if "b" not in pred:
            no_b += 1

    return {
        "exact_match": exact / num_samples,
        "no_b_rate": no_b / num_samples,
    }


def compute_kl_penalty(model, ref_model, seq, device):
    """KL divergence penalty จาก reference SFT model เพื่อป้องกัน catastrophic forgetting"""
    tokens = ["<BOS>"] + seq + ["<SEP>"]
    ids = encode(tokens)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    pad_mask = x == PAD_ID
    with torch.no_grad():
        ref_logits = ref_model(x, pad_mask=pad_mask)
    logits = model(x, pad_mask=pad_mask)

    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()
    return kl


def rl_finetune(
    model,
    ref_model,
    device,
    min_len=2,
    max_len=10,
    num_steps=40000,
    rl_lr=1e-4,
    log_every=100,
    batch_size=4,
    grpo_g=4,
    kl_coef=0.1,
    entropy_coef=0.01,
    curriculum_steps=2000,
    checkpoint_path="best_mini_gpt_reverse_skip_b_rl.pth",
    model_config=None,
):
    """RL fine-tuning ด้วย GRPO + Curriculum Learning

    GRPO: ต่อ 1 prompt จะ rollout grpo_g ครั้ง แล้วคำนวณ advantage ภายใน group
    Curriculum: เริ่ม max_len เล็ก แล้วค่อยๆ เพิ่มจนถึง max_len จริง
    """
    model.train()
    ref_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rl_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=rl_lr * 0.1)

    best_exact_match = -1.0
    total_rollouts = batch_size * grpo_g

    for step in range(1, num_steps + 1):
        # Curriculum: เริ่มจาก min_len+1 แล้วค่อยๆ เพิ่มทุก curriculum_steps
        num_stages = max_len - (min_len + 1)
        cur_max_len = min(min_len + 1 + (step - 1) // curriculum_steps, max_len) if num_stages > 0 else max_len

        all_log_prob_sums = []
        all_advantages = []
        all_entropy_means = []
        all_kl_penalties = []
        mean_reward = 0.0
        last_seq = None
        last_generated = None

        for _ in range(batch_size):
            seq = sample_seq_mixed(min_len, cur_max_len, prob_has_b=0.7)

            # GRPO: rollout grpo_g ครั้งต่อ prompt เดียวกัน
            group_log_probs = []
            group_rewards = []
            group_entropies = []
            group_last_generated = None

            for _ in range(grpo_g):
                generated_tokens, log_prob_sum, entropy_mean = rollout_with_logprobs(model, seq, device)
                reward = compute_reward(seq, generated_tokens)
                group_log_probs.append(log_prob_sum)
                group_rewards.append(reward)
                group_entropies.append(entropy_mean)
                group_last_generated = generated_tokens

            # คำนวณ advantage ภายใน group (แทน EMA baseline)
            g_mean = sum(group_rewards) / grpo_g
            g_std = (sum((r - g_mean) ** 2 for r in group_rewards) / grpo_g) ** 0.5
            group_advantages = [(r - g_mean) / (g_std + 1e-8) for r in group_rewards]

            all_log_prob_sums.extend(group_log_probs)
            all_advantages.extend(group_advantages)
            all_entropy_means.extend(group_entropies)
            mean_reward += g_mean

            kl = compute_kl_penalty(model, ref_model, seq, device)
            all_kl_penalties.append(kl)

            last_seq = seq
            last_generated = group_last_generated

        mean_reward /= batch_size

        policy_loss = -sum(lp * adv for lp, adv in zip(all_log_prob_sums, all_advantages)) / total_rollouts
        kl_loss = kl_coef * sum(all_kl_penalties) / batch_size
        entropy_bonus = -entropy_coef * sum(all_entropy_means) / total_rollouts
        loss = policy_loss + kl_loss + entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            target = target_skip_b(last_seq)
            mean_kl = sum(k.item() for k in all_kl_penalties) / batch_size
            mean_entropy = sum(e.item() for e in all_entropy_means) / total_rollouts
            print(
                f"[RL] step={step} "
                f"cur_max_len={cur_max_len} "
                f"reward={mean_reward:.4f} "
                f"kl={mean_kl:.4f} "
                f"entropy={mean_entropy:.4f} "
                f"loss={loss.item():.4f} "
                f"sample={''.join(last_seq)} "
                f"target={' '.join(target)} "
                f"pred={' '.join(last_generated)}"
            )

            metrics = evaluate_skip_b_behavior(
                model=model,
                device=device,
                num_samples=100,
                min_len=min_len,
                max_len=max_len,
            )

            wandb.log(
                {
                    "rl/step": step,
                    "rl/reward": mean_reward,
                    "rl/cur_max_len": cur_max_len,
                    "rl/loss": loss.item(),
                    "rl/kl": mean_kl,
                    "rl/entropy": mean_entropy,
                    "rl/lr": optimizer.param_groups[0]["lr"],
                    "eval/exact_match_skip_b": metrics["exact_match"],
                    "eval/no_b_rate": metrics["no_b_rate"],
                }
            )

            if metrics["exact_match"] > best_exact_match:
                best_exact_match = metrics["exact_match"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "stoi": stoi,
                        "itos": itos,
                        "config": {
                            "min_len": min_len,
                            "max_len": max_len,
                            "d_model": (model_config or {}).get("d_model", 256),
                            "nhead": (model_config or {}).get("nhead", 8),
                            "num_layers": (model_config or {}).get("num_layers", 4),
                            "dim_ff": (model_config or {}).get("dim_ff", 512),
                            "dropout": (model_config or {}).get("dropout", 0.1),
                        },
                        "best_exact_match": best_exact_match,
                        "step": step,
                    },
                    checkpoint_path,
                )
                print(f"  -> saved best RL model (exact_match={best_exact_match:.4f})")

    return model


# =========================================================
# Main
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    sft_checkpoint_path = "best_mini_gpt_reverse.pth"
    rl_checkpoint_path = "best_mini_gpt_reverse_skip_b_rl.pth"

    checkpoint = torch.load(sft_checkpoint_path, map_location=device)
    base_config = checkpoint["config"]

    config = {
        "project_name": "mini-gpt-reverse-sequence-rl",
        "min_len": 2,
        "max_len": 6,
        "d_model": base_config["d_model"],
        "nhead": base_config["nhead"],
        "num_layers": base_config["num_layers"],
        "dim_ff": base_config["dim_ff"],
        "dropout": base_config["dropout"],
        "num_steps": 15000,
        "rl_lr": 3e-5,
        "rl_batch_size": 8,
        "grpo_g": 4,
        "kl_coef": 0.2,
        "entropy_coef": 0.005,
        "curriculum_steps": 3000,
        "log_every": 100,
        "device": str(device),
    }

    wandb.init(project=config["project_name"], config=config)
    cfg = wandb.config

    # Fixed test sets (seed=42) — สร้างครั้งเดียว ใช้ตลอด reproducible
    fixed_test_indist = make_fixed_test_set(500, cfg.min_len, cfg.max_len, seed=42)
    fixed_test_ood = make_fixed_test_set(500, 7, 15, seed=42)

    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=checkpoint["config"]["d_model"],
        nhead=checkpoint["config"]["nhead"],
        num_layers=checkpoint["config"]["num_layers"],
        dim_ff=checkpoint["config"]["dim_ff"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])

    # Reference model (frozen SFT) สำหรับ KL penalty
    ref_model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=checkpoint["config"]["d_model"],
        nhead=checkpoint["config"]["nhead"],
        num_layers=checkpoint["config"]["num_layers"],
        dim_ff=checkpoint["config"]["dim_ff"],
        dropout=0.0,
    ).to(device)
    ref_model.load_state_dict(checkpoint["model_state"])
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print("=== Before RL ===")
    before_words = ["tesbt", "abcde", "bbbbb", "ababa", "bomb", "table", "robot"]
    for word in before_words:
        seq = list(word)
        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]
        print(f"Input={word:>8} | SFT={''.join(pred):<12} | Target={''.join(target)}")

    before_metrics = evaluate_on_fixed_test(model, device, fixed_test_indist)
    print("Before RL metrics (fixed test):", before_metrics)

    wandb.log(
        {
            "before_rl/exact_match_skip_b": before_metrics["exact_match"],
            "before_rl/no_b_rate": before_metrics["no_b_rate"],
        }
    )

    model = rl_finetune(
        model=model,
        ref_model=ref_model,
        device=device,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        num_steps=cfg.num_steps,
        rl_lr=cfg.rl_lr,
        log_every=cfg.log_every,
        batch_size=cfg.rl_batch_size,
        grpo_g=cfg.grpo_g,
        kl_coef=cfg.kl_coef,
        entropy_coef=cfg.entropy_coef,
        curriculum_steps=cfg.curriculum_steps,
        checkpoint_path=rl_checkpoint_path,
        model_config={
            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_layers": cfg.num_layers,
            "dim_ff": cfg.dim_ff,
            "dropout": cfg.dropout,
        },
    )

    print("\n=== After RL ===")
    after_words = ["tesbt", "abcde", "bbbbb", "ababa", "bomb", "table", "robot"]
    for word in after_words:
        seq = list(word)
        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]
        print(f"Input={word:>8} | RL ={''.join(pred):<12} | Target={''.join(target)}")

    after_metrics = evaluate_on_fixed_test(model, device, fixed_test_indist)
    print("After RL metrics (fixed test, in-distribution):", after_metrics)

    wandb.log(
        {
            "after_rl/exact_match_skip_b": after_metrics["exact_match"],
            "after_rl/no_b_rate": after_metrics["no_b_rate"],
        }
    )

    # Generalization test — sequence ยาวกว่าที่ train (fixed, seed=42)
    print("\n=== Generalization test (len 7-15, fixed test) ===")
    gen_metrics = evaluate_on_fixed_test(model, device, fixed_test_ood)
    print("Generalization metrics:", gen_metrics)

    wandb.log(
        {
            "generalization/exact_match_skip_b": gen_metrics["exact_match"],
            "generalization/no_b_rate": gen_metrics["no_b_rate"],
        }
    )

    long_words = ["basketball", "abcdefghib", "bananabread", "robotbattle", "bbbbbbbbb"]
    for word in long_words:
        seq = list(word)
        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]
        print(f"Input={word:>14} | RL={''.join(pred):<14} | Target={''.join(target)}")

    # Manual test
    test_seq = list("tesbt")
    result = generate_reversed(model, test_seq, device)
    pred = extract_prediction(result)

    print("\nManual test")
    print("Input :", "".join(test_seq))
    print("Pred  :", "".join(pred))
    print("Target:", "".join([ch for ch in reversed(test_seq) if ch != "b"]))

    wandb.finish()


if __name__ == "__main__":
    main()
