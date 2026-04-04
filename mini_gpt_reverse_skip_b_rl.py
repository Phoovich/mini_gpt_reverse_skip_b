import math
import random
import string

import torch
import torch.nn as nn

import wandb

# =========================================================
# Vocabulary
# =========================================================
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


# =========================================================
# Model
# =========================================================
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


# =========================================================
# Inference helpers
# =========================================================
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
        max_new_tokens = len(seq) + 2

    sampled_ids = []
    log_probs = []

    for _ in range(max_new_tokens):
        pad_mask = x == PAD_ID
        logits = model(x, pad_mask=pad_mask)
        next_token_logits = logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        next_id = dist.sample()
        log_prob = dist.log_prob(next_id)

        sampled_ids.append(next_id.item())
        log_probs.append(log_prob)

        x = torch.cat([x, next_id.unsqueeze(1)], dim=1)

        if next_id.item() == EOS_ID:
            break

    output_tokens = decode(sampled_ids)
    log_prob_sum = torch.stack(log_probs).sum()

    return output_tokens, log_prob_sum


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

    # 2) partial token match
    min_len = min(len(pred), len(target))
    matches = sum(1 for i in range(min_len) if pred[i] == target[i])
    reward += 0.2 * matches / max(len(target), 1)

    # 3) ลงโทษตัว b แรง
    num_b = pred.count("b")
    reward -= 1.0 * num_b

    # 4) ความยาวผิดลงโทษ
    reward -= 0.1 * abs(len(pred) - len(target))

    # 5) ถ้าไม่มี EOS ลงโทษเพิ่ม
    if "<EOS>" not in generated_tokens:
        reward -= 0.5

    return reward


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


def sample_seq_mixed(min_len, max_len, prob_has_b=0.7):
    n = random.randint(min_len, max_len)
    seq = [random.choice(CHARS) for _ in range(n)]

    if random.random() < prob_has_b:
        b_count = random.randint(1, max(1, n // 2))
        positions = random.sample(range(n), k=b_count)
        for pos in positions:
            seq[pos] = "b"

    return seq


def rl_finetune(
    model,
    device,
    min_len=2,
    max_len=6,
    num_steps=20000,
    rl_lr=1e-4,
    log_every=100,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rl_lr)

    reward_baseline = 0.0
    beta = 0.9

    for step in range(1, num_steps + 1):
        n = random.randint(min_len, max_len)
        seq = [random.choice(CHARS) for _ in range(n)]

        generated_tokens, log_prob_sum = rollout_with_logprobs(model, seq, device)
        reward = compute_reward(seq, generated_tokens)

        reward_baseline = beta * reward_baseline + (1 - beta) * reward
        advantage = reward - reward_baseline

        loss = -log_prob_sum * advantage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            target = target_skip_b(seq)
            print(
                f"[RL] step={step} "
                f"reward={reward:.4f} "
                f"baseline={reward_baseline:.4f} "
                f"adv={advantage:.4f} "
                f"loss={loss.item():.4f} "
                f"sample={''.join(seq)} "
                f"target={' '.join(target)} "
                f"pred={' '.join(generated_tokens)}"
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
                    "rl/reward": reward,
                    "rl/baseline": reward_baseline,
                    "rl/advantage": advantage,
                    "rl/loss": loss.item(),
                    "rl/lr": optimizer.param_groups[0]["lr"],
                    "eval/exact_match_skip_b": metrics["exact_match"],
                    "eval/no_b_rate": metrics["no_b_rate"],
                }
            )

    return model


# =========================================================
# Main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "num_steps": 10000,
        "rl_lr": 5e-5,
        "log_every": 100,
        "device": str(device),
    }

    wandb.init(project=config["project_name"], config=config)
    cfg = wandb.config

    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=checkpoint["config"]["d_model"],
        nhead=checkpoint["config"]["nhead"],
        num_layers=checkpoint["config"]["num_layers"],
        dim_ff=checkpoint["config"]["dim_ff"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])

    print("=== Before RL ===")
    before_words = ["tesbt", "abcde", "bbbbb", "ababa", "bomb", "table", "robot"]
    for word in before_words:
        seq = list(word)
        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]
        print(f"Input={word:>8} | SFT={''.join(pred):<12} | Target={''.join(target)}")

    before_metrics = evaluate_skip_b_behavior(
        model=model,
        device=device,
        num_samples=200,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
    )
    print("Before RL metrics:", before_metrics)

    wandb.log(
        {
            "before_rl/exact_match_skip_b": before_metrics["exact_match"],
            "before_rl/no_b_rate": before_metrics["no_b_rate"],
        }
    )

    model = rl_finetune(
        model=model,
        device=device,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        num_steps=cfg.num_steps,
        rl_lr=cfg.rl_lr,
        log_every=cfg.log_every,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": dict(cfg),
        },
        rl_checkpoint_path,
    )

    print("\n=== After RL ===")
    after_words = ["tesbt", "abcde", "bbbbb", "ababa", "bomb", "table", "robot"]
    for word in after_words:
        seq = list(word)
        result = generate_reversed(model, seq, device)
        pred = extract_prediction(result)
        target = [ch for ch in reversed(seq) if ch != "b"]
        print(f"Input={word:>8} | RL ={''.join(pred):<12} | Target={''.join(target)}")

    after_metrics = evaluate_skip_b_behavior(
        model=model,
        device=device,
        num_samples=300,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
    )

    print("After RL metrics:", after_metrics)

    wandb.log(
        {
            "after_rl/exact_match_skip_b": after_metrics["exact_match"],
            "after_rl/no_b_rate": after_metrics["no_b_rate"],
        }
    )

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
