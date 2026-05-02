# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-stage training of a Transformer Decoder (MiniGPT) for sequence reversal:

1. **SFT** (`mini_gpt_reverse.py`): Train to reverse character sequences (length 2–10)
2. **RL** (`mini_gpt_reverse_skip_b_rl.py`): Fine-tune with GRPO to skip letter "b" in reversed output (length 2–6)

Sequence format: `<BOS> h e l l o <SEP> o l l e h <EOS>`

---

## Commands

### Environment setup

```bash
uv sync  # requires Python >= 3.11.13
```

### Train SFT model

```bash
python mini_gpt_reverse.py
```

Saves checkpoint to `checkpoints/best_mini_gpt_reverse.pth` (saved when val_loss improves). Requires wandb login (`wandb login`). wandb project: `mini-gpt-reverse-sequence`.

### Train RL model

```bash
python mini_gpt_reverse_skip_b_rl.py
```

Loads `checkpoints/best_mini_gpt_reverse.pth`, saves RL checkpoint to `checkpoints/best_mini_gpt_reverse_skip_b_rl.pth` (saved when exact_match improves). wandb project: `mini-gpt-reverse-sequence-rl`.

### Evaluate both models

```bash
python evaluate_model.py
```

Runs sanity check, in-dist eval (len 2–6), OOD eval (len 7–15), per-length breakdown, and edge cases. This script imports helpers from `mini_gpt_reverse_skip_b_rl.py`.

### Quick inference (SFT)

```bash
python load_model_mini_gpt_reverse.py
```

### Quick inference (RL)

```bash
python load_model_mini_gpt_reverse_skip_b_rl.py
```

---

## Architecture

### Files

- `model.py` — `MiniGPT` class + `generate_reversed()` (greedy, for inference) + `extract_prediction()`
- `vocab.py` — fixed vocabulary: 4 special tokens (`<PAD>`, `<BOS>`, `<SEP>`, `<EOS>`) + 26 lowercase letters (vocab size = 30)
- `mini_gpt_reverse.py` — SFT training loop with wandb logging
- `mini_gpt_reverse_skip_b_rl.py` — RL helpers (`compute_reward`, `rollout_with_logprobs`, `rl_finetune`) and main; also exports `evaluate_on_fixed_test`, `make_fixed_test_set` used by `evaluate_model.py`
- `evaluate_model.py` — comprehensive evaluation script
- `load_model_mini_gpt_reverse.py` — standalone SFT inference script
- `load_model_mini_gpt_reverse_skip_b_rl.py` — standalone RL inference script

### MiniGPT model

Despite the name "GPT", the model uses `nn.TransformerEncoder` (not Decoder) with a manually applied causal mask (`torch.triu`). Architecture defaults: `d_model=256`, `nhead=8`, `num_layers=4`, `dim_ff=512`.

### Inference vs. RL rollout

Two decoding paths exist:
- `generate_reversed()` — greedy argmax, used in all evaluation and inference scripts
- `rollout_with_logprobs()` — categorical sampling, used only during RL training to collect trajectories with log-probabilities

### SFT training

Key defaults: `num_train_samples=50000`, `batch_size=64`, `lr=3e-4`, `num_epochs=40`, `max_len=10`. Uses `ReduceLROnPlateau` (factor=0.5, patience=2) and gradient clipping (max norm=1.0).

### RL method (GRPO + KL penalty)

Key defaults: `num_steps=15000`, `rl_lr=3e-5`, `batch_size=8`, `grpo_g=4`, `kl_coef=0.2`, `entropy_coef=0.005`, `curriculum_steps=3000`.

- **GRPO**: For each prompt, roll out `grpo_g` times; compute advantage within the group (normalized by group mean/std)
- **KL penalty**: Penalizes divergence from frozen SFT reference model to prevent catastrophic forgetting
- **Curriculum learning**: `cur_max_len` grows from `min_len+1` to `max_len` over `curriculum_steps` steps
- **Sequence sampling**: 70% of training sequences contain at least one "b" (`prob_has_b=0.7` in `sample_seq_mixed`)
- **Reward** (`compute_reward`): exact match (+5.0), positional match (+0.2×), character coverage (+0.1×), "b" penalty (−1.0×count), length mismatch (−0.1×), no EOS (−0.5), PAD tokens (−2.0×count), unexpected `<SEP>`/`<BOS>` in output (−2.0×count)

### Checkpoints

Both `.pth` files are stored under `checkpoints/` and contain `{"model_state", "stoi", "itos", "config"}`. The RL checkpoint also stores `"best_exact_match"` and `"step"`. SFT checkpoint is saved on best val_loss; RL checkpoint is saved on best exact_match.

---

## Ground Rules

- For any file search or grep in this repo, use `fff` MCP tools (not manual grep/find).
- Do NOT modify `exact_match` or `no_b_rate` metric definitions without permission.
- Always use the correct checkpoint per stage (SFT vs RL).
- Training uses wandb — ensure `wandb` is configured before running training scripts.

---

## Evaluation Protocol

When evaluating, always report:

| Setting                        | Samples    | Seed |
| ------------------------------ | ---------- | ---- |
| In-distribution                | 500        | 42   |
| Out-of-distribution (len 7–15) | 500        | 42   |
| Per-length (len 2–15)          | 100/length | 42   |

Metrics: `exact_match` and `no_b_rate`. Use `evaluate_model.py` — do not invent new evaluation logic.

---

## Known Limitations

- SFT is trained on lengths 2–10, but RL fine-tunes only on lengths 2–6; OOD performance degrades sharply beyond length 6
- Model may learn to skip "b" but regress on reversal accuracy
- `ReverseSequenceDataset` generates samples on-the-fly (no fixed seed per sample in SFT)
