# MiniGPT: Sequence Reversal with Reinforcement Learning

Mini-project for **AI Frontiers (Year 2, Semester 2, CEDT)**.  
Two-stage training of a Transformer decoder to (1) reverse character sequences and (2) skip the letter **"b"** in the reversed output using GRPO reinforcement learning.

---

## Demo & Visualizations

| Resource | Description |
|---|---|
| [`teaching-site/index.html`](teaching-site/index.html) | Interactive explainer — architecture, SFT, RL, results |
| [`embedding_viz.html`](embedding_viz.html) | Token embeddings & positional encoding visualization |
| [`viz.html`](viz.html) | Training curves & reward landscape |
| [`report.pdf`](report.pdf) | Full project report (Thai) |

---

## Task Definition

**Stage 1 — Supervised Fine-Tuning (SFT)**

Given a sequence of lowercase English letters, train the model to reverse it:

```
Input:  h e l l o
Output: o l l e h
```

Full sequence format: `<BOS> h e l l o <SEP> o l l e h <EOS>`

**Stage 2 — Reinforcement Learning (GRPO)**

Fine-tune with GRPO + KL penalty to additionally skip the letter **"b"**:

```
Input:  a b c d e
Output: e d c a        ← "b" is omitted
```

---

## Architecture

The model is a **Transformer Encoder with a causal mask** (equivalent to a decoder), implemented in [`model.py`](model.py).

| Hyperparameter | Value |
|---|---|
| `d_model` | 256 |
| `nhead` | 8 |
| `num_layers` | 4 |
| `dim_feedforward` | 512 |
| Vocabulary size | 30 (4 special + 26 letters) |

---

## Files

```
.
├── model.py                            # MiniGPT class + greedy decode
├── vocab.py                            # Fixed vocabulary (30 tokens)
├── mini_gpt_reverse.py                 # Stage 1: SFT training loop
├── mini_gpt_reverse_skip_b_rl.py       # Stage 2: GRPO RL training
├── evaluate_model.py                   # Comprehensive evaluation
├── load_model_mini_gpt_reverse.py      # Quick SFT inference
├── load_model_mini_gpt_reverse_skip_b_rl.py  # Quick RL inference
├── demo.py                             # Interactive demo
├── best_mini_gpt_reverse.pth           # SFT checkpoint
├── best_mini_gpt_reverse_skip_b_rl.pth # RL checkpoint (step 9,400)
├── report.typ / report.pdf             # Project report
├── teaching-site/                      # Interactive web explainer
├── embedding_viz.html                  # Embedding visualization
└── viz.html                            # Training visualization
```

---

## Training Details

### SFT (Stage 1)

| Setting | Value |
|---|---|
| Training samples | 50,000 |
| Sequence length | 2–10 characters |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 3e-4 (ReduceLROnPlateau) |
| Gradient clipping | max norm = 1.0 |

### RL (Stage 2 — GRPO + KL Penalty)

| Setting | Value |
|---|---|
| Training steps | 15,000 |
| Sequence length | 2–6 characters |
| RL learning rate | 3e-5 |
| GRPO group size (`G`) | 4 |
| KL coefficient | 0.2 |
| Entropy coefficient | 0.005 |
| Curriculum steps | 3,000 |
| Prob. of "b" in sequence | 70% |

**Reward function** (`compute_reward`):

| Signal | Value |
|---|---|
| Exact match | +5.0 |
| Per-position match | +0.2 each |
| Character coverage | +0.1 each |
| "b" in output (penalty) | −1.0 each |
| Length mismatch | −0.1 per char |
| Missing `<EOS>` | −0.5 |
| PAD tokens in output | −2.0 each |
| Unexpected `<SEP>`/`<BOS>` | −2.0 each |

---

## Evaluation Protocol

Always evaluated with seed 42.

| Setting | Samples |
|---|---|
| In-distribution (len 2–6) | 500 |
| Out-of-distribution (len 7–15) | 500 |
| Per-length breakdown (len 2–15) | 1,000 / length |

Metrics: `exact_match` and `no_b_rate` (fraction of outputs with zero "b"s).

---

## Setup & Usage

**Prerequisites:** Python ≥ 3.11.13, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Train SFT model (requires wandb login)
wandb login
python mini_gpt_reverse.py

# Train RL model (loads SFT checkpoint)
python mini_gpt_reverse_skip_b_rl.py

# Evaluate both models
python evaluate_model.py

# Quick inference
python load_model_mini_gpt_reverse.py           # SFT
python load_model_mini_gpt_reverse_skip_b_rl.py # RL
```

Checkpoints (`*.pth`) are included in the repo — you can run inference and evaluation without retraining.

---

## Known Limitations

- SFT is trained on lengths 2–10; RL fine-tunes only on 2–6 → OOD performance degrades beyond length 6
- Model occasionally regresses on reversal accuracy after RL fine-tuning
- `ReverseSequenceDataset` generates samples on-the-fly (no fixed seed per SFT sample)

---

## wandb Projects

- SFT: [`mini-gpt-reverse-sequence`](https://wandb.ai/phoovich/mini-gpt-reverse-sequence)
- RL: [`mini-gpt-reverse-sequence-rl`](https://wandb.ai/phoovich/mini-gpt-reverse-sequence-rl)
