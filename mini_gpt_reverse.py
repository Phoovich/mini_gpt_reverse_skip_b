import random

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset

from vocab import CHARS, PAD_ID, VOCAB_SIZE, stoi, itos, encode
from model import MiniGPT, generate_reversed


# =========================================================
# Dataset
# =========================================================
class ReverseSequenceDataset(Dataset):
    def __init__(self, num_samples=50000, min_len=3, max_len=15):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n = random.randint(self.min_len, self.max_len)
        seq = [random.choice(CHARS) for _ in range(n)]
        rev = list(reversed(seq))

        full_tokens = ["<BOS>"] + seq + ["<SEP>"] + rev + ["<EOS>"]
        token_ids = encode(full_tokens)

        x = token_ids[:-1]
        y = token_ids[1:]

        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    x_padded = torch.full((len(xs), max_len), PAD_ID, dtype=torch.long)
    y_padded = torch.full((len(ys), max_len), PAD_ID, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_padded[i, : len(x)] = x
        y_padded[i, : len(y)] = y

    return x_padded, y_padded


# =========================================================
# Helper functions
# =========================================================
def compute_loss(model, x, y, device):
    pad_mask = x == PAD_ID
    logits = model(x, pad_mask=pad_mask)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        y.reshape(-1),
        ignore_index=PAD_ID,
    )
    return loss


def train_one_epoch(model, loader, optimizer, epoch_idx, device):
    model.train()
    total_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        loss = compute_loss(model, x, y, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        global_step = epoch_idx * len(loader) + step
        wandb.log(
            {
                "train/batch_loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
        )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        loss = compute_loss(model, x, y, device)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def exact_match_accuracy(model, device, min_len, max_len, num_samples=200):
    model.eval()
    correct = 0

    for _ in range(num_samples):
        n = random.randint(min_len, max_len)
        seq = [random.choice(CHARS) for _ in range(n)]
        pred_tokens = generate_reversed(model, seq, device)

        try:
            sep_index = pred_tokens.index("<SEP>")
            generated_part = pred_tokens[sep_index + 1:]
        except ValueError:
            generated_part = []

        expected = list(reversed(seq)) + ["<EOS>"]

        if generated_part == expected:
            correct += 1

    return correct / num_samples


# =========================================================
# Main
# =========================================================
def main():
    config = {
        "num_train_samples": 50000,
        "num_val_samples": 5000,
        "num_test_samples": 5000,
        "min_len": 2,
        "max_len": 6,
        "batch_size": 64,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_ff": 512,
        "dropout": 0.1,
        "lr": 3e-4,
        "num_epochs": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "project_name": "mini-gpt-reverse-sequence",
    }

    wandb.init(project=config["project_name"], config=config)
    cfg = wandb.config

    device = torch.device(cfg.device)

    # Data
    train_ds = ReverseSequenceDataset(cfg.num_train_samples, cfg.min_len, cfg.max_len)
    val_ds = ReverseSequenceDataset(cfg.num_val_samples, cfg.min_len, cfg.max_len)
    test_ds = ReverseSequenceDataset(cfg.num_test_samples, cfg.min_len, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_model_path = "best_mini_gpt_reverse.pth"

    # Training
    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
        val_loss = evaluate_loss(model, val_loader, device)
        val_acc = exact_match_accuracy(model, device, cfg.min_len, cfg.max_len, num_samples=200)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_exact_match={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
            "val/exact_match": val_acc,
            "train/lr_epoch": current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": dict(cfg),
            }, best_model_path)

            wandb.log({"best/val_loss": best_val_loss, "best/epoch": epoch + 1})
            wandb.save(best_model_path)
            print("  -> saved best model")

    # Load best & final test
    checkpoint = torch.load(best_model_path, map_location=device)
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

    test_loss = evaluate_loss(model, test_loader, device)
    test_acc = exact_match_accuracy(model, device, cfg.min_len, cfg.max_len, num_samples=300)

    print(f"\nFinal test loss: {test_loss:.4f}")
    print(f"Final test exact match: {test_acc:.4f}")

    wandb.log({"test/loss": test_loss, "test/exact_match": test_acc})

    test_seq = list("moviethunpun")
    result = generate_reversed(model, test_seq, device)
    print("Input sequence :", test_seq)
    print("Generated tokens:", result)

    wandb.log({
        "examples/test_sequence": "".join(test_seq),
        "examples/generated_tokens": " ".join(result),
    })

    wandb.finish()


if __name__ == "__main__":
    main()
