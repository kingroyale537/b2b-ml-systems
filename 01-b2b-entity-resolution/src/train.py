"""Training Pipeline for B2B Company Bi-Encoder with Triplet Margin Loss.

Trains a lightweight Transformer Bi-Encoder on corporate triplet representations
with online hard negative mining and AdamW + Cosine Annealing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import B2BDataGenerator, CompanyRecord
from .model import CompanyBiEncoder, TripletMarginLossWithHardNegatives


class TripletCompanyDataset(Dataset):
    """PyTorch Dataset yielding tokenized (anchor, positive, negative) triples."""

    def __init__(
        self,
        triplets: List[Tuple[CompanyRecord, CompanyRecord, CompanyRecord]],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor, pos, neg = self.triplets[idx]

        a_enc = self.tokenizer(
            anchor.to_representation(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        p_enc = self.tokenizer(
            pos.to_representation(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        n_enc = self.tokenizer(
            neg.to_representation(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "a_ids": a_enc["input_ids"].squeeze(0),
            "a_mask": a_enc["attention_mask"].squeeze(0),
            "p_ids": p_enc["input_ids"].squeeze(0),
            "p_mask": p_enc["attention_mask"].squeeze(0),
            "n_ids": n_enc["input_ids"].squeeze(0),
            "n_mask": n_enc["attention_mask"].squeeze(0),
        }


def train_biencoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_canonical: int = 100,
    num_triplets: int = 1000,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    margin: float = 0.35,
    output_dir: str = "checkpoints",
    device: str = "auto",
) -> Dict[str, Any]:
    """Train the Bi-Encoder on synthesized enterprise corporate records."""
    if device == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    print(f"[*] Training CompanyBiEncoder on device: {dev}")

    # Generate synthetic training & validation data
    gen = B2BDataGenerator(seed=42)
    records, clusters = gen.generate_corpus(
        num_canonical=num_canonical, duplicates_per_entity=4
    )
    triplets = gen.generate_triplets(records, num_triplets=num_triplets, hard_negative_prob=0.5)

    split_idx = int(len(triplets) * 0.85)
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = TripletCompanyDataset(train_triplets, tokenizer)
    val_ds = TripletCompanyDataset(val_triplets, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CompanyBiEncoder(model_name=model_name)
    model.to(dev)

    criterion = TripletMarginLossWithHardNegatives(margin=margin, distance_metric="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            a_ids, a_mask = batch["a_ids"].to(dev), batch["a_mask"].to(dev)
            p_ids, p_mask = batch["p_ids"].to(dev), batch["p_mask"].to(dev)
            n_ids, n_mask = batch["n_ids"].to(dev), batch["n_mask"].to(dev)

            optimizer.zero_grad()
            emb_a = model.encode(a_ids, a_mask)
            emb_p = model.encode(p_ids, p_mask)
            emb_n = model.encode(n_ids, n_mask)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                a_ids, a_mask = batch["a_ids"].to(dev), batch["a_mask"].to(dev)
                p_ids, p_mask = batch["p_ids"].to(dev), batch["p_mask"].to(dev)
                n_ids, n_mask = batch["n_ids"].to(dev), batch["n_mask"].to(dev)

                emb_a = model.encode(a_ids, a_mask)
                emb_p = model.encode(p_ids, p_mask)
                emb_n = model.encode(n_ids, n_mask)

                v_loss = criterion(emb_a, emb_p, emb_n)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        duration = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Time: {duration:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "duration_sec": duration,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(output_dir, "best_biencoder.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Saved new best model checkpoint to {ckpt_path}")

    # Write training log
    log_path = os.path.join(output_dir, "training_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    return {"best_val_loss": best_val_loss, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Company Bi-Encoder Matcher")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--triplets", type=int, default=500)
    args = parser.parse_args()

    train_biencoder(epochs=args.epochs, batch_size=args.batch_size, num_triplets=args.triplets)
