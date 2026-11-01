"""Training Script for Multilingual Job Title Normalizer with MNRL and Hard Negatives."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import TitleDataGenerator, TitleSample
from .loss import HierarchicalMultiTaskLoss
from .model import HierarchicalTitleNormalizer


class TitleDataset(Dataset):
    """PyTorch Dataset yielding (raw_title, canonical_title, hard_negative, dep_idx, sen_idx)."""

    def __init__(
        self,
        samples: List[TitleSample],
        generator: TitleDataGenerator,
    ) -> None:
        self.samples = samples
        self.generator = generator

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        hard_negs = self.generator.mine_hard_negatives(s)
        neg_title = hard_negs[0] if hard_negs else "Intern"

        return {
            "raw_title": s.raw_title,
            "pos_title": s.canonical_title,
            "neg_title": neg_title,
            "dep_idx": s.department_idx,
            "sen_idx": s.seniority_idx,
        }


def collate_fn(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    raw_titles = [b["raw_title"] for b in batch]
    pos_titles = [b["pos_title"] for b in batch]
    neg_titles = [b["neg_title"] for b in batch]

    q_enc = tokenizer(raw_titles, padding=True, truncation=True, max_length=64, return_tensors="pt")
    p_enc = tokenizer(pos_titles, padding=True, truncation=True, max_length=64, return_tensors="pt")
    n_enc = tokenizer(neg_titles, padding=True, truncation=True, max_length=64, return_tensors="pt")

    dep_targets = torch.tensor([b["dep_idx"] for b in batch], dtype=torch.long)
    sen_targets = torch.tensor([b["sen_idx"] for b in batch], dtype=torch.long)

    return {
        "q_ids": q_enc["input_ids"],
        "q_mask": q_enc["attention_mask"],
        "p_ids": p_enc["input_ids"],
        "p_mask": p_enc["attention_mask"],
        "n_ids": n_enc["input_ids"],
        "n_mask": n_enc["attention_mask"],
        "dep_targets": dep_targets,
        "sen_targets": sen_targets,
    }


def train_title_normalizer(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    samples_per_role: int = 30,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 3e-5,
    output_dir: str = "checkpoints",
    device: str = "auto",
) -> Dict[str, Any]:
    """Train Hierarchical Multilingual Title Normalizer."""
    if device == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    print(f"[*] Training TitleNormalizer on device: {dev}")

    gen = TitleDataGenerator(seed=42)
    corpus = gen.generate_corpus(samples_per_role=samples_per_role)

    split_idx = int(len(corpus) * 0.85)
    train_samples = corpus[:split_idx]
    val_samples = corpus[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = TitleDataset(train_samples, gen)
    val_ds = TitleDataset(val_samples, gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    model = HierarchicalTitleNormalizer(model_name=model_name)
    model.to(dev)

    criterion = HierarchicalMultiTaskLoss(temperature=0.05, weight_mnrl=1.0, weight_dep=0.5, weight_sen=0.5)
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

        for batch in train_loader:
            q_ids, q_mask = batch["q_ids"].to(dev), batch["q_mask"].to(dev)
            p_ids, p_mask = batch["p_ids"].to(dev), batch["p_mask"].to(dev)
            n_ids, n_mask = batch["n_ids"].to(dev), batch["n_mask"].to(dev)
            dep_targets = batch["dep_targets"].to(dev)
            sen_targets = batch["sen_targets"].to(dev)

            optimizer.zero_grad()

            q_emb, dept_logits, sen_logits = model(q_ids, q_mask)
            p_emb = model.encode(p_ids, p_mask)
            n_emb = model.encode(n_ids, n_mask)

            loss, metrics = criterion(
                q_emb, p_emb, n_emb, dept_logits, sen_logits, dep_targets, sen_targets
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                q_ids, q_mask = batch["q_ids"].to(dev), batch["q_mask"].to(dev)
                p_ids, p_mask = batch["p_ids"].to(dev), batch["p_mask"].to(dev)
                n_ids, n_mask = batch["n_ids"].to(dev), batch["n_mask"].to(dev)
                dep_targets = batch["dep_targets"].to(dev)
                sen_targets = batch["sen_targets"].to(dev)

                q_emb, dept_logits, sen_logits = model(q_ids, q_mask)
                p_emb = model.encode(p_ids, p_mask)
                n_emb = model.encode(n_ids, n_mask)

                v_loss, _ = criterion(
                    q_emb, p_emb, n_emb, dept_logits, sen_logits, dep_targets, sen_targets
                )
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
            ckpt_path = os.path.join(output_dir, "best_title_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Saved new best model to {ckpt_path}")

    log_path = os.path.join(output_dir, "title_training_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    return {"best_val_loss": best_val_loss, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Job Title Normalizer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-role", type=int, default=20)
    args = parser.parse_args()

    train_title_normalizer(
        epochs=args.epochs, batch_size=args.batch_size, samples_per_role=args.samples_per_role
    )
