# MDPCR — training/dataset.py
# Loads and preprocesses the 5000 Q&A pairs.
# Each example becomes a (prompt_cloud, target_cloud) pair for training.

import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokeniser.vocab import Vocabulary
from tokeniser.context import ContextualEncoder
import config


class QAPairDataset(Dataset):
    """
    Dataset of (prompt_positions, prompt_densities,
                target_positions, target_densities) tensors.

    Each example is one Q&A pair where:
    - prompt = the question cloud
    - target = the answer cloud (what the model should converge toward)
    """

    def __init__(
        self,
        pairs   : list[dict],
        encoder : ContextualEncoder,
        device  : str = config.DEVICE,
    ):
        self.encoder = encoder
        self.device  = device
        self.examples : list[dict] = []

        print(f"Encoding {len(pairs)} Q&A pairs into clouds...")
        skipped = 0

        for pair in pairs:
            try:
                p_pos, p_den, p_toks = encoder.encode_text(pair["q"], device)
                t_pos, t_den, t_toks = encoder.encode_text(pair["a"], device)

                if len(p_toks) == 0 or len(t_toks) == 0:
                    skipped += 1
                    continue

                self.examples.append({
                    "prompt_positions"  : p_pos.detach(),
                    "prompt_densities"  : p_den.detach(),
                    "target_positions"  : t_pos.detach(),
                    "target_densities"  : t_den.detach(),
                    "prompt_text"       : pair["q"],
                    "target_text"       : pair["a"],
                })
            except Exception as e:
                skipped += 1

        print(f"Encoded {len(self.examples)} examples, skipped {skipped}.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: pads clouds to the same number of points within a batch.
    Clouds can have different numbers of points (different text lengths).
    """
    def pad_to_max(tensors: list[torch.Tensor], pad_val: float = 0.0):
        max_len = max(t.shape[0] for t in tensors)
        padded  = []
        masks   = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if t.dim() == 2:
                # positions: (M, N) → pad rows
                p = torch.cat([t, torch.full((pad_len, t.shape[1]), pad_val,
                                              device=t.device)], dim=0)
            else:
                # densities: (M,) → pad scalars
                p = torch.cat([t, torch.full((pad_len,), pad_val,
                                              device=t.device)], dim=0)
            padded.append(p)
            masks.append(torch.cat([
                torch.ones(t.shape[0], dtype=torch.bool, device=t.device),
                torch.zeros(pad_len,   dtype=torch.bool, device=t.device),
            ]))
        return torch.stack(padded), torch.stack(masks)

    pp, pm = pad_to_max([b["prompt_positions"] for b in batch])
    pd, _  = pad_to_max([b["prompt_densities"] for b in batch], pad_val=1e-4)
    tp, tm = pad_to_max([b["target_positions"] for b in batch])
    td, _  = pad_to_max([b["target_densities"] for b in batch], pad_val=1e-4)

    return {
        "prompt_positions" : pp,   # (B, M_p, N)
        "prompt_densities" : pd,   # (B, M_p)
        "prompt_mask"      : pm,   # (B, M_p)
        "target_positions" : tp,   # (B, M_t, N)
        "target_densities" : td,   # (B, M_t)
        "target_mask"      : tm,   # (B, M_t)
    }


def build_dataloaders(
    pairs    : list[dict],
    encoder  : ContextualEncoder,
    device   : str = config.DEVICE,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from raw pairs."""
    dataset   = QAPairDataset(pairs, encoder, device)
    n_train   = int(len(dataset) * config.TRAIN_SPLIT)
    n_val     = len(dataset) - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
    )
    return train_loader, val_loader


def load_pairs(path: str) -> list[dict]:
    """Load Q&A pairs from JSON file."""
    with open(path) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} Q&A pairs from {path}")
    return pairs
