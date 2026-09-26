"""
data.py — everything related to data

Responsibilities:
  - Download the Shakespeare text
  - Build the character vocabulary
  - Parse text by speaking role (McMahan et al. 2017 style)
  - Per-role 80/20 line split (chronological, matching the paper)
  - CharDataset: sliding window (x, y, client_id) pairs
  - Aggregate all roles into centralized train/test loaders
"""

import os
import re
import random
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ── Download ──────────────────────────────────────────────────────────────────

def download_shakespeare(path="shakespeare.txt"):
    """Download The Complete Works of Shakespeare from Project Gutenberg."""
    if os.path.exists(path):
        print(f"[data] found '{path}' locally")
        return path
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    print(f"[data] downloading Shakespeare...")
    urllib.request.urlretrieve(url, path)
    print(f"[data] saved to '{path}'")
    return path


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(text):
    """
    Build character-level vocabulary from raw text.
    Returns:
        char2idx : dict  char -> int
        idx2char : dict  int  -> char
    """
    vocab    = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


# ── Role parsing ──────────────────────────────────────────────────────────────

# Speaker lines in Gutenberg Shakespeare look like:
#   "  HAMLET." or "  FIRST WITCH." — 2-space indent, ALL CAPS, ends with period
_SPEAKER_RE = re.compile(r'^([A-Z][A-Z\s\.\'\-]{1,40})\.\s*$', re.MULTILINE)

def parse_roles(text):
    """
    Parse raw Shakespeare text into per-role dialogue blocks.

    Each match of _SPEAKER_RE marks a new speaker; everything between
    that match and the next speaker match is that speaker's dialogue.

    Returns:
        roles : dict  {role_name: [dialogue_block, ...]}
                Only roles with >= 2 dialogue blocks are kept (paper criterion).
        role2idx : dict  {role_name: int}  stable sorted mapping
    """
    matches = list(_SPEAKER_RE.finditer(text))
    roles   = {}

    for i, match in enumerate(matches):
        role      = match.group(1).strip()
        start     = match.end()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        dialogue  = text[start:end].strip()
        if dialogue:
            roles.setdefault(role, []).append(dialogue)

    # paper criterion: keep roles with at least 2 lines
    roles    = {r: lines for r, lines in roles.items() if len(lines) >= 2}
    role2idx = {r: i for i, r in enumerate(sorted(roles))}
    return roles, role2idx


# ── Dataset ───────────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    """
    Sliding window over encoded text for one speaking role.

    For every position i in the text:
        x         = chars[i     : i + seq_len]   <- input sequence
        y         = chars[i + 1 : i + seq_len+1] <- target (shifted by 1)
        client_id = scalar tensor identifying the speaking role

    client_id is metadata — training ignores it, but it's available for
    federated simulation, per-role analysis, etc.
    """
    def __init__(self, text, seq_len, char2idx, client_id=0):
        self.seq_len   = seq_len
        self.client_id = torch.tensor(client_id, dtype=torch.long)
        encoded        = [char2idx[c] for c in text if c in char2idx]
        self.data      = torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y, self.client_id


# ── Per-role splits ───────────────────────────────────────────────────────────

def build_role_datasets(roles, role2idx, seq_len, char2idx, train_frac=0.8):
    """
    For each role, split its dialogue blocks chronologically (first 80% train,
    last 20% test — at least 1 block in test), then build a CharDataset for
    each split tagged with that role's client_id.

    Returns:
        train_datasets : list of CharDataset (one per role, train split)
        test_datasets  : list of CharDataset (one per role, test split)
    """
    train_datasets, test_datasets = [], []

    for role, lines in roles.items():
        client_id  = role2idx[role]
        split      = max(1, int(len(lines) * train_frac))
        train_text = "\n".join(lines[:split])
        test_text  = "\n".join(lines[split:])

        train_ds = CharDataset(train_text, seq_len, char2idx, client_id)
        test_ds  = CharDataset(test_text,  seq_len, char2idx, client_id)

        # only include if enough chars to form at least one sample
        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        if len(test_ds) > 0:
            test_datasets.append(test_ds)

    return train_datasets, test_datasets


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_loaders(train_datasets, test_datasets, batch_size, seed=42):
    """
    Aggregate per-role datasets and return (train_loader, test_loader).
    Train loader shuffles across all roles; test loader preserves order.
    num_workers=4 + pin_memory overlap CPU prep with GPU compute.
    worker_init_fn seeds each worker for reproducibility.
    """
    train_ds = ConcatDataset(train_datasets)
    test_ds  = ConcatDataset(test_datasets)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader


# ── Convenience: load everything in one call ──────────────────────────────────

def load_data(seq_len, batch_size, train_frac=0.8, path="shakespeare.txt", seed=42):
    """
    Full pipeline: download -> vocab -> parse by role -> per-role splits
                   -> aggregate -> loaders.

    Returns: train_loader, test_loader, char2idx, idx2char, vocab_size, role2idx
    """
    path = download_shakespeare(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    char2idx, idx2char = build_vocab(text)
    roles, role2idx    = parse_roles(text)

    train_datasets, test_datasets = build_role_datasets(
        roles, role2idx, seq_len, char2idx, train_frac
    )

    train_chars = sum(len(ds.data) for ds in train_datasets)
    test_chars  = sum(len(ds.data) for ds in test_datasets)
    train_samples = sum(len(ds) for ds in train_datasets)
    test_samples  = sum(len(ds) for ds in test_datasets)

    print(f"[data] total chars   : {len(text):,}")
    print(f"[data] roles found   : {len(roles):,}  (>= 2 lines)")
    print(f"[data] train chars   : {train_chars:,}")
    print(f"[data] test  chars   : {test_chars:,}")
    print(f"[data] vocab size    : {len(char2idx)}")
    print(f"[data] train samples : {train_samples:,}")
    print(f"[data] test  samples : {test_samples:,}")

    train_loader, test_loader = get_loaders(
        train_datasets, test_datasets, batch_size, seed=seed
    )
    return train_loader, test_loader, char2idx, idx2char, len(char2idx), role2idx
