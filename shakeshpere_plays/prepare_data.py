# -*- coding: utf-8 -*-
"""
prepare_data.py - prepare train/val/test/retain/forget splits

Saves raw text files per split (no pre-sliced tensors).
Sliding window sampling happens at load time in data_loader.py.

Output layout (itr=1):
  data_splits_itr1/
    retain.txt, train.txt, val.txt, test.txt, full.txt
    forget/
      forget.txt
    meta.json

Output layout (itr=N):
  data_splits_itrN/
    retain.txt, train.txt, val.txt, test.txt, full.txt
    forget/
      forget_0.txt ... forget_{N-1}.txt
    meta.json

  forget_*.txt are built by shuffling all forget dialogue blocks
  (natural speaker units) and splitting into N equal chunks.
"""

import os
import re
import json
import numpy as np
from pathlib import Path


def download_shakespeare(path="shakespeare.txt"):
    """Download if not exists."""
    if os.path.exists(path):
        print(f"[prepare] found '{path}' locally")
        return path
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    print(f"[prepare] downloading Shakespeare...")
    import urllib.request
    urllib.request.urlretrieve(url, path)
    print(f"[prepare] saved to '{path}'")
    return path


def build_vocab(text):
    """Return sorted list of unique characters."""
    return sorted(set(text))


_SPEAKER_RE = re.compile(r'^([A-Z][A-Z\s\.\'\-]{1,40})\.\s*$', re.MULTILINE)


def parse_roles(text):
    """Parse text by role; collect dialogue blocks per role."""
    matches = list(_SPEAKER_RE.finditer(text))
    roles = {}

    for i, match in enumerate(matches):
        role = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        dialogue = text[start:end].strip()
        if dialogue:
            roles.setdefault(role, []).append(dialogue)

    n_before = len(roles)
    roles = {r: lines for r, lines in roles.items() if len(lines) >= 2}
    n_after = len(roles)

    print(f"[parse] total speaker mentions: {len(matches):,}")
    print(f"[parse] unique speakers:        {n_before:,}")
    print(f"[parse] speakers with >= 2 blocks: {n_after:,} (filtered {n_before - n_after:,})")

    role2idx = {r: i for i, r in enumerate(sorted(roles))}
    return roles, role2idx


def split_role_data(roles):
    """Split each role's dialogue chronologically (80/10/10).

    Returns:
        train_by_role       : role -> joined text string
        train_blocks_by_role: role -> list of raw dialogue blocks (for shuffled splitting)
        val_by_role         : role -> joined text string
        test_by_role        : role -> joined text string
    """
    train_by_role = {}
    train_blocks_by_role = {}
    val_by_role = {}
    test_by_role = {}

    for role, lines in roles.items():
        n = len(lines)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_lines = lines[:n_train]
        val_lines = lines[n_train : n_train + n_val]
        test_lines = lines[n_train + n_val:]

        if train_lines:
            train_by_role[role] = "\n".join(train_lines)
            train_blocks_by_role[role] = train_lines
        if val_lines:
            val_by_role[role] = "\n".join(val_lines)
        if test_lines:
            test_by_role[role] = "\n".join(test_lines)

    return train_by_role, train_blocks_by_role, val_by_role, test_by_role


def split_retain_forget(train_by_role, forget_frac=0.1, seed=42):
    """Partition training roles into retain/forget greedily by character count."""
    rng = np.random.RandomState(seed)

    roles = list(train_by_role.keys())
    total_chars = sum(len(t) for t in train_by_role.values())
    target_forget = int(forget_frac * total_chars)

    shuffled = rng.permutation(roles).tolist()

    forget_roles = []
    forget_chars = 0
    for role in shuffled:
        n = len(train_by_role[role])
        if forget_chars + n <= target_forget:
            forget_roles.append(role)
            forget_chars += n

    retain_roles = [r for r in roles if r not in forget_roles]
    retain_chars = sum(len(train_by_role[r]) for r in retain_roles)

    actual_forget_pct = 100.0 * forget_chars / total_chars
    print(f"[prepare] retain roles: {len(retain_roles):,} ({100*len(retain_roles)/len(roles):.1f}%)")
    print(f"[prepare] forget roles: {len(forget_roles):,} ({100*len(forget_roles)/len(roles):.1f}%)")
    print(f"[prepare] retain chars: {retain_chars:,} ({100 - actual_forget_pct:.2f}%)")
    print(f"[prepare] forget chars: {forget_chars:,} ({actual_forget_pct:.2f}%)")

    return retain_roles, forget_roles


def make_forget_splits(forget_roles, train_blocks_by_role, forget_splits, seed):
    """Shuffle forget dialogue blocks and split into forget_splits chunks.

    Shuffling is at the dialogue-block level (one speaker's continuous
    speech) so within-block coherence is preserved while temporal/
    contextual order across blocks is broken.

    Uses seed+1 so the shuffle is independently reproducible from the
    retain/forget role assignment (which uses seed).

    Returns list of forget_splits text strings (forget_0 ... forget_{N-1}).
    """
    rng = np.random.RandomState(seed + 1)

    all_blocks = []
    for role in forget_roles:
        all_blocks.extend(train_blocks_by_role[role])

    rng.shuffle(all_blocks)

    chunks = []
    n = len(all_blocks)
    for i in range(forget_splits):
        start = (i * n) // forget_splits
        end = ((i + 1) * n) // forget_splits
        chunks.append("\n".join(all_blocks[start:end]))

    return chunks


def prepare_data(
    path="shakespeare.txt",
    seq_len=80,
    seed=42,
    output_dir="data_splits",
    forget_splits=1,
    num_speakers=None,
):
    """Full pipeline: parse -> split -> save as txt files + meta.json."""
    suffix = f"_speakers{num_speakers}" if num_speakers is not None else ""
    suffix += f"_fs{forget_splits}" if forget_splits > 1 else ""
    out = Path(f"{output_dir}{suffix}")
    out.mkdir(exist_ok=True)

    print(f"\n[prepare] downloading...")
    path = download_shakespeare(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    print(f"[prepare] building vocabulary...")
    vocab = build_vocab(text)

    print(f"[prepare] parsing roles...")
    roles, role2idx = parse_roles(text)
    print(f"[prepare] found {len(roles)} roles")

    if num_speakers is not None:
        rng = np.random.RandomState(seed)
        all_roles = sorted(roles.keys())
        selected_role_names = rng.choice(all_roles, size=num_speakers, replace=False)
        roles = {r: roles[r] for r in selected_role_names}
        role2idx = {r: i for i, r in enumerate(sorted(roles.keys()))}
        print(f"[prepare] subset: randomly selected {num_speakers} speakers ({', '.join(sorted(selected_role_names)[:3])}...)")


    print(f"\n[prepare] splitting per-role (80/10/10)...")
    train_by_role, train_blocks_by_role, val_by_role, test_by_role = split_role_data(roles)

    print(f"\n[prepare] splitting train into retain/forget...")
    retain_roles, forget_roles = split_retain_forget(
        train_by_role, forget_frac=0.1, seed=seed
    )

    retain_text = "\n".join(train_by_role[r] for r in retain_roles)
    train_text = retain_text + "\n" + "\n".join(train_by_role[r] for r in forget_roles)
    val_text = "\n".join(val_by_role.values())
    test_text = "\n".join(test_by_role.values())

    print(f"\n[prepare] saving splits to '{out}'...")

    # Fixed splits
    for name, content in [
        ("retain", retain_text),
        ("train", train_text),
        ("val", val_text),
        ("test", test_text),
        ("full", text),
    ]:
        fpath = out / f"{name}.txt"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        tag = " (debug)" if name == "full" else ""
        print(f"[prepare]   {name}.txt:  {len(content):,} chars{tag}")

    # Create forget subfolder
    forget_dir = out / "forget"
    forget_dir.mkdir(exist_ok=True)

    # Forget split(s)
    if forget_splits == 1:
        forget_text = "\n".join(train_by_role[r] for r in forget_roles)
        fpath = forget_dir / "forget.txt"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(forget_text)
        print(f"[prepare]   forget/forget.txt:  {len(forget_text):,} chars")
        forget_split_names = ["forget"]
    else:
        print(f"\n[prepare] splitting forget into {forget_splits} chunks (block-level shuffle, seed={seed+1})...")
        forget_chunks = make_forget_splits(forget_roles, train_blocks_by_role, forget_splits, seed)
        forget_split_names = []
        for i, chunk in enumerate(forget_chunks):
            name = f"forget_{i}"
            fpath = forget_dir / f"{name}.txt"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(chunk)
            print(f"[prepare]   forget/{name}.txt:  {len(chunk):,} chars")
            forget_split_names.append(name)

    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "role2idx": role2idx,
        "retain_roles": retain_roles,
        "forget_roles": forget_roles,
        "forget_split_names": forget_split_names,
        "seq_len": seq_len,
        "forget_splits": forget_splits,
        "num_speakers": num_speakers,
    }
    meta_path = out / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare]   meta.json: {len(vocab)} vocab chars, {len(role2idx)} roles")

    print(f"\n[prepare] summary:")
    print(f"  output dir:   {out}")
    print(f"  vocab size:   {len(vocab)}")
    print(f"  total roles:  {len(roles)}")
    print(f"  retain roles: {len(retain_roles)}")
    print(f"  forget roles: {len(forget_roles)}")
    print(f"  forget splits: {forget_splits}")

    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--output_dir", default="data_splits")
    parser.add_argument("--forget_splits", type=int, default=1,
                        help="number of forget splits (>1 shuffles blocks and chunks)")
    parser.add_argument("--num_speakers", type=int, default=300,
                        help="randomly select N speakers for faster iteration (seeded for reproducibility). E.g., --num_speakers 5")
    args = parser.parse_args()

    prepare_data(
        seq_len=args.seq_len,
        seed=args.seed,
        output_dir=args.output_dir,
        forget_splits=args.forget_splits,
        num_speakers=args.num_speakers,
    )
