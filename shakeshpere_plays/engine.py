"""
engine.py — training, evaluation, and text generation

Functions:
    train_one_epoch  — one full pass over training data, returns loss/acc/ppl
    evaluate         — frozen pass over any loader, returns loss/acc/ppl
    generate         — sample text character by character from a trained model
"""

import math
import torch
import torch.nn as nn


# exp(x) overflows for x > ~709.78 in float64; return inf instead of crashing.
_MAX_LOG_FLOAT = math.log(float("1.7976931348623157e+308"))


def _safe_perplexity(avg_loss):
    """Compute perplexity robustly for extreme losses."""
    if not math.isfinite(avg_loss):
        return float("inf")
    if avg_loss > _MAX_LOG_FLOAT:
        return float("inf")
    return math.exp(avg_loss)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, clip=1.0):
    """
    One full pass over the training DataLoader.

    Uses truncated BPTT: hidden state is reset (detached) at the start
    of each batch so gradients don't flow across batch boundaries.
    Gradient clipping (default max norm = 1.0) prevents exploding gradients,
    which are common in LSTMs trained with SGD.

    Returns:
        avg_loss   : float  — average cross-entropy per character
        accuracy   : float  — fraction of correct next-char predictions
        perplexity : float  — exp(avg_loss), lower is better
    """
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_chars   = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x, None)  # PyTorch LSTM inits hidden to zeros when None

        # logits: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
        # y:      (batch, seq_len)        -> (batch*seq_len,)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        n = y.numel()
        total_loss    += loss.item() * n
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_chars   += n

    avg_loss   = total_loss / total_chars
    accuracy   = total_correct / total_chars
    perplexity = _safe_perplexity(avg_loss)
    return avg_loss, accuracy, perplexity


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """
    Evaluate on a DataLoader with no weight updates.

    Metrics match the paper:
        accuracy   = fraction of chars where top-1 prediction is correct
                     (paper target: ~54% on Shakespeare non-IID)
        perplexity = exp(avg cross-entropy)
                     (standard NLP metric; paper doesn't report it but useful)

    Returns:
        avg_loss, accuracy, perplexity
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_chars   = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, None)

            loss           = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            n              = y.numel()
            total_loss    += loss.item() * n
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_chars   += n

    avg_loss   = total_loss / total_chars
    accuracy   = total_correct / total_chars
    perplexity = _safe_perplexity(avg_loss)
    return avg_loss, accuracy, perplexity


# ── Text generation ───────────────────────────────────────────────────────────

def generate(model, seed_text, char2idx, idx2char,
             n_chars=300, temperature=0.8, device="cpu"):
    """
    Generate text by sampling one character at a time.

    Steps:
        1. Feed the seed into the model to build up a hidden state
           (the model 'reads' the context without producing output)
        2. From the last seed character, sample the next character
           from the output probability distribution
        3. Feed that character back in, repeat n_chars times

    temperature:
        < 1.0  ->  sharper distribution, more repetitive / confident
        > 1.0  ->  flatter distribution, more random / creative
        = 1.0  ->  raw model probabilities

    Returns: seed_text + generated continuation (string)
    """
    model.eval()
    hidden = model.init_hidden(1, device)

    # Warm up: feed all seed chars except the last to build hidden state
    with torch.no_grad():
        for c in seed_text[:-1]:
            if c not in char2idx:
                continue
            x = torch.tensor([[char2idx[c]]], device=device)
            _, hidden = model(x, hidden)

    result = seed_text
    c = seed_text[-1]

    with torch.no_grad():
        for _ in range(n_chars):
            if c not in char2idx:
                break
            x              = torch.tensor([[char2idx[c]]], device=device)
            logits, hidden = model(x, hidden)           # (1, 1, vocab)
            logits         = logits.squeeze() / temperature
            probs          = torch.softmax(logits, dim=-1)
            next_id        = torch.multinomial(probs, 1).item()
            c              = idx2char[next_id]
            result        += c

    return result
