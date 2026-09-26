"""
model.py — model architecture

The exact architecture described in McMahan et al. 2017:
    Embedding  : vocab_size  ->  8
    LSTM layer 1:    8       -> 256
    LSTM layer 2:  256       -> 256
    Linear     :  256        -> vocab_size

Total parameters: ~866,578 (with vocab_size ~86)
"""

import torch
import torch.nn as nn


class ShakespeareLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim=8, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True        # input/output shape: (batch, seq_len, features)
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len)  — integer character indices
        hidden : (h, c) tuple or None

        Returns:
            logits : (batch, seq_len, vocab_size)  — raw scores before softmax
            hidden : updated (h, c) tuple
        """
        emb            = self.embedding(x)       # (batch, seq_len, embed_dim)
        out, hidden    = self.lstm(emb, hidden)  # (batch, seq_len, hidden_size)
        logits         = self.fc(out)            # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Return zeroed (h_0, c_0) for the start of a sequence."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
