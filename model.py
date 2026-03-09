"""
model_mingru.py — Drop-in replacement for model.py
Backbone: MinGRU (nn.GRU) replacing BytewiseMamba

Changes from original:
  - Removed all Mamba / mamba_ssm / mambapy dependencies
  - Replaced MambaBlock with MinGRUBlock (standard nn.GRU)
  - Works on both CPU and CUDA with no special installs
  - All interfaces preserved: forward(), step(), init_stream()
  - make_splits() and ByteDataloader unchanged
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# MinGRU Block — replaces MambaBlock
# ---------------------------------------------------------------------------

class MinGRUBlock(nn.Module):
    """
    A single recurrent block using a GRU as the sequence mixer.

    Mirrors the MambaBlock interface exactly:
      - forward(x, inference_params=None) for training   [B, L, D] -> [B, L, D]
      - init_cache(batch_size, device)    for streaming  -> hidden state tensor
      - step(x, cache)                   for streaming   [B, D]    -> [B, D], cache
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)

        # Core recurrent mixer — input and hidden both d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,   # input shape: [B, L, D]
        )

        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward sublayer (same as original)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, inference_params=None):
        """
        Training / teacher-forced forward pass.
        x: [B, L, D]
        inference_params: ignored (kept for interface compatibility)
        returns: [B, L, D]
        """
        y = self.ln1(x)               # pre-norm
        y, _ = self.gru(y)            # [B, L, D], hidden discarded
        y = self.ln2(y)
        y = self.ff(y)
        return x + y                  # residual

    def init_cache(self, batch_size: int, device):
        """
        Initialise GRU hidden state for autoregressive streaming.
        Returns h_0: [1, B, D]  (num_layers=1, batch, hidden)
        """
        return torch.zeros(1, batch_size, self.d_model, device=device)

    def step(self, x, cache):
        """
        Single-step autoregressive inference.
        x:     [B, D]         — current input token embedding
        cache: [1, B, D]      — GRU hidden state
        returns: ([B, D], new_cache)
        """
        y = self.ln1(x)
        y = y.unsqueeze(1)            # [B, 1, D]  — add seq dim
        y, new_cache = self.gru(y, cache)
        y = y.squeeze(1)              # [B, D]     — remove seq dim
        y = self.ln2(y)
        y = self.ff(y)
        return x + y, new_cache       # residual + updated cache


# ---------------------------------------------------------------------------
# BoaConstrictor factory — identical signature to original
# ---------------------------------------------------------------------------

def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    """
    Construct a BoaBytePredictor with MinGRU backbone.

    Drop-in replacement for the original Mamba-based BoaConstrictor.
    No CUDA-specific libraries required — runs on CPU and GPU.
    """

    # -----------------------------------------------------------------------
    class BoaBytePredictor(nn.Module):
        """
        MinGRU model adapted to predict the next byte in a sequence.

        Architecture:
          byte tokens
              ↓
          Embedding  [B, L] -> [B, L, D]
              ↓
          MinGRUBlock x num_layers
              ↓
          Head (Linear -> ReLU -> Linear)  -> [B, L, 256]
        """

        def __init__(self, d_model=256, num_layers=4, vocab_size=256):
            super().__init__()
            self.d_model    = d_model
            self.num_layers = num_layers
            self.vocab_size = vocab_size

            self.embedding = nn.Embedding(vocab_size, d_model)

            self.blocks = nn.ModuleList(
                [MinGRUBlock(d_model) for _ in range(num_layers)]
            )

            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size),
            )

        # -------------------------------------------------------------------
        # Training path
        # -------------------------------------------------------------------

        def forward(self, x, inference_params=None):
            """
            x: [B, L]  — batch of byte token sequences
            returns: [B, L, 256] — logits over next byte
            inference_params: ignored, kept for interface compatibility
            """
            h = self.embedding(x)                       # [B, L, D]
            for blk in self.blocks:
                h = blk(h, inference_params=None)       # [B, L, D]
            return self.head(h)                         # [B, L, 256]

        # -------------------------------------------------------------------
        # Streaming / autoregressive inference
        # -------------------------------------------------------------------

        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1,
                        device=None, dtype=None):
            """
            Initialise per-block GRU caches for streaming inference.
            Returns a list of hidden states, one per block.
            max_len is accepted for interface compatibility but not used
            (GRU has no fixed-length state unlike Mamba's conv cache).
            """
            dev = device or next(self.parameters()).device
            return [blk.init_cache(batch_size, dev) for blk in self.blocks]

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
            """
            Single autoregressive step.
            byte_t: [B]             — current byte tokens
            caches: list of [1,B,D] — per-block GRU hidden states (mutated)
            returns: [B, 256]       — logits for next byte
            """
            h = self.embedding(byte_t)                  # [B, D]
            for i, blk in enumerate(self.blocks):
                h, caches[i] = blk.step(h, caches[i])  # O(1) per token
            return self.head(h)                         # [B, 256]

    # -----------------------------------------------------------------------
    model = BoaBytePredictor(
        d_model=d_model,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    return model


# ---------------------------------------------------------------------------
# Data utilities — unchanged from original
# ---------------------------------------------------------------------------

def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    """Number of usable bytes that fit whole (batch_size * seq_len) chunks."""
    block = seq_len * batch_size
    return (n_bytes // block) * block


def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n       = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    return (
        buf[i0:i1].tobytes(),
        buf[i1:i2].tobytes(),
        buf[i2:i2 + n_test].tobytes(),
    )


class ByteDataloader:
    """Simple dataloader that yields batches of bytes."""

    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.pos        = 0
        self.device     = device

    def __len__(self):
        return len(self.data_bytes) // (self.seq_len * self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        end = self.pos + self.seq_len * self.batch_size
        if end > len(self.data_bytes):
            self.pos = 0
            raise StopIteration

        indices = (
            np.arange(self.pos, end)
              .reshape(self.batch_size, self.seq_len)
        )
        self.pos += self.seq_len * self.batch_size
        batch = self.data_bytes[indices]
        return torch.tensor(batch, dtype=torch.long).to(self.device)