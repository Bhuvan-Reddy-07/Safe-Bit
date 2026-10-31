"""
Continuous Temporal Positional Encoder based on Bochner's Theorem.
Projects continuous time deltas into a d-dimensional harmonic embedding space:
    Phi(t) = [cos(w_1 * t), sin(w_1 * t), ..., cos(w_k * t), sin(w_k * t)]^T
    where w_k = 1 / (10000^(2k / d))
"""

import math
import numpy as np
import torch
import torch.nn as nn


class ContinuousTimeEncoder(nn.Module):
    """
    TorchScript-compatible Continuous-Time Positional Encoder.
    Applies Bochner harmonic Fourier projections to continuous time deltas.
    """

    def __init__(self, dimension: int = 16):
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError(f"Temporal dimension must be even, got {dimension}")
        self.dimension = dimension
        self.num_frequencies = dimension // 2

        # w_k = 1.0 / (10000 ** (2 * k / dimension))
        frequencies = [
            1.0 / (10000.0 ** (2.0 * k / float(dimension)))
            for k in range(self.num_frequencies)
        ]
        # Register as buffer so it moves with device and traces into TorchScript
        self.register_buffer(
            "freq_tensor", torch.tensor(frequencies, dtype=torch.float32)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for continuous time projection.
        Args:
            t: [N] or [N, 1] tensor of timestamps or time deltas (float32).
        Returns:
            [N, dimension] tensor of sinusoidal temporal embeddings.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]

        # outer product: [N, 1] * [1, num_frequencies] -> [N, num_frequencies]
        angles = t * self.freq_tensor.unsqueeze(0)
        cos_emb = torch.cos(angles)
        sin_emb = torch.sin(angles)

        # Interleave or concatenate: [cos, sin] -> [N, dimension]
        # Using cat for maximum JIT and vectorization efficiency
        out = torch.cat([cos_emb, sin_emb], dim=-1)
        return out


def encode_time_numpy(t_array: np.ndarray, dimension: int = 16) -> np.ndarray:
    """
    High-performance NumPy implementation for streaming buffer use without PyTorch overhead.
    Args:
        t_array: 1D NumPy array of time values.
        dimension: Embedding dimension (must be even).
    Returns:
        [N, dimension] float32 array.
    """
    if dimension % 2 != 0:
        raise ValueError(f"Dimension must be even, got {dimension}")
    num_freq = dimension // 2
    k = np.arange(num_freq, dtype=np.float32)
    freqs = 1.0 / (10000.0 ** (2.0 * k / float(dimension)))

    t = np.asarray(t_array, dtype=np.float32).reshape(-1, 1)
    angles = t * freqs.reshape(1, -1)
    cos_part = np.cos(angles)
    sin_part = np.sin(angles)
    return np.concatenate([cos_part, sin_part], axis=-1)
