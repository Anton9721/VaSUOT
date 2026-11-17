"""Neural parameterizations of VarEOT scalar functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from config import ModelConfig, VaSUOTConfig


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Fully connected network operating on the final tensor dimension.

    VarEOT uses scalar-valued functions: the transport potential and an
    auxiliary normalizer.  Keeping all leading dimensions intact lets the
    same model evaluate ordinary batches ``(B, D)`` and conditional samples
    ``(B, K, D)`` without reshaping at call sites.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        activation_cls = _ACTIVATIONS[activation.lower()]

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), activation_cls()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        """Evaluate the MLP while preserving every leading batch dimension."""
        return self.network(inputs)


def build_scalar_mlp(config: ModelConfig) -> MLP:
    """Build a scalar potential or normalizer network from ``ModelConfig``."""
    return MLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=1,
        activation=config.activation,
    )


def build_potential_mlp(config: ModelConfig) -> MLP:
    """Build the dual potential g_theta for a selected capacity ablation."""
    return build_scalar_mlp(config)


def build_normalizer_mlp(config: ModelConfig) -> MLP:
    """Build the auxiliary log-normalizer xi_phi independently of g_theta."""
    return build_scalar_mlp(config)


def build_vasuot_networks(config: VaSUOTConfig) -> tuple[MLP, MLP]:
    """Create independently configured potential and normalizer networks."""
    return (
        build_potential_mlp(config.resolved_potential_model()),
        build_normalizer_mlp(config.resolved_normalizer_model()),
    )


__all__ = [
    "MLP",
    "build_normalizer_mlp",
    "build_potential_mlp",
    "build_scalar_mlp",
    "build_vasuot_networks",
]
