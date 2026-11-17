"""Neural parameterizations of VarEOT scalar functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from config import ModelConfig


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

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("All hidden layer widths must be positive.")

        activation_name = activation.lower()
        try:
            activation_cls = _ACTIVATIONS[activation_name]
        except KeyError as error:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(
                f"Unsupported activation {activation!r}. Choose one of: {choices}."
            ) from error

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
        if inputs.ndim == 0:
            raise ValueError("inputs must have a feature dimension.")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final dimension {self.input_dim}, got {inputs.shape[-1]}."
            )
        return self.network(inputs)


def build_scalar_mlp(config: ModelConfig) -> MLP:
    """Build a scalar potential or normalizer network from ``ModelConfig``."""
    return MLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=1,
        activation=config.activation,
    )


__all__ = ["MLP", "build_scalar_mlp"]
