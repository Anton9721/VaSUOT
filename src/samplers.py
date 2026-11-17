"""Independent samplers for source and target distributions in VaSUOT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Sampler(ABC):
    """Interface for sampling an empirical or analytic distribution.

    VaSUOT receives independent draws from the source distribution ``p0`` and
    the reference target distribution ``p1``.  The interface deliberately has
    no pairing or label argument: couplings are learned by the objective.
    """

    @property
    @abstractmethod
    def event_dim(self) -> int:
        """Dimension of one sample."""

    @abstractmethod
    def sample(self, batch_size: int, *, device: Optional[torch.device | str] = None) -> Tensor:
        """Return independent samples with shape ``(batch_size, event_dim)``."""


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")


class TensorSampler(Sampler):
    """Uniform sampler over rows of an in-memory empirical dataset.

    Sampling is with replacement by default, which yields independent draws
    from the empirical measure even when a training batch is larger than the
    dataset.  The underlying data is detached so the sampler cannot retain an
    accidental autograd graph from preprocessing.
    """

    def __init__(
        self,
        data: Tensor,
        *,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not isinstance(data, Tensor):
            raise TypeError("data must be a torch.Tensor.")
        if data.ndim != 2:
            raise ValueError(
                f"data must have shape (num_samples, event_dim), got {tuple(data.shape)}."
            )
        if data.shape[0] == 0:
            raise ValueError("data must contain at least one sample.")
        if data.shape[1] == 0:
            raise ValueError("data must have a positive event dimension.")
        if generator is not None and generator.device != data.device:
            raise ValueError("generator and data must be on the same device.")

        self._data = data.detach()
        self.replacement = replacement
        self.generator = generator

    @property
    def event_dim(self) -> int:
        return self._data.shape[1]

    @property
    def num_samples(self) -> int:
        return self._data.shape[0]

    def sample(self, batch_size: int, *, device: Optional[torch.device | str] = None) -> Tensor:
        _validate_batch_size(batch_size)
        if not self.replacement and batch_size > self.num_samples:
            raise ValueError(
                "batch_size cannot exceed dataset size when replacement=False."
            )

        if self.replacement:
            indices = torch.randint(
                self.num_samples,
                (batch_size,),
                device=self._data.device,
                generator=self.generator,
            )
        else:
            indices = torch.randperm(
                self.num_samples,
                device=self._data.device,
                generator=self.generator,
            )[:batch_size]

        batch = self._data.index_select(0, indices)
        return batch if device is None else batch.to(device)


class StandardNormalSampler(Sampler):
    """Sampler for an isotropic Gaussian distribution in ``R^event_dim``."""

    def __init__(
        self,
        event_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if event_dim <= 0:
            raise ValueError(f"event_dim must be positive, got {event_dim}.")
        self._event_dim = event_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = generator

    @property
    def event_dim(self) -> int:
        return self._event_dim

    def sample(self, batch_size: int, *, device: Optional[torch.device | str] = None) -> Tensor:
        _validate_batch_size(batch_size)
        output_device = self.device if device is None else torch.device(device)
        if self.generator is not None and self.generator.device != output_device:
            raise ValueError("generator and sampling device must be the same.")
        return torch.randn(
            batch_size,
            self.event_dim,
            device=output_device,
            dtype=self.dtype,
            generator=self.generator,
        )


class GaussianMixtureSampler(Sampler):
    """Sampler for a full-covariance Gaussian mixture.

    It is shared by the class-imbalance and target-outlier benchmarks. Component
    weights define the empirical target mass that semi-unbalanced transport may
    deliberately not reproduce exactly.
    """

    def __init__(
        self,
        means: Tensor,
        covariances: Tensor,
        *,
        weights: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not isinstance(means, Tensor) or not isinstance(covariances, Tensor):
            raise TypeError("means and covariances must be torch.Tensors.")
        if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
            raise ValueError("means must have shape (positive_num_components, event_dim).")
        expected_shape = (means.shape[0], means.shape[1], means.shape[1])
        if tuple(covariances.shape) != expected_shape:
            raise ValueError(
                f"covariances must have shape {expected_shape}, got {tuple(covariances.shape)}."
            )
        if means.device != covariances.device or means.dtype != covariances.dtype:
            raise ValueError("means and covariances must have the same device and dtype.")
        if not torch.is_floating_point(means):
            raise TypeError("means and covariances must have floating-point dtypes.")
        if generator is not None and generator.device != means.device:
            raise ValueError("generator and mixture parameters must be on the same device.")
        if not torch.allclose(covariances, covariances.transpose(-1, -2)):
            raise ValueError("Each covariance matrix must be symmetric.")

        if weights is None:
            normalized_weights = torch.full(
                (means.shape[0],),
                1.0 / means.shape[0],
                device=means.device,
                dtype=means.dtype,
            )
        else:
            if tuple(weights.shape) != (means.shape[0],):
                raise ValueError("weights must have one non-negative value per component.")
            if weights.device != means.device or weights.dtype != means.dtype:
                raise ValueError("weights must have the same device and dtype as means.")
            if (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("weights must be non-negative and sum to a positive value.")
            normalized_weights = weights / weights.sum()

        try:
            cholesky = torch.linalg.cholesky(covariances)
        except RuntimeError as error:
            raise ValueError("Every covariance matrix must be positive definite.") from error

        self.means = means.detach()
        self.cholesky = cholesky.detach()
        self.weights = normalized_weights.detach()
        self.generator = generator

    @property
    def event_dim(self) -> int:
        return self.means.shape[1]

    @property
    def num_components(self) -> int:
        return self.means.shape[0]

    def sample(self, batch_size: int, *, device: Optional[torch.device | str] = None) -> Tensor:
        _validate_batch_size(batch_size)
        component_indices = torch.multinomial(
            self.weights, batch_size, replacement=True, generator=self.generator
        )
        standard_noise = torch.randn(
            batch_size,
            self.event_dim,
            device=self.means.device,
            dtype=self.means.dtype,
            generator=self.generator,
        )
        factors = self.cholesky.index_select(0, component_indices)
        samples = self.means.index_select(0, component_indices) + torch.bmm(
            factors, standard_noise.unsqueeze(-1)
        ).squeeze(-1)
        return samples if device is None else samples.to(device)


class SwissRollSampler(Sampler):
    """Two-dimensional Swiss-roll distribution used for synthetic validation."""

    def __init__(
        self,
        *,
        noise: float = 0.8,
        scale: float = 7.5,
        device: torch.device | str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}.")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")
        self.noise = noise
        self.scale = scale
        self.device = torch.device(device)
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)

    @property
    def event_dim(self) -> int:
        return 2

    def sample(self, batch_size: int, *, device: Optional[torch.device | str] = None) -> Tensor:
        _validate_batch_size(batch_size)
        output_device = self.device if device is None else torch.device(device)
        t = 1.5 * torch.pi * (1 + 2 * torch.rand(batch_size, generator=self._generator))
        samples = torch.stack((t * torch.cos(t), t * torch.sin(t)), dim=1)
        if self.noise:
            samples = samples + self.noise * torch.randn(
                samples.shape, generator=self._generator
            )
        return (samples / self.scale).to(output_device)


__all__ = [
    "GaussianMixtureSampler",
    "Sampler",
    "StandardNormalSampler",
    "SwissRollSampler",
    "TensorSampler",
]
