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
        self._event_dim = event_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = generator

    @property
    def event_dim(self) -> int:
        return self._event_dim

    def sample(self, batch_size: int, *, device: Optional[torch.device | str] = None) -> Tensor:
        output_device = self.device if device is None else torch.device(device)
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
        if weights is None:
            normalized_weights = torch.full(
                (means.shape[0],),
                1.0 / means.shape[0],
                device=means.device,
                dtype=means.dtype,
            )
        else:
            normalized_weights = weights / weights.sum()

        cholesky = torch.linalg.cholesky(covariances)

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
