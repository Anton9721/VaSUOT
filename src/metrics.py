"""Evaluation metrics shared by VaSUOT experiments.

The functions operate on evaluation samples only and do not construct autograd
graphs. They accept PyTorch tensors or NumPy arrays and return Python floats or
small tensors, making them suitable for logging and aggregation across seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor


ArrayLike: TypeAlias = Tensor | np.ndarray
OutlierPredicate: TypeAlias = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class MetricSummary:
    """Mean and sample standard deviation of one metric across independent runs."""

    mean: float
    std: float
    count: int


def transport_cost(source: ArrayLike, transported: ArrayLike) -> float:
    """Estimate E[0.5 * ||x - y||²] from paired joint-plan samples."""
    source_tensor, transported_tensor = _paired_samples(source, transported, "source", "transported")
    return float(0.5 * (source_tensor - transported_tensor).square().sum(dim=1).mean())


def wasserstein2_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    max_samples: int | None = None,
    seed: int | None = None,
) -> float:
    """Estimate empirical W2 using exact discrete OT from POT.

    The result is the square root of the optimal expected squared Euclidean
    cost. Exact discrete OT is quadratic in both memory and time; use
    max_samples for a reproducible evaluation subsample on large datasets.
    """
    first_tensor, second_tensor = _sample_sets(first, second, "first", "second")
    first_tensor = _subsample(first_tensor, max_samples=max_samples, seed=seed)
    second_tensor = _subsample(
        second_tensor, max_samples=max_samples, seed=None if seed is None else seed + 1
    )

    try:
        import ot
    except ImportError as error:
        raise ImportError(
            "wasserstein2_distance requires the optional POT package. "
            "Install it with pip install POT."
        ) from error

    cost_matrix = torch.cdist(first_tensor, second_tensor).square().numpy()
    first_weights = np.full(first_tensor.shape[0], 1.0 / first_tensor.shape[0])
    second_weights = np.full(second_tensor.shape[0], 1.0 / second_tensor.shape[0])
    squared_distance = float(ot.emd2(first_weights, second_weights, cost_matrix))
    return math.sqrt(max(squared_distance, 0.0))


def barycentric_projection(conditional_samples: ArrayLike) -> Tensor:
    """Estimate T(x)=E[y|x] from samples shaped (N, K, D)."""
    samples = _conditional_samples(conditional_samples)
    return samples.mean(dim=1)


def barycentric_projection_error(
    estimate: ArrayLike, reference: ArrayLike
) -> float:
    """Compute BME: mean squared error between two barycentric projections."""
    estimate_tensor, reference_tensor = _paired_samples(
        estimate, reference, "estimate", "reference"
    )
    return float((estimate_tensor - reference_tensor).square().sum(dim=1).mean())


def conditional_variance(conditional_samples: ArrayLike) -> float:
    """Estimate E_x[tr Cov(y|x)] using the unbiased sample covariance."""
    samples = _conditional_samples(conditional_samples)
    num_conditional_samples = samples.shape[1]
    if num_conditional_samples < 2:
        raise ValueError("At least two conditional samples per source point are required.")
    centered = samples - samples.mean(dim=1, keepdim=True)
    trace_covariance = centered.square().sum(dim=2).sum(dim=1) / (
        num_conditional_samples - 1
    )
    return float(trace_covariance.mean())


def energy_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    max_samples: int | None = None,
    seed: int | None = None,
    chunk_size: int = 2048,
) -> float:
    """Compute multivariate Energy Distance between two empirical samples.

    It returns the non-negative square root of
    2 E||X-Y|| - E||X-X'|| - E||Y-Y'||. Within-set expectations use
    off-diagonal pairs, yielding an unbiased finite-sample estimator before the
    final non-negativity clamp.
    """
    _validate_chunk_size(chunk_size)
    first_tensor, second_tensor = _sample_sets(first, second, "first", "second")
    first_tensor = _subsample(first_tensor, max_samples=max_samples, seed=seed)
    second_tensor = _subsample(
        second_tensor, max_samples=max_samples, seed=None if seed is None else seed + 1
    )
    if first_tensor.shape[0] < 2 or second_tensor.shape[0] < 2:
        raise ValueError("Energy Distance needs at least two samples in each set.")

    cross = _mean_pairwise_distance(first_tensor, second_tensor, chunk_size=chunk_size)
    first_within = _mean_pairwise_distance(
        first_tensor, first_tensor, exclude_diagonal=True, chunk_size=chunk_size
    )
    second_within = _mean_pairwise_distance(
        second_tensor, second_tensor, exclude_diagonal=True, chunk_size=chunk_size
    )
    squared_energy_distance = 2.0 * cross - first_within - second_within
    return math.sqrt(max(squared_energy_distance, 0.0))


def nearest_component_assignments(samples: ArrayLike, means: ArrayLike) -> Tensor:
    """Assign each sample to its nearest Gaussian-component mean."""
    sample_tensor, means_tensor = _sample_sets(samples, means, "samples", "means")
    return torch.cdist(sample_tensor, means_tensor).argmin(dim=1)


def component_masses(samples: ArrayLike, means: ArrayLike) -> Tensor:
    """Estimate mass per nearest-mean component, ordered as the supplied means."""
    sample_tensor, means_tensor = _sample_sets(samples, means, "samples", "means")
    assignments = nearest_component_assignments(sample_tensor, means_tensor)
    counts = torch.bincount(assignments, minlength=means_tensor.shape[0])
    return counts.to(torch.float64) / sample_tensor.shape[0]


def outlier_region_mass(samples: ArrayLike, predicate: OutlierPredicate) -> float:
    """Estimate probability mass in an arbitrary outlier region.

    predicate receives a detached CPU float64 tensor of shape (N, D) and must
    return a boolean tensor of shape (N,).
    """
    sample_tensor = _as_samples(samples, "samples")
    mask = predicate(sample_tensor)
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (
        sample_tensor.shape[0],
    ):
        raise ValueError("predicate must return a boolean tensor with shape (num_samples,).")
    return float(mask.to(torch.float64).mean())


def ball_outlier_mass(samples: ArrayLike, center: ArrayLike, radius: float) -> float:
    """Estimate mass inside a Euclidean ball used as a declared outlier region."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")
    sample_tensor = _as_samples(samples, "samples")
    center_tensor = _as_vector(center, event_dim=sample_tensor.shape[1], name="center")
    distances = (sample_tensor - center_tensor).square().sum(dim=1).sqrt()
    return float((distances <= radius).to(torch.float64).mean())


def aggregate_runs(
    runs: Sequence[Mapping[str, float]], *, ddof: int = 1
) -> dict[str, MetricSummary]:
    """Aggregate identically keyed metric dictionaries from independent seeds."""
    if not runs:
        raise ValueError("runs must contain at least one metric dictionary.")
    metric_names = set(runs[0])
    if not metric_names:
        raise ValueError("metric dictionaries must not be empty.")
    if any(set(run) != metric_names for run in runs):
        raise ValueError("Every run must report exactly the same metric names.")
    if ddof < 0 or ddof >= len(runs):
        raise ValueError("ddof must satisfy 0 <= ddof < number of runs.")

    summaries: dict[str, MetricSummary] = {}
    for name in sorted(metric_names):
        values = np.asarray([run[name] for run in runs], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {name!r} contains NaN or infinity.")
        summaries[name] = MetricSummary(
            mean=float(values.mean()),
            std=float(values.std(ddof=ddof)),
            count=len(values),
        )
    return summaries


def _as_samples(values: ArrayLike, name: str) -> Tensor:
    tensor = _to_float64_cpu(values, name)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape (positive_num_samples, positive_event_dim), "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


def _as_vector(values: ArrayLike, *, event_dim: int, name: str) -> Tensor:
    tensor = _to_float64_cpu(values, name)
    if tensor.ndim != 1 or tensor.shape[0] != event_dim:
        raise ValueError(f"{name} must have shape ({event_dim},), got {tuple(tensor.shape)}.")
    return tensor


def _conditional_samples(values: ArrayLike) -> Tensor:
    tensor = _to_float64_cpu(values, "conditional_samples")
    if tensor.ndim != 3 or 0 in tensor.shape:
        raise ValueError(
            "conditional_samples must have shape "
            f"(positive_num_sources, positive_num_samples, positive_event_dim), got {tuple(tensor.shape)}."
        )
    return tensor


def _paired_samples(
    first: ArrayLike, second: ArrayLike, first_name: str, second_name: str
) -> tuple[Tensor, Tensor]:
    first_tensor, second_tensor = _sample_sets(first, second, first_name, second_name)
    if first_tensor.shape[0] != second_tensor.shape[0]:
        raise ValueError(
            f"{first_name} and {second_name} must contain the same number of paired samples."
        )
    return first_tensor, second_tensor


def _sample_sets(
    first: ArrayLike, second: ArrayLike, first_name: str, second_name: str
) -> tuple[Tensor, Tensor]:
    first_tensor = _as_samples(first, first_name)
    second_tensor = _as_samples(second, second_name)
    if first_tensor.shape[1] != second_tensor.shape[1]:
        raise ValueError(f"{first_name} and {second_name} must have the same event dimension.")
    return first_tensor, second_tensor


def _to_float64_cpu(values: ArrayLike, name: str) -> Tensor:
    if isinstance(values, Tensor):
        tensor = values.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(values, np.ndarray):
        tensor = torch.as_tensor(values, device="cpu", dtype=torch.float64)
    else:
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return tensor


def _subsample(samples: Tensor, *, max_samples: int | None, seed: int | None) -> Tensor:
    if max_samples is None or samples.shape[0] <= max_samples:
        return samples
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive or None, got {max_samples}.")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    indices = torch.randperm(samples.shape[0], generator=generator)[:max_samples]
    return samples.index_select(0, indices)


def _mean_pairwise_distance(
    first: Tensor, second: Tensor, *, exclude_diagonal: bool = False, chunk_size: int
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    same_tensor = exclude_diagonal and first.data_ptr() == second.data_ptr()

    for start in range(0, first.shape[0], chunk_size):
        stop = min(start + chunk_size, first.shape[0])
        distances = torch.cdist(first[start:stop], second)
        if same_tensor:
            local_indices = torch.arange(start, stop)
            distances[torch.arange(stop - start), local_indices] = 0.0
        total += distances.sum()
        count += distances.numel() - (stop - start if same_tensor else 0)
    if count <= 0:
        raise ValueError("At least two samples are needed for off-diagonal distances.")
    return float(total / count)


def _validate_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")


__all__ = [
    "MetricSummary",
    "aggregate_runs",
    "ball_outlier_mass",
    "barycentric_projection",
    "barycentric_projection_error",
    "component_masses",
    "conditional_variance",
    "energy_distance",
    "nearest_component_assignments",
    "outlier_region_mass",
    "transport_cost",
    "wasserstein2_distance",
]
