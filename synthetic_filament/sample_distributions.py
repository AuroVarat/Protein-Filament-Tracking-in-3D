from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .utils import load_json


@dataclass
class DistributionSpec:
    name: str
    strategy: str
    values: np.ndarray
    params: dict[str, float]


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def choose_distribution(name: str, values: list[float] | np.ndarray) -> DistributionSpec:
    arr = _finite(values)
    if arr.size == 0:
        return DistributionSpec(name=name, strategy="constant", values=arr, params={"value": 0.0})

    if np.allclose(arr, arr[0]):
        return DistributionSpec(name=name, strategy="constant", values=arr, params={"value": float(arr[0])})

    if np.all((arr >= 0.0) & (arr <= 1.0)) and arr.size >= 20:
        eps = 1e-4
        clipped = np.clip(arr, eps, 1.0 - eps)
        mean = float(np.mean(clipped))
        var = float(np.var(clipped))
        if var > 1e-6 and mean * (1 - mean) > var:
            common = mean * (1 - mean) / var - 1
            alpha = max(mean * common, eps)
            beta = max((1 - mean) * common, eps)
            return DistributionSpec(name=name, strategy="beta", values=arr, params={"alpha": alpha, "beta": beta})

    strictly_positive = np.all(arr > 0)
    unique_count = len(np.unique(np.round(arr, 6)))
    if strictly_positive and arr.size >= 25:
        log_arr = np.log(arr)
        if np.std(log_arr) > 1e-6:
            return DistributionSpec(
                name=name,
                strategy="lognormal",
                values=arr,
                params={"mu": float(np.mean(log_arr)), "sigma": float(np.std(log_arr))},
            )

    if unique_count >= 8 and arr.size >= 40:
        return DistributionSpec(name=name, strategy="kde", values=arr, params={})

    if arr.size >= 10:
        return DistributionSpec(
            name=name,
            strategy="truncnorm",
            values=arr,
            params={"mean": float(np.mean(arr)), "std": float(np.std(arr) + 1e-6), "min": float(np.min(arr)), "max": float(np.max(arr))},
        )

    return DistributionSpec(name=name, strategy="empirical", values=arr, params={})


def sample_from_spec(spec: DistributionSpec, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    size = size or 1
    if spec.strategy == "constant":
        return np.full(size, spec.params["value"], dtype=float)
    if spec.strategy == "empirical":
        return rng.choice(spec.values, size=size, replace=True).astype(float)
    if spec.strategy == "kde":
        kde = stats.gaussian_kde(spec.values)
        samples = kde.resample(size, seed=rng).reshape(-1)
        lo = np.min(spec.values)
        hi = np.max(spec.values)
        return np.clip(samples, lo, hi)
    if spec.strategy == "lognormal":
        return rng.lognormal(mean=spec.params["mu"], sigma=spec.params["sigma"], size=size)
    if spec.strategy == "beta":
        return rng.beta(spec.params["alpha"], spec.params["beta"], size=size)
    if spec.strategy == "truncnorm":
        std = max(spec.params["std"], 1e-6)
        a = (spec.params["min"] - spec.params["mean"]) / std
        b = (spec.params["max"] - spec.params["mean"]) / std
        return stats.truncnorm.rvs(a, b, loc=spec.params["mean"], scale=std, size=size, random_state=rng)
    raise ValueError(f"Unsupported strategy: {spec.strategy}")


class StatsSampler:
    def __init__(self, stats_dir: Path):
        self.stats_dir = stats_dir
        self.summary = load_json(stats_dir / "stats_summary.json")
        self.sample_df = pd.read_csv(stats_dir / "sample_measurements.csv")
        self.distribution_specs: dict[str, DistributionSpec] = {}
        for name, payload in self.summary.get("distributions", {}).items():
            self.distribution_specs[name] = choose_distribution(name, payload.get("values", []))

    def has(self, name: str) -> bool:
        return name in self.distribution_specs

    def sample(self, name: str, rng: np.random.Generator) -> float:
        spec = self.distribution_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown distribution: {name}")
        return float(sample_from_spec(spec, rng, size=1)[0])

    def sample_or_default(self, name: str, rng: np.random.Generator, default: float) -> float:
        if not self.has(name):
            return float(default)
        return self.sample(name, rng)

    def sample_image_shape(self, rng: np.random.Generator) -> tuple[int, int]:
        heights = self.sample_df["image_height"].dropna().to_numpy(dtype=int)
        widths = self.sample_df["image_width"].dropna().to_numpy(dtype=int)
        if len(heights) == 0:
            return 128, 128
        idx = int(rng.integers(0, len(heights)))
        return int(heights[idx]), int(widths[idx])

    def conditional_resample(self, target: str, condition: str, condition_value: float, rng: np.random.Generator, bandwidth_quantile: float = 0.2) -> float:
        cols = self.sample_df[[target, condition]].dropna()
        if len(cols) < 8:
            return self.sample(target, rng)
        distances = np.abs(cols[condition].to_numpy(dtype=float) - condition_value)
        k = max(3, int(len(cols) * bandwidth_quantile))
        nearest = cols.iloc[np.argsort(distances)[:k]]
        return float(rng.choice(nearest[target].to_numpy(dtype=float)))

    def strategy_table(self) -> pd.DataFrame:
        rows = []
        for name, spec in sorted(self.distribution_specs.items()):
            rows.append({"parameter": name, "strategy": spec.strategy, "count": int(len(spec.values))})
        return pd.DataFrame(rows)
