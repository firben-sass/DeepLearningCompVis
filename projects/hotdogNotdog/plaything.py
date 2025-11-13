#!/usr/bin/env python3
"""Utility script demonstrating a standard Least Mean Squares (LMS) optimiser.

The script can consume a simple CSV dataset (features columns followed by a
single target column) or fall back to a synthetic regression dataset. The LMS
algorithm is the stochastic counterpart to ordinary least squares and performs
online gradient updates as it iterates over the samples.

Example usage (synthetic data)::

    python plaything.py --epochs 200 --learning-rate 0.05 --verbose

Example usage (CSV input)::

    python plaything.py --data data.csv --learning-rate 0.01 --epochs 500

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Dataset:
    """Simple container for feature matrix and target vector."""

    features: np.ndarray
    targets: np.ndarray


@dataclass
class TrainingResult:
    """Holds the model parameters and loss history."""

    weights: np.ndarray
    bias: float
    loss_history: list[float]


def load_csv_dataset(path: Path, normalize: bool) -> Dataset:
    """Load a CSV dataset assuming the last column is the target."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected CSV with at least one feature column and one target column")

    features = data[:, :-1]
    targets = data[:, -1]

    if normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        features = (features - mean) / std

    return Dataset(features=features, targets=targets)


def make_synthetic_dataset(
    *,
    n_samples: int,
    n_features: int,
    noise: float,
    random_state: int,
    normalize: bool,
) -> Dataset:
    """Create a synthetic linear regression problem with optional noise."""

    rng = np.random.default_rng(random_state)
    features = rng.normal(size=(n_samples, n_features))
    true_weights = rng.normal(size=n_features)
    bias = rng.normal()
    noise_term = rng.normal(scale=noise, size=n_samples)
    targets = features @ true_weights + bias + noise_term

    if normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        features = (features - mean) / std

    return Dataset(features=features, targets=targets)


def train_test_split(dataset: Dataset, train_fraction: float, random_state: int) -> tuple[Dataset, Dataset]:
    """Split dataset into train and test portions."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in the open interval (0, 1)")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(dataset.features))
    cutoff = int(len(indices) * train_fraction)
    train_idx, test_idx = indices[:cutoff], indices[cutoff:]

    train = Dataset(features=dataset.features[train_idx], targets=dataset.targets[train_idx])
    test = Dataset(features=dataset.features[test_idx], targets=dataset.targets[test_idx])
    return train, test


def lms_train(
    dataset: Dataset,
    *,
    learning_rate: float,
    epochs: int,
    tolerance: float | None,
    shuffle: bool,
    random_state: int,
) -> TrainingResult:
    """Run the Least Mean Squares algorithm on the provided dataset."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    n_samples, n_features = dataset.features.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history: list[float] = []
    rng = np.random.default_rng(random_state)

    for epoch in range(epochs):
        if shuffle:
            indices = rng.permutation(n_samples)
            features = dataset.features[indices]
            targets = dataset.targets[indices]
        else:
            features = dataset.features
            targets = dataset.targets

        squared_error = 0.0
        for feature_vector, target in zip(features, targets):
            prediction = float(feature_vector @ weights + bias)
            error = target - prediction
            weights += learning_rate * error * feature_vector
            bias += learning_rate * error
            squared_error += error * error

        mse = squared_error / n_samples
        loss_history.append(mse)

        if tolerance is not None and mse <= tolerance:
            break

    return TrainingResult(weights=weights, bias=bias, loss_history=loss_history)


def predict(features: np.ndarray, *, weights: np.ndarray, bias: float) -> np.ndarray:
    """Compute predictions for the given feature matrix."""

    return features @ weights + bias


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error between targets and predictions."""

    residual = y_true - y_pred
    return float(np.mean(residual * residual))


def iter_loss_history(loss_history: list[float], *, every: int = 1) -> Iterable[tuple[int, float]]:
    """Yield epoch and loss pairs for logging purposes."""

    for epoch, loss in enumerate(loss_history, start=1):
        if epoch % every == 0:
            yield epoch, loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal LMS trainer for regression problems")
    parser.add_argument("--data", type=Path, help="Path to CSV file with features and target")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Stochastic step size (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs (default: 100)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Stop early when training MSE falls below this value",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of samples used for training (default: 0.8)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable epoch-wise shuffling (enabled by default)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Standardise feature columns to zero mean / unit variance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for reproducibility (default: 13)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force generation of a synthetic regression dataset",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=256,
        help="Number of synthetic samples to generate (default: 256)",
    )
    parser.add_argument(
        "--synthetic-features",
        type=int,
        default=4,
        help="Number of synthetic features to generate (default: 4)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Standard deviation of noise added to synthetic targets (default: 0.1)",
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=10,
        help="Log loss every N epochs when verbose (default: 10)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print loss curve information")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data is None and not args.synthetic:
        print("No dataset provided. Generating a synthetic regression problem.")

    if args.data is not None and args.synthetic:
        print("Both --data and --synthetic provided; defaulting to the external dataset.")

    if args.data is not None:
        dataset = load_csv_dataset(args.data, normalize=args.normalize)
    else:
        dataset = make_synthetic_dataset(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_features,
            noise=args.noise,
            random_state=args.seed,
            normalize=args.normalize,
        )

    train_set, test_set = train_test_split(dataset, args.train_fraction, args.seed)

    result = lms_train(
        train_set,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        tolerance=args.tolerance,
        shuffle=not args.no_shuffle,
        random_state=args.seed,
    )

    train_predictions = predict(train_set.features, weights=result.weights, bias=result.bias)
    test_predictions = predict(test_set.features, weights=result.weights, bias=result.bias)

    train_mse = mean_squared_error(train_set.targets, train_predictions)
    test_mse = mean_squared_error(test_set.targets, test_predictions)

    print(f"Trained for {len(result.loss_history)} epochs")
    print(f"Final weights: {result.weights}")
    print(f"Final bias: {result.bias:.6f}")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")

    if args.verbose and result.loss_history:
        print("\nEpoch-wise training loss (MSE):")
        for epoch, loss in iter_loss_history(result.loss_history, every=max(1, args.log_frequency)):
            print(f"  epoch {epoch:4d} | loss {loss:.6f}")


if __name__ == "__main__":
    main()
