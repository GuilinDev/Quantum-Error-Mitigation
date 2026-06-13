"""Metrics for evaluating quantum error mitigation performance."""

from typing import Optional, Tuple
import numpy as np


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute quantum fidelity between two density matrices.

    F(rho, sigma) = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Fidelity value between 0 and 1.
    """
    # For pure states or diagonal matrices, use simplified formula
    if np.allclose(rho, np.diag(np.diag(rho))) and np.allclose(
        sigma, np.diag(np.diag(sigma))
    ):
        return float(np.sum(np.sqrt(np.diag(rho) * np.diag(sigma))) ** 2)

    # General case
    sqrt_rho = _matrix_sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = _matrix_sqrt(product)
    return float(np.real(np.trace(sqrt_product)) ** 2)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute trace distance between two density matrices.

    T(rho, sigma) = 0.5 * Tr[|rho - sigma|]

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Trace distance between 0 and 1.
    """
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def energy_error(
    measured: float,
    exact: float,
    absolute: bool = True,
) -> float:
    """Compute energy estimation error.

    Args:
        measured: Measured/mitigated energy value.
        exact: Exact/ideal energy value.
        absolute: If True, return absolute error. Otherwise, relative.

    Returns:
        Energy error.
    """
    if absolute:
        return abs(measured - exact)
    else:
        return abs(measured - exact) / (abs(exact) + 1e-10)


def chemical_accuracy(
    measured: float, exact: float, threshold: float = 0.0016
) -> bool:
    """Check if energy estimate achieves chemical accuracy.

    Chemical accuracy is typically defined as 1 kcal/mol ~ 0.0016 Hartree.

    Args:
        measured: Measured energy.
        exact: Exact energy.
        threshold: Accuracy threshold in Hartree.

    Returns:
        True if within chemical accuracy.
    """
    return abs(measured - exact) < threshold


def approximation_ratio(cost: float, optimal_cost: float) -> float:
    """Compute approximation ratio for optimization problems.

    Args:
        cost: Achieved cost/objective value.
        optimal_cost: Optimal cost value.

    Returns:
        Approximation ratio (higher is better for MaxCut).
    """
    if optimal_cost == 0:
        return 1.0 if cost == 0 else 0.0
    return cost / optimal_cost


def improvement_ratio(
    mitigated_error: float,
    unmitigated_error: float,
) -> float:
    """Compute relative improvement from error mitigation.

    Args:
        mitigated_error: Error after mitigation.
        unmitigated_error: Error before mitigation.

    Returns:
        Fractional improvement (1.0 = perfect mitigation, 0.0 = no improvement).
    """
    if unmitigated_error < 1e-10:
        return 1.0 if mitigated_error < 1e-10 else 0.0
    return (unmitigated_error - mitigated_error) / unmitigated_error


def sampling_overhead(
    shots_mitigated: int,
    shots_standard: int,
) -> float:
    """Compute sampling overhead of mitigation method.

    Args:
        shots_mitigated: Shots required with mitigation.
        shots_standard: Shots required without mitigation.

    Returns:
        Overhead factor (>1 means more shots needed).
    """
    return shots_mitigated / shots_standard


def statistical_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Compute comprehensive statistical metrics.

    Args:
        predictions: Array of predicted values.
        targets: Array of target values.

    Returns:
        Dictionary of metrics.
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)

    return {
        "mae": float(np.mean(abs_errors)),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(abs_errors)),
        "median_error": float(np.median(abs_errors)),
        "std_error": float(np.std(errors)),
        "mean_bias": float(np.mean(errors)),
        "r2": float(1 - np.var(errors) / (np.var(targets) + 1e-10)),
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Data array.
        statistic: Statistic function to compute.
        confidence: Confidence level (0 to 1).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - confidence) / 2

    point_estimate = statistic(data)
    lower = np.percentile(bootstrap_stats, 100 * alpha)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))

    return float(point_estimate), float(lower), float(upper)
