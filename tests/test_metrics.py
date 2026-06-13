"""Unit tests for evaluation metrics."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import (
    fidelity,
    trace_distance,
    energy_error,
    chemical_accuracy,
    approximation_ratio,
    improvement_ratio,
    sampling_overhead,
    statistical_metrics,
    bootstrap_confidence_interval,
)


class TestQuantumMetrics:
    """Tests for quantum state metrics."""

    def test_fidelity_identical(self):
        """Test fidelity of identical states."""
        rho = np.diag([0.5, 0.5])
        f = fidelity(rho, rho)

        assert np.isclose(f, 1.0)

    def test_fidelity_orthogonal(self):
        """Test fidelity of orthogonal states."""
        rho = np.diag([1.0, 0.0])
        sigma = np.diag([0.0, 1.0])
        f = fidelity(rho, sigma)

        assert np.isclose(f, 0.0)

    def test_fidelity_bounds(self):
        """Test that fidelity is between 0 and 1."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            # Create random density matrices
            p = rng.random()
            rho = np.diag([p, 1 - p])
            q = rng.random()
            sigma = np.diag([q, 1 - q])

            f = fidelity(rho, sigma)
            assert 0 <= f <= 1

    def test_trace_distance_identical(self):
        """Test trace distance of identical states."""
        rho = np.diag([0.5, 0.5])
        d = trace_distance(rho, rho)

        assert np.isclose(d, 0.0)

    def test_trace_distance_orthogonal(self):
        """Test trace distance of orthogonal states."""
        rho = np.diag([1.0, 0.0])
        sigma = np.diag([0.0, 1.0])
        d = trace_distance(rho, sigma)

        assert np.isclose(d, 1.0)


class TestEnergyMetrics:
    """Tests for energy-related metrics."""

    def test_energy_error_absolute(self):
        """Test absolute energy error."""
        err = energy_error(-1.2, -1.0, absolute=True)
        assert np.isclose(err, 0.2)

    def test_energy_error_relative(self):
        """Test relative energy error."""
        err = energy_error(-1.2, -1.0, absolute=False)
        assert np.isclose(err, 0.2)

    def test_chemical_accuracy_achieved(self):
        """Test chemical accuracy check - achieved."""
        result = chemical_accuracy(-1.1000, -1.1010, threshold=0.0016)
        assert result == True

    def test_chemical_accuracy_not_achieved(self):
        """Test chemical accuracy check - not achieved."""
        result = chemical_accuracy(-1.1000, -1.1050, threshold=0.0016)
        assert result == False


class TestOptimizationMetrics:
    """Tests for optimization metrics."""

    def test_approximation_ratio_perfect(self):
        """Test perfect approximation ratio."""
        ratio = approximation_ratio(10.0, 10.0)
        assert np.isclose(ratio, 1.0)

    def test_approximation_ratio_partial(self):
        """Test partial approximation."""
        ratio = approximation_ratio(8.0, 10.0)
        assert np.isclose(ratio, 0.8)

    def test_approximation_ratio_zero_optimal(self):
        """Test with zero optimal cost."""
        ratio = approximation_ratio(0.0, 0.0)
        assert ratio == 1.0

        ratio = approximation_ratio(1.0, 0.0)
        assert ratio == 0.0


class TestMitigationMetrics:
    """Tests for mitigation effectiveness metrics."""

    def test_improvement_ratio_perfect(self):
        """Test perfect improvement."""
        ratio = improvement_ratio(0.0, 0.1)
        assert np.isclose(ratio, 1.0)

    def test_improvement_ratio_none(self):
        """Test no improvement."""
        ratio = improvement_ratio(0.1, 0.1)
        assert np.isclose(ratio, 0.0)

    def test_improvement_ratio_partial(self):
        """Test partial improvement."""
        ratio = improvement_ratio(0.05, 0.1)
        assert np.isclose(ratio, 0.5)

    def test_sampling_overhead(self):
        """Test sampling overhead calculation."""
        overhead = sampling_overhead(10000, 5000)
        assert np.isclose(overhead, 2.0)


class TestStatisticalMetrics:
    """Tests for statistical metrics."""

    def test_statistical_metrics_perfect(self):
        """Test statistical metrics for perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])

        metrics = statistical_metrics(predictions, targets)

        assert np.isclose(metrics["mae"], 0.0)
        assert np.isclose(metrics["mse"], 0.0)
        assert np.isclose(metrics["rmse"], 0.0)

    def test_statistical_metrics_with_error(self):
        """Test statistical metrics with errors."""
        predictions = np.array([1.1, 2.0, 2.9])
        targets = np.array([1.0, 2.0, 3.0])

        metrics = statistical_metrics(predictions, targets)

        assert metrics["mae"] > 0
        assert metrics["mse"] > 0
        assert np.isclose(metrics["rmse"], np.sqrt(metrics["mse"]))

    def test_statistical_metrics_all_keys(self):
        """Test that all expected keys are present."""
        predictions = np.random.randn(100)
        targets = np.random.randn(100)

        metrics = statistical_metrics(predictions, targets)

        expected_keys = ["mae", "mse", "rmse", "max_error", "median_error",
                        "std_error", "mean_bias", "r2"]
        for key in expected_keys:
            assert key in metrics


class TestBootstrap:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        np.random.seed(42)
        data = np.random.randn(100)

        point, lower, upper = bootstrap_confidence_interval(
            data, confidence=0.95, n_bootstrap=100, seed=42
        )

        assert lower < point < upper

    def test_bootstrap_reproducibility(self):
        """Test bootstrap reproducibility with seed."""
        data = np.random.randn(100)

        result1 = bootstrap_confidence_interval(data, seed=42)
        result2 = bootstrap_confidence_interval(data, seed=42)

        assert result1 == result2

    def test_bootstrap_custom_statistic(self):
        """Test bootstrap with custom statistic."""
        data = np.random.randn(100)

        point, lower, upper = bootstrap_confidence_interval(
            data, statistic=np.median, seed=42
        )

        assert lower < point < upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
