"""Pytest configuration and fixtures for SQAI 2026 tests."""

import pytest
import numpy as np
import torch


@pytest.fixture(scope="session")
def seed():
    """Set random seeds for reproducibility."""
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    return seed_value


@pytest.fixture
def device():
    """Get compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_circuit_features():
    """Generate small circuit feature vector."""
    return np.random.randn(20).astype(np.float32)


@pytest.fixture
def small_noise_features():
    """Generate small noise feature vector."""
    return np.random.randn(8).astype(np.float32)


@pytest.fixture
def batch_data():
    """Generate batch of test data."""
    batch_size = 8
    circuit_dim = 20
    noise_dim = 8

    return {
        "circuit_features": np.random.randn(batch_size, circuit_dim).astype(np.float32),
        "noise_features": np.random.randn(batch_size, noise_dim).astype(np.float32),
        "noisy_values": np.random.randn(batch_size).astype(np.float32),
        "ideal_values": np.random.randn(batch_size).astype(np.float32),
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
