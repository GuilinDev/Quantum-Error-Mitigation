"""Unit tests for neural network models."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.error_predictor import (
    CircuitEncoder,
    NoiseEncoder,
    ErrorPredictor,
    EnsembleErrorPredictor,
    CrossAttention,
)
from src.models.mitigation_net import (
    MitigationNetwork,
    AdaptiveMitigationNetwork,
    HybridMitigationModel,
    IterativeMitigationNetwork,
    create_mitigation_model,
)


class TestCircuitEncoder:
    """Tests for circuit encoder module."""

    def test_forward_shape(self):
        """Test output shape of circuit encoder."""
        encoder = CircuitEncoder(input_dim=20, hidden_dims=[64, 128], latent_dim=32)

        x = torch.randn(16, 20)  # batch of 16, 20 features
        output = encoder(x)

        assert output.shape == (16, 32)

    def test_different_hidden_dims(self):
        """Test with various hidden dimensions."""
        encoder = CircuitEncoder(
            input_dim=50, hidden_dims=[128, 256, 128], latent_dim=64
        )

        x = torch.randn(8, 50)
        output = encoder(x)

        assert output.shape == (8, 64)


class TestNoiseEncoder:
    """Tests for noise encoder module."""

    def test_forward_shape(self):
        """Test noise encoder output shape."""
        encoder = NoiseEncoder(noise_dim=8, hidden_dim=64, latent_dim=32)

        x = torch.randn(16, 8)
        output = encoder(x)

        assert output.shape == (16, 32)


class TestCrossAttention:
    """Tests for cross-attention module."""

    def test_forward_shape(self):
        """Test cross-attention output shape."""
        attention = CrossAttention(query_dim=64, key_dim=32, num_heads=4)

        query = torch.randn(8, 1, 64)  # batch=8, seq=1, dim=64
        key_value = torch.randn(8, 1, 32)

        output = attention(query, key_value)

        assert output.shape == (8, 1, 64)

    def test_multi_head(self):
        """Test multi-head attention."""
        attention = CrossAttention(query_dim=128, key_dim=64, num_heads=8)

        query = torch.randn(4, 2, 128)
        key_value = torch.randn(4, 3, 64)

        output = attention(query, key_value)

        assert output.shape == (4, 2, 128)


class TestErrorPredictor:
    """Tests for error predictor model."""

    def test_forward(self):
        """Test error predictor forward pass."""
        model = ErrorPredictor(
            circuit_dim=20,
            noise_dim=8,
            hidden_dims=[64, 128, 64],
            output_dim=1,
        )

        circuit_features = torch.randn(16, 20)
        noise_params = torch.randn(16, 8)

        output = model(circuit_features, noise_params)

        assert output.shape == (16, 1)

    def test_without_attention(self):
        """Test error predictor without attention."""
        model = ErrorPredictor(
            circuit_dim=20,
            noise_dim=8,
            use_attention=False,
        )

        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        output = model(circuit_features, noise_params)

        assert output.shape == (8, 1)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = ErrorPredictor(circuit_dim=20, noise_dim=8)

        circuit_features = torch.randn(4, 20, requires_grad=True)
        noise_params = torch.randn(4, 8, requires_grad=True)

        output = model(circuit_features, noise_params)
        loss = output.sum()
        loss.backward()

        assert circuit_features.grad is not None
        assert noise_params.grad is not None


class TestEnsembleErrorPredictor:
    """Tests for ensemble error predictor."""

    def test_forward_with_uncertainty(self):
        """Test ensemble with uncertainty estimation."""
        model = EnsembleErrorPredictor(circuit_dim=20, noise_dim=8, n_models=3)

        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        mean, std = model(circuit_features, noise_params, return_uncertainty=True)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)
        assert (std >= 0).all()  # Std should be non-negative

    def test_forward_without_uncertainty(self):
        """Test ensemble without uncertainty."""
        model = EnsembleErrorPredictor(circuit_dim=20, noise_dim=8, n_models=3)

        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        mean, std = model(circuit_features, noise_params, return_uncertainty=False)

        assert mean.shape == (8, 1)
        assert std is None


class TestMitigationNetwork:
    """Tests for mitigation network."""

    def test_forward(self):
        """Test mitigation network forward pass."""
        model = MitigationNetwork(
            circuit_dim=20,
            noise_dim=8,
            num_observables=1,
        )

        noisy_values = torch.randn(16, 1)
        circuit_features = torch.randn(16, 20)
        noise_params = torch.randn(16, 8)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (16, 1)

    def test_residual_connection(self):
        """Test residual connection mode."""
        model = MitigationNetwork(
            circuit_dim=20, noise_dim=8, residual=True
        )

        noisy_values = torch.randn(8, 1)
        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        # With residual, output should be noisy + correction
        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == noisy_values.shape

    def test_multi_observable(self):
        """Test mitigation with multiple observables."""
        model = MitigationNetwork(
            circuit_dim=20, noise_dim=8, num_observables=3
        )

        noisy_values = torch.randn(8, 3)
        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (8, 3)


class TestAdaptiveMitigationNetwork:
    """Tests for adaptive mitigation network."""

    def test_forward(self):
        """Test adaptive mitigation forward pass."""
        model = AdaptiveMitigationNetwork(
            circuit_dim=20, noise_dim=8, num_noise_levels=5
        )

        noisy_values = torch.randn(8, 1)
        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (8, 1)


class TestHybridMitigationModel:
    """Tests for hybrid mitigation model."""

    def test_forward_without_zne(self):
        """Test hybrid model without ZNE values."""
        model = HybridMitigationModel(circuit_dim=20, noise_dim=8, num_zne_points=3)

        noisy_values = torch.randn(8, 1)
        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (8, 1)

    def test_forward_with_zne(self):
        """Test hybrid model with ZNE values."""
        model = HybridMitigationModel(circuit_dim=20, noise_dim=8, num_zne_points=3)

        noisy_values = torch.randn(8, 1)
        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)
        zne_values = torch.randn(8, 3, 1)  # 3 ZNE points, 1 observable

        output = model(noisy_values, circuit_features, noise_params, zne_values)

        assert output.shape == (8, 1)


class TestIterativeMitigationNetwork:
    """Tests for iterative mitigation network."""

    def test_forward(self):
        """Test iterative mitigation forward pass."""
        model = IterativeMitigationNetwork(
            circuit_dim=20, noise_dim=8, num_iterations=3
        )

        noisy_values = torch.randn(8, 1)
        circuit_features = torch.randn(8, 20)
        noise_params = torch.randn(8, 8)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (8, 1)

    def test_return_intermediate(self):
        """Test returning intermediate values."""
        model = IterativeMitigationNetwork(
            circuit_dim=20, noise_dim=8, num_iterations=3
        )

        noisy_values = torch.randn(4, 1)
        circuit_features = torch.randn(4, 20)
        noise_params = torch.randn(4, 8)

        intermediates = model(
            noisy_values, circuit_features, noise_params, return_intermediate=True
        )

        # Should have num_iterations + 1 values (initial + each iteration)
        assert len(intermediates) == 4
        for inter in intermediates:
            assert inter.shape == (4, 1)


class TestModelFactory:
    """Tests for model factory function."""

    def test_create_standard(self):
        """Test creating standard model."""
        model = create_mitigation_model("standard", circuit_dim=20, noise_dim=8)
        assert isinstance(model, MitigationNetwork)

    def test_create_adaptive(self):
        """Test creating adaptive model."""
        model = create_mitigation_model("adaptive", circuit_dim=20, noise_dim=8)
        assert isinstance(model, AdaptiveMitigationNetwork)

    def test_create_hybrid(self):
        """Test creating hybrid model."""
        model = create_mitigation_model("hybrid", circuit_dim=20, noise_dim=8)
        assert isinstance(model, HybridMitigationModel)

    def test_create_iterative(self):
        """Test creating iterative model."""
        model = create_mitigation_model("iterative", circuit_dim=20, noise_dim=8)
        assert isinstance(model, IterativeMitigationNetwork)

    def test_invalid_type(self):
        """Test error for invalid model type."""
        with pytest.raises(ValueError):
            create_mitigation_model("invalid", circuit_dim=20, noise_dim=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
