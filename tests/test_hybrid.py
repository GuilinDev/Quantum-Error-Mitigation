"""Unit tests for hybrid mitigation pipeline."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mitigation_net import create_mitigation_model
from src.hybrid.neural_mitigator import NeuralMitigator, HybridPipeline


class TestNeuralMitigator:
    """Tests for NeuralMitigator class."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return create_mitigation_model(
            model_type="standard",
            circuit_dim=20,
            noise_dim=8,
        )

    @pytest.fixture
    def mitigator(self, model):
        """Create a NeuralMitigator instance."""
        return NeuralMitigator(model, device="cpu")

    def test_initialization(self, model):
        """Test NeuralMitigator initialization."""
        mitigator = NeuralMitigator(model, device="cpu")

        assert mitigator.model is not None
        assert mitigator.device == "cpu"

    def test_mitigate_single(self, mitigator):
        """Test single value mitigation."""
        noisy_value = -0.8
        circuit_features = np.random.randn(20).astype(np.float32)
        noise_features = np.random.randn(8).astype(np.float32)

        result = mitigator.mitigate(noisy_value, circuit_features, noise_features)

        assert isinstance(result, float)

    def test_mitigate_batch(self, mitigator):
        """Test batch mitigation."""
        batch_size = 8
        noisy_values = np.random.randn(batch_size).astype(np.float32)
        circuit_features = np.random.randn(batch_size, 20).astype(np.float32)
        noise_features = np.random.randn(batch_size, 8).astype(np.float32)

        result = mitigator.mitigate_batch(
            noisy_values, circuit_features, noise_features
        )

        assert result.shape == (batch_size,)


class TestHybridPipeline:
    """Tests for HybridPipeline class."""

    def test_initialization_no_neural(self):
        """Test pipeline initialization without neural model."""
        pipeline = HybridPipeline(
            neural_mitigator=None,
            use_zne_features=True,
            shots=1000,
        )

        assert pipeline.neural_mitigator is None
        assert pipeline.use_zne_features

    def test_initialization_with_neural(self):
        """Test pipeline initialization with neural model."""
        model = create_mitigation_model("standard", circuit_dim=20, noise_dim=8)
        mitigator = NeuralMitigator(model, device="cpu")

        pipeline = HybridPipeline(
            neural_mitigator=mitigator,
            use_zne_features=False,
            shots=1000,
        )

        assert pipeline.neural_mitigator is not None


class TestCreateHybridPipeline:
    """Tests for create_hybrid_pipeline factory."""

    def test_create_without_checkpoint(self):
        """Test creating pipeline without model checkpoint."""
        from src.hybrid.neural_mitigator import create_hybrid_pipeline

        pipeline = create_hybrid_pipeline(
            checkpoint_path=None,
            use_zne=True,
            shots=1000,
        )

        assert isinstance(pipeline, HybridPipeline)
        assert pipeline.neural_mitigator is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
