"""Unit tests for the direct prediction baseline network."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.direct_predictor import DirectPredictionNet
from src.training.trainer import Trainer, TrainingConfig

CIRCUIT_DIM = 20
NOISE_DIM = 8


class _SyntheticMitigationDataset(Dataset):
    """Synthetic dataset matching QuantumDataset's batch dict format.

    The ideal value is a deterministic function of the features so that
    a model predicting from features alone can fit it.
    """

    def __init__(self, num_samples: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.circuit_features = torch.randn(
            num_samples, CIRCUIT_DIM, generator=generator
        )
        self.noise_features = torch.randn(num_samples, NOISE_DIM, generator=generator)

        # Learnable target: smooth function of circuit + noise features
        w_circuit = torch.randn(CIRCUIT_DIM, 1, generator=generator)
        w_noise = torch.randn(NOISE_DIM, 1, generator=generator)
        self.ideal_values = torch.tanh(
            self.circuit_features @ w_circuit + 0.5 * self.noise_features @ w_noise
        )
        self.noisy_values = self.ideal_values + 0.1 * torch.randn(
            num_samples, 1, generator=generator
        )

    def __len__(self) -> int:
        return len(self.ideal_values)

    def __getitem__(self, idx: int):
        return {
            "circuit_features": self.circuit_features[idx],
            "noise_features": self.noise_features[idx],
            "noisy_value": self.noisy_values[idx],
            "ideal_value": self.ideal_values[idx],
        }


def _make_trainer(model, num_epochs: int, save_dir: str) -> Trainer:
    """Build a Trainer on synthetic data with test-friendly settings."""
    config = TrainingConfig(
        circuit_dim=CIRCUIT_DIM,
        noise_dim=NOISE_DIM,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=num_epochs,
        loss_type="huber",
        device="cpu",
        use_wandb=False,
        save_dir=save_dir,
    )
    train_loader = DataLoader(
        _SyntheticMitigationDataset(200, seed=0),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        _SyntheticMitigationDataset(64, seed=1),
        batch_size=config.batch_size,
    )
    return Trainer(model, config, train_loader, val_loader)


class TestDirectPredictionNet:
    """Tests for the direct prediction network."""

    def test_forward_shape(self):
        """Test output shape of forward pass."""
        model = DirectPredictionNet(
            circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM, num_observables=1
        )

        noisy_values = torch.randn(16, 1)
        circuit_features = torch.randn(16, CIRCUIT_DIM)
        noise_params = torch.randn(16, NOISE_DIM)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (16, 1)

    def test_multi_observable_shape(self):
        """Test output shape with multiple observables."""
        model = DirectPredictionNet(
            circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM, num_observables=3
        )

        noisy_values = torch.randn(8, 3)
        circuit_features = torch.randn(8, CIRCUIT_DIM)
        noise_params = torch.randn(8, NOISE_DIM)

        output = model(noisy_values, circuit_features, noise_params)

        assert output.shape == (8, 3)

    def test_forward_with_uncertainty(self):
        """Test uncertainty head output."""
        model = DirectPredictionNet(circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM)

        noisy_values = torch.randn(8, 1)
        circuit_features = torch.randn(8, CIRCUIT_DIM)
        noise_params = torch.randn(8, NOISE_DIM)

        predicted, uncertainty = model(
            noisy_values, circuit_features, noise_params, return_uncertainty=True
        )

        assert predicted.shape == (8, 1)
        assert uncertainty.shape == (8, 1)
        assert (uncertainty >= 0).all()

    def test_forward_accepts_none_noisy_values(self):
        """Test that noisy_values may be None (it is ignored)."""
        model = DirectPredictionNet(circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM)

        circuit_features = torch.randn(4, CIRCUIT_DIM)
        noise_params = torch.randn(4, NOISE_DIM)

        output = model(None, circuit_features, noise_params)

        assert output.shape == (4, 1)

    def test_gradient_flow(self):
        """Test gradients flow to feature inputs but NOT to noisy values."""
        model = DirectPredictionNet(circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM)

        noisy_values = torch.randn(4, 1, requires_grad=True)
        circuit_features = torch.randn(4, CIRCUIT_DIM, requires_grad=True)
        noise_params = torch.randn(4, NOISE_DIM, requires_grad=True)

        output = model(noisy_values, circuit_features, noise_params)
        output.sum().backward()

        assert circuit_features.grad is not None
        assert noise_params.grad is not None
        # noisy_values must not participate in the computation graph
        assert noisy_values.grad is None

        # All model parameters receive gradients (except the unused
        # uncertainty head, which is only active with return_uncertainty)
        for name, param in model.named_parameters():
            if name.startswith("uncertainty_head"):
                continue
            assert param.grad is not None, f"No gradient for {name}"

    def test_output_independent_of_noisy_value(self):
        """Same features with different noisy values give identical outputs."""
        model = DirectPredictionNet(circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM)
        model.eval()  # Disable dropout for deterministic comparison

        circuit_features = torch.randn(8, CIRCUIT_DIM)
        noise_params = torch.randn(8, NOISE_DIM)
        noisy_a = torch.randn(8, 1)
        noisy_b = torch.randn(8, 1)

        with torch.no_grad():
            output_a = model(noisy_a, circuit_features, noise_params)
            output_b = model(noisy_b, circuit_features, noise_params)

        assert torch.equal(output_a, output_b)

    def test_trainer_single_step(self, tmp_path):
        """Test one training epoch with the existing Trainer."""
        torch.manual_seed(42)
        model = DirectPredictionNet(circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM)
        trainer = _make_trainer(model, num_epochs=1, save_dir=str(tmp_path))

        train_loss = trainer.train_epoch()
        val_metrics = trainer.validate()

        assert torch.isfinite(torch.tensor(train_loss))
        assert torch.isfinite(torch.tensor(val_metrics["val_loss"]))

    def test_fits_random_data(self, tmp_path):
        """Sanity check: loss decreases over 5 epochs on 200 fake samples."""
        torch.manual_seed(42)
        model = DirectPredictionNet(circuit_dim=CIRCUIT_DIM, noise_dim=NOISE_DIM)
        trainer = _make_trainer(model, num_epochs=5, save_dir=str(tmp_path))

        losses = [trainer.train_epoch() for _ in range(5)]

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
