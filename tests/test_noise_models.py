"""Unit tests for noise model implementations."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.noise_models import (
    NoiseParameters,
    NoiseModel,
    DepolarizingNoise,
    ThermalRelaxationNoise,
    RealisticDeviceNoise,
    VariableNoiseModel,
    IBMQ_MONTREAL_NOISE,
    IBMQ_TORONTO_NOISE,
    get_device_noise,
)


class TestNoiseParameters:
    """Tests for NoiseParameters dataclass."""

    def test_default_parameters(self):
        """Test default noise parameters."""
        params = NoiseParameters()

        assert params.single_qubit_error == 0.001
        assert params.two_qubit_error == 0.01
        assert params.t1 == 100.0
        assert params.t2 == 80.0

    def test_custom_parameters(self):
        """Test custom noise parameters."""
        params = NoiseParameters(
            single_qubit_error=0.002,
            two_qubit_error=0.02,
            t1=50.0,
        )

        assert params.single_qubit_error == 0.002
        assert params.two_qubit_error == 0.02
        assert params.t1 == 50.0


class TestDepolarizingNoise:
    """Tests for depolarizing noise model."""

    def test_build_noise_model(self):
        """Test noise model construction."""
        noise = DepolarizingNoise()
        aer_noise = noise.build_noise_model()

        # Should be an AerNoiseModel
        assert aer_noise is not None

    def test_custom_parameters(self):
        """Test noise model with custom parameters."""
        params = NoiseParameters(single_qubit_error=0.005, two_qubit_error=0.05)
        noise = DepolarizingNoise(params)

        assert noise.params.single_qubit_error == 0.005

    def test_noise_strength(self):
        """Test noise strength calculation."""
        params = NoiseParameters(
            single_qubit_error=0.001,
            two_qubit_error=0.01,
            readout_error_0=0.02,
            readout_error_1=0.03,
        )
        noise = DepolarizingNoise(params)

        strength = noise.get_noise_strength()
        expected = 0.001 + 0.01 + (0.02 + 0.03) / 2
        assert abs(strength - expected) < 1e-10

    def test_feature_dict(self):
        """Test conversion to feature dictionary."""
        noise = DepolarizingNoise()
        features = noise.to_feature_dict()

        assert "single_qubit_error" in features
        assert "two_qubit_error" in features
        assert "t1" in features
        assert len(features) == 8


class TestThermalRelaxationNoise:
    """Tests for thermal relaxation noise model."""

    def test_build_noise_model(self):
        """Test thermal relaxation model construction."""
        noise = ThermalRelaxationNoise()
        aer_noise = noise.build_noise_model()

        assert aer_noise is not None

    def test_t1_t2_constraint(self):
        """Test that T2 <= 2*T1 constraint is handled."""
        # T2 > 2*T1 would be unphysical, but qiskit handles this
        params = NoiseParameters(t1=50.0, t2=80.0)  # T2 < 2*T1 is fine
        noise = ThermalRelaxationNoise(params)

        aer_noise = noise.build_noise_model()
        assert aer_noise is not None


class TestRealisticDeviceNoise:
    """Tests for realistic device noise model."""

    def test_build_noise_model(self):
        """Test realistic noise model construction."""
        noise = RealisticDeviceNoise()
        aer_noise = noise.build_noise_model()

        assert aer_noise is not None

    def test_crosstalk_option(self):
        """Test crosstalk configuration."""
        noise_with = RealisticDeviceNoise(include_crosstalk=True)
        noise_without = RealisticDeviceNoise(include_crosstalk=False)

        assert noise_with.include_crosstalk == True
        assert noise_without.include_crosstalk == False


class TestVariableNoiseModel:
    """Tests for variable noise model."""

    def test_sample_parameters(self):
        """Test parameter sampling."""
        noise = VariableNoiseModel(error_range=(0.001, 0.05), seed=42)

        params1 = noise.sample_parameters()
        params2 = noise.sample_parameters()

        # Different samples should give different parameters
        assert params1.single_qubit_error != params2.single_qubit_error

    def test_sample_reproducibility(self):
        """Test reproducibility with seed."""
        noise1 = VariableNoiseModel(seed=42)
        noise2 = VariableNoiseModel(seed=42)

        params1 = noise1.sample_parameters()
        params2 = noise2.sample_parameters()

        assert params1.single_qubit_error == params2.single_qubit_error

    def test_sample_and_build(self):
        """Test combined sample and build."""
        noise = VariableNoiseModel(seed=42)

        aer_noise, params = noise.sample_and_build()

        assert aer_noise is not None
        assert isinstance(params, NoiseParameters)

    def test_error_range(self):
        """Test that sampled errors are within range."""
        noise = VariableNoiseModel(error_range=(0.01, 0.1), seed=42)

        for _ in range(10):
            params = noise.sample_parameters()
            assert 0.001 <= params.single_qubit_error <= 0.1
            assert 0.01 <= params.two_qubit_error <= 0.1


class TestPredefinedProfiles:
    """Tests for predefined noise profiles."""

    def test_montreal_profile(self):
        """Test IBMQ Montreal noise profile."""
        assert IBMQ_MONTREAL_NOISE.single_qubit_error == 0.0003
        assert IBMQ_MONTREAL_NOISE.two_qubit_error == 0.008

    def test_toronto_profile(self):
        """Test IBMQ Toronto noise profile."""
        assert IBMQ_TORONTO_NOISE.single_qubit_error == 0.0004

    def test_get_device_noise(self):
        """Test device noise factory function."""
        noise = get_device_noise("montreal")

        assert isinstance(noise, RealisticDeviceNoise)
        assert noise.params.single_qubit_error == IBMQ_MONTREAL_NOISE.single_qubit_error

    def test_get_device_noise_invalid(self):
        """Test error for invalid device."""
        with pytest.raises(ValueError):
            get_device_noise("invalid_device")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
