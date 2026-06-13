"""Unit tests for training data generation.

All tests are intentionally fast: small circuits (n <= 6), few samples,
and small trajectory counts.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit.quantum_info import Statevector

from src.quantum.circuits import VQECircuit
from src.quantum.noise_models import (
    RealisticDeviceNoise,
    IBMQ_MONTREAL_NOISE,
)
from src.training.data_generator import (
    DENSITY_MATRIX_MAX_QUBITS,
    DataSample,
    MitigationDataset,
    MultiTaskDataGenerator,
    QuantumDataGenerator,
)
from src.utils.qiskit_compat import run_estimation


class TestSampleGeneration:
    """Tests for QuantumDataGenerator.generate_sample output."""

    def test_sample_shapes_and_dtypes(self):
        """Test feature shapes, dtypes, and value types of a sample."""
        num_qubits, num_layers = 4, 2
        gen = QuantumDataGenerator(
            circuit_type="vqe",
            num_qubits=num_qubits,
            num_layers=num_layers,
            shots=256,
            seed=42,
        )
        sample = gen.generate_sample()

        assert isinstance(sample, DataSample)
        # [num_qubits, num_layers, num_parameters] + raw params
        expected_len = 3 + gen.circuit_template.num_parameters
        assert sample.circuit_features.shape == (expected_len,)
        assert sample.circuit_features.dtype == np.float32
        assert sample.noise_features.shape == (8,)
        assert isinstance(sample.noisy_value, float)
        assert isinstance(sample.ideal_value, float)
        assert sample.error == pytest.approx(
            sample.noisy_value - sample.ideal_value
        )

    def test_values_in_observable_range(self):
        """Test expectation values stay within observable bounds."""
        gen = QuantumDataGenerator(num_qubits=3, num_layers=1, shots=256, seed=0)
        sample = gen.generate_sample()

        # Observable is a convex sum of Z terms: spectrum within [-1, 1].
        assert -1.0 <= sample.ideal_value <= 1.0
        assert -1.0 <= sample.noisy_value <= 1.0

    def test_generate_dataset_length(self):
        """Test dataset generation returns requested number of samples."""
        gen = QuantumDataGenerator(num_qubits=3, num_layers=1, shots=128, seed=1)
        samples = gen.generate_dataset(3, show_progress=False)

        assert len(samples) == 3
        dataset = MitigationDataset(samples)
        assert len(dataset) == 3

    def test_invalid_sim_method_raises(self):
        """Test invalid sim_method is rejected."""
        with pytest.raises(ValueError, match="sim_method"):
            QuantumDataGenerator(num_qubits=3, sim_method="matrix_product_state")


class TestSimMethodSelection:
    """Tests for the noisy simulation method resolution."""

    def test_auto_uses_density_matrix_for_small_circuits(self):
        """Test 'auto' resolves to exact density_matrix at small n."""
        gen = QuantumDataGenerator(num_qubits=4, sim_method="auto")
        assert gen.noisy_sim_method == "density_matrix"

    def test_auto_uses_statevector_for_large_circuits(self):
        """Test 'auto' resolves to trajectory sampling above the cutoff."""
        gen = QuantumDataGenerator(
            num_qubits=DENSITY_MATRIX_MAX_QUBITS + 2, sim_method="auto"
        )
        assert gen.noisy_sim_method == "statevector"

    def test_explicit_method_is_respected(self):
        """Test explicit sim_method overrides the size-based choice."""
        gen = QuantumDataGenerator(num_qubits=4, sim_method="statevector")
        assert gen.noisy_sim_method == "statevector"


class TestTrajectoryAccuracy:
    """Tests for statevector trajectory sampling vs exact noisy values."""

    def test_trajectory_matches_exact_noisy_expectation(self):
        """Test trajectory estimate agrees with exact density matrix."""
        num_qubits = 4
        vqe = VQECircuit(num_qubits, 2)
        rng = np.random.default_rng(11)
        circuit = vqe.bind_parameters(
            rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        )
        noise = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE).build_noise_model()

        gen = QuantumDataGenerator(num_qubits=num_qubits, num_layers=2)
        observable = gen.observable

        exact_noisy = run_estimation(
            circuit, observable, noise_model=noise, exact=True
        )
        trajectory = run_estimation(
            circuit,
            observable,
            shots=2048,
            noise_model=noise,
            method="statevector",
            seed=7,
        )

        # Loose tolerance: measured trajectory error is ~0.001 at this size;
        # 0.05 stays far above statistical fluctuations (1/sqrt(2048)=0.022).
        assert trajectory == pytest.approx(exact_noisy, abs=0.05)

    def test_density_matrix_is_shot_independent(self):
        """Test exact density-matrix value does not depend on shots."""
        num_qubits = 3
        vqe = VQECircuit(num_qubits, 1)
        rng = np.random.default_rng(5)
        circuit = vqe.bind_parameters(
            rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        )
        noise = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE).build_noise_model()

        gen = QuantumDataGenerator(num_qubits=num_qubits, num_layers=1)
        value_1 = run_estimation(
            circuit, gen.observable, shots=1,
            noise_model=noise, method="density_matrix",
        )
        value_2 = run_estimation(
            circuit, gen.observable, shots=512,
            noise_model=noise, method="density_matrix",
        )

        assert value_1 == value_2


class TestExactIdealLabels:
    """Tests for the exact (shot-noise-free) ideal label path."""

    def test_ideal_labels_are_reproducible(self):
        """Test the same parameters give bit-identical ideal labels."""
        gen = QuantumDataGenerator(num_qubits=4, num_layers=2, shots=128, seed=3)
        params = np.linspace(0.1, 2.0, gen.circuit_template.num_parameters)

        sample_a = gen.generate_sample(params=params)
        sample_b = gen.generate_sample(params=params)

        assert sample_a.ideal_value == sample_b.ideal_value

    def test_ideal_label_matches_statevector(self):
        """Test ideal labels equal exact Statevector expectation values."""
        gen = QuantumDataGenerator(num_qubits=4, num_layers=2, shots=128, seed=4)
        params = np.linspace(0.0, np.pi, gen.circuit_template.num_parameters)
        circuit = gen.circuit_template.bind_parameters(params)

        sample = gen.generate_sample(params=params)
        exact = float(
            np.real(Statevector(circuit).expectation_value(gen.observable))
        )

        assert sample.ideal_value == pytest.approx(exact, abs=1e-12)

    def test_run_estimation_exact_ignores_shots(self):
        """Test exact ideal estimation is independent of the shots setting."""
        vqe = VQECircuit(3, 1)
        rng = np.random.default_rng(9)
        circuit = vqe.bind_parameters(
            rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        )
        gen = QuantumDataGenerator(num_qubits=3, num_layers=1)

        value_exact = run_estimation(
            circuit, gen.observable, noise_model=None, exact=True
        )
        value_shots_none = run_estimation(
            circuit, gen.observable, shots=None, noise_model=None
        )

        assert value_exact == value_shots_none


class TestMultiTaskGenerator:
    """Tests for MultiTaskDataGenerator."""

    def test_generates_valid_samples(self):
        """Test multi-task generation produces well-formed samples."""
        gen = MultiTaskDataGenerator(
            qubit_range=(2, 4),
            layer_range=(1, 2),
            circuit_types=["vqe"],
            shots=128,
            seed=21,
        )
        sample = gen.generate_sample()

        assert isinstance(sample, DataSample)
        assert sample.noise_features.shape == (8,)
        assert sample.error == pytest.approx(
            sample.noisy_value - sample.ideal_value
        )

    def test_invalid_sim_method_raises(self):
        """Test invalid sim_method is rejected."""
        with pytest.raises(ValueError, match="sim_method"):
            MultiTaskDataGenerator(sim_method="bogus")
