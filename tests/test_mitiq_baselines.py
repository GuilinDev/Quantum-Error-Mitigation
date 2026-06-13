"""Unit tests for mitiq-based ZNE and CDR baselines."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit.quantum_info import SparsePauliOp, Statevector

from src.quantum.circuits import VQECircuit
from src.quantum.noise_models import DepolarizingNoise, NoiseParameters
from src.utils.qiskit_compat import run_estimation
from src.classical.baselines import MitigationResult
from src.classical.mitiq_baselines import (
    CDR_BASIS_GATES,
    mitiq_cdr,
    mitiq_zne,
    transpile_to_cdr_basis,
)

# Reduced shot count for fast smoke tests.
FAST_SHOTS = 1024


@pytest.fixture(scope="module")
def vqe_setup():
    """4-qubit VQE circuit with fixed parameters and systematic noise."""
    vqe = VQECircuit(num_qubits=4, num_layers=2)
    rng = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
    circuit = vqe.bind_parameters(params)

    observable = SparsePauliOp(["ZIII", "IZII", "IIZI", "IIIZ"], [1.0] * 4)

    noise = DepolarizingNoise(NoiseParameters(two_qubit_error=0.02))
    aer_noise = noise.build_noise_model()

    ideal = float(np.real(Statevector(circuit).expectation_value(observable)))
    return circuit, observable, aer_noise, ideal


class TestMitiqZNE:
    """Smoke tests for the mitiq digital-ZNE wrapper."""

    def test_richardson_runs_and_tracks_cost(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup
        scale_factors = [1.0, 2.0, 3.0]

        result = mitiq_zne(
            circuit,
            observable,
            aer_noise,
            extrapolation="richardson",
            scale_factors=scale_factors,
            shots=FAST_SHOTS,
        )

        assert isinstance(result, MitigationResult)
        assert np.isfinite(result.mitigated_value)
        assert result.method == "mitiq-ZNE-richardson"
        assert len(result.raw_values) == len(scale_factors)
        assert all(np.isfinite(v) for v in result.raw_values)
        assert result.metadata["n_circuit_executions"] == len(scale_factors)
        assert result.metadata["total_shots"] == len(scale_factors) * FAST_SHOTS

    def test_exponential_runs_and_tracks_cost(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup
        scale_factors = [1.0, 2.0, 3.0]

        result = mitiq_zne(
            circuit,
            observable,
            aer_noise,
            extrapolation="exponential",
            scale_factors=scale_factors,
            shots=FAST_SHOTS,
        )

        assert np.isfinite(result.mitigated_value)
        assert result.method == "mitiq-ZNE-exponential"
        assert result.metadata["n_circuit_executions"] == len(scale_factors)
        assert result.metadata["total_shots"] == len(scale_factors) * FAST_SHOTS

    def test_adaptive_runs_and_tracks_cost(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup
        steps = 4

        result = mitiq_zne(
            circuit,
            observable,
            aer_noise,
            extrapolation="adaptive",
            shots=FAST_SHOTS,
            adaptive_steps=steps,
        )

        assert np.isfinite(result.mitigated_value)
        assert result.method == "mitiq-ZNE-adaptive"
        assert result.metadata["n_circuit_executions"] == steps
        assert result.metadata["total_shots"] == steps * FAST_SHOTS
        # Adaptive ZNE chooses its own scale factors; first is always 1.
        assert result.metadata["scale_factors"][0] == pytest.approx(1.0)

    def test_random_folding_runs(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup

        result = mitiq_zne(
            circuit,
            observable,
            aer_noise,
            extrapolation="richardson",
            shots=FAST_SHOTS,
            fold_method="random",
            seed=42,
        )

        assert np.isfinite(result.mitigated_value)
        assert result.metadata["fold_method"] == "random"

    def test_invalid_extrapolation_raises(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup
        with pytest.raises(ValueError, match="extrapolation"):
            mitiq_zne(circuit, observable, aer_noise, extrapolation="cubic")

    def test_invalid_fold_method_raises(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup
        with pytest.raises(ValueError, match="fold_method"):
            mitiq_zne(circuit, observable, aer_noise, fold_method="bogus")

    def test_richardson_beats_unmitigated_on_systematic_noise(self, vqe_setup):
        """Deterministic: global folding + exact-expectation executor."""
        circuit, observable, aer_noise, ideal = vqe_setup

        raw = run_estimation(
            circuit, observable, shots=8192, noise_model=aer_noise
        )
        raw_error = abs(raw - ideal)

        result = mitiq_zne(
            circuit,
            observable,
            aer_noise,
            extrapolation="richardson",
            shots=8192,
        )
        zne_error = abs(result.mitigated_value - ideal)

        assert zne_error < 0.8 * raw_error


class TestMitiqCDR:
    """Tests for the mitiq CDR wrapper."""

    def test_transpile_to_cdr_basis_preserves_expectation(self, vqe_setup):
        circuit, observable, _, ideal = vqe_setup

        transpiled, diff = transpile_to_cdr_basis(circuit, observable)

        assert diff < 1e-7
        assert set(transpiled.count_ops()).issubset(set(CDR_BASIS_GATES))
        transpiled_value = float(
            np.real(Statevector(transpiled).expectation_value(observable))
        )
        assert transpiled_value == pytest.approx(ideal, abs=1e-7)

    def test_cdr_runs_and_tracks_cost(self, vqe_setup):
        circuit, observable, aer_noise, _ = vqe_setup
        num_training = 8

        result = mitiq_cdr(
            circuit,
            observable,
            aer_noise,
            num_training_circuits=num_training,
            fraction_non_clifford=0.5,
            shots=FAST_SHOTS,
            seed=42,
        )

        assert isinstance(result, MitigationResult)
        assert np.isfinite(result.mitigated_value)
        assert result.method == "mitiq-CDR"
        # Quantum cost: training circuits + the target circuit.
        assert result.metadata["n_circuit_executions"] == num_training + 1
        assert result.metadata["total_shots"] == (num_training + 1) * FAST_SHOTS
        assert result.metadata["n_simulator_executions"] == num_training
        assert result.overhead == num_training + 1

    @pytest.mark.slow
    def test_cdr_beats_unmitigated_on_systematic_noise(self, vqe_setup):
        circuit, observable, aer_noise, ideal = vqe_setup

        raw = run_estimation(
            circuit, observable, shots=8192, noise_model=aer_noise
        )
        raw_error = abs(raw - ideal)

        result = mitiq_cdr(
            circuit,
            observable,
            aer_noise,
            num_training_circuits=20,
            fraction_non_clifford=0.5,
            shots=8192,
            seed=42,
        )
        cdr_error = abs(result.mitigated_value - ideal)

        # Loose factor to avoid flakiness; measured ratio is ~0.3x.
        assert cdr_error < 0.8 * raw_error
