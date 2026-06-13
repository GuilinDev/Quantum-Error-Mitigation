"""Unit tests for classical baseline methods."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.circuits import VQECircuit
from src.quantum.noise_models import RealisticDeviceNoise, NoiseParameters, IBMQ_MONTREAL_NOISE
from src.classical.baselines import (
    ZeroNoiseExtrapolation,
    ProbabilisticErrorCancellation,
    DynamicalDecoupling,
    MitigationResult,
    run_baseline_comparison,
    compute_ideal_value,
)


class TestZeroNoiseExtrapolation:
    """Tests for ZNE implementation."""

    @pytest.fixture
    def simple_circuit(self):
        """Create a simple VQE circuit for testing."""
        vqe = VQECircuit(num_qubits=2, num_layers=1)
        params = np.array([0.5, 0.5, 0.5, 0.5])
        return vqe.bind_parameters(params)

    @pytest.fixture
    def noise_model(self):
        """Create a noise model for testing."""
        params = NoiseParameters(
            single_qubit_error=0.01,
            two_qubit_error=0.05,
        )
        return RealisticDeviceNoise(params)

    def test_initialization(self):
        """Test ZNE initialization."""
        zne = ZeroNoiseExtrapolation(
            scale_factors=[1.0, 2.0, 3.0],
            extrapolation="richardson",
            shots=1000,
        )

        assert zne.scale_factors == [1.0, 2.0, 3.0]
        assert zne.extrapolation == "richardson"
        assert zne.shots == 1000

    def test_extrapolation_linear(self):
        """Test linear extrapolation."""
        zne = ZeroNoiseExtrapolation(extrapolation="linear")

        scale_factors = np.array([1.0, 2.0, 3.0])
        values = np.array([-0.8, -0.7, -0.6])  # Linear relationship

        result = zne._extrapolate(scale_factors, values)

        # Linear extrapolation to 0 should give -0.9
        assert np.isclose(result, -0.9, atol=0.01)

    def test_extrapolation_richardson(self):
        """Test Richardson extrapolation."""
        zne = ZeroNoiseExtrapolation(extrapolation="richardson")

        scale_factors = np.array([1.0, 2.0, 3.0])
        values = np.array([-0.8, -0.7, -0.6])

        result = zne._extrapolate(scale_factors, values)

        # Should give a reasonable extrapolation
        assert -1.0 <= result <= 0.0

    def test_mitigate_result_type(self, simple_circuit, noise_model):
        """Test that mitigate returns MitigationResult."""
        from qiskit.quantum_info import SparsePauliOp

        observable = SparsePauliOp(["ZI", "IZ"], [0.5, 0.5])
        zne = ZeroNoiseExtrapolation(shots=100)  # Low shots for speed

        result = zne.mitigate(simple_circuit, observable, noise_model)

        assert isinstance(result, MitigationResult)
        assert result.method == "ZNE"
        assert len(result.raw_values) == len(zne.scale_factors)


class TestDynamicalDecoupling:
    """Tests for dynamical decoupling implementation."""

    def test_initialization(self):
        """Test DD initialization."""
        dd = DynamicalDecoupling(dd_sequence="XY4", shots=1000)

        assert dd.dd_sequence == "XY4"
        assert dd.shots == 1000

    def test_insert_x_sequence(self):
        """Test X DD sequence insertion."""
        from qiskit import QuantumCircuit

        dd = DynamicalDecoupling(dd_sequence="X")
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        dd_circuit = dd._insert_dd_sequence(circuit)

        # Should have more gates than original
        assert dd_circuit.size() > circuit.size()

    def test_insert_xy4_sequence(self):
        """Test XY4 DD sequence insertion."""
        from qiskit import QuantumCircuit

        dd = DynamicalDecoupling(dd_sequence="XY4")
        circuit = QuantumCircuit(2)
        circuit.h(0)

        dd_circuit = dd._insert_dd_sequence(circuit)

        # Should have X, Y, X, Y gates for each qubit
        assert dd_circuit.size() > circuit.size()


class TestProbabilisticErrorCancellation:
    """Tests for PEC implementation."""

    def test_initialization(self):
        """Test PEC initialization."""
        pec = ProbabilisticErrorCancellation(shots=1000, num_samples=50)

        assert pec.shots == 1000
        assert pec.num_samples == 50

    def test_estimate_gamma(self):
        """Test gamma estimation."""
        params = NoiseParameters(single_qubit_error=0.01, two_qubit_error=0.05)
        noise = RealisticDeviceNoise(params)
        pec = ProbabilisticErrorCancellation()

        gamma_1q, gamma_2q = pec._estimate_gamma(noise)

        # Gamma should be > 1 for non-zero noise
        assert gamma_1q > 1.0
        assert gamma_2q > 1.0


class TestMitigationResult:
    """Tests for MitigationResult dataclass."""

    def test_creation(self):
        """Test MitigationResult creation."""
        result = MitigationResult(
            mitigated_value=-0.85,
            raw_values=[-0.8, -0.75, -0.7],
            method="ZNE",
            overhead=3.0,
            metadata={"extrapolation": "linear"},
        )

        assert result.mitigated_value == -0.85
        assert result.method == "ZNE"
        assert result.overhead == 3.0


class TestComputeIdealValue:
    """Tests for ideal value computation."""

    def test_compute_ideal(self):
        """Test ideal value computation."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        # Observable
        observable = SparsePauliOp(["ZZ"], [1.0])

        ideal = compute_ideal_value(circuit, observable, shots=1000)

        # Bell state |00> + |11> should have <ZZ> = 1
        assert np.isclose(ideal, 1.0, atol=0.1)


class TestRunBaselineComparison:
    """Tests for baseline comparison function."""

    def test_run_comparison(self):
        """Test running baseline comparison."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        observable = SparsePauliOp(["ZZ"], [1.0])
        noise = RealisticDeviceNoise(NoiseParameters())

        # Only test 'none' for speed
        results = run_baseline_comparison(
            circuit, observable, noise,
            methods=["none"],
            shots=100,
        )

        assert "none" in results
        assert isinstance(results["none"], MitigationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
