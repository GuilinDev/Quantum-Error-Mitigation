"""Tests for shot-sampled expectation-value estimation."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from src.quantum.circuits import VQECircuit
from src.quantum.noise_models import DepolarizingNoise, NoiseParameters
from src.utils.qiskit_compat import run_estimation, run_estimation_sampled


@pytest.fixture
def setup_4q():
    vqe = VQECircuit(4, 2)
    rng = np.random.default_rng(3)
    circuit = vqe.bind_parameters(rng.uniform(0, 2 * np.pi, vqe.num_parameters))
    obs = SparsePauliOp(
        ["ZIII", "IZII", "IIZI", "IIIZ"], [0.25] * 4
    )
    return circuit, obs


class TestRunEstimationSampled:
    def test_matches_exact_within_shot_noise(self, setup_4q):
        circuit, obs = setup_4q
        nm = DepolarizingNoise(
            NoiseParameters(two_qubit_error=0.03)
        ).build_noise_model()
        exact = run_estimation(circuit, obs, noise_model=nm, exact=True)
        sampled = run_estimation_sampled(
            circuit, obs, shots=8192, noise_model=nm, seed=1
        )
        assert abs(sampled - exact) < 5 / np.sqrt(8192)

    def test_seed_reproducible(self, setup_4q):
        circuit, obs = setup_4q
        a = run_estimation_sampled(circuit, obs, shots=512, seed=9)
        b = run_estimation_sampled(circuit, obs, shots=512, seed=9)
        assert a == b

    def test_shot_noise_scales(self, setup_4q):
        circuit, obs = setup_4q
        exact = run_estimation(circuit, obs, noise_model=None, exact=True)
        small = [
            abs(run_estimation_sampled(circuit, obs, shots=64, seed=s) - exact)
            for s in range(30)
        ]
        large = [
            abs(run_estimation_sampled(circuit, obs, shots=4096, seed=s) - exact)
            for s in range(30)
        ]
        assert np.mean(large) < np.mean(small)

    def test_rejects_non_diagonal_observable(self, setup_4q):
        circuit, _ = setup_4q
        obs_x = SparsePauliOp(["XIII"], [1.0])
        with pytest.raises(ValueError, match="diagonal"):
            run_estimation_sampled(circuit, obs_x, shots=128)

    def test_identity_term_contributes_constant(self):
        circuit = QuantumCircuit(2)  # |00>
        obs = SparsePauliOp(["II", "ZI"], [0.5, 0.5])
        val = run_estimation_sampled(circuit, obs, shots=256, seed=0)
        # |00>: <II> = 1, <ZI> = 1 -> 0.5 + 0.5 = 1
        assert abs(val - 1.0) < 1e-9

    def test_bit_ordering(self):
        # Prepare |01> (qubit 0 = 1): Z on qubit 0 gives -1, on qubit 1 gives +1
        circuit = QuantumCircuit(2)
        circuit.x(0)
        z0 = SparsePauliOp(["IZ"], [1.0])  # rightmost char = qubit 0
        z1 = SparsePauliOp(["ZI"], [1.0])
        v0 = run_estimation_sampled(circuit, z0, shots=128, seed=0)
        v1 = run_estimation_sampled(circuit, z1, shots=128, seed=0)
        assert abs(v0 - (-1.0)) < 1e-9
        assert abs(v1 - 1.0) < 1e-9
