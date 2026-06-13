#!/usr/bin/env python3
"""Prototype: which physically-motivated noise structures break folding ZNE?

Tests three candidate hard regimes against gate-folding ZNE
(exponential + Richardson extrapolation):

  A. Coherent over-rotation: every RY/RZ applies an extra fixed rotation
     delta (calibration drift). Coherent errors interfere; the bias is
     oscillatory in the fold factor rather than smoothly amplified.
  B. Duration-dependent heating (non-Markovian): per-gate error rate
     grows with total circuit duration, so folding (3x duration) makes
     noise super-linear in the scale factor.
  C. A + B combined plus asymmetric readout (not amplified by folding).

For each regime, reports MAE of raw / ZNE-exp / ZNE-rich over random
parameter draws, with the noise realization fixed per instance.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import (
    NoiseModel as AerNoiseModel,
    ReadoutError,
    coherent_unitary_error,
    depolarizing_error,
)

from mitiq.zne import execute_with_zne
from mitiq.zne.inference import ExpFactory, RichardsonFactory

from src.quantum.circuits import VQECircuit
from src.utils.qiskit_compat import run_estimation

N_QUBITS = 6
N_INSTANCES = 30
SHOTS = 8192
SCALE_FACTORS = [1.0, 2.0, 3.0]


def coherent_overrotation_model(delta: float) -> AerNoiseModel:
    """Calibration-drift noise: extra fixed rotation after RY/RZ gates."""
    nm = AerNoiseModel()
    nm.add_all_qubit_quantum_error(
        coherent_unitary_error(RYGate(delta).to_matrix()), ["ry"]
    )
    nm.add_all_qubit_quantum_error(
        coherent_unitary_error(RZGate(delta).to_matrix()), ["rz"]
    )
    return nm


def heating_noise_model(circuit: QuantumCircuit, base_2q: float, kappa: float,
                        base_gates: int) -> AerNoiseModel:
    """Noise whose per-gate strength grows with total circuit length.

    Models device heating / TLS activation: a circuit with G gates sees
    an effective per-gate error of base * (1 + kappa * (G/base_gates - 1)).
    Folding triples G, so noise grows super-linearly in the scale factor.
    """
    n_gates = sum(1 for _ in circuit.data)
    factor = 1.0 + kappa * max(n_gates / base_gates - 1.0, 0.0)
    nm = AerNoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(min(0.1 * base_2q * factor, 0.4), 1),
        ["rx", "ry", "rz", "h", "x", "y", "z"],
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(min(base_2q * factor, 0.4), 2), ["cx", "cz"]
    )
    return nm


def add_readout(nm: AerNoiseModel, p01: float, p10: float) -> AerNoiseModel:
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    )
    return nm


def zne_value(circuit, observable, executor, factory):
    return execute_with_zne(circuit, executor, factory=factory)


def evaluate_regime(name, make_executor, make_plain_noise, vqe, observable, rng):
    """MAE of raw / ZNE-exp / ZNE-rich over random instances."""
    errs = {"raw": [], "zne_exp": [], "zne_rich": []}
    for _ in range(N_INSTANCES):
        params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        circuit = vqe.bind_parameters(params)
        ideal = run_estimation(circuit, observable, shots=SHOTS,
                               noise_model=None, method="statevector")
        executor = make_executor(circuit)
        noisy = executor(circuit)
        errs["raw"].append(abs(noisy - ideal))
        for key, factory in (
            ("zne_exp", ExpFactory(scale_factors=SCALE_FACTORS, asymptote=0.0)),
            ("zne_rich", RichardsonFactory(scale_factors=SCALE_FACTORS)),
        ):
            val = zne_value(circuit, observable, executor, factory)
            errs[key].append(abs(val - ideal))

    raw_mae = np.mean(errs["raw"])
    print(f"\n--- {name} ---")
    for k, v in errs.items():
        m = np.mean(v)
        imp = (raw_mae - m) / raw_mae * 100
        print(f"{k:9s} MAE={m:.5f}  improvement={imp:+6.1f}%")
    return {k: float(np.mean(v)) for k, v in errs.items()}


def main():
    vqe = VQECircuit(N_QUBITS, 2)
    obs = SparsePauliOp(
        ["".join("Z" if j == i else "I" for j in range(N_QUBITS))
         for i in range(N_QUBITS)],
        [1.0 / N_QUBITS] * N_QUBITS,
    )
    base_gates = sum(1 for _ in vqe.bind_parameters(
        np.zeros(vqe.num_parameters)).data)
    rng = np.random.default_rng(2026)

    # A: coherent over-rotation (fixed noise model per instance)
    delta = 0.06
    nm_coherent = coherent_overrotation_model(delta)

    def exec_coherent(_circuit):
        def executor(c):
            return run_estimation(c, obs, shots=SHOTS, noise_model=nm_coherent)
        return executor

    evaluate_regime(f"A: coherent over-rotation (delta={delta})",
                    exec_coherent, None, vqe, obs, rng)

    # B: duration-dependent heating (noise rebuilt from the folded circuit)
    base_2q, kappa = 0.02, 1.5

    def exec_heating(_circuit):
        def executor(c):
            nm = heating_noise_model(c, base_2q, kappa, base_gates)
            return run_estimation(c, obs, shots=SHOTS, noise_model=nm)
        return executor

    evaluate_regime(f"B: heating (base={base_2q}, kappa={kappa})",
                    exec_heating, None, vqe, obs, rng)

    # C: combined + asymmetric readout
    def exec_combo(_circuit):
        def executor(c):
            nm = heating_noise_model(c, base_2q, kappa, base_gates)
            nm.add_all_qubit_quantum_error(
                coherent_unitary_error(RYGate(delta).to_matrix()), ["ry"],
                warnings=False,
            )
            nm.add_all_qubit_quantum_error(
                coherent_unitary_error(RZGate(delta).to_matrix()), ["rz"],
                warnings=False,
            )
            add_readout(nm, 0.03, 0.08)
            return run_estimation(c, obs, shots=SHOTS, noise_model=nm)
        return executor

    evaluate_regime("C: coherent + heating + asymmetric readout",
                    exec_combo, None, vqe, obs, rng)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time() - t0:.0f}s")
