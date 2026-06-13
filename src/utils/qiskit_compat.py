"""Qiskit compatibility utilities.

This module provides wrapper functions for Qiskit primitives
to handle API changes across versions.

Simulation method notes (measured with qiskit-aer 0.17.2):
    Aer's ``automatic`` method selection is shots-dependent for noisy
    simulation: it picks ``density_matrix`` when ``shots`` exceeds roughly
    ``2**num_qubits`` (and the density matrix fits in memory), and
    ``statevector`` trajectory sampling otherwise. Density-matrix memory
    grows as ``4**num_qubits`` (64 GB at 16 qubits), so for circuits beyond
    ~10-12 qubits the explicit ``statevector`` method (one noise trajectory
    per shot) must be used. Pass ``method`` explicitly to make the choice
    deterministic instead of relying on Aer's heuristic.

Exactness notes:
    The Aer ``EstimatorV2`` evaluates observables via
    ``save_expectation_value`` with ``default_precision=0.0``, so no
    measurement shot noise is ever added on top of the simulated state:
    - ``noise_model=None`` with the ``statevector`` method yields the exact
      ideal expectation value (bit-reproducible, independent of ``shots``).
    - A noisy ``density_matrix`` simulation yields the exact noisy
      expectation value (noise channels are applied deterministically).
    - A noisy ``statevector`` simulation is a Monte Carlo average over
      ``shots`` noise trajectories with statistical error ~1/sqrt(shots).
"""

from typing import Optional, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.noise import NoiseModel as AerNoiseModel

#: Simulation methods supported by :func:`create_estimator`.
SUPPORTED_METHODS = ("automatic", "density_matrix", "statevector")


def create_estimator(
    shots: int = 8192,
    noise_model: Optional[AerNoiseModel] = None,
    method: str = "automatic",
    seed: Optional[int] = None,
) -> EstimatorV2:
    """Create an AER Estimator with the correct API.

    Args:
        shots: Number of trajectories (statevector) or shots (automatic).
            Irrelevant for the density_matrix method, which is exact.
        noise_model: Optional noise model.
        method: Aer simulation method ('automatic', 'density_matrix',
            or 'statevector'). See module docstring for trade-offs.
        seed: Optional simulator seed for reproducible trajectory sampling.

    Returns:
        Configured EstimatorV2 instance.

    Raises:
        ValueError: If method is not one of SUPPORTED_METHODS.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown simulation method: {method}. "
            f"Supported: {SUPPORTED_METHODS}"
        )

    run_options = {"shots": shots}
    if seed is not None:
        run_options["seed_simulator"] = seed

    backend_options = {}
    if method != "automatic":
        backend_options["method"] = method
    if noise_model is not None:
        backend_options["noise_model"] = noise_model

    options = {"run_options": run_options}
    if backend_options:
        options["backend_options"] = backend_options

    return EstimatorV2(options=options)


def run_estimation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    shots: Optional[int] = 8192,
    noise_model: Optional[AerNoiseModel] = None,
    method: str = "automatic",
    exact: bool = False,
    seed: Optional[int] = None,
) -> float:
    """Run expectation value estimation.

    Args:
        circuit: Quantum circuit to execute.
        observable: Observable to measure.
        shots: Number of noise trajectories for statevector simulation.
            ``None`` is equivalent to ``exact=True``.
        noise_model: Optional noise model.
        method: Aer simulation method ('automatic', 'density_matrix',
            or 'statevector').
        exact: If True, compute the exact expectation value without
            trajectory sampling: statevector for ideal circuits
            (noise_model=None), density_matrix for noisy circuits.
            Overrides ``method`` and ``shots``.
        seed: Optional simulator seed for reproducible trajectory sampling.

    Returns:
        Expectation value as float.
    """
    if exact or shots is None:
        # save_expectation_value is deterministic on the final state, so a
        # single execution suffices: ideal statevector evolution is exact,
        # and density_matrix applies noise channels deterministically.
        method = "statevector" if noise_model is None else "density_matrix"
        shots = 1

    estimator = create_estimator(
        shots=shots, noise_model=noise_model, method=method, seed=seed
    )
    job = estimator.run([(circuit, observable)])
    result = job.result()
    return float(result[0].data.evs)


#: Largest qubit count for which density-matrix sampling is used by the
#: 'auto' method of :func:`run_estimation_sampled`.
_SAMPLED_DM_MAX_QUBITS = 10


def run_estimation_sampled(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    shots: int,
    noise_model: Optional[AerNoiseModel] = None,
    method: str = "auto",
    seed: Optional[int] = None,
) -> float:
    """Estimate a diagonal observable from S measured shots.

    Unlike :func:`run_estimation` (which returns expectation values with
    no measurement noise), this performs an actual S-shot measurement in
    the computational basis and estimates the observable from counts —
    the physically faithful estimator for shot-budget comparisons. With
    the density_matrix method the counts are sampled from the exact
    noisy distribution; with statevector, each shot is an independent
    noise trajectory plus a measurement.

    Args:
        circuit: Bound quantum circuit (without measurements).
        observable: Diagonal observable — every Pauli term must contain
            only I and Z factors.
        shots: Number of measurement shots.
        noise_model: Optional Aer noise model.
        method: 'auto' (density_matrix up to 10 qubits, statevector
            above), or an explicit Aer method.
        seed: Optional simulator seed.

    Returns:
        Estimated expectation value.

    Raises:
        ValueError: If the observable contains X or Y factors.
    """
    labels = [str(p) for p in observable.paulis]
    if any(ch in label for label in labels for ch in ("X", "Y")):
        raise ValueError(
            "run_estimation_sampled supports diagonal (I/Z) observables "
            f"only; got {labels}"
        )

    if method == "auto":
        method = (
            "density_matrix"
            if circuit.num_qubits <= _SAMPLED_DM_MAX_QUBITS
            else "statevector"
        )

    measured = circuit.copy()
    measured.measure_all()

    backend_kwargs = {"method": method}
    if noise_model is not None:
        backend_kwargs["noise_model"] = noise_model
    simulator = AerSimulator(**backend_kwargs)
    run_kwargs = {"shots": shots}
    if seed is not None:
        run_kwargs["seed_simulator"] = seed
    counts = simulator.run(measured, **run_kwargs).result().get_counts()

    coeffs = observable.coeffs.real
    total = 0.0
    n_shots = sum(counts.values())
    for bitstring, count in counts.items():
        # Both the Pauli label and the counts key list qubit (n-1) first.
        bits = bitstring.replace(" ", "")
        value = 0.0
        for label, w in zip(labels, coeffs):
            sign = 1.0
            for pauli_ch, bit_ch in zip(label, bits):
                if pauli_ch == "Z" and bit_ch == "1":
                    sign = -sign
            value += w * sign
        total += value * count
    return total / n_shots
