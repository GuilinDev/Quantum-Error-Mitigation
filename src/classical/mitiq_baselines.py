"""Mitiq-based error mitigation baselines: robust ZNE variants and CDR.

This module wraps mitiq 1.0.0 implementations of digital Zero-Noise
Extrapolation (Richardson, exponential, and adaptive-exponential
extrapolation with unitary gate folding) and Clifford Data Regression
(CDR) so they can be compared against the neural mitigation models
using the shared :class:`~src.classical.baselines.MitigationResult`
interface.

Unlike :class:`~src.classical.baselines.ZeroNoiseExtrapolation`, which
rescales the *noise-model parameters*, the ZNE implemented here scales
noise digitally via gate folding (G -> G G^dag G), exactly as it would
run on real hardware. This is the publication-relevant comparison.

Implementation notes (validated empirically against mitiq 1.0.0):
    * ``fold_global`` with integer scale factors [1, 2, 3] is used by
      default. Fractional scale factors (e.g. 1.5) under global folding
      scale only part of the circuit, which produces non-uniform noise
      amplification and destabilizes Richardson extrapolation.
    * ``ExpFactory``/``AdaExpFactory`` with ``asymptote=None`` perform a
      non-linear fit that frequently fails to converge
      (``ExtrapolationError``) on few-point data. We default to
      ``asymptote=0.0``, which is the exact infinite-noise limit for
      traceless observables (e.g. sums of Pauli-Z terms) under
      depolarizing noise.
    * The executors return scalar expectation values computed via
      :func:`~src.utils.qiskit_compat.run_estimation`, so no separate
      observable is passed to mitiq.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.noise import NoiseModel as AerNoiseModel

from mitiq.cdr import execute_with_cdr
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import AdaExpFactory, ExpFactory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random, fold_global

from .baselines import MitigationResult
from ..utils.qiskit_compat import (
    run_estimation,
    run_estimation_hardware,
    run_estimation_sampled,
)

# Gate set required by mitiq CDR: all non-Clifford gates must be RZ
# rotations (IBM native basis {Rz, sqrt(X), X, CNOT}).
CDR_BASIS_GATES: List[str] = ["rz", "sx", "x", "cx"]

# Default digital-ZNE scale factors. Integer factors are realized
# exactly by global unitary folding; see module docstring.
DEFAULT_SCALE_FACTORS: List[float] = [1.0, 2.0, 3.0]

_ZNE_EXTRAPOLATIONS = ("richardson", "exponential", "adaptive")


class _CountingExecutor:
    """Scalar-returning mitiq executor that counts circuit executions.

    Mitiq hands back qiskit circuits when given qiskit input, so the
    executor can evaluate them directly with the Aer estimator.
    """

    def __init__(
        self,
        observable: SparsePauliOp,
        noise_model: Optional[AerNoiseModel],
        shots: int,
        circuit_transform: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
        sampled: bool = False,
        backend=None,
        hw_mode=None,
    ):
        """Initialize executor.

        Args:
            observable: Observable whose expectation value is returned.
            noise_model: Aer noise model (None for an ideal simulator).
            shots: Number of measurement shots per execution.
            circuit_transform: Optional transform applied to every circuit
                before execution (e.g. coherent angle miscalibration that
                models execute-time calibration error). Folded ZNE
                circuits and CDR training circuits all pass through it,
                so every method faces the same quantum process.
            sampled: If True, estimate the observable from ``shots``
                measured shots (physically faithful, includes measurement
                sampling noise) instead of the noiseless-readout
                expectation value.
            backend: A qiskit-ibm-runtime ``BackendV2`` (real QPU or fake
                backend). When set, executions run on hardware via
                :func:`run_estimation_hardware` and ``noise_model`` /
                ``sampled`` are ignored (the device is the noise source).
                No ``circuit_transform`` should be passed for hardware,
                since the device's intrinsic miscalibration replaces the
                injected one.
        """
        self.observable = observable
        self.noise_model = noise_model
        self.shots = shots
        self.circuit_transform = circuit_transform
        self.sampled = sampled
        self.backend = backend
        self.hw_mode = hw_mode
        self.n_executions = 0

    def __call__(self, circuit: QuantumCircuit) -> float:
        """Execute a circuit and return the expectation value."""
        self.n_executions += 1
        if self.circuit_transform is not None:
            circuit = self.circuit_transform(circuit)
        if self.backend is not None:
            return run_estimation_hardware(
                circuit,
                self.observable,
                shots=self.shots,
                backend=self.backend,
                mode=self.hw_mode,
            )
        if self.sampled:
            return run_estimation_sampled(
                circuit,
                self.observable,
                shots=self.shots,
                noise_model=self.noise_model,
            )
        return run_estimation(
            circuit,
            self.observable,
            shots=self.shots,
            noise_model=self.noise_model,
        )


def _make_zne_factory(
    extrapolation: str,
    scale_factors: List[float],
    asymptote: Optional[float],
    adaptive_steps: int,
):
    """Create a fresh (stateful) mitiq inference factory.

    Args:
        extrapolation: One of 'richardson', 'exponential', 'adaptive'.
        scale_factors: Noise scale factors (ignored by 'adaptive', which
            chooses its own factors).
        asymptote: Infinite-noise limit for exponential models.
        adaptive_steps: Number of adaptive steps (>= 3) for 'adaptive'.

    Returns:
        A mitiq Factory instance.

    Raises:
        ValueError: If extrapolation is not a supported method.
    """
    if extrapolation == "richardson":
        return RichardsonFactory(scale_factors=scale_factors)
    if extrapolation == "exponential":
        return ExpFactory(scale_factors=scale_factors, asymptote=asymptote)
    if extrapolation == "adaptive":
        # AdaExpFactory picks scale factors adaptively; it takes the
        # number of steps and an optional asymptote, not scale_factors.
        return AdaExpFactory(
            steps=adaptive_steps,
            scale_factor=2.0,
            asymptote=asymptote,
        )
    raise ValueError(
        f"Unknown extrapolation method: {extrapolation!r}. "
        f"Expected one of {_ZNE_EXTRAPOLATIONS}."
    )


def mitiq_zne(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    aer_noise_model: AerNoiseModel,
    extrapolation: str = "richardson",
    scale_factors: Optional[List[float]] = None,
    shots: int = 8192,
    fold_method: str = "global",
    asymptote: Optional[float] = 0.0,
    adaptive_steps: int = 4,
    seed: Optional[int] = None,
    circuit_transform: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
    sampled: bool = False,
    backend=None,
    hw_mode=None,
) -> MitigationResult:
    """Apply mitiq digital ZNE with gate folding to mitigate errors.

    Args:
        circuit: Bound (parameter-free) quantum circuit.
        observable: Observable to measure (e.g. sum of Z terms).
        aer_noise_model: Aer noise model used by the noisy executor.
        extrapolation: 'richardson', 'exponential', or 'adaptive'.
        scale_factors: Noise scale factors (default [1.0, 2.0, 3.0]).
            Ignored for 'adaptive'. Integer factors recommended with
            global folding.
        shots: Shots per circuit execution.
        fold_method: 'global' (deterministic, fold_global) or 'random'
            (fold_gates_at_random).
        asymptote: Infinite-noise limit for exponential/adaptive
            extrapolation. 0.0 is exact for traceless observables under
            depolarizing noise; None triggers a non-linear fit that can
            fail to converge.
        adaptive_steps: Number of executions for 'adaptive' (>= 3).
        seed: Seed for random gate folding (only used when
            fold_method='random').
        circuit_transform: Optional execute-time transform applied to
            every (folded) circuit, e.g. coherent angle miscalibration.
        sampled: If True, every execution estimates the observable from
            ``shots`` measured shots rather than a noiseless-readout
            expectation value.

    Returns:
        MitigationResult with the zero-noise extrapolated value. The
        metadata records ``n_circuit_executions`` and ``total_shots``
        for shot-budget accounting.

    Raises:
        ValueError: If extrapolation or fold_method is unknown.
    """
    if scale_factors is None:
        scale_factors = list(DEFAULT_SCALE_FACTORS)

    if fold_method == "global":
        scale_noise = fold_global
    elif fold_method == "random":

        def scale_noise(circ: QuantumCircuit, factor: float) -> QuantumCircuit:
            return fold_gates_at_random(circ, factor, seed=seed)

    else:
        raise ValueError(
            f"Unknown fold_method: {fold_method!r}. Expected 'global' or 'random'."
        )

    factory = _make_zne_factory(
        extrapolation, scale_factors, asymptote, adaptive_steps
    )
    executor = _CountingExecutor(
        observable, aer_noise_model, shots,
        circuit_transform=circuit_transform, sampled=sampled, backend=backend,
        hw_mode=hw_mode,
    )

    mitigated = execute_with_zne(
        circuit,
        executor,
        factory=factory,
        scale_noise=scale_noise,
    )

    # Actual scale factors / values used (informative for 'adaptive').
    used_scale_factors = [float(s) for s in factory.get_scale_factors()]
    raw_values = [float(v) for v in factory.get_expectation_values()]

    return MitigationResult(
        mitigated_value=float(mitigated),
        raw_values=raw_values,
        method=f"mitiq-ZNE-{extrapolation}",
        overhead=float(executor.n_executions),
        metadata={
            "library": "mitiq",
            "extrapolation": extrapolation,
            "fold_method": fold_method,
            "scale_factors": used_scale_factors,
            "asymptote": asymptote,
            "n_circuit_executions": executor.n_executions,
            "total_shots": executor.n_executions * shots,
        },
    )


def transpile_to_cdr_basis(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    atol: float = 1e-7,
) -> Tuple[QuantumCircuit, float]:
    """Transpile a circuit to the CDR gate set and verify equivalence.

    Mitiq CDR requires every non-Clifford gate to be an RZ rotation, so
    the circuit is transpiled to the {rz, sx, x, cx} basis. Equivalence
    is checked by comparing exact statevector expectation values.

    Args:
        circuit: Bound quantum circuit.
        observable: Observable used for the equivalence check.
        atol: Absolute tolerance for the expectation-value check.

    Returns:
        Tuple of (transpiled circuit, |expectation difference|).

    Raises:
        RuntimeError: If the transpiled circuit's exact expectation
            value deviates from the original by more than atol.
    """
    transpiled = transpile(
        circuit, basis_gates=CDR_BASIS_GATES, optimization_level=1
    )
    original_value = float(
        np.real(Statevector(circuit).expectation_value(observable))
    )
    transpiled_value = float(
        np.real(Statevector(transpiled).expectation_value(observable))
    )
    diff = abs(transpiled_value - original_value)
    if diff > atol:
        raise RuntimeError(
            f"Transpilation to CDR basis changed the expectation value by "
            f"{diff:.2e} (> atol={atol:.2e})."
        )
    return transpiled, diff


def mitiq_cdr(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    aer_noise_model: AerNoiseModel,
    num_training_circuits: int = 30,
    fraction_non_clifford: float = 0.5,
    shots: int = 8192,
    seed: Optional[int] = None,
    circuit_transform: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
    sampled: bool = False,
    skip_transpile: bool = False,
    backend=None,
    hw_mode=None,
) -> MitigationResult:
    """Apply mitiq Clifford Data Regression (CDR) to mitigate errors.

    The input circuit is first transpiled to the {rz, sx, x, cx} basis
    (required by CDR: non-Clifford content must live in RZ rotations)
    with an exact expectation-value equivalence check. Near-Clifford
    training circuits are executed on the noisy executor and simulated
    classically (noise-free) to fit a linear noisy-to-exact map.

    Args:
        circuit: Bound quantum circuit.
        observable: Observable to measure (must be diagonal, e.g. a sum
            of Z terms, for the classical near-Clifford simulation).
        aer_noise_model: Aer noise model used by the noisy executor.
        num_training_circuits: Number of near-Clifford training circuits.
        fraction_non_clifford: Fraction of non-Clifford RZ gates kept in
            each training circuit.
        shots: Shots per circuit execution.
        seed: Random state for training-circuit generation.
        circuit_transform: Optional execute-time transform applied to
            every noisily-executed circuit (target and near-Clifford
            training circuits), e.g. coherent angle miscalibration. The
            noise-free classical simulator is NOT transformed — it
            represents exact classical simulation of the requested
            circuit.
        sampled: If True, noisy executions estimate the observable from
            ``shots`` measured shots. The classical simulator stays
            exact.
        skip_transpile: If True, trust that the circuit is already in
            the CDR {rz, sx, x, cx} basis and skip re-transpilation.

    Returns:
        MitigationResult with the CDR-corrected value. The metadata
        records ``n_circuit_executions`` (noisy quantum executions:
        num_training_circuits + 1) and ``total_shots``.
    """
    if skip_transpile:
        transpiled, transpile_diff = circuit, 0.0
    else:
        transpiled, transpile_diff = transpile_to_cdr_basis(circuit, observable)

    noisy_executor = _CountingExecutor(
        observable, aer_noise_model, shots,
        circuit_transform=circuit_transform, sampled=sampled, backend=backend,
        hw_mode=hw_mode,
    )
    # The classical CDR training labels stay exact, noise-free, and
    # in-simulation regardless of where the noisy executor runs.
    ideal_simulator = _CountingExecutor(observable, None, shots)

    # Raw noisy value of the (transpiled) target circuit, recorded for
    # diagnostics; not counted toward the CDR quantum cost.
    raw_noisy_value = noisy_executor(transpiled)
    noisy_executor.n_executions -= 1

    mitigated = execute_with_cdr(
        transpiled,
        noisy_executor,
        simulator=ideal_simulator,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        random_state=seed,
    )

    return MitigationResult(
        mitigated_value=float(mitigated),
        raw_values=[raw_noisy_value],
        method="mitiq-CDR",
        overhead=float(num_training_circuits + 1),
        metadata={
            "library": "mitiq",
            "num_training_circuits": num_training_circuits,
            "fraction_non_clifford": fraction_non_clifford,
            "basis_gates": list(CDR_BASIS_GATES),
            "transpile_expectation_diff": transpile_diff,
            "n_circuit_executions": noisy_executor.n_executions,
            "total_shots": noisy_executor.n_executions * shots,
            "n_simulator_executions": ideal_simulator.n_executions,
        },
    )
