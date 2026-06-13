"""Noise models for simulating NISQ device behavior.

This module provides configurable noise models that closely approximate
real quantum hardware noise characteristics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer.noise import (
    NoiseModel as AerNoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
    pauli_error,
)


@dataclass
class NoiseParameters:
    """Container for noise model parameters."""

    # Single-qubit gate errors
    single_qubit_error: float = 0.001
    # Two-qubit gate errors
    two_qubit_error: float = 0.01
    # Readout errors
    readout_error_0: float = 0.02  # P(1|0)
    readout_error_1: float = 0.03  # P(0|1)
    # Relaxation times (in microseconds)
    t1: float = 100.0
    t2: float = 80.0
    # Gate times (in microseconds)
    single_gate_time: float = 0.05
    two_gate_time: float = 0.3


class NoiseModel(ABC):
    """Abstract base class for quantum noise models."""

    def __init__(self, params: Optional[NoiseParameters] = None):
        """Initialize noise model with given parameters.

        Args:
            params: Noise parameters. Uses defaults if None.
        """
        self.params = params or NoiseParameters()

    @abstractmethod
    def build_noise_model(self) -> AerNoiseModel:
        """Build and return a Qiskit Aer noise model.

        Returns:
            Configured AerNoiseModel.
        """
        pass

    def get_noise_strength(self) -> float:
        """Return a scalar representing overall noise strength.

        Returns:
            Scalar noise strength for feature engineering.
        """
        return (
            self.params.single_qubit_error
            + self.params.two_qubit_error
            + (self.params.readout_error_0 + self.params.readout_error_1) / 2
        )

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert noise parameters to a feature dictionary.

        Returns:
            Dictionary of noise features for neural network input.
        """
        return {
            "single_qubit_error": self.params.single_qubit_error,
            "two_qubit_error": self.params.two_qubit_error,
            "readout_error_0": self.params.readout_error_0,
            "readout_error_1": self.params.readout_error_1,
            "t1": self.params.t1,
            "t2": self.params.t2,
            "single_gate_time": self.params.single_gate_time,
            "two_gate_time": self.params.two_gate_time,
        }


class DepolarizingNoise(NoiseModel):
    """Depolarizing noise model.

    Applies depolarizing errors to single and two-qubit gates,
    plus readout errors.
    """

    def build_noise_model(self) -> AerNoiseModel:
        """Build depolarizing noise model.

        Returns:
            AerNoiseModel with depolarizing errors.
        """
        noise_model = AerNoiseModel()

        # Single-qubit depolarizing error
        error_1q = depolarizing_error(self.params.single_qubit_error, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["rx", "ry", "rz", "h", "x", "y", "z", "s", "t"]
        )

        # Two-qubit depolarizing error
        error_2q = depolarizing_error(self.params.two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])

        # Readout errors
        readout_error = ReadoutError(
            [
                [1 - self.params.readout_error_0, self.params.readout_error_0],
                [self.params.readout_error_1, 1 - self.params.readout_error_1],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model


class ThermalRelaxationNoise(NoiseModel):
    """Thermal relaxation noise model.

    Models T1/T2 relaxation along with gate errors.
    """

    def build_noise_model(self) -> AerNoiseModel:
        """Build thermal relaxation noise model.

        Returns:
            AerNoiseModel with thermal relaxation errors.
        """
        noise_model = AerNoiseModel()

        # Single-qubit thermal relaxation
        error_1q = thermal_relaxation_error(
            t1=self.params.t1,
            t2=self.params.t2,
            time=self.params.single_gate_time,
        )
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["rx", "ry", "rz", "h", "x", "y", "z", "s", "t"]
        )

        # Two-qubit thermal relaxation
        error_2q_1 = thermal_relaxation_error(
            t1=self.params.t1,
            t2=self.params.t2,
            time=self.params.two_gate_time,
        )
        error_2q = error_2q_1.tensor(error_2q_1)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])

        # Readout errors
        readout_error = ReadoutError(
            [
                [1 - self.params.readout_error_0, self.params.readout_error_0],
                [self.params.readout_error_1, 1 - self.params.readout_error_1],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model


class RealisticDeviceNoise(NoiseModel):
    """Realistic device noise model combining multiple noise sources.

    This model combines depolarizing noise, thermal relaxation,
    and coherent errors to approximate real IBMQ device noise.
    """

    def __init__(
        self,
        params: Optional[NoiseParameters] = None,
        include_crosstalk: bool = True,
        crosstalk_strength: float = 0.001,
    ):
        """Initialize realistic noise model.

        Args:
            params: Base noise parameters.
            include_crosstalk: Whether to include crosstalk errors.
            crosstalk_strength: Strength of crosstalk between neighboring qubits.
        """
        super().__init__(params)
        self.include_crosstalk = include_crosstalk
        self.crosstalk_strength = crosstalk_strength

    def build_noise_model(self) -> AerNoiseModel:
        """Build realistic device noise model.

        Returns:
            AerNoiseModel approximating real device behavior.
        """
        noise_model = AerNoiseModel()

        # Combined single-qubit error (depolarizing + thermal)
        depol_1q = depolarizing_error(self.params.single_qubit_error * 0.7, 1)
        thermal_1q = thermal_relaxation_error(
            t1=self.params.t1,
            t2=self.params.t2,
            time=self.params.single_gate_time,
        )
        error_1q = depol_1q.compose(thermal_1q)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["rx", "ry", "rz", "h", "x", "y", "z"]
        )

        # Combined two-qubit error
        depol_2q = depolarizing_error(self.params.two_qubit_error * 0.7, 2)
        thermal_2q_1 = thermal_relaxation_error(
            t1=self.params.t1,
            t2=self.params.t2,
            time=self.params.two_gate_time,
        )
        thermal_2q = thermal_2q_1.tensor(thermal_2q_1)
        error_2q = depol_2q.compose(thermal_2q)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

        # Readout errors
        readout_error = ReadoutError(
            [
                [1 - self.params.readout_error_0, self.params.readout_error_0],
                [self.params.readout_error_1, 1 - self.params.readout_error_1],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model


class VariableNoiseModel(NoiseModel):
    """Noise model with variable parameters for training data generation.

    Allows sampling noise parameters from distributions to generate
    diverse training data for neural error mitigation.
    """

    def __init__(
        self,
        base_params: Optional[NoiseParameters] = None,
        error_range: Tuple[float, float] = (0.001, 0.05),
        seed: Optional[int] = None,
    ):
        """Initialize variable noise model.

        Args:
            base_params: Base noise parameters.
            error_range: Range for sampling error rates.
            seed: Random seed for reproducibility.
        """
        super().__init__(base_params)
        self.error_range = error_range
        self.rng = np.random.default_rng(seed)

    def sample_parameters(self) -> NoiseParameters:
        """Sample random noise parameters within specified ranges.

        Returns:
            Randomly sampled NoiseParameters.
        """
        low, high = self.error_range

        # Sample T1 first, then ensure T2 <= 2*T1 (physical constraint)
        # Scale T1/T2 inversely with error rate (higher noise = shorter coherence)
        t1_scale = 1.0 - (low + high) / 2  # Reduce T1 for higher noise
        t1 = self.rng.uniform(30 * t1_scale + 20, 150 * t1_scale + 50)
        t2_max = min(2 * t1, 200)  # T2 cannot exceed 2*T1
        t2 = self.rng.uniform(max(15, t2_max * 0.3), t2_max * 0.9)

        # Compute sub-ranges ensuring low <= high for all parameters
        single_low = max(low / 10, 0.0001)
        single_high = max(high / 10, single_low * 2)

        readout_low = max(low / 2, 0.001)
        readout_high = max(high / 2, readout_low * 2)

        return NoiseParameters(
            single_qubit_error=self.rng.uniform(single_low, single_high),
            two_qubit_error=self.rng.uniform(low, high),
            readout_error_0=self.rng.uniform(readout_low, readout_high),
            readout_error_1=self.rng.uniform(readout_low, min(readout_high * 1.5, 0.2)),
            t1=t1,
            t2=t2,
            single_gate_time=self.rng.uniform(0.02, 0.1),
            two_gate_time=self.rng.uniform(0.1, 0.5),
        )

    def build_noise_model(self) -> AerNoiseModel:
        """Build noise model with current parameters.

        Returns:
            AerNoiseModel with current noise parameters.
        """
        # Use RealisticDeviceNoise for building
        builder = RealisticDeviceNoise(self.params)
        return builder.build_noise_model()

    def sample_and_build(self) -> Tuple[AerNoiseModel, NoiseParameters]:
        """Sample new parameters and build corresponding noise model.

        Returns:
            Tuple of (noise model, sampled parameters).
        """
        self.params = self.sample_parameters()
        return self.build_noise_model(), self.params


class NonLinearCorrelatedNoise(NoiseModel):
    """Non-linear correlated noise that violates ZNE scaling assumptions.

    Combines three effects that break smooth noise amplification:
    stochastic gate-error fluctuations (a fresh realization per circuit
    instance), correlated ZZ crosstalk on two-qubit gates, and asymmetric
    readout error whose magnitude grows with the nonlinearity setting.
    Readout bias in particular is not amplified by gate folding, which
    genuinely limits folding-based ZNE in this regime.

    The feature vector exposes the regime parameters (nonlinearity,
    correlation) and the base device profile, but NOT the per-instance
    random draws: those are aleatoric variation that every mitigation
    method must face at test time.
    """

    def __init__(
        self,
        params: Optional[NoiseParameters] = None,
        base_error: float = 0.02,
        nonlinearity: float = 0.3,
        correlation: float = 0.2,
        seed: Optional[int] = None,
    ):
        """Initialize non-linear correlated noise.

        Args:
            params: Base device profile (T1/T2, gate times). Defaults used
                if None; the gate/readout errors below override the profile.
            base_error: Base two-qubit error rate.
            nonlinearity: Strength of stochastic error fluctuation and
                readout asymmetry, in [0, 1].
            correlation: ZZ crosstalk probability as a fraction of
                base_error.
            seed: Random seed for the per-instance realizations.
        """
        super().__init__(params)
        self.base_error = base_error
        self.nonlinearity = nonlinearity
        self.correlation = correlation
        self.rng = np.random.default_rng(seed)

    def build_noise_model(self) -> AerNoiseModel:
        """Build one noise realization (fresh random draws per call).

        Returns:
            AerNoiseModel with non-linear correlated errors.
        """
        noise_model = AerNoiseModel()

        # Stochastic single-qubit error
        eff_1q = self.base_error * 0.1 * (1 + self.nonlinearity * self.rng.random())
        error_1q = depolarizing_error(eff_1q, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["rx", "ry", "rz", "h", "x", "y", "z"]
        )

        # Stochastic two-qubit error with correlated ZZ component
        eff_2q = self.base_error * (1 + self.nonlinearity * self.rng.random())
        error_2q = depolarizing_error(eff_2q, 2)
        if self.correlation > 0:
            zz_prob = self.correlation * self.base_error
            correlated = pauli_error([("II", 1 - zz_prob), ("ZZ", zz_prob)])
            error_2q = error_2q.compose(correlated)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

        # Asymmetric readout error scaling with nonlinearity
        ro_01 = 0.02 + self.nonlinearity * self.rng.random() * 0.05
        ro_10 = 0.05 + self.nonlinearity * self.rng.random() * 0.08
        readout_error = ReadoutError(
            [[1 - ro_01, ro_01], [ro_10, 1 - ro_10]]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

    def to_feature_dict(self) -> Dict[str, float]:
        """Return noise features (regime parameters, not realizations)."""
        return {
            "base_error": self.base_error,
            "nonlinearity": self.nonlinearity,
            "correlation": self.correlation,
            "expected_2q": self.base_error * (1 + self.nonlinearity * 0.5),
            "expected_readout_asym": 0.03 + self.nonlinearity * 0.065,
            "t1": self.params.t1,
            "t2": self.params.t2,
            "two_gate_time": self.params.two_gate_time,
        }


# Predefined noise profiles approximating real devices
IBMQ_MONTREAL_NOISE = NoiseParameters(
    single_qubit_error=0.0003,
    two_qubit_error=0.008,
    readout_error_0=0.012,
    readout_error_1=0.025,
    t1=120.0,
    t2=90.0,
    single_gate_time=0.035,
    two_gate_time=0.26,
)

IBMQ_TORONTO_NOISE = NoiseParameters(
    single_qubit_error=0.0004,
    two_qubit_error=0.012,
    readout_error_0=0.015,
    readout_error_1=0.030,
    t1=100.0,
    t2=70.0,
    single_gate_time=0.035,
    two_gate_time=0.30,
)

HIGH_NOISE_PROFILE = NoiseParameters(
    single_qubit_error=0.002,
    two_qubit_error=0.05,
    readout_error_0=0.05,
    readout_error_1=0.08,
    t1=50.0,
    t2=30.0,
    single_gate_time=0.05,
    two_gate_time=0.4,
)


def get_device_noise(device_name: str) -> NoiseModel:
    """Get a noise model approximating a specific device.

    Args:
        device_name: Name of the device ('montreal', 'toronto', 'high_noise').

    Returns:
        NoiseModel for the specified device.
    """
    profiles = {
        "montreal": IBMQ_MONTREAL_NOISE,
        "toronto": IBMQ_TORONTO_NOISE,
        "high_noise": HIGH_NOISE_PROFILE,
    }

    if device_name.lower() not in profiles:
        raise ValueError(
            f"Unknown device: {device_name}. Available: {list(profiles.keys())}"
        )

    return RealisticDeviceNoise(profiles[device_name.lower()])
