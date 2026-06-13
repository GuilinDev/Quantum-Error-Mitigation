"""Data generation for training error mitigation models.

This module provides utilities to generate training data by running
quantum circuits under various noise conditions.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..quantum.circuits import VQECircuit, QAOACircuit, VariationalCircuit
from ..quantum.noise_models import (
    NoiseModel,
    VariableNoiseModel,
    NoiseParameters,
    DepolarizingNoise,
)
from ..utils.qiskit_compat import run_estimation

#: Largest qubit count for which exact density-matrix simulation of the
#: noisy circuit is cheap (memory grows as 4**n: 16 MB at 10 qubits but
#: 64 GB at 16 qubits). Beyond this, statevector trajectory sampling is used.
DENSITY_MATRIX_MAX_QUBITS = 10

#: Simulation methods accepted by QuantumDataGenerator.
SIM_METHODS = ("auto", "density_matrix", "statevector")


@dataclass
class DataSample:
    """Single data sample for training."""

    circuit_features: np.ndarray
    noise_features: np.ndarray
    noisy_value: float
    ideal_value: float
    error: float


class QuantumDataGenerator:
    """Generator for quantum error mitigation training data.

    Generates paired (noisy, ideal) expectation values for training
    neural error mitigation models.
    """

    def __init__(
        self,
        circuit_type: str = "vqe",
        num_qubits: int = 4,
        num_layers: int = 2,
        noise_model: Optional[NoiseModel] = None,
        shots: int = 8192,
        sim_method: str = "auto",
        seed: Optional[int] = None,
    ):
        """Initialize data generator.

        Args:
            circuit_type: Type of circuit ('vqe' or 'qaoa').
            num_qubits: Number of qubits.
            num_layers: Number of variational layers.
            noise_model: Noise model for noisy simulations.
            shots: Number of noise trajectories for statevector sampling
                of the noisy expectation value. Ignored when the noisy
                simulation is exact (density_matrix method).
            sim_method: Method for the noisy simulation. 'density_matrix'
                computes the exact noisy expectation (memory grows as 4**n),
                'statevector' Monte Carlo averages over ``shots`` noise
                trajectories (memory 2**n, error ~1/sqrt(shots)), and
                'auto' picks density_matrix up to
                DENSITY_MATRIX_MAX_QUBITS qubits and statevector above.
                Ideal labels are always computed exactly via noiseless
                statevector simulation, independent of this setting.
            seed: Random seed.

        Raises:
            ValueError: If sim_method is not one of SIM_METHODS.
        """
        if sim_method not in SIM_METHODS:
            raise ValueError(
                f"Unknown sim_method: {sim_method}. Supported: {SIM_METHODS}"
            )

        self.circuit_type = circuit_type
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.shots = shots
        self.sim_method = sim_method
        self.rng = np.random.default_rng(seed)

        # Set up noise model
        if noise_model is None:
            self.noise_model = VariableNoiseModel(seed=seed)
        else:
            self.noise_model = noise_model

        # Create circuit template
        if circuit_type == "vqe":
            self.circuit_template = VQECircuit(num_qubits, num_layers)
            self.observable = self._create_vqe_observable()
        elif circuit_type == "qaoa":
            self.circuit_template = QAOACircuit(num_qubits, num_layers)
            self.observable = self.circuit_template.get_cost_hamiltonian()
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

    @property
    def noisy_sim_method(self) -> str:
        """Aer method used for noisy simulation after resolving 'auto'.

        Returns:
            'density_matrix' (exact noisy expectation) for small circuits,
            'statevector' (trajectory sampling) for large ones.
        """
        if self.sim_method == "auto":
            if self.num_qubits <= DENSITY_MATRIX_MAX_QUBITS:
                return "density_matrix"
            return "statevector"
        return self.sim_method

    def _create_vqe_observable(self) -> SparsePauliOp:
        """Create a simple observable for VQE."""
        # Use sum of Z operators as observable
        terms = []
        coeffs = []
        for i in range(self.num_qubits):
            pauli = ["I"] * self.num_qubits
            pauli[i] = "Z"
            terms.append("".join(reversed(pauli)))
            coeffs.append(1.0 / self.num_qubits)
        return SparsePauliOp(terms, coeffs)

    def generate_sample(
        self,
        params: Optional[np.ndarray] = None,
        sample_noise: bool = True,
    ) -> DataSample:
        """Generate a single training sample.

        Args:
            params: Circuit parameters. Random if None.
            sample_noise: Whether to sample new noise parameters.

        Returns:
            DataSample with circuit features, noise features, and values.
        """
        # Generate random parameters if not provided
        if params is None:
            params = self.rng.uniform(
                0, 2 * np.pi, size=self.circuit_template.num_parameters
            )

        # Build circuit with parameters
        circuit = self.circuit_template.bind_parameters(params)

        # Sample noise parameters if using variable noise
        if sample_noise and isinstance(self.noise_model, VariableNoiseModel):
            noise_model, noise_params = self.noise_model.sample_and_build()
        else:
            noise_model = self.noise_model.build_noise_model()
            noise_params = self.noise_model.params

        # Run ideal simulation (exact, no shot noise)
        ideal_value = self._run_ideal(circuit)

        # Run noisy simulation
        noisy_value = self._run_noisy(circuit, noise_model)

        # Compute error
        error = noisy_value - ideal_value

        # Extract features
        circuit_features = self.circuit_template.to_feature_vector(params)
        noise_features = np.array(list(self.noise_model.to_feature_dict().values()))

        return DataSample(
            circuit_features=circuit_features,
            noise_features=noise_features,
            noisy_value=noisy_value,
            ideal_value=ideal_value,
            error=error,
        )

    def _run_ideal(self, circuit: QuantumCircuit) -> float:
        """Compute the exact ideal expectation value.

        Uses noiseless statevector simulation with no shot sampling, so
        ideal labels are exact and bit-reproducible.

        Args:
            circuit: Quantum circuit to simulate.

        Returns:
            Exact ideal expectation value of the observable.
        """
        return run_estimation(
            circuit, self.observable, noise_model=None, exact=True
        )

    def _run_noisy(self, circuit: QuantumCircuit, noise_model) -> float:
        """Compute the noisy expectation value.

        Uses the method resolved by :attr:`noisy_sim_method`: exact
        density-matrix simulation where cheap, or a Monte Carlo average
        over ``self.shots`` statevector noise trajectories for larger
        circuits (statistical error ~1/sqrt(shots)).

        Args:
            circuit: Quantum circuit to simulate.
            noise_model: Aer noise model to apply.

        Returns:
            Noisy expectation value of the observable.
        """
        method = self.noisy_sim_method
        if method == "density_matrix":
            # Exact noisy expectation: one deterministic execution suffices.
            return run_estimation(
                circuit,
                self.observable,
                shots=1,
                noise_model=noise_model,
                method=method,
            )
        return run_estimation(
            circuit,
            self.observable,
            shots=self.shots,
            noise_model=noise_model,
            method=method,
        )

    def generate_dataset(
        self,
        n_samples: int,
        show_progress: bool = True,
    ) -> List[DataSample]:
        """Generate multiple training samples.

        Args:
            n_samples: Number of samples to generate.
            show_progress: Whether to show progress bar.

        Returns:
            List of DataSample objects.
        """
        samples = []
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating data")

        for _ in iterator:
            sample = self.generate_sample()
            samples.append(sample)

        return samples


class MitigationDataset(Dataset):
    """PyTorch Dataset for error mitigation training."""

    def __init__(
        self,
        samples: Optional[List[DataSample]] = None,
        data_file: Optional[str] = None,
    ):
        """Initialize dataset.

        Args:
            samples: List of DataSample objects.
            data_file: Path to saved data file (alternative to samples).
        """
        if samples is not None:
            self.circuit_features = np.stack([s.circuit_features for s in samples])
            self.noise_features = np.stack([s.noise_features for s in samples])
            self.noisy_values = np.array([s.noisy_value for s in samples])
            self.ideal_values = np.array([s.ideal_value for s in samples])
            self.errors = np.array([s.error for s in samples])
        elif data_file is not None:
            self._load_from_file(data_file)
        else:
            raise ValueError("Either samples or data_file must be provided")

    def _load_from_file(self, path: str):
        """Load dataset from file."""
        data = np.load(path)
        self.circuit_features = data["circuit_features"]
        self.noise_features = data["noise_features"]
        self.noisy_values = data["noisy_values"]
        self.ideal_values = data["ideal_values"]
        self.errors = data["errors"]

    def save(self, path: str):
        """Save dataset to file."""
        np.savez(
            path,
            circuit_features=self.circuit_features,
            noise_features=self.noise_features,
            noisy_values=self.noisy_values,
            ideal_values=self.ideal_values,
            errors=self.errors,
        )

    def __len__(self) -> int:
        return len(self.noisy_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "circuit_features": torch.tensor(
                self.circuit_features[idx], dtype=torch.float32
            ),
            "noise_features": torch.tensor(
                self.noise_features[idx], dtype=torch.float32
            ),
            "noisy_value": torch.tensor(
                self.noisy_values[idx], dtype=torch.float32
            ).unsqueeze(0),
            "ideal_value": torch.tensor(
                self.ideal_values[idx], dtype=torch.float32
            ).unsqueeze(0),
            "error": torch.tensor(self.errors[idx], dtype=torch.float32).unsqueeze(0),
        }


def create_dataloaders(
    train_samples: List[DataSample],
    val_samples: List[DataSample],
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        train_samples: Training samples.
        val_samples: Validation samples.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = MitigationDataset(train_samples)
    val_dataset = MitigationDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class MultiTaskDataGenerator(QuantumDataGenerator):
    """Data generator for multiple circuit types and sizes.

    Generates diverse training data spanning different circuit
    configurations for better generalization.
    """

    def __init__(
        self,
        qubit_range: Tuple[int, int] = (2, 8),
        layer_range: Tuple[int, int] = (1, 4),
        circuit_types: List[str] = ["vqe", "qaoa"],
        shots: int = 8192,
        sim_method: str = "auto",
        seed: Optional[int] = None,
    ):
        """Initialize multi-task generator.

        Args:
            qubit_range: Range of qubit counts (min, max).
            layer_range: Range of layer counts (min, max).
            circuit_types: List of circuit types to include.
            shots: Number of noise trajectories per noisy simulation.
            sim_method: Noisy simulation method ('auto', 'density_matrix',
                or 'statevector'); see QuantumDataGenerator.
            seed: Random seed.

        Raises:
            ValueError: If sim_method is not one of SIM_METHODS.
        """
        if sim_method not in SIM_METHODS:
            raise ValueError(
                f"Unknown sim_method: {sim_method}. Supported: {SIM_METHODS}"
            )

        self.qubit_range = qubit_range
        self.layer_range = layer_range
        self.circuit_types = circuit_types
        self.shots = shots
        self.sim_method = sim_method
        self.rng = np.random.default_rng(seed)

        # Create noise model
        self.noise_model = VariableNoiseModel(seed=seed)

    def generate_sample(self) -> DataSample:
        """Generate a sample with random circuit configuration."""
        # Random configuration
        num_qubits = self.rng.integers(self.qubit_range[0], self.qubit_range[1] + 1)
        num_layers = self.rng.integers(self.layer_range[0], self.layer_range[1] + 1)
        circuit_type = self.rng.choice(self.circuit_types)

        # Create generator for this configuration
        generator = QuantumDataGenerator(
            circuit_type=circuit_type,
            num_qubits=num_qubits,
            num_layers=num_layers,
            noise_model=self.noise_model,
            shots=self.shots,
            sim_method=self.sim_method,
        )

        return generator.generate_sample()
