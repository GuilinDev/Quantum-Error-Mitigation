"""Variational quantum circuit implementations for VQE and QAOA.

This module provides quantum circuit classes optimized for neural error mitigation
experiments on NISQ devices.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


class VariationalCircuit(ABC):
    """Abstract base class for variational quantum circuits."""

    def __init__(self, num_qubits: int, num_layers: int = 1):
        """Initialize a variational circuit.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_layers: Number of variational layers.
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._circuit: Optional[QuantumCircuit] = None
        self._parameters: Optional[ParameterVector] = None

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Return the number of variational parameters."""
        pass

    @abstractmethod
    def build_circuit(self) -> QuantumCircuit:
        """Build and return the parameterized quantum circuit."""
        pass

    def get_circuit(self) -> QuantumCircuit:
        """Get the quantum circuit, building it if necessary."""
        if self._circuit is None:
            self._circuit = self.build_circuit()
        return self._circuit

    def bind_parameters(self, params: np.ndarray) -> QuantumCircuit:
        """Bind parameter values to the circuit.

        Args:
            params: Array of parameter values.

        Returns:
            Quantum circuit with bound parameters.
        """
        circuit = self.get_circuit()
        if len(params) != self.num_parameters:
            raise ValueError(
                f"Expected {self.num_parameters} parameters, got {len(params)}"
            )
        param_dict = dict(zip(self._parameters, params))
        return circuit.assign_parameters(param_dict)

    def to_feature_vector(self, params: np.ndarray) -> np.ndarray:
        """Convert circuit configuration to a feature vector for neural network input.

        Args:
            params: Parameter values.

        Returns:
            Feature vector containing circuit structure information.
        """
        features = [
            self.num_qubits,
            self.num_layers,
            self.num_parameters,
        ]
        features.extend(params.tolist())
        return np.array(features, dtype=np.float32)


class VQECircuit(VariationalCircuit):
    """Variational Quantum Eigensolver circuit with hardware-efficient ansatz.

    This implementation uses a hardware-efficient ansatz suitable for
    molecular ground state energy estimation on NISQ devices.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 2,
        entanglement: str = "linear",
        rotation_gates: List[str] = None,
    ):
        """Initialize VQE circuit.

        Args:
            num_qubits: Number of qubits.
            num_layers: Number of variational layers.
            entanglement: Entanglement pattern ('linear', 'full', 'circular').
            rotation_gates: List of rotation gates to use (default: ['ry', 'rz']).
        """
        super().__init__(num_qubits, num_layers)
        self.entanglement = entanglement
        self.rotation_gates = rotation_gates or ["ry", "rz"]

    @property
    def num_parameters(self) -> int:
        """Return number of variational parameters."""
        rotations_per_layer = self.num_qubits * len(self.rotation_gates)
        return rotations_per_layer * self.num_layers

    def _add_rotation_layer(
        self, circuit: QuantumCircuit, params: ParameterVector, offset: int
    ) -> int:
        """Add a layer of rotation gates.

        Args:
            circuit: Quantum circuit to modify.
            params: Parameter vector.
            offset: Starting index in parameter vector.

        Returns:
            Updated offset.
        """
        idx = offset
        for qubit in range(self.num_qubits):
            for gate in self.rotation_gates:
                if gate == "rx":
                    circuit.rx(params[idx], qubit)
                elif gate == "ry":
                    circuit.ry(params[idx], qubit)
                elif gate == "rz":
                    circuit.rz(params[idx], qubit)
                idx += 1
        return idx

    def _add_entanglement_layer(self, circuit: QuantumCircuit):
        """Add entanglement layer based on specified pattern.

        Args:
            circuit: Quantum circuit to modify.
        """
        if self.entanglement == "linear":
            for i in range(self.num_qubits - 1):
                circuit.cx(i, i + 1)
        elif self.entanglement == "circular":
            for i in range(self.num_qubits - 1):
                circuit.cx(i, i + 1)
            if self.num_qubits > 2:
                circuit.cx(self.num_qubits - 1, 0)
        elif self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    circuit.cx(i, j)

    def build_circuit(self) -> QuantumCircuit:
        """Build the VQE circuit with hardware-efficient ansatz.

        Returns:
            Parameterized quantum circuit.
        """
        self._parameters = ParameterVector("θ", self.num_parameters)
        circuit = QuantumCircuit(self.num_qubits)

        param_idx = 0
        for layer in range(self.num_layers):
            param_idx = self._add_rotation_layer(circuit, self._parameters, param_idx)
            if layer < self.num_layers - 1:
                self._add_entanglement_layer(circuit)

        return circuit


class QAOACircuit(VariationalCircuit):
    """Quantum Approximate Optimization Algorithm circuit.

    Implements QAOA for combinatorial optimization problems,
    particularly MaxCut.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        graph_edges: Optional[List[Tuple[int, int]]] = None,
        edge_weights: Optional[List[float]] = None,
    ):
        """Initialize QAOA circuit.

        Args:
            num_qubits: Number of qubits (nodes in graph).
            num_layers: Number of QAOA layers (p parameter).
            graph_edges: List of edges as (i, j) tuples.
            edge_weights: Weights for each edge (default: all 1.0).
        """
        super().__init__(num_qubits, num_layers)
        self.graph_edges = graph_edges or self._default_graph()
        self.edge_weights = edge_weights or [1.0] * len(self.graph_edges)

    def _default_graph(self) -> List[Tuple[int, int]]:
        """Create a default linear graph."""
        return [(i, i + 1) for i in range(self.num_qubits - 1)]

    @property
    def num_parameters(self) -> int:
        """Return number of QAOA parameters (2 per layer: gamma and beta)."""
        return 2 * self.num_layers

    def _add_cost_layer(self, circuit: QuantumCircuit, gamma: Parameter):
        """Add cost unitary layer (problem Hamiltonian).

        Args:
            circuit: Quantum circuit to modify.
            gamma: Gamma parameter for this layer.
        """
        for (i, j), weight in zip(self.graph_edges, self.edge_weights):
            circuit.cx(i, j)
            circuit.rz(2 * gamma * weight, j)
            circuit.cx(i, j)

    def _add_mixer_layer(self, circuit: QuantumCircuit, beta: Parameter):
        """Add mixer unitary layer.

        Args:
            circuit: Quantum circuit to modify.
            beta: Beta parameter for this layer.
        """
        for qubit in range(self.num_qubits):
            circuit.rx(2 * beta, qubit)

    def build_circuit(self) -> QuantumCircuit:
        """Build the QAOA circuit.

        Returns:
            Parameterized QAOA circuit.
        """
        self._parameters = ParameterVector("θ", self.num_parameters)
        circuit = QuantumCircuit(self.num_qubits)

        # Initial state: uniform superposition
        circuit.h(range(self.num_qubits))

        # Alternating cost and mixer layers
        for p in range(self.num_layers):
            gamma = self._parameters[2 * p]
            beta = self._parameters[2 * p + 1]
            self._add_cost_layer(circuit, gamma)
            self._add_mixer_layer(circuit, beta)

        return circuit

    def get_cost_hamiltonian(self) -> SparsePauliOp:
        """Get the MaxCut cost Hamiltonian as a SparsePauliOp.

        Returns:
            Cost Hamiltonian for MaxCut problem.
        """
        pauli_terms = []
        coeffs = []

        for (i, j), weight in zip(self.graph_edges, self.edge_weights):
            # MaxCut: C = 0.5 * sum_{(i,j)} w_ij * (1 - Z_i Z_j)
            # The constant term is often ignored for optimization
            pauli_str = ["I"] * self.num_qubits
            pauli_str[i] = "Z"
            pauli_str[j] = "Z"
            pauli_terms.append("".join(reversed(pauli_str)))
            coeffs.append(-0.5 * weight)

        return SparsePauliOp(pauli_terms, coeffs)


class MolecularVQECircuit(VQECircuit):
    """VQE circuit specialized for molecular simulations.

    Includes methods for generating molecular Hamiltonians
    for H2 and LiH molecules.
    """

    def __init__(
        self,
        molecule: str = "H2",
        num_layers: int = 2,
        bond_distance: float = 0.735,
    ):
        """Initialize molecular VQE circuit.

        Args:
            molecule: Molecule identifier ('H2' or 'LiH').
            num_layers: Number of variational layers.
            bond_distance: Bond distance in Angstroms.
        """
        self.molecule = molecule
        self.bond_distance = bond_distance

        # Determine qubit count based on molecule
        if molecule == "H2":
            num_qubits = 4  # Minimal basis for H2
        elif molecule == "LiH":
            num_qubits = 10  # Reduced active space for LiH
        else:
            raise ValueError(f"Unsupported molecule: {molecule}")

        super().__init__(num_qubits, num_layers, entanglement="linear")

    def get_molecular_hamiltonian(self) -> SparsePauliOp:
        """Get the molecular Hamiltonian using Jordan-Wigner transformation.

        Returns:
            Molecular Hamiltonian as SparsePauliOp.

        Note:
            This is a simplified Hamiltonian for demonstration.
            For production, use qiskit_nature with PySCF.
        """
        if self.molecule == "H2":
            return self._h2_hamiltonian()
        elif self.molecule == "LiH":
            return self._lih_hamiltonian()

    def _h2_hamiltonian(self) -> SparsePauliOp:
        """Generate simplified H2 Hamiltonian.

        This uses approximate coefficients for H2 at equilibrium geometry.
        """
        # Simplified H2 Hamiltonian coefficients (approximate)
        # Real implementation would use qiskit_nature
        terms = [
            ("IIII", -0.8105),
            ("IIIZ", 0.1715),
            ("IIZI", -0.2222),
            ("IZII", 0.1715),
            ("ZIII", -0.2222),
            ("IIZZ", 0.1209),
            ("IZIZ", 0.1686),
            ("IZZI", 0.0453),
            ("ZIIZ", 0.0453),
            ("ZIZI", 0.1686),
            ("ZZII", 0.1209),
            ("XXXX", 0.0453),
            ("XXYY", 0.0453),
            ("YYXX", 0.0453),
            ("YYYY", 0.0453),
        ]

        paulis = [t[0] for t in terms]
        coeffs = [t[1] for t in terms]
        return SparsePauliOp(paulis, coeffs)

    def _lih_hamiltonian(self) -> SparsePauliOp:
        """Generate simplified LiH Hamiltonian (reduced active space).

        This is a placeholder - real implementation would compute
        from molecular integrals.
        """
        # Simplified placeholder - would need qiskit_nature for real Hamiltonian
        n = self.num_qubits
        terms = [("I" * n, -7.8)]  # Approximate ground state energy

        # Add some single-qubit terms
        for i in range(n):
            pauli = ["I"] * n
            pauli[i] = "Z"
            terms.append(("".join(pauli), 0.1 * (i + 1) / n))

        paulis = [t[0] for t in terms]
        coeffs = [t[1] for t in terms]
        return SparsePauliOp(paulis, coeffs)


def create_random_graph(
    num_nodes: int, edge_probability: float = 0.5, seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """Create a random Erdos-Renyi graph for QAOA experiments.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_probability: Probability of edge between any two nodes.
        seed: Random seed for reproducibility.

    Returns:
        List of edges as (i, j) tuples.
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_probability:
                edges.append((i, j))
    return edges


def create_regular_graph(num_nodes: int, degree: int = 3) -> List[Tuple[int, int]]:
    """Create a d-regular graph for QAOA experiments.

    Args:
        num_nodes: Number of nodes.
        degree: Degree of each node.

    Returns:
        List of edges (approximately d-regular).
    """
    if degree >= num_nodes:
        raise ValueError("Degree must be less than number of nodes")

    edges = []
    for i in range(num_nodes):
        for d in range(1, degree // 2 + 1):
            j = (i + d) % num_nodes
            if (i, j) not in edges and (j, i) not in edges:
                edges.append((min(i, j), max(i, j)))
    return list(set(edges))
