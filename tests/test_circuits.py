"""Unit tests for quantum circuit implementations."""

import pytest
import numpy as np
from qiskit import QuantumCircuit

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.circuits import (
    VQECircuit,
    QAOACircuit,
    MolecularVQECircuit,
    create_random_graph,
    create_regular_graph,
)


class TestVQECircuit:
    """Tests for VQE circuit implementation."""

    def test_initialization(self):
        """Test VQE circuit initialization."""
        vqe = VQECircuit(num_qubits=4, num_layers=2)
        assert vqe.num_qubits == 4
        assert vqe.num_layers == 2

    def test_num_parameters(self):
        """Test parameter count calculation."""
        # Default rotation gates: ['ry', 'rz']
        vqe = VQECircuit(num_qubits=4, num_layers=2)
        expected_params = 4 * 2 * 2  # 4 qubits, 2 rotations, 2 layers
        assert vqe.num_parameters == expected_params

    def test_build_circuit(self):
        """Test circuit construction."""
        vqe = VQECircuit(num_qubits=3, num_layers=1)
        circuit = vqe.get_circuit()

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 3

    def test_bind_parameters(self):
        """Test parameter binding."""
        vqe = VQECircuit(num_qubits=2, num_layers=1)
        params = np.random.uniform(0, 2 * np.pi, vqe.num_parameters)

        bound_circuit = vqe.bind_parameters(params)
        assert isinstance(bound_circuit, QuantumCircuit)
        # Bound circuit should have no free parameters
        assert len(bound_circuit.parameters) == 0

    def test_wrong_parameter_count(self):
        """Test error on wrong parameter count."""
        vqe = VQECircuit(num_qubits=2, num_layers=1)
        wrong_params = np.zeros(vqe.num_parameters + 1)

        with pytest.raises(ValueError):
            vqe.bind_parameters(wrong_params)

    def test_entanglement_linear(self):
        """Test linear entanglement pattern."""
        vqe = VQECircuit(num_qubits=4, num_layers=2, entanglement="linear")
        circuit = vqe.get_circuit()

        # Count CNOT gates
        cnot_count = sum(1 for inst in circuit.data if inst.operation.name == "cx")
        # Linear entanglement: (n-1) CNOTs per layer, (num_layers-1) entanglement layers
        expected_cnots = (4 - 1) * (2 - 1)
        assert cnot_count == expected_cnots

    def test_to_feature_vector(self):
        """Test feature vector generation."""
        vqe = VQECircuit(num_qubits=4, num_layers=2)
        params = np.random.uniform(0, 2 * np.pi, vqe.num_parameters)

        features = vqe.to_feature_vector(params)

        # Features should include: num_qubits, num_layers, num_parameters, then params
        expected_length = 3 + vqe.num_parameters
        assert len(features) == expected_length
        assert features[0] == 4  # num_qubits
        assert features[1] == 2  # num_layers


class TestQAOACircuit:
    """Tests for QAOA circuit implementation."""

    def test_initialization(self):
        """Test QAOA circuit initialization."""
        qaoa = QAOACircuit(num_qubits=5, num_layers=2)
        assert qaoa.num_qubits == 5
        assert qaoa.num_layers == 2

    def test_num_parameters(self):
        """Test QAOA parameter count (2 per layer)."""
        qaoa = QAOACircuit(num_qubits=5, num_layers=3)
        assert qaoa.num_parameters == 6  # 2 * 3 layers

    def test_build_circuit(self):
        """Test QAOA circuit construction."""
        qaoa = QAOACircuit(num_qubits=4, num_layers=2)
        circuit = qaoa.get_circuit()

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

        # Should start with Hadamard gates
        first_gates = [inst.operation.name for inst in circuit.data[:4]]
        assert all(g == "h" for g in first_gates)

    def test_custom_graph(self):
        """Test QAOA with custom graph."""
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        qaoa = QAOACircuit(num_qubits=4, num_layers=1, graph_edges=edges)

        assert qaoa.graph_edges == edges
        circuit = qaoa.get_circuit()
        assert circuit.num_qubits == 4

    def test_cost_hamiltonian(self):
        """Test MaxCut Hamiltonian generation."""
        edges = [(0, 1), (1, 2)]
        qaoa = QAOACircuit(num_qubits=3, num_layers=1, graph_edges=edges)

        hamiltonian = qaoa.get_cost_hamiltonian()

        # Should have terms for each edge
        assert len(hamiltonian.paulis) == 2


class TestMolecularVQECircuit:
    """Tests for molecular VQE circuit."""

    def test_h2_circuit(self):
        """Test H2 molecular circuit."""
        vqe = MolecularVQECircuit(molecule="H2", num_layers=2)

        assert vqe.num_qubits == 4  # Minimal basis H2
        assert vqe.molecule == "H2"

    def test_lih_circuit(self):
        """Test LiH molecular circuit."""
        vqe = MolecularVQECircuit(molecule="LiH", num_layers=2)

        assert vqe.num_qubits == 10  # Reduced active space LiH

    def test_h2_hamiltonian(self):
        """Test H2 Hamiltonian generation."""
        vqe = MolecularVQECircuit(molecule="H2")
        hamiltonian = vqe.get_molecular_hamiltonian()

        # Should be a SparsePauliOp
        assert hasattr(hamiltonian, "paulis")

    def test_unsupported_molecule(self):
        """Test error for unsupported molecule."""
        with pytest.raises(ValueError):
            MolecularVQECircuit(molecule="CH4")


class TestGraphGeneration:
    """Tests for graph generation utilities."""

    def test_random_graph(self):
        """Test random graph generation."""
        edges = create_random_graph(5, edge_probability=0.5, seed=42)

        assert isinstance(edges, list)
        for edge in edges:
            assert len(edge) == 2
            assert edge[0] < edge[1]  # Canonical ordering
            assert 0 <= edge[0] < 5
            assert 0 <= edge[1] < 5

    def test_random_graph_seed(self):
        """Test random graph reproducibility."""
        edges1 = create_random_graph(5, seed=42)
        edges2 = create_random_graph(5, seed=42)

        assert edges1 == edges2

    def test_regular_graph(self):
        """Test regular graph generation."""
        edges = create_regular_graph(6, degree=2)

        # Check connectivity
        node_degrees = {i: 0 for i in range(6)}
        for i, j in edges:
            node_degrees[i] += 1
            node_degrees[j] += 1

        # All nodes should have approximately the same degree
        degrees = list(node_degrees.values())
        assert max(degrees) - min(degrees) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
