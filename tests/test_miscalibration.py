"""Tests for coherent angle-miscalibration utilities."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile

from src.quantum.circuits import VQECircuit
from src.quantum.miscalibration import (
    DEFAULT_EXEMPT_GATES,
    FIXED_PULSE_ANGLES,
    make_miscalibration_transform,
    miscalibrate_circuit,
)


@pytest.fixture
def bound_circuit():
    vqe = VQECircuit(4, 2)
    params = np.linspace(0.1, 2.0, vqe.num_parameters)
    return vqe.bind_parameters(params)


def rotation_angles(circuit, names=("rx", "ry", "rz")):
    return [
        float(inst.operation.params[0])
        for inst in circuit.data
        if inst.operation.name in names and inst.operation.params
    ]


class TestMiscalibrateCircuit:
    def test_snapshot_deterministic(self, bound_circuit):
        a = miscalibrate_circuit(bound_circuit, 0.05, 0.02, snapshot_seed=7)
        b = miscalibrate_circuit(bound_circuit, 0.05, 0.02, snapshot_seed=7)
        assert np.allclose(rotation_angles(a), rotation_angles(b))

    def test_different_snapshots_differ(self, bound_circuit):
        a = miscalibrate_circuit(bound_circuit, 0.05, 0.02, snapshot_seed=7)
        b = miscalibrate_circuit(bound_circuit, 0.05, 0.02, snapshot_seed=8)
        assert not np.allclose(rotation_angles(a), rotation_angles(b))

    def test_zero_delta_zero_sigma_is_identity(self, bound_circuit):
        out = miscalibrate_circuit(bound_circuit, 0.0, 0.0, snapshot_seed=7)
        assert np.allclose(
            rotation_angles(out), rotation_angles(bound_circuit)
        )

    def test_mean_offset_matches_delta_bias(self, bound_circuit):
        # rz is exempt by default (virtual gate), so measure ry only.
        delta = 0.05
        offsets = []
        for seed in range(200):
            out = miscalibrate_circuit(bound_circuit, delta, 0.01, seed)
            offsets.extend(
                np.array(rotation_angles(out, names=("ry",)))
                - np.array(rotation_angles(bound_circuit, names=("ry",)))
            )
        assert abs(np.mean(offsets) - delta) < 0.005

    def test_exempt_gates_untouched(self, bound_circuit):
        native = transpile(
            bound_circuit, basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
        )
        out = miscalibrate_circuit(native, 0.05, 0.0, snapshot_seed=3)
        rz_in = rotation_angles(native, names=("rz",))
        rz_out = rotation_angles(out, names=("rz",))
        assert np.allclose(rz_in, rz_out), "virtual rz must stay exact"

    def test_fixed_pulses_become_rx(self, bound_circuit):
        native = transpile(
            bound_circuit, basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
        )
        n_sx = sum(1 for i in native.data if i.operation.name in FIXED_PULSE_ANGLES)
        assert n_sx > 0
        out = miscalibrate_circuit(native, 0.05, 0.0, snapshot_seed=3)
        n_rx = sum(1 for i in out.data if i.operation.name == "rx")
        assert n_rx == n_sx
        rx_angles = rotation_angles(out, names=("rx",))
        # sx -> rx(pi/2 + delta): every angle offset by exactly delta
        assert all(
            any(abs(a - (nominal + 0.05)) < 1e-12
                for nominal in FIXED_PULSE_ANGLES.values())
            for a in rx_angles
        )

    def test_gate_count_preserved(self, bound_circuit):
        out = miscalibrate_circuit(bound_circuit, 0.05, 0.02, snapshot_seed=1)
        assert len(out.data) == len(bound_circuit.data)


class TestTransform:
    def test_transform_closure(self, bound_circuit):
        tf = make_miscalibration_transform(0.04, 0.01, snapshot_seed=11)
        a, b = tf(bound_circuit), tf(bound_circuit)
        assert np.allclose(rotation_angles(a), rotation_angles(b))

    def test_default_exemptions(self):
        assert "rz" in DEFAULT_EXEMPT_GATES
