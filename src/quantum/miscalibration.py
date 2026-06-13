"""Coherent angle-miscalibration noise applied at the circuit level.

Models pulse-amplitude / rotation-angle calibration error: every
rotation gate executes theta + delta_g, where delta_g is drawn once per
device snapshot from N(delta_bias, sigma) indexed by gate position.
Because the miscalibration lives in the executed unitary, it applies
identically to any circuit handed to the executor — folded ZNE circuits,
CDR near-Clifford training circuits, and the target circuit all face the
same quantum process.

Under unitary folding the miscalibration of the G-dagger segment does
not cancel (RY(theta+d) RY(-theta+d) = RY(2d)), so the coherent angle
error grows linearly with the scale factor while the expectation value
responds non-monotonically — the documented mechanism by which coherent
errors defeat extrapolation-based mitigation.
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate

#: Parametrized rotation gates subject to angle miscalibration.
ROTATION_GATES = frozenset({"rx", "ry", "rz"})

#: Fixed-angle physical pulses subject to the same control offset,
#: mapped to their nominal RX angle.
FIXED_PULSE_ANGLES = {"sx": np.pi / 2, "sxdg": -np.pi / 2, "x": np.pi}


#: Gates exempt from miscalibration by default: virtual-Z rotations are
#: implemented as exact software frame updates on superconducting
#: hardware and carry no control error.
DEFAULT_EXEMPT_GATES = frozenset({"rz"})


def miscalibrate_circuit(
    circuit: QuantumCircuit,
    delta_bias: float,
    sigma: float,
    snapshot_seed: int,
    exempt_gates: frozenset = DEFAULT_EXEMPT_GATES,
) -> QuantumCircuit:
    """Apply per-gate angle-offset miscalibration to a bound circuit.

    Models a constant control offset (plus gate-to-gate variation): the
    g-th pulse executes its nominal angle plus an offset drawn from
    N(delta_bias, sigma). Offsets are ADDITIVE with a fixed sign,
    matching control-offset errors: under unitary folding the inverse
    segment's offsets do not cancel (e.g. RY(t+d) RY(-t+d) = RY(2d)),
    so the coherent error accumulates with the scale factor while the
    expectation value responds non-monotonically — defeating
    extrapolation. Fixed physical pulses (sx, sxdg, x) receive the same
    treatment via their nominal RX angle, so the mechanism is identical
    in the {rz, sx, x, cx} native basis used by CDR.

    Args:
        circuit: Circuit with bound (numeric) rotation angles.
        delta_bias: Mean angle offset per pulse (radians).
        sigma: Standard deviation of the per-gate offsets.
        snapshot_seed: Seed fixing the device snapshot; the g-th pulse
            in execution order always receives the same offset for a
            given seed, so repeated executions (e.g. ZNE scale points)
            see a consistent calibration state.
        exempt_gates: Gate names excluded from miscalibration (default:
            virtual-Z rotations, which are exact frame updates).

    Returns:
        A new circuit with perturbed rotations.
    """
    snap = np.random.default_rng(snapshot_seed)
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        op = inst.operation
        if op.name in exempt_gates:
            out.append(op, inst.qubits, inst.clbits)
        elif op.name in ROTATION_GATES and len(op.params) == 1:
            offset = snap.normal(delta_bias, sigma)
            new_op = op.copy()
            new_op.params = [float(op.params[0]) + offset]
            out.append(new_op, inst.qubits, inst.clbits)
        elif op.name in FIXED_PULSE_ANGLES:
            offset = snap.normal(delta_bias, sigma)
            out.append(
                RXGate(FIXED_PULSE_ANGLES[op.name] + offset),
                inst.qubits, inst.clbits,
            )
        else:
            out.append(op, inst.qubits, inst.clbits)
    return out


def make_miscalibration_transform(
    delta_bias: float,
    sigma: float,
    snapshot_seed: int,
):
    """Return a circuit transform closure for executor pipelines."""

    def transform(circuit: QuantumCircuit) -> QuantumCircuit:
        return miscalibrate_circuit(circuit, delta_bias, sigma, snapshot_seed)

    return transform
