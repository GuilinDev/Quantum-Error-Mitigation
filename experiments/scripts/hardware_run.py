#!/usr/bin/env python3
"""Real-device validation of budget-aware neural error mitigation.

Sim-to-real protocol:
  1. Read a backend's calibration (a real IBM QPU, or a fake backend that
     carries a real device snapshot) and build an Aer "digital twin" via
     NoiseModel.from_backend.
  2. Train the budget-conditioned NEM model on the twin (cheap, classical).
  3. Test NEM, ZNE (Richardson), and CDR on the ACTUAL backend under the
     equal-budget protocol, in the coherent-miscalibration regime that is
     intrinsic to real hardware (no injected transform on the device).

The same script runs against a fake backend with no IBM account (for
plumbing validation) and against a real QPU once credentials are saved
(QiskitRuntimeService.save_account run once in the user's own terminal).

Usage:
  # local validation, no token (real device snapshot, runs on Aer):
  python experiments/scripts/hardware_run.py --fake FakeTorino --qubits 4 \
      --train-samples 300 --instances 3 --epochs 10 --smoke

  # real hardware, after save_account:
  python experiments/scripts/hardware_run.py --backend least_busy --qubits 4 6 \
      --train-samples 2000 --instances 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from qiskit import transpile
from qiskit_aer.noise import NoiseModel

from src.classical.mitiq_baselines import mitiq_cdr, mitiq_zne
from src.models.mitigation_net import create_mitigation_model
from src.quantum.circuits import VQECircuit
from src.training.data_generator import DataSample, MitigationDataset
from src.utils.qiskit_compat import run_estimation, run_estimation_hardware

from scaling_cell import make_observable, train_model

# Native-basis gate set for the twin simulation (Heron = CZ; we keep the
# circuit at n qubits rather than embedding it into the full device, so
# the Aer twin stays tractable). Eagle devices (ECR) also reduce cleanly
# to this set for the twin; the hardware path uses the backend's own ISA.
TWIN_BASIS = ["rz", "sx", "x", "cz", "id"]
LOG2_S_RANGE = (8, 16)  # budget-conditioning: training shots ~ 2**U(8,16)


def get_backend(args):
    """Return a runtime BackendV2: a fake backend or a real QPU."""
    if args.fake:
        from qiskit_ibm_runtime import fake_provider
        return getattr(fake_provider, args.fake)()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    if args.backend == "least_busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(args.backend)


def _median(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.median(vals)) if vals else 0.0


def device_feature_vector(backend, n):
    """Eight device-characterization features from backend.target medians.

    Matches src/quantum/noise_models.py::to_feature_dict ordering:
    [single_qubit_error, two_qubit_error, readout_error_0, readout_error_1,
     t1, t2, single_gate_time, two_gate_time].
    """
    t = backend.target
    qubits = list(range(min(n, backend.num_qubits)))
    t1 = _median([t.qubit_properties[q].t1 for q in qubits]) * 1e6  # s -> us
    t2 = _median([t.qubit_properties[q].t2 for q in qubits]) * 1e6
    sx_err = _median([t["sx"][(q,)].error for q in qubits if (q,) in t["sx"]])
    sx_dur = _median([t["sx"][(q,)].duration for q in qubits if (q,) in t["sx"]]) * 1e6
    two_q = "cz" if "cz" in t.operation_names else (
        "ecr" if "ecr" in t.operation_names else "cx")
    pairs = [k for k in t[two_q].keys() if max(k) < max(qubits) + 2][:8] or list(t[two_q].keys())[:8]
    cz_err = _median([t[two_q][p].error for p in pairs])
    cz_dur = _median([t[two_q][p].duration for p in pairs]) * 1e6
    ro = _median([t["measure"][(q,)].error for q in qubits if (q,) in t["measure"]])
    return np.array([sx_err, cz_err, ro, ro, t1, t2, sx_dur, cz_dur], dtype=np.float32)


def budget_feature(shots):
    return 1.0 / np.sqrt(shots)


def to_twin_basis(circuit):
    """Transpile to native basis at n qubits (no device embedding).

    Uses optimization_level=0 to match the hardware executor
    (run_estimation_hardware, level 0 to preserve mitiq folds), so the
    twin the NEM model trains on has the same gate count / noise exposure
    as the circuits actually executed on the device. The VQE ansatz is
    already nearest-neighbour, so level 0 adds no SWAPs on a connected
    qubit line.
    """
    return transpile(circuit, basis_gates=TWIN_BASIS, optimization_level=0)


def generate_twin_data(vqe, obs, twin, dev_feat, n_samples, seed, desc):
    """Training pairs from the Aer digital twin, with budget conditioning."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        circuit = vqe.bind_parameters(params)
        native = to_twin_basis(circuit)
        shots = int(2 ** rng.uniform(*LOG2_S_RANGE))
        from src.utils.qiskit_compat import run_estimation_sampled
        ideal = run_estimation(circuit, obs, noise_model=None, exact=True)
        noisy = run_estimation_sampled(native, obs, shots=shots,
                                       noise_model=twin, method="density_matrix")
        nf = np.append(dev_feat, budget_feature(shots)).astype(np.float32)
        samples.append(DataSample(
            circuit_features=vqe.to_feature_vector(params),
            noise_features=nf, noisy_value=noisy, ideal_value=ideal,
            error=noisy - ideal))
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fake", type=str, help="Fake backend class, e.g. FakeTorino")
    g.add_argument("--backend", type=str, help="Real backend name or 'least_busy'")
    p.add_argument("--qubits", type=int, nargs="+", default=[4, 6])
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument("--instances", type=int, default=15)
    p.add_argument("--shots", type=int, default=8192, help="Total budget B per evaluation")
    p.add_argument("--cdr-training-circuits", type=int, default=31)
    p.add_argument("--skip-cdr", action="store_true",
                   help="Omit CDR (its 32x executions dominate the hardware job count)")
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Tiny run for plumbing checks")
    p.add_argument("--out", type=str, default="experiments/results/hardware")
    args = p.parse_args()

    device = "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(args)
    backend_name = getattr(backend, "name", str(backend))
    print(f"=== Hardware run on {backend_name} ({backend.num_qubits} qubits) ===")
    twin = NoiseModel.from_backend(backend)
    B = args.shots
    results = {"backend": str(backend_name), "budget": B, "cells": {}}

    for n in args.qubits:
        t0 = time.time()
        vqe = VQECircuit(n, args.layers)
        obs = make_observable(vqe)
        dev_feat = device_feature_vector(backend, n)

        train = generate_twin_data(vqe, obs, twin, dev_feat,
                                   args.train_samples, args.seed, "train")
        val = generate_twin_data(vqe, obs, twin, dev_feat,
                                 args.val_samples, args.seed + 1, "val")
        train_ds, val_ds = MitigationDataset(train), MitigationDataset(val)
        torch.manual_seed(args.seed)
        model = create_mitigation_model(
            "standard", circuit_dim=train_ds.circuit_features.shape[1],
            noise_dim=train_ds.noise_features.shape[1],
            hidden_dims=[256, 512, 256], dropout=0.15)
        model, val_loss = train_model(model, train_ds, val_ds, args.epochs, device)
        print(f"n={n}: twin-trained NEM (val={val_loss:.5f}); now testing on {backend_name}")

        # Test on the actual backend under the equal-budget protocol.
        rng = np.random.default_rng(args.seed + 2)
        rows = {"raw": [], "neural": [], "zne_richardson": [], "ideal": []}
        if not args.skip_cdr:
            rows["cdr"] = []

        # On a real backend, group every execution into one Batch so the
        # whole sweep shares a single queue reservation (Open plan allows
        # Batch/job mode, not Session). Fake backends run locally with no
        # batching (mode=None -> the backend itself).
        use_batch = args.backend is not None
        if use_batch:
            from qiskit_ibm_runtime import Batch
            batch_ctx = Batch(backend=backend)
        else:
            from contextlib import nullcontext
            batch_ctx = nullcontext()

        with batch_ctx as batch:
            mode = batch if use_batch else None
            for i in range(args.instances):
                params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
                circuit = vqe.bind_parameters(params)
                ideal = run_estimation(circuit, obs, noise_model=None, exact=True)
                rows["ideal"].append(ideal)
                # raw + NEM: 1 execution at full budget B
                noisy = run_estimation_hardware(circuit, obs, shots=B,
                                                backend=backend, mode=mode)
                rows["raw"].append(noisy)
                nf = np.append(dev_feat, budget_feature(B)).astype(np.float32)
                with torch.no_grad():
                    pred = float(model(
                        torch.tensor([[noisy]], dtype=torch.float32),
                        torch.tensor(vqe.to_feature_vector(params), dtype=torch.float32).unsqueeze(0),
                        torch.tensor(nf, dtype=torch.float32).unsqueeze(0)).numpy()[0, 0])
                rows["neural"].append(pred)
                # ZNE Richardson: 3 executions at B/3
                rows["zne_richardson"].append(mitiq_zne(
                    circuit, obs, None, extrapolation="richardson",
                    shots=max(B // 3, 1), backend=backend, hw_mode=mode).mitigated_value)
                line = (f"  inst {i}: ideal={ideal:+.4f} raw={noisy:+.4f} "
                        f"NEM={rows['neural'][-1]:+.4f} ZNE={rows['zne_richardson'][-1]:+.4f}")
                if not args.skip_cdr:
                    # CDR: 32 executions at B/32 (noisy on hardware, labels classical-exact)
                    rows["cdr"].append(mitiq_cdr(
                        circuit, obs, None, num_training_circuits=args.cdr_training_circuits,
                        shots=max(B // (args.cdr_training_circuits + 1), 1),
                        backend=backend, hw_mode=mode).mitigated_value)
                    line += f" CDR={rows['cdr'][-1]:+.4f}"
                print(line, flush=True)

        ideal = np.array(rows["ideal"])
        cell = {"n": n, "raw_mae": float(np.mean(np.abs(np.array(rows["raw"]) - ideal)))}
        metric_methods = ["neural", "zne_richardson"] + ([] if args.skip_cdr else ["cdr"])
        for m in metric_methods:
            err = np.abs(np.array(rows[m]) - ideal)
            raw_err = np.abs(np.array(rows["raw"]) - ideal)
            cell[m + "_mae"] = float(np.mean(err))
            cell[m + "_improvement_pct"] = float((raw_err.mean() - err.mean()) / raw_err.mean() * 100)
            cell[m + "_worse_rate"] = float((err > raw_err).mean())
        cell["per_instance"] = {k: list(map(float, v)) for k, v in rows.items()}
        cell["seconds"] = time.time() - t0
        results["cells"][f"n{n}"] = cell
        ordering = (f"n={n} ORDERING: NEM {cell['neural_improvement_pct']:+.0f}% | "
                    f"ZNE {cell['zne_richardson_improvement_pct']:+.0f}% "
                    f"(worse-than-raw {cell['zne_richardson_worse_rate']*100:.0f}%)")
        if not args.skip_cdr:
            ordering += f" | CDR {cell['cdr_improvement_pct']:+.0f}%"
        print(ordering, flush=True)

    tag = "smoke" if args.smoke else "run"
    out_path = out_dir / f"hardware_{backend_name}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
