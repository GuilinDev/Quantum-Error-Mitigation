#!/usr/bin/env python3
"""Phase-boundary sweep: mitigation quality vs coherent-error strength.

Sweeps the angle-miscalibration bias delta from 0 (purely stochastic
noise) to 0.06 rad while holding the stochastic components fixed, and
records the MAE of raw execution, the neural mitigator (one model
trained across the full delta range, conditioned on delta via its noise
features), folding ZNE (exponential and adaptive), and CDR at each
point. Maps the boundary at which extrapolation-based mitigation stops
helping and starts hurting. All methods use 8192 shots per execution;
budget effects are studied separately in budget_sweep.py.

Usage:
    python experiments/scripts/phase_boundary.py
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
from tqdm import tqdm

from src.classical.mitiq_baselines import mitiq_cdr, mitiq_zne
from src.models.mitigation_net import create_mitigation_model
from src.quantum.circuits import VQECircuit
from src.quantum.miscalibration import make_miscalibration_transform
from src.training.data_generator import DataSample, MitigationDataset
from src.utils.qiskit_compat import run_estimation, run_estimation_sampled

from scaling_cell import (
    NATIVE_BASIS,
    build_miscal_stochastic_noise,
    make_observable,
    train_model,
)

DELTA_POINTS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08)
TRAIN_DELTA_RANGE = (0.0, 0.09)
SIGMA_FRACTION = 0.5
FIXED_P2Q = 0.015
FIXED_RO = (0.01, 0.03)


def make_instance(vqe, observable, params, delta, snapshot_seed, shots):
    circuit = transpile(
        vqe.bind_parameters(params), basis_gates=NATIVE_BASIS,
        optimization_level=0,
    )
    aer_nm = build_miscal_stochastic_noise(FIXED_P2Q, *FIXED_RO)
    transform = (
        make_miscalibration_transform(delta, SIGMA_FRACTION * delta, snapshot_seed)
        if delta > 0 else None
    )
    ideal = run_estimation(circuit, observable, noise_model=None, exact=True)
    executed = transform(circuit) if transform is not None else circuit
    noisy = run_estimation_sampled(
        executed, observable, shots=shots, noise_model=aer_nm
    )
    feats = np.array(
        [delta, SIGMA_FRACTION * delta, FIXED_P2Q, FIXED_RO[0], FIXED_RO[1],
         0.0, 0.0, 0.0], dtype=np.float32,
    )
    return circuit, aer_nm, transform, feats, ideal, noisy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--instances-per-point", type=int, default=80)
    parser.add_argument("--cdr-instances", type=int, default=25)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results/boundary")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vqe = VQECircuit(args.qubits, args.layers)
    observable = make_observable(vqe)
    timings = {}

    # Train one delta-conditioned model across the full range.
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    samples = {"train": [], "val": []}
    for split, n in (("train", args.train_samples), ("val", args.val_samples)):
        for i in tqdm(range(n), desc=f"{split} data"):
            params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
            delta = rng.uniform(*TRAIN_DELTA_RANGE)
            _, _, _, feats, ideal, noisy = make_instance(
                vqe, observable, params, delta,
                snapshot_seed=args.seed * 1_000_000 + len(samples[split]) +
                (0 if split == "train" else 10_000_000) + i,
                shots=args.shots,
            )
            samples[split].append(DataSample(
                circuit_features=vqe.to_feature_vector(params),
                noise_features=feats,
                noisy_value=noisy, ideal_value=ideal, error=noisy - ideal,
            ))
    train_ds = MitigationDataset(samples["train"])
    val_ds = MitigationDataset(samples["val"])
    timings["data_generation_s"] = time.time() - t0

    t0 = time.time()
    torch.manual_seed(args.seed)
    model = create_mitigation_model(
        "standard",
        circuit_dim=train_ds.circuit_features.shape[1],
        noise_dim=train_ds.noise_features.shape[1],
        hidden_dims=[256, 512, 256], dropout=0.15,
    )
    model, best_val = train_model(model, train_ds, val_ds, args.epochs, device)
    torch.save(model.state_dict(), out_dir / "boundary_model.pt")
    timings["training_s"] = time.time() - t0
    print(f"delta-conditioned model trained (val={best_val:.6f})")

    sweep = {}
    t0 = time.time()
    for delta in DELTA_POINTS:
        rng_pt = np.random.default_rng(args.seed + int(delta * 10_000))
        rows = {"raw": [], "neural": [], "zne_exponential": [],
                "zne_adaptive": [], "cdr": []}
        ideals = []
        for i in tqdm(range(args.instances_per_point), desc=f"delta={delta}"):
            params = rng_pt.uniform(0, 2 * np.pi, vqe.num_parameters)
            circuit, aer_nm, transform, feats, ideal, noisy = make_instance(
                vqe, observable, params, delta,
                snapshot_seed=20_000_000 + int(delta * 10_000) * 1000 + i,
                shots=args.shots,
            )
            ideals.append(ideal)
            rows["raw"].append(noisy)
            with torch.no_grad():
                pred = float(model(
                    torch.tensor([[noisy]], dtype=torch.float32).to(device),
                    torch.tensor(vqe.to_feature_vector(params),
                                 dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device),
                ).cpu().numpy()[0, 0])
            rows["neural"].append(pred)
            rows["zne_exponential"].append(mitiq_zne(
                circuit, observable, aer_nm, extrapolation="exponential",
                asymptote=0.0, shots=args.shots,
                circuit_transform=transform, sampled=True,
            ).mitigated_value)
            rows["zne_adaptive"].append(mitiq_zne(
                circuit, observable, aer_nm, extrapolation="adaptive",
                asymptote=0.0, shots=args.shots,
                circuit_transform=transform, sampled=True,
            ).mitigated_value)
            if i < args.cdr_instances:
                rows["cdr"].append(mitiq_cdr(
                    circuit, observable, aer_nm, shots=args.shots,
                    circuit_transform=transform, sampled=True,
                    skip_transpile=True,
                ).mitigated_value)

        ideals = np.array(ideals)
        raw_mae = float(np.mean(np.abs(np.array(rows["raw"]) - ideals)))
        entry = {"maes": {}, "improvement_pct": {},
                 "per_instance": {k: list(map(float, v)) for k, v in rows.items()},
                 "ideal": ideals.tolist()}
        for name, vals in rows.items():
            ref = ideals[: len(vals)]
            m = float(np.mean(np.abs(np.array(vals) - ref)))
            ref_raw = float(np.mean(np.abs(np.array(rows["raw"][: len(vals)]) - ref)))
            entry["maes"][name] = m
            entry["improvement_pct"][name] = (ref_raw - m) / ref_raw * 100
        sweep[str(delta)] = entry
        print(f"delta={delta}: " + "  ".join(
            f"{k}={entry['maes'][k]:.5f}({entry['improvement_pct'][k]:+.0f}%)"
            for k in rows
        ))
    timings["sweep_s"] = time.time() - t0

    payload = {
        "config": vars(args),
        "delta_points": list(DELTA_POINTS),
        "fixed_stochastic": {"p2q": FIXED_P2Q, "ro01": FIXED_RO[0],
                             "ro10": FIXED_RO[1]},
        "sweep": sweep,
        "timings": timings,
    }
    out_path = out_dir / "phase_boundary.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
