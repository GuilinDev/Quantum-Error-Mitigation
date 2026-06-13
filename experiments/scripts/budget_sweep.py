#!/usr/bin/env python3
"""Equal-quantum-budget Pareto comparison of mitigation methods.

For each total shot budget B per evaluation point, every method spends
exactly B shots, split across its required circuit executions:

    raw / NEM        1 execution  x B shots
    ZNE-exponential  3 executions x B/3 shots
    ZNE-adaptive     4 executions x B/4 shots
    CDR              32 executions x B/32 shots (31 training + target)

NEM uses a single budget-conditioned network per regime: training
samples are measured with a random per-sample shot count S (log-uniform)
and the shot-noise scale 1/sqrt(S) is appended to the noise features, so
one model serves every budget. Test instances are shared across budgets
(fixed circuit + noise realization; fresh measurements per budget).

Usage:
    python experiments/scripts/budget_sweep.py --regime miscal
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from src.classical.mitiq_baselines import mitiq_cdr, mitiq_zne
from src.models.mitigation_net import create_mitigation_model
from src.training.data_generator import DataSample, MitigationDataset
from src.utils.qiskit_compat import run_estimation

from scaling_cell import CellContext, train_model  # reuse cell machinery

BUDGETS = (1024, 4096, 16384, 65536)
LOG2_S_RANGE = (8, 16)  # training-time shot counts: 2**U(8,16)


def budget_feature(shots: int) -> float:
    return 1.0 / np.sqrt(shots)


def generate_budget_samples(ctx, n_samples, seed, desc):
    """Training samples with randomized per-sample shot budgets."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in tqdm(range(n_samples), desc=desc):
        params = rng.uniform(0, 2 * np.pi, ctx.vqe.num_parameters)
        circuit = ctx.bound_circuit(params)
        aer_nm, transform, feats = ctx.draw_noise(
            rng, snapshot_seed=seed * 1_000_000 + i
        )
        shots = int(2 ** rng.uniform(*LOG2_S_RANGE))
        ideal = ctx.exact_ideal(circuit)
        noisy = ctx.measure_noisy(circuit, aer_nm, transform, shots)
        feats = np.append(feats, budget_feature(shots)).astype(np.float32)
        samples.append(DataSample(
            circuit_features=ctx.vqe.to_feature_vector(params),
            noise_features=feats,
            noisy_value=noisy, ideal_value=ideal, error=noisy - ideal,
        ))
    return samples


def mae(pred, ideal):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ideal))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("systematic", "nonlinear", "miscal"),
                        required=True)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--test-instances", type=int, default=100)
    parser.add_argument("--cdr-training-circuits", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results/budget")
    # placeholder so CellContext sees a shots attr; per-call values vary
    parser.add_argument("--shots", type=int, default=8192)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = CellContext(args)
    timings = {}

    t0 = time.time()
    train_samples = generate_budget_samples(ctx, args.train_samples, args.seed, "train")
    val_samples = generate_budget_samples(ctx, args.val_samples, args.seed + 1, "val")
    timings["data_generation_s"] = time.time() - t0

    train_ds = MitigationDataset(train_samples)
    val_ds = MitigationDataset(val_samples)
    circuit_dim = train_ds.circuit_features.shape[1]
    noise_dim = train_ds.noise_features.shape[1]

    t0 = time.time()
    torch.manual_seed(args.seed)
    model = create_mitigation_model(
        "standard", circuit_dim=circuit_dim, noise_dim=noise_dim,
        hidden_dims=[256, 512, 256], dropout=0.15,
    )
    model, best_val = train_model(model, train_ds, val_ds, args.epochs, device)
    torch.save(model.state_dict(), out_dir / f"budget_model_{args.regime}.pt")
    timings["training_s"] = time.time() - t0
    print(f"budget-conditioned model trained (val={best_val:.6f})")

    # Fixed test instances (circuit + noise realization). Measurements
    # are taken fresh per budget below.
    rng = np.random.default_rng(args.seed + 2)
    instances = []
    for i in range(args.test_instances):
        params = rng.uniform(0, 2 * np.pi, ctx.vqe.num_parameters)
        circuit = ctx.bound_circuit(params)
        aer_nm, transform, feats = ctx.draw_noise(
            rng, snapshot_seed=(args.seed + 2) * 1_000_000 + i
        )
        instances.append({
            "circuit": circuit, "aer_noise": aer_nm, "transform": transform,
            "circuit_features": ctx.vqe.to_feature_vector(params),
            "noise_features": feats,
            "ideal": ctx.exact_ideal(circuit),
        })
    ideal = np.array([inst["ideal"] for inst in instances])

    sweep = {}
    t0 = time.time()
    for B in BUDGETS:
        rows = {"raw": [], "neural": [], "zne_exponential": [],
                "zne_adaptive": [], "cdr": []}
        for inst in tqdm(instances, desc=f"budget {B}"):
            raw = ctx.measure_noisy(
                inst["circuit"], inst["aer_noise"], inst["transform"], B
            )
            rows["raw"].append(raw)

            nf = np.append(inst["noise_features"], budget_feature(B)).astype(np.float32)
            with torch.no_grad():
                pred = float(model(
                    torch.tensor([[raw]], dtype=torch.float32).to(device),
                    torch.tensor(inst["circuit_features"], dtype=torch.float32
                                 ).unsqueeze(0).to(device),
                    torch.tensor(nf, dtype=torch.float32).unsqueeze(0).to(device),
                ).cpu().numpy()[0, 0])
            rows["neural"].append(pred)

            rows["zne_exponential"].append(mitiq_zne(
                inst["circuit"], ctx.observable, inst["aer_noise"],
                extrapolation="exponential", asymptote=0.0,
                shots=max(B // 3, 1), circuit_transform=inst["transform"],
                sampled=True,
            ).mitigated_value)
            rows["zne_adaptive"].append(mitiq_zne(
                inst["circuit"], ctx.observable, inst["aer_noise"],
                extrapolation="adaptive", asymptote=0.0,
                shots=max(B // 4, 1), circuit_transform=inst["transform"],
                sampled=True,
            ).mitigated_value)
            rows["cdr"].append(mitiq_cdr(
                inst["circuit"], ctx.observable, inst["aer_noise"],
                num_training_circuits=args.cdr_training_circuits,
                shots=max(B // (args.cdr_training_circuits + 1), 1),
                circuit_transform=inst["transform"], sampled=True,
                skip_transpile=(args.regime == "miscal"),
            ).mitigated_value)

        entry = {"maes": {}, "improvement_pct": {}, "p_vs_raw": {}}
        raw_mae = mae(rows["raw"], ideal)
        for name, vals in rows.items():
            m = mae(vals, ideal)
            entry["maes"][name] = m
            entry["improvement_pct"][name] = (raw_mae - m) / raw_mae * 100
            ea = np.abs(np.array(vals) - ideal)
            eb = np.abs(np.array(rows["raw"]) - ideal)
            entry["p_vs_raw"][name] = (
                1.0 if np.allclose(ea, eb)
                else float(stats.ttest_rel(ea, eb).pvalue)
            )
        entry["per_instance"] = {k: list(map(float, v)) for k, v in rows.items()}
        sweep[str(B)] = entry
        print(f"\nB={B}: " + "  ".join(
            f"{k}={entry['maes'][k]:.5f}({entry['improvement_pct'][k]:+.0f}%)"
            for k in rows
        ))
    timings["sweep_s"] = time.time() - t0

    payload = {
        "regime": args.regime,
        "config": vars(args),
        "budgets": list(BUDGETS),
        "executions_per_method": {
            "raw": 1, "neural": 1, "zne_exponential": 3,
            "zne_adaptive": 4, "cdr": args.cdr_training_circuits + 1,
        },
        "ideal": ideal.tolist(),
        "sweep": sweep,
        "timings": timings,
    }
    out_path = out_dir / f"budget_sweep_{args.regime}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
