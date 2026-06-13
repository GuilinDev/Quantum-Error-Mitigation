#!/usr/bin/env python3
"""Model-size ablation: does the mitigation network need 400K+ parameters?

Trains the standard architecture at three widths on identical n=8
miscal-regime data and reports test MAE, answering the reviewer concern
that a ~10^5-parameter network on a ~30-dim input is overparameterized.

Usage:
    python experiments/scripts/model_size_ablation.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from src.models.mitigation_net import MitigationNetwork
from src.training.data_generator import MitigationDataset

from scaling_cell import (
    CellContext,
    generate_samples,
    generate_test_instances,
    neural_predictions,
    train_model,
)

CONFIGS = {
    "tiny":     {"hidden_dims": [32, 32],        "enc_scale": 0.125},
    "small":    {"hidden_dims": [64, 128],       "enc_scale": 0.5},
    "standard": {"hidden_dims": [256, 512, 256], "enc_scale": 1.0},
}


def build_model(circuit_dim, noise_dim, hidden_dims, dropout=0.15):
    return MitigationNetwork(
        circuit_dim=circuit_dim, noise_dim=noise_dim,
        hidden_dims=hidden_dims, dropout=dropout,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--regime", default="miscal")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--test-instances", type=int, default=150)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results/ablation")
    # unused but required by CellContext consumers
    parser.add_argument("--cdr-instances", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = CellContext(args)

    train_ds = MitigationDataset(generate_samples(ctx, args.train_samples, args.seed, "train"))
    val_ds = MitigationDataset(generate_samples(ctx, args.val_samples, args.seed + 1, "val"))
    instances = generate_test_instances(ctx, args.test_instances, args.seed + 2)
    ideal = np.array([i["ideal"] for i in instances])
    noisy = np.array([i["noisy"] for i in instances])
    raw_mae = float(np.mean(np.abs(noisy - ideal)))
    circuit_dim = train_ds.circuit_features.shape[1]
    noise_dim = train_ds.noise_features.shape[1]

    results = {"raw_mae": raw_mae, "configs": {}}
    for name, cfg in CONFIGS.items():
        torch.manual_seed(args.seed)
        model = build_model(circuit_dim, noise_dim, cfg["hidden_dims"])
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        model, best_val = train_model(model, train_ds, val_ds, args.epochs, device)
        preds = neural_predictions(model, instances, device)
        m = float(np.mean(np.abs(preds - ideal)))
        results["configs"][name] = {
            "hidden_dims": cfg["hidden_dims"],
            "n_params": n_params,
            "val_loss": best_val,
            "test_mae": m,
            "improvement_pct": (raw_mae - m) / raw_mae * 100,
            "train_time_s": time.time() - t0,
        }
        print(f"{name:9s} params={n_params:>8,}  MAE={m:.5f} "
              f"({results['configs'][name]['improvement_pct']:+.1f}%)  "
              f"val={best_val:.6f}")

    with open(out_dir / "model_size_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"raw MAE {raw_mae:.5f}; saved to {out_dir}/model_size_ablation.json")


if __name__ == "__main__":
    main()
