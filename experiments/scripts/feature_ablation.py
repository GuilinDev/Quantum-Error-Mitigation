#!/usr/bin/env python3
"""Feature ablation under the miscalibration regime (n=8).

Trains the standard mitigation network with feature groups zeroed out:
full model, no noise features, no circuit features, and noisy-value-only
(both groups zeroed). Re-validates the original paper's central ablation
finding under the corrected evaluation pipeline.

Usage:
    python experiments/scripts/feature_ablation.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from src.models.mitigation_net import create_mitigation_model
from src.training.data_generator import MitigationDataset

from scaling_cell import (
    CellContext,
    generate_samples,
    generate_test_instances,
    train_model,
)

VARIANTS = {
    "full": (False, False),
    "no_noise_features": (False, True),
    "no_circuit_features": (True, False),
    "noisy_value_only": (True, True),
}


def mask_dataset(ds, zero_circuit, zero_noise):
    if zero_circuit:
        ds.circuit_features = np.zeros_like(ds.circuit_features)
    if zero_noise:
        ds.noise_features = np.zeros_like(ds.noise_features)
    return ds


def masked_predictions(model, instances, zero_circuit, zero_noise, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for inst in instances:
            cf = inst["circuit_features"] * (0.0 if zero_circuit else 1.0)
            nf = inst["noise_features"] * (0.0 if zero_noise else 1.0)
            preds.append(float(model(
                torch.tensor([[inst["noisy"]]], dtype=torch.float32).to(device),
                torch.tensor(cf, dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor(nf, dtype=torch.float32).unsqueeze(0).to(device),
            ).cpu().numpy()[0, 0]))
    return np.array(preds)


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
    parser.add_argument("--cdr-instances", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = CellContext(args)

    base_train = generate_samples(ctx, args.train_samples, args.seed, "train")
    base_val = generate_samples(ctx, args.val_samples, args.seed + 1, "val")
    instances = generate_test_instances(ctx, args.test_instances, args.seed + 2)
    ideal = np.array([i["ideal"] for i in instances])
    noisy = np.array([i["noisy"] for i in instances])
    raw_mae = float(np.mean(np.abs(noisy - ideal)))

    results = {"raw_mae": raw_mae, "variants": {}}
    full_mae = None
    for name, (zero_c, zero_n) in VARIANTS.items():
        train_ds = mask_dataset(MitigationDataset(base_train), zero_c, zero_n)
        val_ds = mask_dataset(MitigationDataset(base_val), zero_c, zero_n)
        torch.manual_seed(args.seed)
        model = create_mitigation_model(
            "standard",
            circuit_dim=train_ds.circuit_features.shape[1],
            noise_dim=train_ds.noise_features.shape[1],
            hidden_dims=[256, 512, 256], dropout=0.15,
        )
        model, best_val = train_model(model, train_ds, val_ds, args.epochs, device)
        preds = masked_predictions(model, instances, zero_c, zero_n, device)
        m = float(np.mean(np.abs(preds - ideal)))
        if name == "full":
            full_mae = m
        results["variants"][name] = {
            "test_mae": m,
            "improvement_pct": (raw_mae - m) / raw_mae * 100,
            "mae_increase_vs_full_pct": (m - full_mae) / full_mae * 100,
            "val_loss": best_val,
        }
        print(f"{name:22s} MAE={m:.5f} ({results['variants'][name]['improvement_pct']:+.1f}% vs raw, "
              f"{results['variants'][name]['mae_increase_vs_full_pct']:+.1f}% vs full)")

    with open(out_dir / "feature_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir}/feature_ablation.json")


if __name__ == "__main__":
    main()
