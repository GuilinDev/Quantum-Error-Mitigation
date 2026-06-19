#!/usr/bin/env python3
"""Offline robustness sweep for on-device NEM fine-tuning (no QPU).

Loads the device dataset and the cached twin model produced by
finetune_hardware.py, then re-runs the fine-tune + evaluation across a
grid of (lr, epochs, freeze_encoders, seed). Reports held-out improvement
and worse-than-unmitigated rate per config, plus the affine baseline.
Purpose: show the positive on-device result is stable, not a lucky pick.

Usage:
  python experiments/scripts/sweep_finetune.py \
      --dataset experiments/results/hardware/finetune_dataset_ibm_marrakesh_n4L5.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from src.quantum.circuits import VQECircuit
from scaling_cell import make_observable
from finetune_hardware import (
    build_model, to_samples, nem_predict, finetune,
    affine_recalibration, metrics,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="experiments/results/hardware/finetune_sweep.json")
    args = p.parse_args()

    ds = json.load(open(args.dataset))
    n, layers, shots = ds["n"], ds["layers"], ds["shots"]
    dev_feat = np.array(ds["dev_feat"], dtype=np.float32)
    vqe = VQECircuit(n, layers)
    _ = make_observable(vqe)
    train_recs, test_recs = ds["train"], ds["test"]
    raw_test = np.array([r["noisy"] for r in test_recs])
    ideal_test = np.array([r["ideal"] for r in test_recs])

    circuit_dim = len(vqe.to_feature_vector(np.zeros(vqe.num_parameters)))
    noise_dim = len(dev_feat) + 1

    cache = (Path(args.dataset).parent /
             f"twin_model_{ds['backend']}_n{n}L{layers}.pt")
    twin_state = torch.load(cache)

    def load_twin():
        m = build_model(circuit_dim, noise_dim, 42)
        m.load_state_dict(twin_state)
        return m

    twin_model = load_twin()
    twin_pred_test = nem_predict(twin_model, test_recs, vqe, dev_feat, shots)
    twin_pred_train = nem_predict(twin_model, train_recs, vqe, dev_feat, shots)
    twin_m = metrics(twin_pred_test, raw_test, ideal_test)

    a, b = affine_recalibration(
        twin_pred_train, [r["noisy"] for r in train_recs],
        [r["ideal"] for r in train_recs])
    affine_pred = raw_test + a * (twin_pred_test - raw_test) + b
    affine_m = metrics(affine_pred, raw_test, ideal_test)

    ft_samples = to_samples(train_recs, vqe, dev_feat, shots)

    grid = []
    for lr in (1e-4, 2e-4, 5e-4):
        for epochs in (40, 60, 100):
            for freeze in (False, True):
                imps, worses = [], []
                for seed in (0, 1, 2, 3, 4):
                    m = build_model(circuit_dim, noise_dim, 42)
                    m.load_state_dict(twin_state)
                    m, _ = finetune(m, ft_samples, epochs, lr, freeze, 0.2, seed)
                    pred = nem_predict(m, test_recs, vqe, dev_feat, shots)
                    mm = metrics(pred, raw_test, ideal_test)
                    imps.append(mm["improvement_pct"])
                    worses.append(mm["worse_rate"])
                grid.append({
                    "lr": lr, "epochs": epochs, "freeze_encoders": freeze,
                    "impr_mean": float(np.mean(imps)),
                    "impr_std": float(np.std(imps)),
                    "impr_min": float(np.min(imps)),
                    "worse_mean": float(np.mean(worses)),
                })
                print(f"lr={lr:.0e} ep={epochs:3d} freeze={int(freeze)} | "
                      f"impr {np.mean(imps):+6.1f}% (±{np.std(imps):4.1f}, "
                      f"min {np.min(imps):+6.1f}) worse {np.mean(worses)*100:4.0f}%",
                      flush=True)

    summary = {
        "dataset": args.dataset, "n": n, "layers": layers, "shots": shots,
        "raw_mae": float(np.abs(raw_test - ideal_test).mean()),
        "twin": twin_m, "affine": {**affine_m, "a": a, "b": b},
        "grid": grid,
        "grid_impr_mean": float(np.mean([g["impr_mean"] for g in grid])),
        "grid_impr_min": float(np.min([g["impr_min"] for g in grid])),
        "n_configs_positive": int(sum(g["impr_mean"] > 0 for g in grid)),
        "n_configs": len(grid),
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nTwin-only: {twin_m['improvement_pct']:+.1f}% (worse {twin_m['worse_rate']*100:.0f}%)")
    print(f"Affine:    {affine_m['improvement_pct']:+.1f}% (worse {affine_m['worse_rate']*100:.0f}%) a={a:.3f}")
    print(f"Fine-tune grid: {summary['n_configs_positive']}/{summary['n_configs']} configs "
          f"positive; mean {summary['grid_impr_mean']:+.1f}%, worst-case min {summary['grid_impr_min']:+.1f}%")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
