#!/usr/bin/env python3
"""Cross-check: apply the on-device-fine-tuned model to the ORIGINAL
hardware test instances (the ones that scored -94% twin-only in Table 7).

The original hardware_run.py drew its test instances from a fixed RNG but
did not store the circuit parameters, only the measured raw value and the
exact ideal. We reconstruct the parameters by replaying the RNG, VALIDATE
the reconstruction by recomputing the exact ideal and matching it to the
stored value, then apply the twin model, the seed-averaged fine-tuned
model, and the affine recalibration to the STORED raw measurements (no new
QPU). This shows the fix on the very instances of the negative result.
"""

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
    affine_recalibration, metrics, budget_feature,
)

ROOT = Path(__file__).parent.parent.parent
HW = ROOT / "experiments/results/hardware"


def reconstruct_params(n, layers, n_inst, stored_ideal, obs, vqe, tol=1e-6):
    """Find the RNG seed whose first n_inst draws reproduce stored_ideal."""
    from src.utils.qiskit_compat import run_estimation
    for seed_base in range(0, 60):  # hardware_run.py used default_rng(args.seed+2)
        rng = np.random.default_rng(seed_base)
        params_list, ideals = [], []
        ok = True
        for i in range(n_inst):
            params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
            params_list.append(params)
            circ = vqe.bind_parameters(params)
            idv = run_estimation(circ, obs, noise_model=None, exact=True)
            ideals.append(idv)
            if abs(idv - stored_ideal[i]) > tol:
                ok = False
                break
        if ok:
            return seed_base, params_list, ideals
    return None, None, None


def main():
    ds = json.load(open(HW / "finetune_dataset_ibm_marrakesh_n4L5.json"))
    n, layers, shots = ds["n"], ds["layers"], ds["shots"]
    dev_feat = np.array(ds["dev_feat"], dtype=np.float32)
    vqe = VQECircuit(n, layers)
    obs = make_observable(vqe)
    circuit_dim = len(vqe.to_feature_vector(np.zeros(vqe.num_parameters)))
    noise_dim = len(dev_feat) + 1

    old = json.load(open(HW / "hardware_marrakesh_n4L5.json"))["cells"]["n4"]
    stored_ideal = old["per_instance"]["ideal"]
    stored_raw = old["per_instance"]["raw"]
    n_inst = len(stored_ideal)
    print(f"Original L5 set: {n_inst} instances, raw_mae={old['raw_mae']:.4f}, "
          f"twin reported {old['neural_improvement_pct']:+.0f}%")

    seed, params_list, ideals = reconstruct_params(
        n, layers, n_inst, stored_ideal, obs, vqe)
    if seed is None:
        print("RECONSTRUCTION FAILED: no RNG seed reproduces the stored ideal "
              "values. Skipping cross-check (use the held-out result only).")
        return
    print(f"Reconstruction VALIDATED at default_rng({seed}): "
          f"max |ideal_recomputed - ideal_stored| < 1e-6")

    # Build records for the old instances (stored raw + reconstructed params).
    recs = [{"params": list(map(float, params_list[i])),
             "noisy": stored_raw[i], "ideal": stored_ideal[i]}
            for i in range(n_inst)]

    twin_state = torch.load(HW / f"twin_model_ibm_marrakesh_n{n}L{layers}.pt")
    twin_model = build_model(circuit_dim, noise_dim, 42)
    twin_model.load_state_dict(twin_state)

    raw = np.array(stored_raw)
    ideal = np.array(stored_ideal)
    twin_pred = nem_predict(twin_model, recs, vqe, dev_feat, shots)
    twin_m = metrics(twin_pred, raw, ideal)

    # Affine recalibration fitted on the device TRAIN set (deterministic).
    train_recs = ds["train"]
    twin_pred_train = nem_predict(twin_model, train_recs, vqe, dev_feat, shots)
    a, b = affine_recalibration(
        twin_pred_train, [r["noisy"] for r in train_recs],
        [r["ideal"] for r in train_recs])
    affine_pred = raw + a * (twin_pred - raw) + b
    affine_m = metrics(affine_pred, raw, ideal)

    # Seed-averaged fine-tuned model (lr=2e-4, ep=60, freeze off), 5 seeds.
    ft_samples = to_samples(train_recs, vqe, dev_feat, shots)
    ft_imps, ft_worses, ft_preds = [], [], []
    for s in (0, 1, 2, 3, 4):
        m = build_model(circuit_dim, noise_dim, 42)
        m.load_state_dict(twin_state)
        m, _ = finetune(m, ft_samples, 60, 2e-4, False, 0.2, s)
        pred = nem_predict(m, recs, vqe, dev_feat, shots)
        mm = metrics(pred, raw, ideal)
        ft_imps.append(mm["improvement_pct"])
        ft_worses.append(mm["worse_rate"])
        ft_preds.append(pred)
    ft_mean_pred = np.mean(ft_preds, axis=0)
    ensemble_m = metrics(ft_mean_pred, raw, ideal)

    print(f"\n=== On the ORIGINAL {n_inst} instances (Table 7 row, twin {old['neural_improvement_pct']:+.0f}%) ===")
    print(f"  raw MAE                  {np.abs(raw-ideal).mean():.5f}")
    print(f"  twin (recomputed)        {twin_m['improvement_pct']:+6.1f}%  worse {twin_m['worse_rate']*100:.0f}%")
    print(f"  affine recalibration     {affine_m['improvement_pct']:+6.1f}%  worse {affine_m['worse_rate']*100:.0f}%  (a={a:.3f})")
    print(f"  fine-tune (seed-avg)     {np.mean(ft_imps):+6.1f}% (±{np.std(ft_imps):.1f}, min {np.min(ft_imps):+.1f})  worse {np.mean(ft_worses)*100:.0f}%")
    print(f"  fine-tune (5-seed ens.)  {ensemble_m['improvement_pct']:+6.1f}%  worse {ensemble_m['worse_rate']*100:.0f}%")

    out = {
        "n_instances": n_inst, "seed": seed,
        "raw_mae": float(np.abs(raw - ideal).mean()),
        "twin": twin_m, "affine": {**affine_m, "a": a, "b": b},
        "finetune_seed_avg": {"impr_mean": float(np.mean(ft_imps)),
                              "impr_std": float(np.std(ft_imps)),
                              "impr_min": float(np.min(ft_imps)),
                              "worse_mean": float(np.mean(ft_worses))},
        "finetune_ensemble": ensemble_m,
    }
    json.dump(out, open(HW / "reconstruct_oldhw_n4L5.json", "w"), indent=2)
    print(f"Saved to {HW / 'reconstruct_oldhw_n4L5.json'}")


if __name__ == "__main__":
    main()
