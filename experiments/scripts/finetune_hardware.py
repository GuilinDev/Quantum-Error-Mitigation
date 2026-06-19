#!/usr/bin/env python3
"""On-device fine-tuning of budget-aware neural error mitigation.

A NEM model trained only on an Aer *digital twin* overcorrects on the
real device: the twin's depolarizing channel overstates the device's true
bias, so the network learns too aggressive a correction (a sim-to-real
shift). This script closes that gap with a small on-device fine-tuning
set, and measures whether it turns the twin-only failure into a gain.

Protocol:
  1. Pretrain NEM on the Aer digital twin (classical, cheap).
  2. Measure a small fine-tuning set + a held-out test set of random VQE
     circuits on the REAL backend, in a SINGLE batched job, and compute
     their ideal labels classically (exact statevector).  This is the only
     QPU phase; the measured dataset is written to disk immediately.
  3. Fine-tune the twin-trained NEM on the real (noisy, ideal) pairs.
  4. Evaluate twin-only vs fine-tuned NEM on the held-out real-device
     instances, against unmitigated execution.  Also fit a 2-parameter
     affine recalibration of the twin model's correction as a robust
     low-data reference.

Because the QPU is touched exactly once, fine-tuning hyperparameters can
be re-explored offline for free with ``--from-dataset <device JSON>``.

Usage:
  # plumbing check, no token (real device snapshot, runs on Aer):
  python experiments/scripts/finetune_hardware.py --fake FakeTorino \
      --qubits 4 --layers 5 --twin-train-samples 400 \
      --finetune-circuits 40 --test-instances 12 --epochs 8 --smoke

  # real hardware (after save_account), single batched job:
  python experiments/scripts/finetune_hardware.py --backend least_busy \
      --qubits 4 --layers 5 --finetune-circuits 120 --test-instances 16

  # re-tune offline on already-measured device data:
  python experiments/scripts/finetune_hardware.py --from-dataset \
      experiments/results/hardware/finetune_dataset_ibm_marrakesh_n4L5.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from qiskit_aer.noise import NoiseModel

from src.models.mitigation_net import create_mitigation_model
from src.quantum.circuits import VQECircuit
from src.training.data_generator import DataSample, MitigationDataset
from src.utils.qiskit_compat import (
    run_estimation,
    run_estimation_hardware_batch,
)

from scaling_cell import make_observable, train_model
from hardware_run import (
    LOG2_S_RANGE,
    budget_feature,
    device_feature_vector,
    generate_twin_data,
)


# --------------------------------------------------------------------------
# Device data collection (the one QPU phase)
# --------------------------------------------------------------------------
def collect_device_dataset(args, backend, vqe, obs, dev_feat):
    """Measure fine-tune + held-out test circuits on the backend in one job.

    Returns a JSON-serializable dict with per-circuit params, the measured
    noisy expectation, and the classically-exact ideal label.
    """
    rng = np.random.default_rng(args.seed + 7)
    n_train, n_test = args.finetune_circuits, args.test_instances
    all_params = [
        rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        for _ in range(n_train + n_test)
    ]
    circuits = [vqe.bind_parameters(p) for p in all_params]
    ideals = [run_estimation(c, obs, noise_model=None, exact=True) for c in circuits]

    # One batched job for every measurement: collapses N jobs into one.
    use_batch = args.backend is not None
    if use_batch:
        from qiskit_ibm_runtime import Batch
        ctx = Batch(backend=backend)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    t0 = time.time()
    with ctx as batch:
        mode = batch if use_batch else None
        noisy = run_estimation_hardware_batch(
            circuits, obs, shots=args.shots, backend=backend, mode=mode
        )
    elapsed = time.time() - t0
    print(f"  measured {len(circuits)} circuits in one batched job "
          f"({elapsed:.0f}s wall)", flush=True)

    def pack(lo, hi):
        return [
            {"params": list(map(float, all_params[i])),
             "noisy": float(noisy[i]), "ideal": float(ideals[i])}
            for i in range(lo, hi)
        ]

    return {
        "backend": str(getattr(backend, "name", backend)),
        "n": args.qubits, "layers": args.layers, "shots": args.shots,
        "dev_feat": list(map(float, dev_feat)),
        "train": pack(0, n_train),
        "test": pack(n_train, n_train + n_test),
    }


# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------
def build_model(circuit_dim, noise_dim, seed):
    torch.manual_seed(seed)
    return create_mitigation_model(
        "standard", circuit_dim=circuit_dim, noise_dim=noise_dim,
        hidden_dims=[256, 512, 256], dropout=0.15)


def to_samples(records, vqe, dev_feat, shots):
    bf = budget_feature(shots)
    samples = []
    for r in records:
        params = np.asarray(r["params"], dtype=np.float32)
        nf = np.append(dev_feat, bf).astype(np.float32)
        samples.append(DataSample(
            circuit_features=vqe.to_feature_vector(params),
            noise_features=nf, noisy_value=r["noisy"],
            ideal_value=r["ideal"], error=r["noisy"] - r["ideal"]))
    return samples


def nem_predict(model, records, vqe, dev_feat, shots):
    """Per-record NEM prediction; returns np.array aligned with records."""
    bf = budget_feature(shots)
    model.eval()
    preds = []
    with torch.no_grad():
        for r in records:
            params = np.asarray(r["params"], dtype=np.float32)
            nf = np.append(dev_feat, bf).astype(np.float32)
            cf = torch.tensor(vqe.to_feature_vector(params), dtype=torch.float32).unsqueeze(0)
            nfv = torch.tensor(nf, dtype=torch.float32).unsqueeze(0)
            nv = torch.tensor([[r["noisy"]]], dtype=torch.float32)
            preds.append(float(model(nv, cf, nfv).numpy()[0, 0]))
    return np.array(preds)


def finetune(model, train_samples, epochs, lr, freeze_encoders, val_frac, seed):
    """Continue training ``model`` on a small device set; best-val state."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_samples))
    n_val = max(1, int(round(val_frac * len(train_samples))))
    val_idx, tr_idx = set(idx[:n_val].tolist()), idx[n_val:]
    tr = [train_samples[i] for i in tr_idx]
    va = [train_samples[i] for i in sorted(val_idx)]

    if freeze_encoders:
        for p in model.circuit_encoder.parameters():
            p.requires_grad_(False)
        for p in model.noise_encoder.parameters():
            p.requires_grad_(False)

    model, best_val = train_model(
        model, MitigationDataset(tr), MitigationDataset(va),
        epochs=epochs, device="cpu", lr=lr)
    return model, best_val


def affine_recalibration(twin_preds_train, noisy_train, ideal_train):
    """Least-squares fit of pred = noisy + a*(twin_pred - noisy) + b.

    A 2-parameter shrink/bias of the twin model's correction: robust with
    very little device data, and directly counters overcorrection (a<1).
    Returns (a, b).
    """
    corr = np.asarray(twin_preds_train) - np.asarray(noisy_train)
    target = np.asarray(ideal_train) - np.asarray(noisy_train)
    A = np.column_stack([corr, np.ones_like(corr)])
    (a, b), *_ = np.linalg.lstsq(A, target, rcond=None)
    return float(a), float(b)


def metrics(pred, raw, ideal):
    pred, raw, ideal = map(np.asarray, (pred, raw, ideal))
    err = np.abs(pred - ideal)
    raw_err = np.abs(raw - ideal)
    return {
        "mae": float(err.mean()),
        "improvement_pct": float((raw_err.mean() - err.mean()) / raw_err.mean() * 100),
        "worse_rate": float((err > raw_err).mean()),
    }


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fake", type=str, help="Fake backend class, e.g. FakeTorino")
    g.add_argument("--backend", type=str, help="Real backend name or 'least_busy'")
    p.add_argument("--from-dataset", type=str,
                   help="Skip QPU; load a previously measured device dataset JSON")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--twin-train-samples", type=int, default=2000)
    p.add_argument("--twin-val-samples", type=int, default=200)
    p.add_argument("--twin-epochs", type=int, default=75)
    p.add_argument("--finetune-circuits", type=int, default=120)
    p.add_argument("--test-instances", type=int, default=16)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=60, help="Fine-tune epochs")
    p.add_argument("--lr", type=float, default=2e-4, help="Fine-tune LR")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--freeze-encoders", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", type=str, default="experiments/results/hardware")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Acquire the device dataset (measure once, or reload) ----
    if args.from_dataset:
        dataset = json.load(open(args.from_dataset))
        args.qubits, args.layers = dataset["n"], dataset["layers"]
        args.shots = dataset["shots"]
        backend_name = dataset["backend"]
        dev_feat = np.array(dataset["dev_feat"], dtype=np.float32)
        vqe = VQECircuit(args.qubits, args.layers)
        obs = make_observable(vqe)
        print(f"=== Offline fine-tune from {args.from_dataset} "
              f"({backend_name}, n={args.qubits}, L={args.layers}) ===")
        twin = None
    else:
        if not (args.fake or args.backend):
            p.error("one of --fake / --backend / --from-dataset is required")
        if args.fake:
            from qiskit_ibm_runtime import fake_provider
            backend = getattr(fake_provider, args.fake)()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            svc = QiskitRuntimeService()
            backend = (svc.least_busy(operational=True, simulator=False)
                       if args.backend == "least_busy" else svc.backend(args.backend))
        backend_name = getattr(backend, "name", str(backend))
        print(f"=== On-device fine-tune on {backend_name} "
              f"({backend.num_qubits} qubits), n={args.qubits}, L={args.layers} ===")
        vqe = VQECircuit(args.qubits, args.layers)
        obs = make_observable(vqe)
        dev_feat = device_feature_vector(backend, args.qubits)
        twin = NoiseModel.from_backend(backend)
        print("Collecting device fine-tune + test set (one batched job)...", flush=True)
        dataset = collect_device_dataset(args, backend, vqe, obs, dev_feat)
        ds_path = out_dir / f"finetune_dataset_{backend_name}_n{args.qubits}L{args.layers}.json"
        json.dump(dataset, open(ds_path, "w"), indent=2)
        print(f"Saved device dataset to {ds_path}", flush=True)

    # ---- Pretrain the twin model (skip if reloading and a twin cache lacks backend) ----
    circuit_dim = len(vqe.to_feature_vector(np.zeros(vqe.num_parameters)))
    noise_dim = len(dev_feat) + 1
    if twin is not None:
        print("Pretraining NEM on the Aer digital twin...", flush=True)
        tr = generate_twin_data(vqe, obs, twin, dev_feat,
                                args.twin_train_samples, args.seed, "train")
        va = generate_twin_data(vqe, obs, twin, dev_feat,
                                args.twin_val_samples, args.seed + 1, "val")
        twin_model = build_model(circuit_dim, noise_dim, args.seed)
        twin_model, vloss = train_model(twin_model, MitigationDataset(tr),
                                        MitigationDataset(va), args.twin_epochs, "cpu")
        print(f"  twin-trained NEM val={vloss:.5f}", flush=True)
        torch.save(twin_model.state_dict(),
                   out_dir / f"twin_model_{backend_name}_n{args.qubits}L{args.layers}.pt")
    else:
        # Offline mode: load a previously saved twin model if present.
        cache = out_dir / f"twin_model_{backend_name}_n{args.qubits}L{args.layers}.pt"
        twin_model = build_model(circuit_dim, noise_dim, args.seed)
        if cache.exists():
            twin_model.load_state_dict(torch.load(cache))
            print(f"Loaded cached twin model from {cache}", flush=True)
        else:
            print("WARNING: no cached twin model; twin baseline is untrained.", flush=True)

    twin_state = {k: v.detach().clone() for k, v in twin_model.state_dict().items()}

    # ---- Fine-tune on the device training set ----
    train_recs, test_recs = dataset["train"], dataset["test"]
    raw_test = np.array([r["noisy"] for r in test_recs])
    ideal_test = np.array([r["ideal"] for r in test_recs])

    twin_pred_test = nem_predict(twin_model, test_recs, vqe, dev_feat, args.shots)
    twin_pred_train = nem_predict(twin_model, train_recs, vqe, dev_feat, args.shots)

    ft_model = build_model(circuit_dim, noise_dim, args.seed)
    ft_model.load_state_dict(twin_state)
    ft_samples = to_samples(train_recs, vqe, dev_feat, args.shots)
    print(f"Fine-tuning on {len(ft_samples)} device circuits "
          f"(lr={args.lr}, epochs={args.epochs}, "
          f"freeze_encoders={args.freeze_encoders})...", flush=True)
    ft_model, ft_val = finetune(ft_model, ft_samples, args.epochs, args.lr,
                                args.freeze_encoders, args.val_frac, args.seed)
    ft_pred_test = nem_predict(ft_model, test_recs, vqe, dev_feat, args.shots)

    # ---- Affine recalibration reference (2 params from the train set) ----
    a, b = affine_recalibration(
        twin_pred_train, [r["noisy"] for r in train_recs],
        [r["ideal"] for r in train_recs])
    affine_pred_test = raw_test + a * (twin_pred_test - raw_test) + b

    # ---- Report ----
    raw_metrics = {"mae": float(np.abs(raw_test - ideal_test).mean())}
    res = {
        "backend": backend_name, "n": args.qubits, "layers": args.layers,
        "shots": args.shots, "n_finetune": len(train_recs),
        "n_test": len(test_recs),
        "finetune": {"lr": args.lr, "epochs": args.epochs,
                     "freeze_encoders": args.freeze_encoders, "val_loss": ft_val},
        "affine": {"a": a, "b": b},
        "raw_mae": raw_metrics["mae"],
        "nem_twin": metrics(twin_pred_test, raw_test, ideal_test),
        "nem_finetuned": metrics(ft_pred_test, raw_test, ideal_test),
        "nem_affine": metrics(affine_pred_test, raw_test, ideal_test),
        "per_instance": {
            "ideal": ideal_test.tolist(), "raw": raw_test.tolist(),
            "nem_twin": twin_pred_test.tolist(),
            "nem_finetuned": ft_pred_test.tolist(),
            "nem_affine": affine_pred_test.tolist(),
        },
    }
    tag = "smoke" if args.smoke else "run"
    out_path = out_dir / f"finetune_{backend_name}_n{args.qubits}L{args.layers}_{tag}.json"
    json.dump(res, open(out_path, "w"), indent=2)

    print(f"\n=== Held-out test ({len(test_recs)} real-device instances) ===")
    print(f"  raw MAE                 {res['raw_mae']:.5f}")
    for k in ("nem_twin", "nem_finetuned", "nem_affine"):
        m = res[k]
        print(f"  {k:22s} MAE={m['mae']:.5f}  "
              f"impr={m['improvement_pct']:+6.1f}%  worse={m['worse_rate']*100:.0f}%")
    print(f"  affine fit: a={a:.3f} b={b:+.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
