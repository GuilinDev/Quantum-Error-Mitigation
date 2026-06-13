#!/usr/bin/env python3
"""Run one (qubit count, noise regime) cell of the scaling study.

For a given circuit size and noise regime this script generates training
data, trains the neural mitigation model (3 seeds, for post-hoc ensemble
analysis) and the direct-prediction baseline, and evaluates them against
raw noisy execution, mitiq ZNE variants (Richardson / exponential /
adaptive gate folding), and mitiq CDR on a shared set of test instances
with fixed noise realizations.

Regimes:
  systematic — VariableNoiseModel: depolarizing + thermal relaxation +
      readout with per-sample device parameters. Smooth noise scaling.
  nonlinear  — NonLinearCorrelatedNoise: stochastic gate-error
      fluctuations + correlated ZZ crosstalk + asymmetric readout.
  miscal     — coherent angle miscalibration applied at execute time to
      the {rz, sx, x, cx} native-basis circuit (per-gate offsets from a
      per-snapshot calibration table) on top of stochastic depolarizing
      + asymmetric readout. Unitary folding cannot cleanly amplify the
      coherent component, which defeats extrapolation-based mitigation.

Fairness design: every quantum execution (raw measurement, NEM input,
each ZNE scale point, each CDR training circuit) is an S-shot measured
estimate under the same fixed noise realization per test instance; ideal
values come from exact noiseless statevector simulation. No silent
fallbacks — baseline failures abort the cell.

Usage:
    python experiments/scripts/scaling_cell.py --qubits 8 --regime miscal
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
from qiskit_aer.noise import (
    NoiseModel as AerNoiseModel,
    ReadoutError,
    depolarizing_error,
)
from scipy import stats
from tqdm import tqdm

from src.classical.mitiq_baselines import mitiq_cdr, mitiq_zne
from src.models.direct_predictor import DirectPredictionNet
from src.models.mitigation_net import create_mitigation_model
from src.quantum.circuits import VQECircuit
from src.quantum.miscalibration import make_miscalibration_transform
from src.quantum.noise_models import (
    NonLinearCorrelatedNoise,
    VariableNoiseModel,
)
from src.training.data_generator import DataSample, MitigationDataset
from src.utils.qiskit_compat import run_estimation, run_estimation_sampled

REGIMES = ("systematic", "nonlinear", "miscal")
NATIVE_BASIS = ["rz", "sx", "x", "cx"]

# Per-sample randomization ranges.
NL_BASE_ERROR_RANGE = (0.01, 0.04)
NL_NONLINEARITY_RANGE = (0.2, 0.6)
NL_CORRELATION_RANGE = (0.1, 0.4)
MC_DELTA_RANGE = (0.02, 0.06)       # mean angle offset per rotation
MC_SIGMA_FRACTION = 0.5             # sigma = fraction * delta_bias
MC_DEPOL_2Q_RANGE = (0.015, 0.05)
MC_RO01_RANGE = (0.01, 0.04)
MC_RO10_RANGE = (0.03, 0.10)


def default_shots(num_qubits: int) -> int:
    """Per-execution shot budget, scaled down at large sizes for cost."""
    if num_qubits <= 10:
        return 8192
    if num_qubits <= 16:
        return 1024
    return 512


def make_observable(vqe: VQECircuit):
    from qiskit.quantum_info import SparsePauliOp

    n = vqe.num_qubits
    return SparsePauliOp(
        ["".join("Z" if j == i else "I" for j in range(n)) for i in range(n)],
        [1.0 / n] * n,
    )


def build_miscal_stochastic_noise(p2q, ro01, ro10) -> AerNoiseModel:
    nm = AerNoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ["cx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - ro01, ro01], [ro10, 1 - ro10]])
    )
    return nm


class CellContext:
    """Holds the circuit template, observable, and regime samplers."""

    def __init__(self, args):
        self.args = args
        self.vqe = VQECircuit(args.qubits, args.layers)
        self.observable = make_observable(self.vqe)
        if args.regime == "systematic":
            self.noise = VariableNoiseModel(seed=args.seed)
        elif args.regime == "nonlinear":
            self.noise = NonLinearCorrelatedNoise(seed=args.seed)
        else:
            self.noise = None  # miscal: noise drawn inline

    def bound_circuit(self, params):
        circuit = self.vqe.bind_parameters(params)
        if self.args.regime == "miscal":
            circuit = transpile(
                circuit, basis_gates=NATIVE_BASIS, optimization_level=0
            )
        return circuit

    def draw_noise(self, rng, snapshot_seed):
        """Draw one noise realization; returns (aer_nm, transform, features)."""
        if self.args.regime == "systematic":
            aer_nm, _ = self.noise.sample_and_build()
            feats = np.array(
                list(self.noise.to_feature_dict().values()), dtype=np.float32
            )
            return aer_nm, None, feats
        if self.args.regime == "nonlinear":
            self.noise.base_error = rng.uniform(*NL_BASE_ERROR_RANGE)
            self.noise.nonlinearity = rng.uniform(*NL_NONLINEARITY_RANGE)
            self.noise.correlation = rng.uniform(*NL_CORRELATION_RANGE)
            aer_nm = self.noise.build_noise_model()
            feats = np.array(
                list(self.noise.to_feature_dict().values()), dtype=np.float32
            )
            return aer_nm, None, feats
        # miscal
        delta = rng.uniform(*MC_DELTA_RANGE)
        sigma = MC_SIGMA_FRACTION * delta
        p2q = rng.uniform(*MC_DEPOL_2Q_RANGE)
        ro01 = rng.uniform(*MC_RO01_RANGE)
        ro10 = rng.uniform(*MC_RO10_RANGE)
        aer_nm = build_miscal_stochastic_noise(p2q, ro01, ro10)
        transform = make_miscalibration_transform(delta, sigma, snapshot_seed)
        feats = np.array(
            [delta, sigma, p2q, ro01, ro10, 0.0, 0.0, 0.0], dtype=np.float32
        )
        return aer_nm, transform, feats

    def measure_noisy(self, circuit, aer_nm, transform, shots, seed=None):
        executed = transform(circuit) if transform is not None else circuit
        return run_estimation_sampled(
            executed, self.observable, shots=shots,
            noise_model=aer_nm, seed=seed,
        )

    def exact_ideal(self, circuit):
        return run_estimation(
            circuit, self.observable, noise_model=None, exact=True
        )


def generate_samples(ctx, n_samples, seed, desc):
    """Generate supervised (noisy, ideal) samples with S-shot noisy inputs."""
    rng = np.random.default_rng(seed)
    shots = ctx.args.shots
    samples = []
    for i in tqdm(range(n_samples), desc=desc):
        params = rng.uniform(0, 2 * np.pi, ctx.vqe.num_parameters)
        circuit = ctx.bound_circuit(params)
        aer_nm, transform, feats = ctx.draw_noise(
            rng, snapshot_seed=seed * 1_000_000 + i
        )
        ideal = ctx.exact_ideal(circuit)
        noisy = ctx.measure_noisy(circuit, aer_nm, transform, shots)
        samples.append(DataSample(
            circuit_features=ctx.vqe.to_feature_vector(params),
            noise_features=feats,
            noisy_value=noisy, ideal_value=ideal, error=noisy - ideal,
        ))
    return samples


def generate_test_instances(ctx, n_instances, seed):
    """Test instances with frozen noise realizations for fair comparison."""
    rng = np.random.default_rng(seed)
    shots = ctx.args.shots
    instances = []
    for i in tqdm(range(n_instances), desc="test instances"):
        params = rng.uniform(0, 2 * np.pi, ctx.vqe.num_parameters)
        circuit = ctx.bound_circuit(params)
        aer_nm, transform, feats = ctx.draw_noise(
            rng, snapshot_seed=seed * 1_000_000 + i
        )
        ideal = ctx.exact_ideal(circuit)
        noisy = ctx.measure_noisy(circuit, aer_nm, transform, shots)
        instances.append({
            "circuit": circuit,
            "aer_noise": aer_nm,
            "transform": transform,
            "circuit_features": ctx.vqe.to_feature_vector(params),
            "noise_features": feats,
            "noisy": noisy,
            "ideal": ideal,
        })
    return instances


def train_model(model, train_ds, val_ds, epochs, device, lr=1e-3):
    """Train a model, returning the best-validation-loss state."""
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2
    )
    criterion = torch.nn.HuberLoss()
    best_val, best_state = float("inf"), None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            cf = batch["circuit_features"].to(device)
            nf = batch["noise_features"].to(device)
            nv = batch["noisy_value"].to(device)
            iv = batch["ideal_value"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(nv, cf, nf), iv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cf = batch["circuit_features"].to(device)
                nf = batch["noise_features"].to(device)
                nv = batch["noisy_value"].to(device)
                iv = batch["ideal_value"].to(device)
                val_loss += criterion(model(nv, cf, nf), iv).item()
        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


def neural_predictions(model, instances, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for inst in instances:
            cf = torch.tensor(inst["circuit_features"], dtype=torch.float32).unsqueeze(0).to(device)
            nf = torch.tensor(inst["noise_features"], dtype=torch.float32).unsqueeze(0).to(device)
            nv = torch.tensor([[inst["noisy"]]], dtype=torch.float32).to(device)
            preds.append(float(model(nv, cf, nf).cpu().numpy()[0, 0]))
    return np.array(preds)


def evaluate_classical(ctx, instances):
    """ZNE variants on all instances and CDR on a subset. No fallbacks."""
    args = ctx.args
    zne_variants = {
        "zne_richardson": {"extrapolation": "richardson"},
        "zne_exponential": {"extrapolation": "exponential", "asymptote": 0.0},
        "zne_adaptive": {"extrapolation": "adaptive", "asymptote": 0.0},
    }
    out = {k: [] for k in zne_variants}
    shots_used = {k: 0 for k in zne_variants}
    for inst in tqdm(instances, desc="ZNE eval"):
        for key, kw in zne_variants.items():
            res = mitiq_zne(
                inst["circuit"], ctx.observable, inst["aer_noise"],
                shots=args.shots, circuit_transform=inst["transform"],
                sampled=True, **kw,
            )
            out[key].append(res.mitigated_value)
            shots_used[key] += res.metadata["total_shots"]

    cdr_n = min(args.cdr_instances, len(instances))
    out["cdr"] = []
    shots_used["cdr"] = 0
    for inst in tqdm(instances[:cdr_n], desc="CDR eval"):
        res = mitiq_cdr(
            inst["circuit"], ctx.observable, inst["aer_noise"],
            num_training_circuits=args.cdr_training_circuits,
            shots=args.shots, circuit_transform=inst["transform"],
            sampled=True, skip_transpile=(args.regime == "miscal"),
        )
        out["cdr"].append(res.mitigated_value)
        shots_used["cdr"] += res.metadata["total_shots"]
    return {k: np.array(v) for k, v in out.items()}, shots_used, cdr_n


def mae(pred, ideal):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ideal))))


def paired_p(pred_a, pred_b, ideal):
    ea = np.abs(np.asarray(pred_a) - np.asarray(ideal))
    eb = np.abs(np.asarray(pred_b) - np.asarray(ideal))
    if np.allclose(ea, eb):
        return 1.0
    return float(stats.ttest_rel(ea, eb).pvalue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--test-instances", type=int, default=150)
    parser.add_argument("--cdr-instances", type=int, default=50)
    parser.add_argument("--cdr-training-circuits", type=int, default=30)
    parser.add_argument("--shots", type=int, default=None,
                        help="Per-execution shots (default: size-scaled)")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--model-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results/scaling")
    args = parser.parse_args()
    if args.shots is None:
        args.shots = default_shots(args.qubits)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    cell = f"{args.regime}_n{args.qubits}"
    timings = {}

    print(f"=== Scaling cell: {cell} (device={device}, shots={args.shots}) ===")
    ctx = CellContext(args)

    t0 = time.time()
    train_samples = generate_samples(ctx, args.train_samples, args.seed, "train data")
    val_samples = generate_samples(ctx, args.val_samples, args.seed + 1, "val data")
    timings["data_generation_s"] = time.time() - t0

    t0 = time.time()
    instances = generate_test_instances(ctx, args.test_instances, args.seed + 2)
    timings["test_instances_s"] = time.time() - t0

    train_ds = MitigationDataset(train_samples)
    val_ds = MitigationDataset(val_samples)
    circuit_dim = train_ds.circuit_features.shape[1]
    noise_dim = train_ds.noise_features.shape[1]
    ideal = np.array([i["ideal"] for i in instances])
    noisy = np.array([i["noisy"] for i in instances])

    # Train NEM across seeds (saved individually for ensemble analysis)
    # and the direct-prediction baseline (single seed).
    t0 = time.time()
    seed_models, seed_vals = [], []
    for k in range(args.model_seeds):
        torch.manual_seed(args.seed + k)
        nem = create_mitigation_model(
            "standard", circuit_dim=circuit_dim, noise_dim=noise_dim,
            hidden_dims=[256, 512, 256], dropout=0.15,
        )
        nem, val_nem = train_model(nem, train_ds, val_ds, args.epochs, device)
        seed_models.append(nem)
        seed_vals.append(val_nem)
        torch.save(nem.state_dict(), out_dir / "models" / f"{cell}_neural_seed{k}.pt")

    torch.manual_seed(args.seed)
    direct = DirectPredictionNet(circuit_dim=circuit_dim, noise_dim=noise_dim)
    direct, _ = train_model(direct, train_ds, val_ds, args.epochs, device)
    torch.save(direct.state_dict(), out_dir / "models" / f"{cell}_direct.pt")
    timings["training_s"] = time.time() - t0

    best_idx = int(np.argmin(seed_vals))
    nem_params = sum(p.numel() for p in seed_models[0].parameters())

    t0 = time.time()
    per_seed_preds = [neural_predictions(m, instances, device) for m in seed_models]
    nem_pred = per_seed_preds[best_idx]
    direct_pred = neural_predictions(direct, instances, device)
    classical, shots_used, cdr_n = evaluate_classical(ctx, instances)
    timings["evaluation_s"] = time.time() - t0

    methods = {
        "raw": noisy,
        "neural": nem_pred,
        "direct": direct_pred,
        **classical,
    }
    results = {"maes": {}, "improvement_pct": {}, "p_vs_raw": {}}
    for name, pred in methods.items():
        ref_ideal = ideal[: len(pred)]
        m = mae(pred, ref_ideal)
        ref_raw = mae(noisy[: len(pred)], ref_ideal)
        results["maes"][name] = m
        results["improvement_pct"][name] = (ref_raw - m) / ref_raw * 100
        results["p_vs_raw"][name] = paired_p(pred, noisy[: len(pred)], ref_ideal)
    results["p_neural_vs_zne_adaptive"] = paired_p(
        nem_pred, classical["zne_adaptive"], ideal
    )

    payload = {
        "cell": cell,
        "config": vars(args),
        "circuit_dim": circuit_dim,
        "noise_dim": noise_dim,
        "nem_param_count": nem_params,
        "best_seed": best_idx,
        "seed_val_losses": seed_vals,
        "n_test_instances": len(instances),
        "n_cdr_instances": cdr_n,
        "results": results,
        "shots_per_method": shots_used,
        "timings": timings,
        "per_instance": {
            "ideal": ideal.tolist(),
            "noisy": noisy.tolist(),
            "neural": nem_pred.tolist(),
            "direct": direct_pred.tolist(),
            **{f"neural_seed{k}": p.tolist() for k, p in enumerate(per_seed_preds)},
            **{k: v.tolist() for k, v in classical.items()},
        },
    }
    out_path = out_dir / f"{cell}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== {cell} results (MAE, improvement vs raw) ===")
    for name in methods:
        print(
            f"{name:18s} MAE={results['maes'][name]:.5f} "
            f"({results['improvement_pct'][name]:+7.1f}%)  "
            f"p_vs_raw={results['p_vs_raw'][name]:.4f}"
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
