#!/usr/bin/env python3
"""Can the neural mitigator learn coherent-over-rotation corrections?

Trains the standard mitigation network on the fold-breaking regime
(coherent over-rotation + light depolarizing + asymmetric readout, with
per-sample drift delta) at 6 qubits and compares against raw noisy
execution and folding ZNE on held-out instances.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import (
    NoiseModel as AerNoiseModel,
    ReadoutError,
    coherent_unitary_error,
    depolarizing_error,
)

from mitiq.zne import execute_with_zne
from mitiq.zne.inference import ExpFactory, RichardsonFactory

from src.models.mitigation_net import create_mitigation_model
from src.quantum.circuits import VQECircuit
from src.training.data_generator import DataSample, MitigationDataset
from src.utils.qiskit_compat import run_estimation

N = 6
SHOTS = 8192
TRAIN, VAL, TEST = 4000, 300, 150
EPOCHS = 80
DELTA_RANGE = (0.02, 0.08)
RO_RANGE_01 = (0.01, 0.04)
RO_RANGE_10 = (0.03, 0.10)
DEPOL_2Q_RANGE = (0.015, 0.05)
CDR_TEST = 40


def build_noise(delta, ro01, ro10, p2q):
    nm = AerNoiseModel()
    nm.add_all_qubit_quantum_error(
        coherent_unitary_error(RYGate(delta).to_matrix()), ["ry"]
    )
    nm.add_all_qubit_quantum_error(
        coherent_unitary_error(RZGate(delta).to_matrix()), ["rz"]
    )
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ["cx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - ro01, ro01], [ro10, 1 - ro10]])
    )
    return nm


def noise_features(delta, ro01, ro10, p2q):
    return np.array([delta, ro01, ro10, p2q, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32)


def draw_noise(rng):
    return (
        rng.uniform(*DELTA_RANGE),
        rng.uniform(*RO_RANGE_01),
        rng.uniform(*RO_RANGE_10),
        rng.uniform(*DEPOL_2Q_RANGE),
    )


def gen_samples(vqe, obs, n, rng, desc):
    samples = []
    for _ in tqdm(range(n), desc=desc):
        params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        delta, ro01, ro10, p2q = draw_noise(rng)
        circuit = vqe.bind_parameters(params)
        nm = build_noise(delta, ro01, ro10, p2q)
        ideal = run_estimation(circuit, obs, shots=SHOTS,
                               noise_model=None, method="statevector")
        noisy = run_estimation(circuit, obs, shots=SHOTS, noise_model=nm)
        samples.append(DataSample(
            circuit_features=vqe.to_feature_vector(params),
            noise_features=noise_features(delta, ro01, ro10, p2q),
            noisy_value=noisy, ideal_value=ideal, error=noisy - ideal,
        ))
    return samples


def main():
    rng = np.random.default_rng(11)
    vqe = VQECircuit(N, 2)
    obs = SparsePauliOp(
        ["".join("Z" if j == i else "I" for j in range(N)) for i in range(N)],
        [1.0 / N] * N,
    )

    train = gen_samples(vqe, obs, TRAIN, rng, "train")
    val = gen_samples(vqe, obs, VAL, rng, "val")

    train_ds, val_ds = MitigationDataset(train), MitigationDataset(val)
    model = create_mitigation_model(
        "standard",
        circuit_dim=train_ds.circuit_features.shape[1],
        noise_dim=train_ds.noise_features.shape[1],
        hidden_dims=[256, 512, 256], dropout=0.15,
    )
    tl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)
    crit = torch.nn.HuberLoss()
    best_val, best_state = float("inf"), None
    torch.manual_seed(0)

    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss = crit(model(b["noisy_value"], b["circuit_features"],
                              b["noise_features"]), b["ideal_value"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for b in vl:
                vloss += crit(model(b["noisy_value"], b["circuit_features"],
                                    b["noise_features"]), b["ideal_value"]).item()
        vloss /= len(vl)
        if vloss < best_val:
            best_val, best_state = vloss, {
                k: v.clone() for k, v in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.eval()
    print(f"best val loss: {best_val:.6f}")

    # Held-out evaluation incl. ZNE/CDR on fixed realizations
    from src.classical.mitiq_baselines import mitiq_cdr

    errs = {"raw": [], "neural": [], "zne_exp": [], "zne_rich": [], "cdr": []}
    for i in tqdm(range(TEST), desc="test"):
        params = rng.uniform(0, 2 * np.pi, vqe.num_parameters)
        delta, ro01, ro10, p2q = draw_noise(rng)
        circuit = vqe.bind_parameters(params)
        nm = build_noise(delta, ro01, ro10, p2q)
        ideal = run_estimation(circuit, obs, shots=SHOTS,
                               noise_model=None, method="statevector")

        def executor(c, _nm=nm):
            return run_estimation(c, obs, shots=SHOTS, noise_model=_nm)

        noisy = executor(circuit)
        errs["raw"].append(abs(noisy - ideal))
        with torch.no_grad():
            pred = float(model(
                torch.tensor([[noisy]], dtype=torch.float32),
                torch.tensor(vqe.to_feature_vector(params),
                             dtype=torch.float32).unsqueeze(0),
                torch.tensor(noise_features(delta, ro01, ro10, p2q),
                             dtype=torch.float32).unsqueeze(0),
            ).numpy()[0, 0])
        errs["neural"].append(abs(pred - ideal))
        errs["zne_exp"].append(abs(execute_with_zne(
            circuit, executor,
            factory=ExpFactory(scale_factors=[1.0, 2.0, 3.0], asymptote=0.0),
        ) - ideal))
        errs["zne_rich"].append(abs(execute_with_zne(
            circuit, executor,
            factory=RichardsonFactory(scale_factors=[1.0, 2.0, 3.0]),
        ) - ideal))
        if i < CDR_TEST:
            errs["cdr"].append(abs(mitiq_cdr(
                circuit, obs, nm, shots=SHOTS
            ).mitigated_value - ideal))

    raw = np.mean(errs["raw"])
    print("\n=== Mixed regime (stochastic + coherent drift + readout), n=6 ===")
    for k, v in errs.items():
        m = np.mean(v)
        print(f"{k:9s} MAE={m:.5f}  improvement={(raw - m) / raw * 100:+7.1f}%  (n={len(v)})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total: {time.time() - t0:.0f}s")
