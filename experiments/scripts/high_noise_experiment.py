#!/usr/bin/env python3
"""High noise experiment to demonstrate neural error mitigation advantage.

The key insight: Neural mitigation excels in high-noise regimes where
classical methods like ZNE struggle due to extrapolation instability.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.quantum.circuits import VQECircuit, MolecularVQECircuit
from src.quantum.noise_models import (
    RealisticDeviceNoise, VariableNoiseModel, NoiseParameters,
    HIGH_NOISE_PROFILE, IBMQ_MONTREAL_NOISE
)
from src.models.mitigation_net import create_mitigation_model
from src.training.data_generator import QuantumDataGenerator, MitigationDataset
from src.classical.baselines import ZeroNoiseExtrapolation, compute_ideal_value
from src.utils.qiskit_compat import run_estimation


# Custom high-noise profile for demonstrating neural advantage
VERY_HIGH_NOISE = NoiseParameters(
    single_qubit_error=0.005,
    two_qubit_error=0.08,       # 8% two-qubit error
    readout_error_0=0.08,
    readout_error_1=0.12,
    t1=40.0,                    # Short T1
    t2=25.0,                    # Short T2
    single_gate_time=0.06,
    two_gate_time=0.5,
)


def generate_high_noise_data(n_train=2000, n_val=400, n_test=300):
    """Generate training data with variable high noise."""
    print("\n" + "="*60)
    print("Step 1: Generating High-Noise Training Data")
    print("="*60)

    # Variable noise model spanning moderate to high noise
    noise_model = VariableNoiseModel(
        error_range=(0.02, 0.10),  # 2-10% error range
        seed=42
    )

    generator = QuantumDataGenerator(
        circuit_type='vqe',
        num_qubits=4,
        num_layers=2,
        noise_model=noise_model,
        shots=4096,
        seed=42
    )

    print(f"Generating {n_train} training samples...")
    train_samples = generator.generate_dataset(n_train, show_progress=True)

    print(f"Generating {n_val} validation samples...")
    val_samples = generator.generate_dataset(n_val, show_progress=True)

    print(f"Generating {n_test} test samples...")
    test_samples = generator.generate_dataset(n_test, show_progress=True)

    # Create datasets
    train_dataset = MitigationDataset(train_samples)
    val_dataset = MitigationDataset(val_samples)
    test_dataset = MitigationDataset(test_samples)

    # Save datasets
    data_dir = Path("data/high_noise_experiment")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save(str(data_dir / "train.npz"))
    val_dataset.save(str(data_dir / "val.npz"))
    test_dataset.save(str(data_dir / "test.npz"))

    # Print statistics
    errors = train_dataset.errors
    print(f"\nTraining data statistics:")
    print(f"  Mean error: {errors.mean():.6f}")
    print(f"  Std error: {errors.std():.6f}")
    print(f"  Max |error|: {np.abs(errors).max():.6f}")

    return train_dataset, val_dataset, test_dataset


def train_model(train_dataset, val_dataset, epochs=80):
    """Train the mitigation model with improved architecture."""
    print("\n" + "="*60)
    print("Step 2: Training Neural Mitigation Model")
    print("="*60)

    sample = train_dataset[0]
    circuit_dim = sample['circuit_features'].shape[0]
    noise_dim = sample['noise_features'].shape[0]

    print(f"Circuit feature dim: {circuit_dim}")
    print(f"Noise feature dim: {noise_dim}")

    # Larger model for high-noise regime
    model = create_mitigation_model(
        model_type='standard',
        circuit_dim=circuit_dim,
        noise_dim=noise_dim,
        hidden_dims=[256, 512, 256, 128],  # Deeper network
        dropout=0.15
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Training on: {device}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    # Combined loss: MSE + error-aware term
    def custom_loss(pred, target, noisy):
        mse = torch.nn.functional.mse_loss(pred, target)
        # Penalize predictions that are further from ideal than noisy was
        noisy_error = torch.abs(noisy - target)
        pred_error = torch.abs(pred - target)
        improvement_loss = torch.relu(pred_error - noisy_error).mean()
        return mse + 0.1 * improvement_loss

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            circuit_features = batch['circuit_features'].to(device)
            noise_features = batch['noise_features'].to(device)
            noisy_values = batch['noisy_value'].to(device)
            ideal_values = batch['ideal_value'].to(device)

            optimizer.zero_grad()
            mitigated = model(noisy_values, circuit_features, noise_features)
            loss = custom_loss(mitigated, ideal_values, noisy_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                circuit_features = batch['circuit_features'].to(device)
                noise_features = batch['noise_features'].to(device)
                noisy_values = batch['noisy_value'].to(device)
                ideal_values = batch['ideal_value'].to(device)

                mitigated = model(noisy_values, circuit_features, noise_features)
                loss = custom_loss(mitigated, ideal_values, noisy_values)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'circuit_dim': circuit_dim,
                    'noise_dim': noise_dim,
                    'hidden_dims': [256, 512, 256, 128],
                }
            }, 'experiments/results/high_noise_model.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


def evaluate_on_noise_levels(model, device='cpu'):
    """Evaluate at different noise levels to show where neural method excels."""
    print("\n" + "="*60)
    print("Step 3: Evaluating Across Noise Levels")
    print("="*60)

    model.eval()
    model = model.to(device)

    # Test on different noise profiles
    noise_profiles = {
        'Low (Montreal)': IBMQ_MONTREAL_NOISE,
        'Medium': NoiseParameters(
            single_qubit_error=0.002,
            two_qubit_error=0.03,
            readout_error_0=0.04,
            readout_error_1=0.06,
            t1=70.0, t2=50.0,
            single_gate_time=0.05, two_gate_time=0.35
        ),
        'High': HIGH_NOISE_PROFILE,
        'Very High': VERY_HIGH_NOISE,
    }

    vqe = VQECircuit(num_qubits=4, num_layers=2)
    # Create simple observable (sum of Z operators)
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    coeffs = []
    for i in range(4):
        pauli = ["I"] * 4
        pauli[i] = "Z"
        terms.append("".join(reversed(pauli)))
        coeffs.append(0.25)
    observable = SparsePauliOp(terms, coeffs)
    zne = ZeroNoiseExtrapolation(scale_factors=[1.0, 1.5, 2.0, 2.5], shots=4096)

    results = {}
    n_samples = 50

    for noise_name, noise_params in noise_profiles.items():
        print(f"\nEvaluating on {noise_name} noise...")
        noise_model = RealisticDeviceNoise(noise_params)

        ideal_vals = []
        noisy_vals = []
        neural_vals = []
        zne_vals = []

        np.random.seed(456)

        for i in tqdm(range(n_samples), desc=noise_name):
            params = np.random.uniform(0, 2*np.pi, vqe.num_parameters)
            circuit = vqe.bind_parameters(params)

            # Ideal
            ideal = compute_ideal_value(circuit, observable, shots=4096)
            ideal_vals.append(ideal)

            # Noisy
            aer_noise = noise_model.build_noise_model()
            noisy = run_estimation(circuit, observable, shots=4096, noise_model=aer_noise)
            noisy_vals.append(noisy)

            # ZNE
            try:
                zne_result = zne.mitigate(circuit, observable, noise_model)
                zne_vals.append(zne_result.mitigated_value)
            except Exception:
                # ZNE can fail with high noise
                zne_vals.append(noisy)

            # Neural
            circuit_features = torch.tensor(
                vqe.to_feature_vector(params), dtype=torch.float32
            ).unsqueeze(0).to(device)
            noise_features = torch.tensor(
                list(noise_model.to_feature_dict().values()), dtype=torch.float32
            ).unsqueeze(0).to(device)
            noisy_tensor = torch.tensor([[noisy]], dtype=torch.float32).to(device)

            with torch.no_grad():
                neural = model(noisy_tensor, circuit_features, noise_features)
            neural_vals.append(float(neural.cpu().numpy()[0, 0]))

        ideal_vals = np.array(ideal_vals)
        noisy_vals = np.array(noisy_vals)
        neural_vals = np.array(neural_vals)
        zne_vals = np.array(zne_vals)

        noisy_mae = np.abs(noisy_vals - ideal_vals).mean()
        neural_mae = np.abs(neural_vals - ideal_vals).mean()
        zne_mae = np.abs(zne_vals - ideal_vals).mean()

        results[noise_name] = {
            'noisy_mae': noisy_mae,
            'neural_mae': neural_mae,
            'zne_mae': zne_mae,
            'neural_improvement': (noisy_mae - neural_mae) / noisy_mae * 100,
            'zne_improvement': (noisy_mae - zne_mae) / noisy_mae * 100,
        }

        print(f"  Noisy MAE: {noisy_mae:.6f}")
        print(f"  Neural MAE: {neural_mae:.6f} ({results[noise_name]['neural_improvement']:.1f}% improvement)")
        print(f"  ZNE MAE: {zne_mae:.6f} ({results[noise_name]['zne_improvement']:.1f}% improvement)")

    return results


def plot_noise_comparison(results):
    """Generate comparison plot across noise levels."""
    print("\n" + "="*60)
    print("Step 4: Generating Comparison Figures")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    noise_levels = list(results.keys())
    noisy_maes = [results[n]['noisy_mae'] for n in noise_levels]
    neural_maes = [results[n]['neural_mae'] for n in noise_levels]
    zne_maes = [results[n]['zne_mae'] for n in noise_levels]

    x = np.arange(len(noise_levels))
    width = 0.25

    # MAE comparison
    ax = axes[0]
    bars1 = ax.bar(x - width, noisy_maes, width, label='No Mitigation', color='#d62728')
    bars2 = ax.bar(x, zne_maes, width, label='ZNE', color='#1f77b4')
    bars3 = ax.bar(x + width, neural_maes, width, label='Neural (Ours)', color='#2ca02c')

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Error Mitigation Performance Across Noise Levels', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels, rotation=15, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Improvement comparison
    ax = axes[1]
    neural_imp = [results[n]['neural_improvement'] for n in noise_levels]
    zne_imp = [results[n]['zne_improvement'] for n in noise_levels]

    ax.bar(x - width/2, zne_imp, width, label='ZNE', color='#1f77b4')
    ax.bar(x + width/2, neural_imp, width, label='Neural (Ours)', color='#2ca02c')

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Improvement over Noisy (%)', fontsize=12)
    ax.set_title('Error Reduction by Mitigation Method', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels, rotation=15, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/figures/noise_level_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/noise_level_comparison.pdf', bbox_inches='tight')
    print("Saved noise_level_comparison.png/pdf")

    # Print summary table
    print("\n" + "="*60)
    print("Summary Table (for paper)")
    print("="*60)
    print(f"{'Noise Level':<15} {'Noisy MAE':>12} {'ZNE MAE':>12} {'Neural MAE':>12} {'Neural Imp.':>12}")
    print("-" * 65)
    for n in noise_levels:
        r = results[n]
        print(f"{n:<15} {r['noisy_mae']:>12.6f} {r['zne_mae']:>12.6f} {r['neural_mae']:>12.6f} {r['neural_improvement']:>11.1f}%")


def main():
    print("="*60)
    print("SQAI 2026 - High Noise Neural Error Mitigation Experiment")
    print("="*60)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("paper/figures").mkdir(parents=True, exist_ok=True)

    # Step 1: Generate high-noise training data
    train_dataset, val_dataset, test_dataset = generate_high_noise_data(
        n_train=2000, n_val=400, n_test=300
    )

    # Step 2: Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, train_losses, val_losses = train_model(
        train_dataset, val_dataset, epochs=80
    )

    # Step 3: Evaluate across noise levels
    results = evaluate_on_noise_levels(model, device)

    # Step 4: Generate figures
    plot_noise_comparison(results)

    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
