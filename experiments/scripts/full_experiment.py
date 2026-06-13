#!/usr/bin/env python3
"""Full experiment script with proper data generation.

This runs a more complete experiment with consistent noise model
between training and evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.quantum.circuits import VQECircuit, MolecularVQECircuit
from src.quantum.noise_models import RealisticDeviceNoise, VariableNoiseModel, IBMQ_MONTREAL_NOISE, NoiseParameters
from src.models.mitigation_net import create_mitigation_model, MitigationNetwork
from src.training.data_generator import MitigationDataset
from src.classical.baselines import ZeroNoiseExtrapolation, compute_ideal_value
from src.utils.qiskit_compat import run_estimation
from src.utils.metrics import statistical_metrics


def generate_consistent_data(n_train=2000, n_val=400, n_test=200):
    """Generate training data with consistent noise model."""
    print("\n" + "="*60)
    print("Step 1: Generating Training Data with IBMQ Montreal Noise")
    print("="*60)

    # Use fixed IBMQ Montreal noise for consistency
    noise_model = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE)

    # Create VQE circuit
    vqe = MolecularVQECircuit(molecule='H2', num_layers=2)
    hamiltonian = vqe.get_molecular_hamiltonian()
    aer_noise = noise_model.build_noise_model()

    all_data = {
        'circuit_features': [],
        'noise_features': [],
        'noisy_values': [],
        'ideal_values': [],
        'errors': []
    }

    np.random.seed(42)
    total_samples = n_train + n_val + n_test

    print(f"Generating {total_samples} samples...")
    for i in tqdm(range(total_samples)):
        # Random parameters
        params = np.random.uniform(0, 2*np.pi, vqe.num_parameters)
        circuit = vqe.bind_parameters(params)

        # Ideal value
        ideal = compute_ideal_value(circuit, hamiltonian, shots=4096)

        # Noisy value
        noisy = run_estimation(circuit, hamiltonian, shots=4096, noise_model=aer_noise)

        # Features
        circuit_features = vqe.to_feature_vector(params)
        noise_features = np.array(list(noise_model.to_feature_dict().values()))

        all_data['circuit_features'].append(circuit_features)
        all_data['noise_features'].append(noise_features)
        all_data['noisy_values'].append(noisy)
        all_data['ideal_values'].append(ideal)
        all_data['errors'].append(noisy - ideal)

    # Convert to arrays
    for key in all_data:
        all_data[key] = np.array(all_data[key])

    # Split data
    train_end = n_train
    val_end = n_train + n_val

    train_data = {k: v[:train_end] for k, v in all_data.items()}
    val_data = {k: v[train_end:val_end] for k, v in all_data.items()}
    test_data = {k: v[val_end:] for k, v in all_data.items()}

    # Save data
    data_dir = Path("data/full_experiment")
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(data_dir / "train.npz", **train_data)
    np.savez(data_dir / "val.npz", **val_data)
    np.savez(data_dir / "test.npz", **test_data)

    print(f"\nData saved to {data_dir}")
    print(f"Training samples: {len(train_data['noisy_values'])}")
    print(f"Validation samples: {len(val_data['noisy_values'])}")
    print(f"Test samples: {len(test_data['noisy_values'])}")

    # Print statistics
    print(f"\nData statistics:")
    print(f"  Mean ideal: {all_data['ideal_values'].mean():.6f}")
    print(f"  Mean noisy: {all_data['noisy_values'].mean():.6f}")
    print(f"  Mean error: {all_data['errors'].mean():.6f}")
    print(f"  Std error: {all_data['errors'].std():.6f}")

    return train_data, val_data, test_data


class ArrayDataset(torch.utils.data.Dataset):
    """Dataset from numpy arrays."""
    def __init__(self, data_dict):
        self.circuit_features = data_dict['circuit_features']
        self.noise_features = data_dict['noise_features']
        self.noisy_values = data_dict['noisy_values']
        self.ideal_values = data_dict['ideal_values']

    def __len__(self):
        return len(self.noisy_values)

    def __getitem__(self, idx):
        return {
            'circuit_features': torch.tensor(self.circuit_features[idx], dtype=torch.float32),
            'noise_features': torch.tensor(self.noise_features[idx], dtype=torch.float32),
            'noisy_value': torch.tensor([self.noisy_values[idx]], dtype=torch.float32),
            'ideal_value': torch.tensor([self.ideal_values[idx]], dtype=torch.float32),
        }


def train_model(train_data, val_data, epochs=50):
    """Train the mitigation model."""
    print("\n" + "="*60)
    print("Step 2: Training Neural Mitigation Model")
    print("="*60)

    train_dataset = ArrayDataset(train_data)
    val_dataset = ArrayDataset(val_data)

    circuit_dim = train_data['circuit_features'].shape[1]
    noise_dim = train_data['noise_features'].shape[1]

    print(f"Circuit feature dim: {circuit_dim}")
    print(f"Noise feature dim: {noise_dim}")

    # Create model - use smaller model to avoid overfitting
    model = create_mitigation_model(
        model_type='standard',
        circuit_dim=circuit_dim,
        noise_dim=noise_dim,
        hidden_dims=[64, 128, 64],  # Smaller
        dropout=0.2
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Training on: {device}")

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = torch.nn.MSELoss()

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
            loss = criterion(mitigated, ideal_values)
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
                loss = criterion(mitigated, ideal_values)
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
                    'hidden_dims': [64, 128, 64],
                }
            }, 'experiments/results/full_model.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


def evaluate_comprehensive(test_data):
    """Comprehensive evaluation on test set."""
    print("\n" + "="*60)
    print("Step 3: Comprehensive Evaluation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    checkpoint = torch.load('experiments/results/full_model.pt', map_location=device)
    model = create_mitigation_model(
        model_type='standard',
        circuit_dim=checkpoint['config']['circuit_dim'],
        noise_dim=checkpoint['config']['noise_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Get neural predictions
    circuit_features = torch.tensor(test_data['circuit_features'], dtype=torch.float32).to(device)
    noise_features = torch.tensor(test_data['noise_features'], dtype=torch.float32).to(device)
    noisy_values = torch.tensor(test_data['noisy_values'].reshape(-1, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        neural_mitigated = model(noisy_values, circuit_features, noise_features)
    neural_mitigated = neural_mitigated.cpu().numpy().flatten()

    ideal = test_data['ideal_values']
    noisy = test_data['noisy_values']

    # Compute ZNE on a subset (slower)
    print("\nRunning ZNE on test samples...")
    vqe = MolecularVQECircuit(molecule='H2', num_layers=2)
    noise_model = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE)
    hamiltonian = vqe.get_molecular_hamiltonian()
    zne = ZeroNoiseExtrapolation(scale_factors=[1.0, 1.5, 2.0], shots=4096)

    # Only do ZNE on first 30 samples (slow)
    n_zne = min(30, len(test_data['noisy_values']))
    zne_values = []

    # Reconstruct parameters from features
    # The first 3 features are [num_qubits, num_layers, num_parameters]
    # The rest are the parameters themselves
    for i in tqdm(range(n_zne)):
        # Get parameters from features (after the first 3)
        params = test_data['circuit_features'][i, 3:]
        circuit = vqe.bind_parameters(params)
        zne_result = zne.mitigate(circuit, hamiltonian, noise_model)
        zne_values.append(zne_result.mitigated_value)

    zne_values = np.array(zne_values)

    # Compute metrics
    print("\n" + "="*60)
    print("Results")
    print("="*60)

    noisy_errors = np.abs(noisy - ideal)
    neural_errors = np.abs(neural_mitigated - ideal)
    zne_errors = np.abs(zne_values - ideal[:n_zne])

    print(f"\nNo Mitigation (all {len(ideal)} samples):")
    print(f"  MAE: {noisy_errors.mean():.6f}")
    print(f"  RMSE: {np.sqrt((noisy_errors**2).mean()):.6f}")
    print(f"  Max Error: {noisy_errors.max():.6f}")

    print(f"\nNeural Mitigation (all {len(ideal)} samples):")
    print(f"  MAE: {neural_errors.mean():.6f}")
    print(f"  RMSE: {np.sqrt((neural_errors**2).mean()):.6f}")
    print(f"  Max Error: {neural_errors.max():.6f}")
    improvement = (noisy_errors.mean() - neural_errors.mean()) / noisy_errors.mean()
    print(f"  Improvement: {improvement*100:.1f}%")

    print(f"\nZNE (first {n_zne} samples):")
    print(f"  MAE: {zne_errors.mean():.6f}")
    print(f"  RMSE: {np.sqrt((zne_errors**2).mean()):.6f}")
    print(f"  Max Error: {zne_errors.max():.6f}")
    zne_improvement = (noisy_errors[:n_zne].mean() - zne_errors.mean()) / noisy_errors[:n_zne].mean()
    print(f"  Improvement: {zne_improvement*100:.1f}%")

    # Neural on same ZNE subset
    print(f"\nNeural (same {n_zne} samples as ZNE):")
    neural_subset = neural_errors[:n_zne]
    print(f"  MAE: {neural_subset.mean():.6f}")
    neural_subset_improvement = (noisy_errors[:n_zne].mean() - neural_subset.mean()) / noisy_errors[:n_zne].mean()
    print(f"  Improvement: {neural_subset_improvement*100:.1f}%")

    return {
        'ideal': ideal,
        'noisy': noisy,
        'neural': neural_mitigated,
        'zne': zne_values,
        'noisy_mae': noisy_errors.mean(),
        'neural_mae': neural_errors.mean(),
        'zne_mae': zne_errors.mean(),
    }


def plot_results(results, train_losses, val_losses):
    """Generate result figures."""
    print("\n" + "="*60)
    print("Step 4: Generating Figures")
    print("="*60)

    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Progress')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / 'training_curves.png', dpi=150)
    plt.close()

    # Error comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ['No Mitigation', 'Neural', 'ZNE']
    maes = [results['noisy_mae'], results['neural_mae'], results['zne_mae']]
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    bars = ax.bar(methods, maes, color=colors)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error Mitigation Comparison')
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / 'error_comparison.png', dpi=150)
    plt.close()

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.scatter(results['ideal'], results['noisy'], alpha=0.5, s=20)
    lims = [min(results['ideal'].min(), results['noisy'].min()) - 0.01,
            max(results['ideal'].max(), results['noisy'].max()) + 0.01]
    ax1.plot(lims, lims, 'k--', linewidth=1)
    ax1.set_xlabel('Ideal Energy')
    ax1.set_ylabel('Noisy Energy')
    ax1.set_title('Before Mitigation')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(results['ideal'], results['neural'], alpha=0.5, s=20, color='green')
    ax2.plot(lims, lims, 'k--', linewidth=1)
    ax2.set_xlabel('Ideal Energy')
    ax2.set_ylabel('Mitigated Energy')
    ax2.set_title('After Neural Mitigation')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / 'scatter_comparison.png', dpi=150)
    plt.close()

    print(f"Figures saved to {fig_dir}")


def main():
    print("="*60)
    print("SQAI 2026 - Neural Error Mitigation Full Experiment")
    print("="*60)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)

    # Check if data exists
    data_dir = Path("data/full_experiment")
    if (data_dir / "train.npz").exists():
        print("\nLoading existing data...")
        train_data = dict(np.load(data_dir / "train.npz"))
        val_data = dict(np.load(data_dir / "val.npz"))
        test_data = dict(np.load(data_dir / "test.npz"))
    else:
        train_data, val_data, test_data = generate_consistent_data(
            n_train=1500, n_val=300, n_test=200
        )

    # Train
    model, train_losses, val_losses = train_model(train_data, val_data, epochs=60)

    # Evaluate
    results = evaluate_comprehensive(test_data)

    # Plot
    plot_results(results, train_losses, val_losses)

    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
