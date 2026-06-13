#!/usr/bin/env python3
"""Quick experiment script for demonstration.

This runs a smaller-scale experiment to verify the full pipeline works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.quantum.circuits import VQECircuit, MolecularVQECircuit
from src.quantum.noise_models import RealisticDeviceNoise, VariableNoiseModel, IBMQ_MONTREAL_NOISE
from src.models.mitigation_net import create_mitigation_model, MitigationNetwork
from src.training.data_generator import QuantumDataGenerator, MitigationDataset
from src.classical.baselines import ZeroNoiseExtrapolation, compute_ideal_value
from src.utils.qiskit_compat import run_estimation
from src.utils.metrics import statistical_metrics, improvement_ratio


def generate_quick_dataset(n_train=500, n_val=100, n_test=50):
    """Generate a small dataset for quick experiments."""
    print("\n" + "="*60)
    print("Step 1: Generating Training Data")
    print("="*60)

    # Use variable noise model for diverse training data
    noise_model = VariableNoiseModel(
        error_range=(0.001, 0.03),
        seed=42
    )

    generator = QuantumDataGenerator(
        circuit_type='vqe',
        num_qubits=4,
        num_layers=2,
        noise_model=noise_model,
        shots=2048,  # Reduced for speed
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
    data_dir = Path("data/quick_experiment")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save(str(data_dir / "train.npz"))
    val_dataset.save(str(data_dir / "val.npz"))
    test_dataset.save(str(data_dir / "test.npz"))

    print(f"\nData saved to {data_dir}")

    return train_dataset, val_dataset, test_dataset


def train_model(train_dataset, val_dataset, epochs=30):
    """Train the mitigation model."""
    print("\n" + "="*60)
    print("Step 2: Training Neural Mitigation Model")
    print("="*60)

    # Get feature dimensions
    sample = train_dataset[0]
    circuit_dim = sample['circuit_features'].shape[0]
    noise_dim = sample['noise_features'].shape[0]

    print(f"Circuit feature dim: {circuit_dim}")
    print(f"Noise feature dim: {noise_dim}")

    # Create model
    model = create_mitigation_model(
        model_type='standard',
        circuit_dim=circuit_dim,
        noise_dim=noise_dim,
        hidden_dims=[128, 256, 128],
        dropout=0.1
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Training on: {device}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = torch.nn.HuberLoss()

    # Training loop
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
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'circuit_dim': circuit_dim,
                    'noise_dim': noise_dim,
                    'hidden_dims': [128, 256, 128],
                }
            }, 'experiments/results/quick_model.pt')

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


def evaluate_model(model, test_dataset, device='cpu'):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("Step 3: Evaluating Error Mitigation Performance")
    print("="*60)

    model.eval()
    model = model.to(device)

    ideal_values = []
    noisy_values = []
    neural_mitigated = []

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    with torch.no_grad():
        for batch in test_loader:
            circuit_features = batch['circuit_features'].to(device)
            noise_features = batch['noise_features'].to(device)
            noisy = batch['noisy_value'].to(device)
            ideal = batch['ideal_value'].to(device)

            mitigated = model(noisy, circuit_features, noise_features)

            ideal_values.extend(ideal.cpu().numpy().flatten())
            noisy_values.extend(noisy.cpu().numpy().flatten())
            neural_mitigated.extend(mitigated.cpu().numpy().flatten())

    ideal_values = np.array(ideal_values)
    noisy_values = np.array(noisy_values)
    neural_mitigated = np.array(neural_mitigated)

    # Compute metrics
    noisy_errors = np.abs(noisy_values - ideal_values)
    neural_errors = np.abs(neural_mitigated - ideal_values)

    print("\n--- Results ---")
    print(f"\nNo Mitigation:")
    print(f"  MAE: {noisy_errors.mean():.6f}")
    print(f"  RMSE: {np.sqrt((noisy_errors**2).mean()):.6f}")

    print(f"\nNeural Mitigation:")
    print(f"  MAE: {neural_errors.mean():.6f}")
    print(f"  RMSE: {np.sqrt((neural_errors**2).mean()):.6f}")

    improvement = (noisy_errors.mean() - neural_errors.mean()) / noisy_errors.mean()
    print(f"\nImprovement: {improvement*100:.1f}%")

    return {
        'ideal': ideal_values,
        'noisy': noisy_values,
        'neural': neural_mitigated,
        'noisy_mae': noisy_errors.mean(),
        'neural_mae': neural_errors.mean(),
        'improvement': improvement
    }


def compare_with_zne(n_samples=20):
    """Compare neural mitigation with ZNE on fresh samples."""
    print("\n" + "="*60)
    print("Step 4: Comparing with ZNE Baseline")
    print("="*60)

    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load('experiments/results/quick_model.pt', map_location=device)

    model = create_mitigation_model(
        model_type='standard',
        circuit_dim=checkpoint['config']['circuit_dim'],
        noise_dim=checkpoint['config']['noise_dim'],
        hidden_dims=checkpoint['config'].get('hidden_dims', [128, 256, 128]),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Set up circuit and noise
    vqe = MolecularVQECircuit(molecule='H2', num_layers=2)
    noise_model = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE)
    hamiltonian = vqe.get_molecular_hamiltonian()

    # ZNE
    zne = ZeroNoiseExtrapolation(scale_factors=[1.0, 1.5, 2.0], shots=2048)

    results = {
        'ideal': [],
        'noisy': [],
        'neural': [],
        'zne': []
    }

    np.random.seed(123)

    for i in tqdm(range(n_samples), desc="Comparing methods"):
        # Random parameters
        params = np.random.uniform(0, 2*np.pi, vqe.num_parameters)
        circuit = vqe.bind_parameters(params)

        # Ideal value
        ideal = compute_ideal_value(circuit, hamiltonian, shots=2048)
        results['ideal'].append(ideal)

        # Noisy value
        aer_noise = noise_model.build_noise_model()
        noisy = run_estimation(circuit, hamiltonian, shots=2048, noise_model=aer_noise)
        results['noisy'].append(noisy)

        # ZNE
        zne_result = zne.mitigate(circuit, hamiltonian, noise_model)
        results['zne'].append(zne_result.mitigated_value)

        # Neural mitigation
        circuit_features = torch.tensor(
            vqe.to_feature_vector(params), dtype=torch.float32
        ).unsqueeze(0).to(device)
        noise_features = torch.tensor(
            list(noise_model.to_feature_dict().values()), dtype=torch.float32
        ).unsqueeze(0).to(device)
        noisy_tensor = torch.tensor([[noisy]], dtype=torch.float32).to(device)

        with torch.no_grad():
            neural = model(noisy_tensor, circuit_features, noise_features)
        results['neural'].append(float(neural.cpu().numpy()[0, 0]))

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Compute errors
    ideal = results['ideal']

    print("\n--- Comparison Results ---")
    for method in ['noisy', 'zne', 'neural']:
        errors = np.abs(results[method] - ideal)
        print(f"\n{method.upper()}:")
        print(f"  MAE: {errors.mean():.6f}")
        print(f"  Max Error: {errors.max():.6f}")

    noisy_mae = np.abs(results['noisy'] - ideal).mean()
    neural_mae = np.abs(results['neural'] - ideal).mean()
    zne_mae = np.abs(results['zne'] - ideal).mean()

    print(f"\nImprovement over Noisy:")
    print(f"  ZNE: {(noisy_mae - zne_mae) / noisy_mae * 100:.1f}%")
    print(f"  Neural: {(noisy_mae - neural_mae) / noisy_mae * 100:.1f}%")

    return results


def main():
    print("="*60)
    print("SQAI 2026 - Neural Error Mitigation Quick Experiment")
    print("="*60)

    # Create results directory
    Path("experiments/results").mkdir(parents=True, exist_ok=True)

    # Step 1: Generate data
    train_dataset, val_dataset, test_dataset = generate_quick_dataset(
        n_train=300, n_val=60, n_test=40
    )

    # Step 2: Train model
    model, train_losses, val_losses = train_model(
        train_dataset, val_dataset, epochs=25
    )

    # Step 3: Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_results = evaluate_model(model, test_dataset, device)

    # Step 4: Compare with ZNE
    comparison = compare_with_zne(n_samples=15)

    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)

    return eval_results, comparison


if __name__ == "__main__":
    main()
