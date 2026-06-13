#!/usr/bin/env python3
"""Train the error predictor neural network.

This script trains a neural network to predict noise-induced errors
from circuit structure and noise parameters.
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.quantum.circuits import VQECircuit, QAOACircuit
from src.quantum.noise_models import VariableNoiseModel, RealisticDeviceNoise, NoiseParameters
from src.models.error_predictor import ErrorPredictor, EnsembleErrorPredictor
from src.training.data_generator import QuantumDataGenerator, MitigationDataset, create_dataloaders
from src.training.trainer import Trainer, TrainingConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train error predictor model")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/noise_learning.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only generate data, don't train",
    )
    parser.add_argument(
        "--load-data",
        type=str,
        default=None,
        help="Path to pre-generated data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create output directories
    output_config = config["output"]
    Path(output_config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_config["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_config["data_dir"]).mkdir(parents=True, exist_ok=True)

    # Generate or load data
    data_config = config["data"]
    data_path = Path(output_config["data_dir"])

    if args.load_data:
        print(f"Loading data from {args.load_data}")
        train_data = MitigationDataset(data_file=f"{args.load_data}/train.npz")
        val_data = MitigationDataset(data_file=f"{args.load_data}/val.npz")
    else:
        print("Generating training data...")

        # Set up noise model
        noise_config = config["noise"]
        noise_model = VariableNoiseModel(
            error_range=tuple(noise_config["error_range"]),
            seed=data_config["seed"],
        )

        # Generate data for different circuit types
        all_train_samples = []
        all_val_samples = []

        circuit_config = config["circuits"]
        for circuit_type in circuit_config["types"]:
            for num_qubits in range(
                circuit_config["qubit_range"][0],
                circuit_config["qubit_range"][1] + 1,
            ):
                for num_layers in range(
                    circuit_config["layer_range"][0],
                    circuit_config["layer_range"][1] + 1,
                ):
                    print(
                        f"Generating data: {circuit_type}, "
                        f"{num_qubits} qubits, {num_layers} layers"
                    )

                    generator = QuantumDataGenerator(
                        circuit_type=circuit_type,
                        num_qubits=num_qubits,
                        num_layers=num_layers,
                        noise_model=noise_model,
                        shots=data_config["shots"],
                        seed=data_config["seed"],
                    )

                    # Proportional samples
                    n_train = data_config["train_samples"] // (
                        len(circuit_config["types"])
                        * len(range(*circuit_config["qubit_range"]))
                        * len(range(*circuit_config["layer_range"]))
                    )
                    n_val = data_config["val_samples"] // (
                        len(circuit_config["types"])
                        * len(range(*circuit_config["qubit_range"]))
                        * len(range(*circuit_config["layer_range"]))
                    )

                    all_train_samples.extend(generator.generate_dataset(n_train))
                    all_val_samples.extend(generator.generate_dataset(n_val))

        # Create datasets
        train_data = MitigationDataset(all_train_samples)
        val_data = MitigationDataset(all_val_samples)

        # Save datasets
        train_data.save(str(data_path / "train.npz"))
        val_data.save(str(data_path / "val.npz"))
        print(f"Saved data to {data_path}")

    if args.data_only:
        print("Data generation complete. Exiting.")
        return

    # Create data loaders
    train_config = config["training"]
    train_loader, val_loader = create_dataloaders(
        train_data.circuit_features,  # Need to adapt for dataset
        val_data.circuit_features,
        batch_size=train_config["batch_size"],
    )

    # Actually create from datasets directly
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Determine feature dimensions from data
    sample = train_data[0]
    circuit_dim = sample["circuit_features"].shape[0]
    noise_dim = sample["noise_features"].shape[0]

    print(f"Circuit feature dimension: {circuit_dim}")
    print(f"Noise feature dimension: {noise_dim}")

    # Create model
    model_config = config["model"]
    model = ErrorPredictor(
        circuit_dim=circuit_dim,
        noise_dim=noise_dim,
        hidden_dims=model_config["fusion"]["hidden_dims"],
        dropout=model_config["dropout"],
        use_attention=model_config["use_attention"],
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create training config
    training_config = TrainingConfig(
        circuit_dim=circuit_dim,
        noise_dim=noise_dim,
        hidden_dims=model_config["fusion"]["hidden_dims"],
        batch_size=train_config["batch_size"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        num_epochs=train_config["num_epochs"],
        warmup_epochs=train_config["warmup_epochs"],
        gradient_clip=train_config["gradient_clip"],
        scheduler=train_config["scheduler"],
        loss_type=train_config["loss_type"],
        dropout=model_config["dropout"],
        device=args.device,
        use_wandb=config["logging"]["use_wandb"],
        project_name=config["logging"]["project_name"],
        run_name=config["logging"]["run_name"],
        save_dir=output_config["checkpoint_dir"],
    )

    # Create trainer and train
    trainer = Trainer(model, training_config, train_loader, val_loader)
    history = trainer.train()

    print("Training complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
