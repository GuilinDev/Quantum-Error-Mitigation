#!/usr/bin/env python3
"""Main entry point for running experiments.

This script provides a unified interface for all project operations.

Usage:
    python run.py train --config experiments/configs/vqe.yaml
    python run.py evaluate --model checkpoints/best_model.pt --benchmark h2
    python run.py generate-data --config experiments/configs/vqe.yaml --output data/
    python run.py plot --results experiments/results/ --output paper/figures/
"""

import argparse
import sys
from pathlib import Path


def setup_paths():
    """Add project root to Python path."""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))


def cmd_train(args):
    """Run model training."""
    from experiments.scripts.train_error_predictor import main as train_main
    sys.argv = [
        "train_error_predictor.py",
        "--config", args.config,
    ]
    if args.device:
        sys.argv.extend(["--device", args.device])
    if args.data_only:
        sys.argv.append("--data-only")
    if args.load_data:
        sys.argv.extend(["--load-data", args.load_data])
    train_main()


def cmd_evaluate(args):
    """Run model evaluation."""
    from experiments.scripts.evaluate_mitigation import main as eval_main
    sys.argv = [
        "evaluate_mitigation.py",
        "--model-path", args.model,
        "--benchmark", args.benchmark,
        "--config", args.config,
        "--output-dir", args.output,
    ]
    eval_main()


def cmd_plot(args):
    """Generate publication figures."""
    from experiments.scripts.plot_results import main as plot_main
    sys.argv = [
        "plot_results.py",
        "--results-dir", args.results,
        "--output-dir", args.output,
    ]
    plot_main()


def cmd_test(args):
    """Run unit tests."""
    import pytest
    pytest_args = ["tests/", "-v"]
    if args.coverage:
        pytest_args.extend(["--cov=src", "--cov-report=html"])
    sys.exit(pytest.main(pytest_args))


def cmd_demo(args):
    """Run a quick demonstration."""
    setup_paths()

    import numpy as np
    from src.quantum.circuits import VQECircuit, MolecularVQECircuit
    from src.quantum.noise_models import RealisticDeviceNoise, IBMQ_MONTREAL_NOISE
    from src.classical.baselines import ZeroNoiseExtrapolation, compute_ideal_value

    print("=" * 60)
    print("Neural Error Mitigation - Quick Demo")
    print("=" * 60)

    # Create VQE circuit
    print("\n1. Creating H2 VQE circuit...")
    vqe = MolecularVQECircuit(molecule='H2', num_layers=2)
    print(f"   Qubits: {vqe.num_qubits}, Parameters: {vqe.num_parameters}")

    # Set up noise
    print("\n2. Setting up IBMQ Montreal noise model...")
    noise_model = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE)
    print(f"   1-qubit error: {noise_model.params.single_qubit_error}")
    print(f"   2-qubit error: {noise_model.params.two_qubit_error}")

    # Random parameters
    np.random.seed(42)
    params = np.random.uniform(0, 2 * np.pi, vqe.num_parameters)
    circuit = vqe.bind_parameters(params)
    hamiltonian = vqe.get_molecular_hamiltonian()

    # Compute values
    print("\n3. Computing expectation values...")
    ideal = compute_ideal_value(circuit, hamiltonian)
    print(f"   Ideal energy: {ideal:.6f} Hartree")

    from src.utils.qiskit_compat import run_estimation
    aer_noise = noise_model.build_noise_model()
    noisy = run_estimation(circuit, hamiltonian, shots=8192, noise_model=aer_noise)
    print(f"   Noisy energy: {noisy:.6f} Hartree")
    print(f"   Error: {abs(noisy - ideal):.6f} Hartree")

    # ZNE
    print("\n4. Applying Zero-Noise Extrapolation...")
    zne = ZeroNoiseExtrapolation(scale_factors=[1.0, 1.5, 2.0], shots=4096)
    zne_result = zne.mitigate(circuit, hamiltonian, noise_model)
    print(f"   ZNE energy: {zne_result.mitigated_value:.6f} Hartree")
    print(f"   ZNE error: {abs(zne_result.mitigated_value - ideal):.6f} Hartree")

    improvement = (abs(noisy - ideal) - abs(zne_result.mitigated_value - ideal)) / abs(noisy - ideal)
    print(f"\n   ZNE Improvement: {improvement * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Demo complete! Train neural models for better mitigation.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SQAI 2026 Neural Error Mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py demo                          # Quick demonstration
  python run.py train --config configs/vqe.yaml
  python run.py evaluate --model best.pt --benchmark h2
  python run.py test --coverage
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demonstration")
    demo_parser.set_defaults(func=cmd_demo)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train error mitigation model")
    train_parser.add_argument(
        "--config", type=str, default="experiments/configs/vqe.yaml",
        help="Path to configuration file"
    )
    train_parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    train_parser.add_argument("--data-only", action="store_true", help="Only generate data")
    train_parser.add_argument("--load-data", type=str, help="Load existing data")
    train_parser.set_defaults(func=cmd_train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate mitigation methods")
    eval_parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    eval_parser.add_argument(
        "--benchmark", type=str, default="h2",
        choices=["h2", "lih", "maxcut", "all"]
    )
    eval_parser.add_argument(
        "--config", type=str, default="experiments/configs/vqe.yaml"
    )
    eval_parser.add_argument(
        "--output", type=str, default="experiments/results/evaluation"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate publication figures")
    plot_parser.add_argument(
        "--results", type=str, default="experiments/results"
    )
    plot_parser.add_argument(
        "--output", type=str, default="paper/figures"
    )
    plot_parser.set_defaults(func=cmd_plot)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run unit tests")
    test_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    setup_paths()
    args.func(args)


if __name__ == "__main__":
    main()
