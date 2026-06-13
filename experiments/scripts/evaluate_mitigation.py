#!/usr/bin/env python3
"""Evaluate error mitigation methods on VQE and QAOA benchmarks.

This script runs comprehensive evaluation comparing neural error mitigation
against baseline methods (ZNE, PEC, etc.).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.quantum.circuits import VQECircuit, QAOACircuit, MolecularVQECircuit
from src.quantum.noise_models import (
    RealisticDeviceNoise,
    NoiseParameters,
    IBMQ_MONTREAL_NOISE,
)
from src.models.mitigation_net import MitigationNetwork, create_mitigation_model
from src.utils.metrics import (
    energy_error,
    improvement_ratio,
    approximation_ratio,
    statistical_metrics,
    bootstrap_confidence_interval,
)
from src.utils.visualization import (
    plot_error_comparison,
    plot_noise_scaling,
    plot_scatter_comparison,
    save_all_figures,
)

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


class BaselineMethods:
    """Implementation of baseline error mitigation methods."""

    @staticmethod
    def no_mitigation(noisy_values: np.ndarray) -> np.ndarray:
        """No mitigation - return noisy values as-is."""
        return noisy_values

    @staticmethod
    def zero_noise_extrapolation(
        circuit,
        observable,
        noise_model,
        scale_factors: List[float] = [1.0, 2.0, 3.0],
        shots: int = 8192,
    ) -> float:
        """Zero Noise Extrapolation (ZNE).

        Runs circuit at multiple noise levels and extrapolates to zero noise.
        """
        expectation_values = []

        for scale in scale_factors:
            # Scale noise parameters
            scaled_params = NoiseParameters(
                single_qubit_error=noise_model.params.single_qubit_error * scale,
                two_qubit_error=noise_model.params.two_qubit_error * scale,
                readout_error_0=noise_model.params.readout_error_0 * scale,
                readout_error_1=noise_model.params.readout_error_1 * scale,
                t1=noise_model.params.t1 / scale,
                t2=noise_model.params.t2 / scale,
                single_gate_time=noise_model.params.single_gate_time,
                two_gate_time=noise_model.params.two_gate_time,
            )

            scaled_noise = RealisticDeviceNoise(scaled_params)
            aer_noise = scaled_noise.build_noise_model()

            estimator = AerEstimator(
                options={"default_shots": shots, "noise_model": aer_noise}
            )
            job = estimator.run([(circuit, observable)])
            result = job.result()
            expectation_values.append(float(result[0].data.evs))

        # Richardson extrapolation
        scale_factors = np.array(scale_factors)
        exp_values = np.array(expectation_values)

        # Linear extrapolation to zero noise
        coeffs = np.polyfit(scale_factors, exp_values, deg=len(scale_factors) - 1)
        zero_noise_value = np.polyval(coeffs, 0)

        return float(zero_noise_value)


def evaluate_vqe(
    config: dict,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Dict[str, dict]:
    """Evaluate mitigation methods on VQE benchmark.

    Args:
        config: Experiment configuration.
        model: Trained mitigation model.
        device: Compute device.

    Returns:
        Dictionary of results for each method.
    """
    print("\nEvaluating VQE Error Mitigation...")

    # Set up circuit
    circuit_config = config["circuit"]
    vqe = MolecularVQECircuit(
        molecule=circuit_config["molecule"],
        num_layers=circuit_config["num_layers"],
        bond_distance=circuit_config["bond_distance"],
    )

    observable = vqe.get_molecular_hamiltonian()

    # Set up noise
    noise = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE)
    aer_noise = noise.build_noise_model()

    # Ideal simulator
    ideal_estimator = AerEstimator(options={"default_shots": config["data"]["shots"]})

    # Noisy simulator
    noisy_estimator = AerEstimator(
        options={"default_shots": config["data"]["shots"], "noise_model": aer_noise}
    )

    # Run evaluation
    n_samples = config["data"]["test_samples"]
    rng = np.random.default_rng(config["data"]["seed"])

    results = {
        "ideal": [],
        "noisy": [],
        "neural": [],
        "zne": [],
    }

    for _ in tqdm(range(n_samples), desc="VQE Evaluation"):
        # Random parameters
        params = rng.uniform(0, 2 * np.pi, size=vqe.num_parameters)
        circuit = vqe.bind_parameters(params)

        # Ideal expectation value
        job = ideal_estimator.run([(circuit, observable)])
        ideal_value = float(job.result()[0].data.evs)
        results["ideal"].append(ideal_value)

        # Noisy expectation value
        job = noisy_estimator.run([(circuit, observable)])
        noisy_value = float(job.result()[0].data.evs)
        results["noisy"].append(noisy_value)

        # Neural mitigation
        circuit_features = torch.tensor(
            vqe.to_feature_vector(params), dtype=torch.float32
        ).unsqueeze(0).to(device)
        noise_features = torch.tensor(
            list(noise.to_feature_dict().values()), dtype=torch.float32
        ).unsqueeze(0).to(device)
        noisy_tensor = torch.tensor(
            [[noisy_value]], dtype=torch.float32
        ).to(device)

        model.eval()
        with torch.no_grad():
            mitigated = model(noisy_tensor, circuit_features, noise_features)
        results["neural"].append(float(mitigated.cpu().numpy()[0, 0]))

        # ZNE
        zne_value = BaselineMethods.zero_noise_extrapolation(
            circuit, observable, noise, shots=config["data"]["shots"]
        )
        results["zne"].append(zne_value)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Compute metrics
    metrics = {}
    ideal = results["ideal"]

    for method in ["noisy", "neural", "zne"]:
        values = results[method]
        metrics[method] = statistical_metrics(values, ideal)
        metrics[method]["improvement"] = improvement_ratio(
            np.abs(values - ideal).mean(),
            np.abs(results["noisy"] - ideal).mean(),
        )

    return {"values": results, "metrics": metrics}


def evaluate_qaoa(
    config: dict,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Dict[str, dict]:
    """Evaluate mitigation methods on QAOA MaxCut benchmark."""
    print("\nEvaluating QAOA Error Mitigation...")

    results_by_size = {}

    for num_qubits in config["scaling"]["qubit_range"]:
        print(f"  Evaluating {num_qubits} qubits...")

        # Create QAOA circuit
        qaoa = QAOACircuit(
            num_qubits=num_qubits,
            num_layers=config["circuit"]["num_layers"],
        )

        observable = qaoa.get_cost_hamiltonian()

        # Noise model
        noise = RealisticDeviceNoise(IBMQ_MONTREAL_NOISE)
        aer_noise = noise.build_noise_model()

        # Simulators
        ideal_estimator = AerEstimator(
            options={"default_shots": config["data"]["shots"]}
        )
        noisy_estimator = AerEstimator(
            options={"default_shots": config["data"]["shots"], "noise_model": aer_noise}
        )

        n_samples = config["scaling"]["samples_per_size"]
        rng = np.random.default_rng(config["data"]["seed"] + num_qubits)

        results = {"ideal": [], "noisy": [], "neural": [], "zne": []}

        for _ in range(n_samples):
            params = rng.uniform(0, np.pi, size=qaoa.num_parameters)
            circuit = qaoa.bind_parameters(params)

            # Ideal
            job = ideal_estimator.run([(circuit, observable)])
            ideal_value = float(job.result()[0].data.evs)
            results["ideal"].append(ideal_value)

            # Noisy
            job = noisy_estimator.run([(circuit, observable)])
            noisy_value = float(job.result()[0].data.evs)
            results["noisy"].append(noisy_value)

            # Neural mitigation
            circuit_features = torch.tensor(
                qaoa.to_feature_vector(params), dtype=torch.float32
            ).unsqueeze(0).to(device)
            noise_features = torch.tensor(
                list(noise.to_feature_dict().values()), dtype=torch.float32
            ).unsqueeze(0).to(device)
            noisy_tensor = torch.tensor(
                [[noisy_value]], dtype=torch.float32
            ).to(device)

            model.eval()
            with torch.no_grad():
                mitigated = model(noisy_tensor, circuit_features, noise_features)
            results["neural"].append(float(mitigated.cpu().numpy()[0, 0]))

            # ZNE (simplified for speed)
            results["zne"].append(
                BaselineMethods.zero_noise_extrapolation(
                    circuit, observable, noise, shots=config["data"]["shots"] // 2
                )
            )

        # Convert to arrays and compute metrics
        for key in results:
            results[key] = np.array(results[key])

        metrics = {}
        ideal = results["ideal"]
        for method in ["noisy", "neural", "zne"]:
            values = results[method]
            metrics[method] = statistical_metrics(values, ideal)

        results_by_size[num_qubits] = {"values": results, "metrics": metrics}

    return results_by_size


def main():
    parser = argparse.ArgumentParser(description="Evaluate error mitigation")
    parser.add_argument(
        "--method",
        type=str,
        default="neural",
        choices=["neural", "all"],
        help="Mitigation method to evaluate",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="h2",
        choices=["h2", "lih", "maxcut", "all"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/vqe.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/evaluation",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path, map_location=device)

    # Recreate model from config
    model = create_mitigation_model(
        model_type="standard",
        circuit_dim=checkpoint["config"]["circuit_dim"],
        noise_dim=checkpoint["config"]["noise_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    figures = {}

    if args.benchmark in ["h2", "all"]:
        vqe_results = evaluate_vqe(config, model, device)

        # Generate figures
        methods = ["noisy", "neural", "zne"]
        errors = [vqe_results["metrics"][m]["mae"] for m in methods]
        errors_std = [vqe_results["metrics"][m]["std_error"] for m in methods]

        fig = plot_error_comparison(
            methods=["No Mitigation", "Neural", "ZNE"],
            errors=errors,
            errors_std=errors_std,
            title="VQE H2 Error Comparison",
        )
        figures["vqe_error_comparison"] = fig

        fig = plot_scatter_comparison(
            ideal_values=vqe_results["values"]["ideal"],
            noisy_values=vqe_results["values"]["noisy"],
            mitigated_values=vqe_results["values"]["neural"],
            title="VQE H2 Neural Mitigation",
        )
        figures["vqe_scatter"] = fig

        # Print results
        print("\n" + "=" * 50)
        print("VQE H2 Results")
        print("=" * 50)
        for method in methods:
            m = vqe_results["metrics"][method]
            print(f"{method.capitalize()}:")
            print(f"  MAE: {m['mae']:.6f}")
            print(f"  RMSE: {m['rmse']:.6f}")
            if method != "noisy":
                print(f"  Improvement: {m['improvement']:.2%}")

    if args.benchmark in ["maxcut", "all"]:
        # Load QAOA config
        qaoa_config_path = args.config.replace("vqe", "qaoa")
        if Path(qaoa_config_path).exists():
            with open(qaoa_config_path, "r") as f:
                qaoa_config = yaml.safe_load(f)
            qaoa_results = evaluate_qaoa(qaoa_config, model, device)

            # Generate scaling figure
            qubit_sizes = list(qaoa_results.keys())
            noise_errors = {
                "No Mitigation": [qaoa_results[n]["metrics"]["noisy"]["mae"] for n in qubit_sizes],
                "Neural": [qaoa_results[n]["metrics"]["neural"]["mae"] for n in qubit_sizes],
                "ZNE": [qaoa_results[n]["metrics"]["zne"]["mae"] for n in qubit_sizes],
            }

            fig = plot_noise_scaling(
                noise_levels=np.array(qubit_sizes),
                errors=noise_errors,
                title="QAOA MaxCut Error vs System Size",
            )
            figures["qaoa_scaling"] = fig

    # Save figures
    save_all_figures(figures, str(output_dir))
    print(f"\nFigures saved to {output_dir}")

    # Save numerical results
    np.savez(
        output_dir / "results.npz",
        **{f"{k}_values": v for k, v in vqe_results["values"].items()} if "vqe_results" in dir() else {},
    )


if __name__ == "__main__":
    main()
