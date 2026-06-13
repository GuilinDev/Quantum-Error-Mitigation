"""Neural network models for quantum error mitigation.

This module implements networks that learn to correct noisy
expectation values from variational quantum algorithms.
"""

from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .error_predictor import CircuitEncoder, NoiseEncoder, CrossAttention


class MitigationNetwork(nn.Module):
    """Neural network for error mitigation.

    Takes noisy expectation values, circuit features, and noise parameters,
    and outputs corrected (mitigated) expectation values.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        hidden_dims: List[int] = [256, 512, 256],
        num_observables: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        """Initialize mitigation network.

        Args:
            circuit_dim: Dimension of circuit feature vector.
            noise_dim: Number of noise parameters.
            hidden_dims: Hidden layer dimensions.
            num_observables: Number of observables to mitigate.
            dropout: Dropout probability.
            residual: Whether to use residual connection (predict correction).
        """
        super().__init__()

        self.residual = residual
        self.num_observables = num_observables

        # Encoders
        self.circuit_encoder = CircuitEncoder(
            input_dim=circuit_dim,
            hidden_dims=[128, 256],
            latent_dim=128,
            dropout=dropout,
        )

        self.noise_encoder = NoiseEncoder(
            noise_dim=noise_dim,
            hidden_dim=64,
            latent_dim=64,
        )

        # Input includes noisy expectation values
        fusion_input_dim = 128 + 64 + num_observables

        # Mitigation layers
        layers = []
        prev_dim = fusion_input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_observables))
        self.mitigation_layers = nn.Sequential(*layers)

        # Uncertainty estimation head (optional)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Linear(hidden_dims[-1] // 2, num_observables),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(
        self,
        noisy_values: torch.Tensor,
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply error mitigation.

        Args:
            noisy_values: Noisy expectation values of shape (batch, num_obs).
            circuit_features: Circuit features of shape (batch, circuit_dim).
            noise_params: Noise parameters of shape (batch, noise_dim).
            return_uncertainty: Whether to return uncertainty estimate.

        Returns:
            Mitigated values of shape (batch, num_obs), and optionally
            uncertainty of shape (batch, num_obs).
        """
        # Encode circuit and noise
        circuit_latent = self.circuit_encoder(circuit_features)
        noise_latent = self.noise_encoder(noise_params)

        # Combine with noisy values
        combined = torch.cat([circuit_latent, noise_latent, noisy_values], dim=-1)

        # Predict correction or absolute value
        output = self.mitigation_layers(combined)

        if self.residual:
            mitigated = noisy_values + output
        else:
            mitigated = output

        if return_uncertainty:
            # Extract features from last hidden layer for uncertainty
            # (would need to modify architecture slightly for production)
            uncertainty = self.uncertainty_head(
                self.mitigation_layers[:-1](combined)
            )
            return mitigated, uncertainty

        return mitigated


class AdaptiveMitigationNetwork(nn.Module):
    """Adaptive mitigation network with noise-level conditioning.

    Adjusts mitigation strength based on estimated noise level,
    providing more aggressive correction for higher noise.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        hidden_dim: int = 256,
        num_observables: int = 1,
        num_noise_levels: int = 5,
    ):
        """Initialize adaptive mitigation network.

        Args:
            circuit_dim: Circuit feature dimension.
            noise_dim: Noise parameter dimension.
            hidden_dim: Hidden layer dimension.
            num_observables: Number of observables.
            num_noise_levels: Number of discrete noise levels for conditioning.
        """
        super().__init__()

        self.num_noise_levels = num_noise_levels

        # Noise level estimator
        self.noise_estimator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_noise_levels),
            nn.Softmax(dim=-1),
        )

        # Separate mitigation networks for each noise level
        self.mitigators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(circuit_dim + noise_dim + num_observables, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_observables),
                )
                for _ in range(num_noise_levels)
            ]
        )

    def forward(
        self,
        noisy_values: torch.Tensor,
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive error mitigation.

        Args:
            noisy_values: Noisy expectation values.
            circuit_features: Circuit features.
            noise_params: Noise parameters.

        Returns:
            Mitigated expectation values.
        """
        # Estimate noise level distribution
        noise_weights = self.noise_estimator(noise_params)

        # Combine inputs
        combined = torch.cat([circuit_features, noise_params, noisy_values], dim=-1)

        # Weighted combination of mitigator outputs
        outputs = torch.stack(
            [mitigator(combined) for mitigator in self.mitigators], dim=1
        )
        # outputs: (batch, num_levels, num_obs)
        # noise_weights: (batch, num_levels)

        mitigated = torch.einsum("bnk,bn->bk", outputs, noise_weights)
        return noisy_values + mitigated


class HybridMitigationModel(nn.Module):
    """Hybrid model combining neural mitigation with classical techniques.

    This model can optionally incorporate classical error mitigation
    results (e.g., ZNE) as additional features.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        num_zne_points: int = 3,
        hidden_dim: int = 256,
        num_observables: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize hybrid mitigation model.

        Args:
            circuit_dim: Circuit feature dimension.
            noise_dim: Noise parameter dimension.
            num_zne_points: Number of ZNE noise scaling points.
            hidden_dim: Hidden layer dimension.
            num_observables: Number of observables.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_zne_points = num_zne_points

        # Encoders
        self.circuit_encoder = CircuitEncoder(
            input_dim=circuit_dim,
            hidden_dims=[128, 128],
            latent_dim=64,
        )

        self.noise_encoder = NoiseEncoder(
            noise_dim=noise_dim,
            hidden_dim=32,
            latent_dim=32,
        )

        # ZNE feature encoder
        self.zne_encoder = nn.Sequential(
            nn.Linear(num_zne_points * num_observables, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # Fusion network
        fusion_dim = 64 + 32 + 32 + num_observables
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_observables),
        )

        # Learnable combination weight between neural and ZNE
        self.combination_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        noisy_values: torch.Tensor,
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
        zne_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply hybrid error mitigation.

        Args:
            noisy_values: Noisy expectation values.
            circuit_features: Circuit features.
            noise_params: Noise parameters.
            zne_values: ZNE results at different noise levels (optional).

        Returns:
            Mitigated expectation values.
        """
        # Encode inputs
        circuit_repr = self.circuit_encoder(circuit_features)
        noise_repr = self.noise_encoder(noise_params)

        if zne_values is not None:
            zne_repr = self.zne_encoder(zne_values.flatten(start_dim=1))
            combined = torch.cat(
                [circuit_repr, noise_repr, zne_repr, noisy_values], dim=-1
            )
        else:
            zne_repr = torch.zeros(noisy_values.shape[0], 32, device=noisy_values.device)
            combined = torch.cat(
                [circuit_repr, noise_repr, zne_repr, noisy_values], dim=-1
            )

        # Neural mitigation
        neural_correction = self.fusion(combined)
        mitigated = noisy_values + neural_correction

        return mitigated


class IterativeMitigationNetwork(nn.Module):
    """Iterative refinement network for error mitigation.

    Applies multiple refinement steps, each improving on
    the previous estimate.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        hidden_dim: int = 128,
        num_observables: int = 1,
        num_iterations: int = 3,
    ):
        """Initialize iterative mitigation network.

        Args:
            circuit_dim: Circuit feature dimension.
            noise_dim: Noise parameter dimension.
            hidden_dim: Hidden dimension.
            num_observables: Number of observables.
            num_iterations: Number of refinement iterations.
        """
        super().__init__()

        self.num_iterations = num_iterations

        # Shared encoders
        self.circuit_encoder = nn.Linear(circuit_dim, hidden_dim)
        self.noise_encoder = nn.Linear(noise_dim, hidden_dim // 2)

        # Refinement cell (shared across iterations)
        self.refinement_cell = nn.GRUCell(
            input_size=num_observables + hidden_dim + hidden_dim // 2,
            hidden_size=hidden_dim,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_observables)

    def forward(
        self,
        noisy_values: torch.Tensor,
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply iterative mitigation.

        Args:
            noisy_values: Noisy expectation values.
            circuit_features: Circuit features.
            noise_params: Noise parameters.
            return_intermediate: Whether to return all iterations.

        Returns:
            Final mitigated values, or list of all iterations.
        """
        batch_size = noisy_values.shape[0]

        # Encode context
        circuit_repr = F.gelu(self.circuit_encoder(circuit_features))
        noise_repr = F.gelu(self.noise_encoder(noise_params))

        # Initialize hidden state
        hidden = torch.zeros(batch_size, circuit_repr.shape[-1], device=noisy_values.device)

        # Current estimate
        current = noisy_values
        intermediates = [current]

        for _ in range(self.num_iterations):
            # Combine current estimate with context
            cell_input = torch.cat([current, circuit_repr, noise_repr], dim=-1)

            # Refinement step
            hidden = self.refinement_cell(cell_input, hidden)
            correction = self.output_proj(hidden)
            current = current + correction

            intermediates.append(current)

        if return_intermediate:
            return intermediates

        return current


def create_mitigation_model(
    model_type: str,
    circuit_dim: int,
    noise_dim: int = 8,
    **kwargs,
) -> nn.Module:
    """Factory function to create mitigation models.

    Args:
        model_type: Type of model ('standard', 'adaptive', 'hybrid', 'iterative').
        circuit_dim: Circuit feature dimension.
        noise_dim: Noise parameter dimension.
        **kwargs: Additional model-specific arguments.

    Returns:
        Initialized mitigation model.
    """
    models = {
        "standard": MitigationNetwork,
        "adaptive": AdaptiveMitigationNetwork,
        "hybrid": HybridMitigationModel,
        "iterative": IterativeMitigationNetwork,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](circuit_dim=circuit_dim, noise_dim=noise_dim, **kwargs)
