"""Direct prediction baseline for quantum error mitigation.

This module implements the "direct prediction" ablation baseline: an MLP
that predicts the IDEAL expectation value from circuit features and noise
features ONLY, without access to the noisy measured expectation value.

Comparing this baseline against :class:`~src.models.mitigation_net.MitigationNetwork`
isolates the contribution of the noisy measurement input — i.e., it answers
in which regime predicting the noise *correction* is easier than predicting
the expectation value directly.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .error_predictor import CircuitEncoder, NoiseEncoder


class DirectPredictionNet(nn.Module):
    """Predicts ideal expectation values without the noisy measurement.

    The architecture is identical to
    :class:`~src.models.mitigation_net.MitigationNetwork` (same circuit
    encoder, noise encoder, fusion MLP, and uncertainty head), EXCEPT that
    the noisy expectation value is excluded from the fusion vector. The
    network therefore predicts the ideal value purely from circuit and
    noise features.

    Trainer compatibility: :class:`~src.training.trainer.Trainer` calls
    ``model(noisy_values, circuit_features, noise_features)`` positionally.
    To work with the trainer unchanged, ``forward`` accepts ``noisy_values``
    as its first argument but IGNORES it — predictions never depend on it,
    and no residual connection from the noisy value is applied.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        hidden_dims: List[int] = [256, 512, 256],
        num_observables: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize direct prediction network.

        Args:
            circuit_dim: Dimension of circuit feature vector.
            noise_dim: Number of noise parameters.
            hidden_dims: Hidden layer dimensions.
            num_observables: Number of observables to predict.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_observables = num_observables

        # Encoders (identical to MitigationNetwork)
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

        # Fusion input EXCLUDES the noisy expectation values
        fusion_input_dim = 128 + 64

        # Prediction layers (identical structure to MitigationNetwork)
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
        self.prediction_layers = nn.Sequential(*layers)

        # Uncertainty estimation head (optional, mirrors MitigationNetwork)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Linear(hidden_dims[-1] // 2, num_observables),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(
        self,
        noisy_values: Optional[torch.Tensor],
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict ideal expectation values directly.

        Args:
            noisy_values: Noisy expectation values of shape (batch, num_obs).
                Accepted only for interface compatibility with the Trainer;
                this argument is IGNORED (may be None).
            circuit_features: Circuit features of shape (batch, circuit_dim).
            noise_params: Noise parameters of shape (batch, noise_dim).
            return_uncertainty: Whether to return uncertainty estimate.

        Returns:
            Predicted ideal values of shape (batch, num_obs), and optionally
            uncertainty of shape (batch, num_obs).
        """
        # noisy_values is intentionally unused: this baseline must predict
        # the ideal value from circuit and noise features alone.
        del noisy_values

        # Encode circuit and noise
        circuit_latent = self.circuit_encoder(circuit_features)
        noise_latent = self.noise_encoder(noise_params)

        # Fuse WITHOUT the noisy values
        combined = torch.cat([circuit_latent, noise_latent], dim=-1)

        # Predict absolute ideal value (no residual: nothing to correct)
        predicted = self.prediction_layers(combined)

        if return_uncertainty:
            uncertainty = self.uncertainty_head(self.prediction_layers[:-1](combined))
            return predicted, uncertainty

        return predicted
