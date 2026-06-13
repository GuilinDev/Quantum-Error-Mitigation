"""Neural network models for quantum circuit error prediction.

This module implements neural networks that learn to predict
noise-induced errors from circuit structure and noise parameters.
"""

from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircuitEncoder(nn.Module):
    """Encoder network for quantum circuit features.

    Transforms circuit structure information into a latent representation
    suitable for error prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 256, 128],
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        """Initialize circuit encoder.

        Args:
            input_dim: Dimension of input circuit features.
            hidden_dims: List of hidden layer dimensions.
            latent_dim: Dimension of output latent representation.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

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

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode circuit features.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Latent representation of shape (batch, latent_dim).
        """
        return self.encoder(x)


class NoiseEncoder(nn.Module):
    """Encoder for noise model parameters.

    Transforms noise parameters into a latent representation
    that captures noise characteristics.
    """

    def __init__(
        self,
        noise_dim: int = 8,
        hidden_dim: int = 64,
        latent_dim: int = 32,
    ):
        """Initialize noise encoder.

        Args:
            noise_dim: Number of noise parameters.
            hidden_dim: Hidden layer dimension.
            latent_dim: Output latent dimension.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, noise_params: torch.Tensor) -> torch.Tensor:
        """Encode noise parameters.

        Args:
            noise_params: Tensor of shape (batch, noise_dim).

        Returns:
            Latent noise representation of shape (batch, latent_dim).
        """
        return self.encoder(noise_params)


class ErrorPredictor(nn.Module):
    """Neural network for predicting quantum circuit execution errors.

    This network takes circuit features and noise parameters as input,
    and predicts the error in the output expectation value.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        hidden_dims: List[int] = [256, 512, 256],
        output_dim: int = 1,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        """Initialize error predictor.

        Args:
            circuit_dim: Dimension of circuit feature vector.
            noise_dim: Number of noise parameters.
            hidden_dims: Hidden layer dimensions for fusion network.
            output_dim: Output dimension (1 for scalar error prediction).
            dropout: Dropout probability.
            use_attention: Whether to use attention mechanism.
        """
        super().__init__()

        self.use_attention = use_attention

        # Circuit encoder
        self.circuit_encoder = CircuitEncoder(
            input_dim=circuit_dim,
            hidden_dims=[128, 256],
            latent_dim=128,
            dropout=dropout,
        )

        # Noise encoder
        self.noise_encoder = NoiseEncoder(
            noise_dim=noise_dim,
            hidden_dim=64,
            latent_dim=64,
        )

        # Feature fusion
        fusion_input_dim = 128 + 64  # circuit latent + noise latent

        if use_attention:
            self.attention = CrossAttention(query_dim=128, key_dim=64, num_heads=4)
            fusion_input_dim = 128 + 64 + 128  # Add attention output

        # Prediction head
        fusion_layers = []
        prev_dim = fusion_input_dim

        for hidden_dim in hidden_dims:
            fusion_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        fusion_layers.append(nn.Linear(prev_dim, output_dim))
        self.fusion_network = nn.Sequential(*fusion_layers)

    def forward(
        self,
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
    ) -> torch.Tensor:
        """Predict error for given circuit and noise parameters.

        Args:
            circuit_features: Circuit features of shape (batch, circuit_dim).
            noise_params: Noise parameters of shape (batch, noise_dim).

        Returns:
            Predicted error of shape (batch, output_dim).
        """
        # Encode inputs
        circuit_latent = self.circuit_encoder(circuit_features)
        noise_latent = self.noise_encoder(noise_params)

        # Fuse features
        if self.use_attention:
            attention_out = self.attention(
                circuit_latent.unsqueeze(1), noise_latent.unsqueeze(1)
            ).squeeze(1)
            fused = torch.cat([circuit_latent, noise_latent, attention_out], dim=-1)
        else:
            fused = torch.cat([circuit_latent, noise_latent], dim=-1)

        # Predict error
        return self.fusion_network(fused)


class CrossAttention(nn.Module):
    """Cross-attention module for feature interaction."""

    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 4):
        """Initialize cross-attention.

        Args:
            query_dim: Dimension of query features.
            key_dim: Dimension of key/value features.
            num_heads: Number of attention heads.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention.

        Args:
            query: Query tensor of shape (batch, seq_q, query_dim).
            key_value: Key/value tensor of shape (batch, seq_kv, key_dim).

        Returns:
            Attention output of shape (batch, seq_q, query_dim).
        """
        batch_size, seq_len, _ = query.shape

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class EnsembleErrorPredictor(nn.Module):
    """Ensemble of error predictors for uncertainty estimation.

    Uses multiple error predictors and aggregates their predictions
    to provide both mean prediction and uncertainty estimate.
    """

    def __init__(
        self,
        circuit_dim: int,
        noise_dim: int = 8,
        n_models: int = 5,
        **kwargs,
    ):
        """Initialize ensemble predictor.

        Args:
            circuit_dim: Dimension of circuit features.
            noise_dim: Number of noise parameters.
            n_models: Number of models in ensemble.
            **kwargs: Additional arguments passed to ErrorPredictor.
        """
        super().__init__()

        self.n_models = n_models
        self.models = nn.ModuleList(
            [
                ErrorPredictor(circuit_dim, noise_dim, **kwargs)
                for _ in range(n_models)
            ]
        )

    def forward(
        self,
        circuit_features: torch.Tensor,
        noise_params: torch.Tensor,
        return_uncertainty: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through ensemble.

        Args:
            circuit_features: Circuit features.
            noise_params: Noise parameters.
            return_uncertainty: Whether to return uncertainty estimate.

        Returns:
            Tuple of (mean_prediction, std_prediction) if return_uncertainty,
            otherwise just mean_prediction.
        """
        predictions = torch.stack(
            [model(circuit_features, noise_params) for model in self.models], dim=0
        )

        mean_pred = predictions.mean(dim=0)

        if return_uncertainty:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        return mean_pred, None


class SequenceErrorPredictor(nn.Module):
    """Error predictor using sequence model for circuit gate sequence.

    This model treats the circuit as a sequence of gates and uses
    a transformer encoder to capture gate interactions.
    """

    def __init__(
        self,
        gate_vocab_size: int = 20,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        noise_dim: int = 8,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        """Initialize sequence error predictor.

        Args:
            gate_vocab_size: Number of unique gate types.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            noise_dim: Noise parameter dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()

        self.gate_embedding = nn.Embedding(gate_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.noise_encoder = NoiseEncoder(noise_dim, embed_dim, embed_dim // 2)

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        gate_sequence: torch.Tensor,
        noise_params: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict error from gate sequence.

        Args:
            gate_sequence: Gate indices of shape (batch, seq_len).
            noise_params: Noise parameters of shape (batch, noise_dim).
            mask: Padding mask of shape (batch, seq_len).

        Returns:
            Predicted error of shape (batch, 1).
        """
        batch_size, seq_len = gate_sequence.shape

        # Embed gates and positions
        positions = torch.arange(seq_len, device=gate_sequence.device)
        gate_embeds = self.gate_embedding(gate_sequence)
        pos_embeds = self.position_embedding(positions).unsqueeze(0)
        x = gate_embeds + pos_embeds

        # Apply transformer
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)

        # Pool sequence
        circuit_repr = x.mean(dim=1)

        # Encode noise and predict
        noise_repr = self.noise_encoder(noise_params)
        combined = torch.cat([circuit_repr, noise_repr], dim=-1)
        return self.prediction_head(combined)
