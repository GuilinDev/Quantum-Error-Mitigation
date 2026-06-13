"""Visualization utilities for error mitigation experiments."""

from typing import Optional, List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# Set publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.8),  # IEEE single column width
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_energy_convergence(
    iterations: np.ndarray,
    energies: Dict[str, np.ndarray],
    exact_energy: Optional[float] = None,
    title: str = "VQE Energy Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot energy convergence during VQE optimization.

    Args:
        iterations: Array of iteration numbers.
        energies: Dictionary mapping method name to energy array.
        exact_energy: Exact ground state energy (optional).
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    colors = plt.cm.tab10.colors
    for i, (method, energy) in enumerate(energies.items()):
        ax.plot(iterations, energy, label=method, color=colors[i], linewidth=1.5)

    if exact_energy is not None:
        ax.axhline(
            exact_energy,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Exact",
        )
        # Add chemical accuracy band
        ax.axhspan(
            exact_energy - 0.0016,
            exact_energy + 0.0016,
            alpha=0.2,
            color="green",
            label="Chemical accuracy",
        )

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_error_comparison(
    methods: List[str],
    errors: List[float],
    errors_std: Optional[List[float]] = None,
    title: str = "Error Comparison",
    ylabel: str = "Mean Absolute Error",
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot bar chart comparing errors across methods.

    Args:
        methods: List of method names.
        errors: List of error values.
        errors_std: Standard deviations (optional).
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Path to save figure.
        colors: Custom colors for bars.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    x = np.arange(len(methods))

    if colors is None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    bars = ax.bar(x, errors, color=colors[: len(methods)], alpha=0.8)

    if errors_std is not None:
        ax.errorbar(
            x,
            errors,
            yerr=errors_std,
            fmt="none",
            color="black",
            capsize=3,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.annotate(
            f"{err:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_noise_scaling(
    noise_levels: np.ndarray,
    errors: Dict[str, np.ndarray],
    title: str = "Error vs Noise Level",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot error as a function of noise level.

    Args:
        noise_levels: Array of noise levels.
        errors: Dictionary mapping method name to error array.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v"]
    colors = plt.cm.tab10.colors

    for i, (method, error) in enumerate(errors.items()):
        ax.plot(
            noise_levels,
            error,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=method,
            linewidth=1.5,
            markersize=5,
        )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Absolute Error")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train", color="#1f77b4", linewidth=1.5)
    ax.plot(
        epochs, val_losses, label="Validation", color="#ff7f0e", linewidth=1.5
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_scatter_comparison(
    ideal_values: np.ndarray,
    noisy_values: np.ndarray,
    mitigated_values: np.ndarray,
    title: str = "Mitigation Performance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot comparing ideal, noisy, and mitigated values.

    Args:
        ideal_values: Ground truth values.
        noisy_values: Noisy estimates.
        mitigated_values: Mitigated estimates.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Ideal vs Noisy
    ax1 = axes[0]
    ax1.scatter(ideal_values, noisy_values, alpha=0.5, s=10, color="#ff7f0e")
    lims = [
        min(ideal_values.min(), noisy_values.min()),
        max(ideal_values.max(), noisy_values.max()),
    ]
    ax1.plot(lims, lims, "k--", linewidth=1, label="Ideal")
    ax1.set_xlabel("Ideal Value")
    ax1.set_ylabel("Noisy Value")
    ax1.set_title("Before Mitigation")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)

    # Ideal vs Mitigated
    ax2 = axes[1]
    ax2.scatter(
        ideal_values, mitigated_values, alpha=0.5, s=10, color="#2ca02c"
    )
    lims = [
        min(ideal_values.min(), mitigated_values.min()),
        max(ideal_values.max(), mitigated_values.max()),
    ]
    ax2.plot(lims, lims, "k--", linewidth=1, label="Ideal")
    ax2.set_xlabel("Ideal Value")
    ax2.set_ylabel("Mitigated Value")
    ax2.set_title("After Mitigation")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_histogram(
    errors: Dict[str, np.ndarray],
    bins: int = 30,
    title: str = "Error Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot histogram of error distributions.

    Args:
        errors: Dictionary mapping method name to error array.
        bins: Number of histogram bins.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    colors = plt.cm.tab10.colors
    alpha = 0.7

    for i, (method, error) in enumerate(errors.items()):
        ax.hist(
            error,
            bins=bins,
            alpha=alpha,
            color=colors[i],
            label=f"{method} (μ={np.mean(error):.4f})",
            density=True,
        )

    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def create_paper_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: str = "single",
) -> Tuple[plt.Figure, np.ndarray]:
    """Create figure with IEEE paper dimensions.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        width: 'single' (3.5in) or 'double' (7.16in) column.

    Returns:
        Figure and axes array.
    """
    if width == "single":
        fig_width = 3.5
    else:
        fig_width = 7.16

    # Maintain reasonable aspect ratio
    fig_height = fig_width * 0.8 * nrows / ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    return fig, axes


def save_all_figures(figures: Dict[str, plt.Figure], output_dir: str):
    """Save all figures in multiple formats.

    Args:
        figures: Dictionary mapping filename to figure.
        output_dir: Output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        # Save as PDF for paper
        fig.savefig(output_path / f"{name}.pdf", format="pdf")
        # Save as PNG for preview
        fig.savefig(output_path / f"{name}.png", format="png", dpi=300)
