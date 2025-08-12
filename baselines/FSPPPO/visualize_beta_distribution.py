#!/usr/bin/env python3
"""
Visualize Beta distribution for recency bias in opponent sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import jax.numpy as jnp
from jax.scipy.stats import beta as jax_beta

def map_alpha_to_beta_params(alpha):
    """
    Map recency bias alpha [0, 1] to Beta distribution parameters.

    Args:
        alpha: Recency bias parameter
            - alpha = 0.0: Only oldest checkpoint (left-skewed)
            - alpha = 0.5: Uniform distribution
            - alpha = 1.0: Only newest checkpoint (right-skewed)

    Returns:
        (a, b): Beta distribution parameters
    """
    if alpha == 0.5:
        # Uniform: Beta(1, 1)
        return 1.0, 1.0
    elif alpha < 0.5:
        # Bias toward older (left side): Beta(a<1, b>1)
        a = 2 * alpha + 0.1  # Avoid a=0, range [0.1, 1.1]
        b = 2.0
        return a, b
    else:
        # Bias toward newer (right side): Beta(a>1, b<1)
        a = 2.0
        b = 2 * (1 - alpha) + 0.1  # Avoid b=0, range [0.1, 1.1]
        return a, b

def calculate_recency_weights_beta(num_checkpoints, alpha):
    """Calculate sampling weights using Beta distribution."""
    if num_checkpoints == 1:
        return np.array([1.0])

    a, b = map_alpha_to_beta_params(alpha)

    # Generate positions in [0, 1] (avoiding exact 0 and 1)
    positions = np.linspace(0.01, 0.99, num_checkpoints)

    # Calculate Beta PDF weights
    weights = beta.pdf(positions, a, b)
    weights = weights / weights.sum()  # Normalize

    return weights, positions, (a, b)

def plot_beta_distributions():
    """Plot Beta distributions for different alpha values."""

    # Test different alpha values
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    num_checkpoints = 10

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Continuous x for smooth curves
    x_continuous = np.linspace(0.01, 0.99, 1000)

    for i, alpha in enumerate(alpha_values):
        ax = axes[i]

        # Get Beta parameters
        a, b = map_alpha_to_beta_params(alpha)

        # Plot continuous Beta PDF
        y_continuous = beta.pdf(x_continuous, a, b)
        ax.plot(x_continuous, y_continuous, 'b-', linewidth=2, alpha=0.7,
                label=f'Beta({a:.1f}, {b:.1f})')

        # Calculate discrete weights for checkpoints
        weights, positions, _ = calculate_recency_weights_beta(num_checkpoints, alpha)

        # Plot discrete sampling weights
        ax.stem(positions, weights, linefmt='r-', markerfmt='ro', basefmt=' ',
                label='Sampling Weights')

        # Formatting
        ax.set_title(f'α = {alpha:.1f}\n({"Oldest" if alpha < 0.5 else "Newest" if alpha > 0.5 else "Uniform"} bias)')
        ax.set_xlabel('Checkpoint Age (0=Oldest, 1=Newest)')
        ax.set_ylabel('Probability Density / Weight')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    # Remove empty subplot
    if len(alpha_values) < len(axes):
        axes[-1].remove()

    plt.tight_layout()
    plt.suptitle('Beta Distribution for Recency-Biased Opponent Sampling',
                 fontsize=16, y=1.02)

    # Save plot
    plt.savefig('/share/code/src/JaxMARL/baselines/FSPPPO/beta_distribution_visualization.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_checkpoint_sampling():
    """Demonstrate how checkpoints would be sampled."""

    print("=== Checkpoint Sampling Demonstration ===\n")

    # Simulate 10 checkpoints
    checkpoint_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_checkpoints = len(checkpoint_steps)

    print(f"Available checkpoints: {checkpoint_steps}")
    print("(Index 0 = oldest, Index {0} = newest)\n".format(num_checkpoints-1))

    # Test different alpha values
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for alpha in alpha_values:
        weights, positions, (a, b) = calculate_recency_weights_beta(num_checkpoints, alpha)

        print(f"α = {alpha:.2f} → Beta({a:.1f}, {b:.1f})")
        print("Checkpoint sampling probabilities:")

        for i, (step, weight) in enumerate(zip(checkpoint_steps, weights)):
            bar = "█" * int(weight * 50)  # Visual bar
            print(f"  Step {step:4d} (idx {i}): {weight:.3f} {bar}")

        # Show most likely checkpoint
        max_idx = np.argmax(weights)
        print(f"  → Most likely: Step {checkpoint_steps[max_idx]} (probability: {weights[max_idx]:.3f})")
        print()

if __name__ == "__main__":
    print("Visualizing Beta Distribution for Opponent Sampling...")

    # Create visualization
    plot_beta_distributions()

    # Demonstrate sampling behavior
    demonstrate_checkpoint_sampling()

    print("\nVisualization saved as: beta_distribution_visualization.png")
