from typing import Tuple

from numpy import random, array, sum, ndarray, any


def dirichlet_sample(alpha: Tuple[float, ...], size: int) -> ndarray:
    """
    Generate samples from a Dirichlet distribution.

    Args:
    alpha (List[float]): Concentration parameters of the Dirichlet distribution.
    size (int): Number of samples to generate.

    Returns:
    ndarray: An array of shape (size, len(alpha)) containing Dirichlet samples.
    """
    # Convert alpha to numpy array if it's not already
    alpha = array(alpha)

    # Check if all alpha values are positive
    if any(alpha <= 0):
        raise ValueError("All alpha values must be positive")

    # Generate gamma samples
    gamma_samples = random.gamma(alpha, 1, size=(size, len(alpha)))

    # Normalize to get Dirichlet samples
    dirichlet_samples = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)

    return dirichlet_samples


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Example 1: Symmetric Dirichlet
    alpha_symmetric = (1., 1., 1.)
    samples_symmetric = dirichlet_sample(alpha_symmetric, size=5)
    print("Symmetric Dirichlet samples:")
    print(samples_symmetric)

    # Verify that each sample sums to 1
    print("\nSum of each sample:")
    print(sum(samples_symmetric, axis=1))

    # Example 2: Asymmetric Dirichlet
    alpha_asymmetric = (0.1, 1, 5)
    samples_asymmetric = dirichlet_sample(alpha_asymmetric, size=5)
    print("\nAsymmetric Dirichlet samples:")
    print(samples_asymmetric)

    # Verify that each sample sums to 1
    print("\nSum of each sample:")
    print(sum(samples_asymmetric, axis=1))
