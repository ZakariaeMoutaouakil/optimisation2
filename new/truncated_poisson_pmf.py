from typing import Dict

from numpy import arange
from scipy.stats import poisson


def truncated_poisson_pmf(lam: float, b: int) -> Dict[int, float]:
    """
    Calculate the PMF of a truncated Poisson distribution.

    Parameters:
    - lam (float): lambda parameter of the Poisson distribution
    - b (int): upper bound of the truncation

    Returns:
    - Dict[int, float]: PMF values for the truncated range
    """
    # Range of values
    x_values = arange(0, b + 1)

    # Calculate the PMF of the Poisson distribution at each point
    pmf_values = poisson.pmf(x_values, lam)

    # Normalize to ensure the sum of PMF over the truncated range is 1
    pmf_values /= pmf_values.sum()

    return dict(zip(x_values, pmf_values))


def main():
    lam = 3.5
    b = 5
    pmf = truncated_poisson_pmf(lam, b)
    print(f"Truncated Poisson PMF (lambda={lam}, b={b}):")
    for k, v in pmf.items():
        print(f"P(X={k}) = {v:.4f}")
    print(f"Sum of PMF: {sum(pmf.values()):.4f}")


# Example usage
if __name__ == "__main__":
    main()
