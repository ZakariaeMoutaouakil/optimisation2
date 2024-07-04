from typing import Dict

from mpmath import mp, fsum, factorial, power, exp
from numpy import arange


def truncated_poisson_pmf(lam: float, b: int, precision: int = 50) -> Dict[int, float]:
    """
    Calculate the PMF of a truncated Poisson distribution with arbitrary precision.

    Parameters:
    - lam (float): lambda parameter of the Poisson distribution
    - b (int): upper bound of the truncation
    - precision (int): number of decimal places for precision

    Returns:
    - Dict[int, float]: PMF values for the truncated range
    """
    # Set the precision
    mp.dps = precision

    # Range of values
    x_values = arange(0, b + 1)

    # Calculate the PMF of the Poisson distribution at each point
    pmf_values = tuple(exp(-lam) * power(lam, k) / factorial(k) for k in x_values)

    # Normalize to ensure the sum of PMF over the truncated range is 1
    total = fsum(pmf_values)
    pmf_values = tuple(p / total for p in pmf_values)

    return dict(zip(x_values, pmf_values))


def main():
    lam = 3.5
    b = 500
    pmf = truncated_poisson_pmf(lam, b, precision=50)
    print(f"Truncated Poisson PMF (lambda={lam}, b={b}):")
    for k, v in pmf.items():
        print(f"P(X={k}) = {v}")
    print(f"Sum of PMF: {fsum(pmf.values())}")


# Example usage
if __name__ == "__main__":
    main()
