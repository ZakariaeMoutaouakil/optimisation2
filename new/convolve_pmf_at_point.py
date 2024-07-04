from typing import Dict

import mpmath

from new.truncated_poisson_pmf import truncated_poisson_pmf


def convolve_pmf_at_point(pmf1: Dict[int, mpmath.mpf], pmf2: Dict[int, mpmath.mpf], target: int) -> mpmath.mpf:
    """
    Calculate the PMF of the sum of two independent variables at a specific point given their PMFs with high precision.

    Parameters:
    - pmf1 (Dict[int, mpmath.mpf]): PMF of the first variable
    - pmf2 (Dict[int, mpmath.mpf]): PMF of the second variable
    - target (int): The target point at which to calculate the convolved PMF

    Returns:
    - mpmath.mpf: PMF value at the target point
    """
    result_value = mpmath.mpf(0)

    for k1, v1 in pmf1.items():
        k2 = target - k1
        if k2 in pmf2:
            result_value += v1 * pmf2[k2]

    return result_value


def main():
    lam1, lam2 = 3.5, 2.0
    b1, b2 = 500, 500
    pmf1 = truncated_poisson_pmf(lam1, b1, precision=50)
    pmf2 = truncated_poisson_pmf(lam2, b2, precision=50)

    target_point = 7  # Example target point
    convolved_value = convolve_pmf_at_point(pmf1, pmf2, target_point)

    print(f"Truncated Poisson PMF (lambda={lam1}, b={b1}):")
    for k, v in pmf1.items():
        print(f"P(X1={k}) = {v}")
    print(f"Sum of PMF1: {mpmath.fsum(pmf1.values())}")

    print(f"\nTruncated Poisson PMF (lambda={lam2}, b={b2}):")
    for k, v in pmf2.items():
        print(f"P(X2={k}) = {v}")
    print(f"Sum of PMF2: {mpmath.fsum(pmf2.values())}")

    print(f"\nPMF of the sum of the two variables at point {target_point}:")
    print(f"P(X1 + X2={target_point}) = {convolved_value}")


# Example usage
if __name__ == "__main__":
    main()
