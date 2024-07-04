from collections import defaultdict
from typing import Dict

import mpmath

from new.truncated_poisson_pmf import truncated_poisson_pmf


def convolve_pmfs(pmf1: Dict[int, mpmath.mpf], pmf2: Dict[int, mpmath.mpf]) -> Dict[int, mpmath.mpf]:
    """
    Calculate the PMF of the sum of two independent variables given their PMFs with high precision.

    Parameters:
    - pmf1 (Dict[int, mpmath.mpf]): PMF of the first variable
    - pmf2 (Dict[int, mpmath.mpf]): PMF of the second variable

    Returns:
    - Dict[int, mpmath.mpf]: PMF of the sum of the two variables
    """
    result_pmf = defaultdict(mpmath.mpf)

    for k1, v1 in pmf1.items():
        for k2, v2 in pmf2.items():
            result_pmf[k1 + k2] += v1 * v2

    return dict(result_pmf)


def main():
    lam1, lam2 = 3.5, 2.0
    b1, b2 = 500, 500
    pmf1 = truncated_poisson_pmf(lam1, b1, precision=50)
    pmf2 = truncated_poisson_pmf(lam2, b2, precision=50)

    sum_pmf = convolve_pmfs(pmf1, pmf2)

    print(f"Truncated Poisson PMF (lambda={lam1}, b={b1}):")
    for k, v in pmf1.items():
        print(f"P(X1={k}) = {v}")
    print(f"Sum of PMF1: {mpmath.fsum(pmf1.values())}")

    print(f"\nTruncated Poisson PMF (lambda={lam2}, b={b2}):")
    for k, v in pmf2.items():
        print(f"P(X2={k}) = {v}")
    print(f"Sum of PMF2: {mpmath.fsum(pmf2.values())}")

    print(f"\nPMF of the sum of the two variables:")
    for k, v in sum_pmf.items():
        print(f"P(X1 + X2={k}) = {v}")
    print(f"Sum of PMF of the sum: {mpmath.fsum(sum_pmf.values())}")


# Example usage
if __name__ == "__main__":
    main()
