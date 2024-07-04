from collections import defaultdict
from typing import Dict

from new.truncated_poisson_pmf import truncated_poisson_pmf


def convolve_pmfs(pmf1: Dict[int, float], pmf2: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate the PMF of the sum of two independent variables given their PMFs.

    Parameters:
    - pmf1 (Dict[int, float]): PMF of the first variable
    - pmf2 (Dict[int, float]): PMF of the second variable

    Returns:
    - Dict[int, float]: PMF of the sum of the two variables
    """
    result_pmf = defaultdict(float)

    for k1, v1 in pmf1.items():
        for k2, v2 in pmf2.items():
            result_pmf[k1 + k2] += v1 * v2

    return dict(result_pmf)


def main():
    lam1, lam2 = 3.5, 2.0
    b1, b2 = 5, 5
    pmf1 = truncated_poisson_pmf(lam1, b1)
    pmf2 = truncated_poisson_pmf(lam2, b2)

    sum_pmf = convolve_pmfs(pmf1, pmf2)

    print(f"Truncated Poisson PMF (lambda={lam1}, b={b1}):")
    for k, v in pmf1.items():
        print(f"P(X1={k}) = {v:.4f}")
    print(f"Sum of PMF1: {sum(pmf1.values()):.4f}")

    print(f"\nTruncated Poisson PMF (lambda={lam2}, b={b2}):")
    for k, v in pmf2.items():
        print(f"P(X2={k}) = {v:.4f}")
    print(f"Sum of PMF2: {sum(pmf2.values()):.4f}")

    print(f"\nPMF of the sum of the two variables:")
    for k, v in sum_pmf.items():
        print(f"P(X1 + X2={k}) = {v:.4f}")
    print(f"Sum of PMF of the sum: {sum(sum_pmf.values()):.4f}")


# Example usage
if __name__ == "__main__":
    main()
