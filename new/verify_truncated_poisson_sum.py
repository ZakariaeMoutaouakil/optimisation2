from new.convolve_pmfs import convolve_pmfs
from new.truncated_poisson_pmf import truncated_poisson_pmf


def verify_truncated_poisson_sum(lam1: float, lam2: float, b1: int, b2: int):
    """
    Verify that the sum of two truncated Poisson variables is a truncated Poisson variable
    with parameters being the sum of the original parameters.

    Parameters:
    - lam1 (float): lambda parameter of the first Poisson distribution
    - lam2 (float): lambda parameter of the second Poisson distribution
    - b (int): upper bound of the truncation
    """
    pmf1 = truncated_poisson_pmf(lam1, b1)
    pmf2 = truncated_poisson_pmf(lam2, b2)
    sum_pmf = convolve_pmfs(pmf1, pmf2)

    expected_pmf = truncated_poisson_pmf(lam1 + lam2, b1 + b2)

    print(f"PMF of the sum of two truncated Poisson variables (lambda1={lam1}, lambda2={lam2}, b1={b1}, b2={b2}):")

    for i in range(b1 + b2 + 1):
        print(f"P(X1+X2={i}) = {sum_pmf.get(i, 0):.4f}")
        print(f"P(X={i})     = {expected_pmf.get(i, 0):.4f}")
        print(f"Difference = {sum_pmf.get(i, 0) - expected_pmf.get(i, 0):.4f}")

    print(f"Sum of PMF of the sum: {sum(sum_pmf.values()):.4f}")
    print(f"Sum of expected PMF: {sum(expected_pmf.values()):.4f}")


def main():
    lam1, lam2 = 3.5, 2.0
    b1, b2 = 5, 5
    verify_truncated_poisson_sum(lam1, lam2, b1, b2)


# Example usage
if __name__ == "__main__":
    main()
