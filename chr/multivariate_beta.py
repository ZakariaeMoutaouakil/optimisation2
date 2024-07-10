from typing import Tuple

from numpy import prod, sum
from scipy.special import gamma


def multivariate_beta(alpha: Tuple[float, ...]) -> float:
    numerator = prod([gamma(a) for a in alpha])
    denominator = gamma(sum(alpha))
    return numerator / denominator


def main():
    # Example usage
    alpha: Tuple[float, ...] = (1.0, 2.0, 3.0)
    result = multivariate_beta(alpha)
    print(f"Multivariate Beta{alpha} = {result}")


if __name__ == "__main__":
    main()
