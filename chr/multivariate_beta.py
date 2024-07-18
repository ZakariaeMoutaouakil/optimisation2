from typing import Tuple

from mpmath import mp, gamma


def multivariate_beta(alpha: Tuple[float, ...], precision: int = 50) -> mp.mpf:
    # Set the precision
    mp.dps = precision

    numerator = mp.mpf(1.)
    denominator = mp.mpf(0.)
    for a in alpha:
        numerator *= gamma(a)
        denominator += a
    return numerator / gamma(denominator)


def main():
    # Calculate multivariate beta function with default precision
    result = multivariate_beta((1, 2, 3))

    print(result)


if __name__ == "__main__":
    main()
