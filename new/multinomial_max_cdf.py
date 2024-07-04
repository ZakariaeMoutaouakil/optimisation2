from itertools import product
from time import time
from typing import Tuple

from scipy.stats import multinomial


def multinomial_max_cdf(x: int, n: int, p: Tuple[float, ...]) -> float:
    k = len(p)  # Number of categories
    # Generate all possible outcomes where the maximum count is <= x
    valid_outcomes = []

    # Iterate over all possible combinations of counts that sum to n and max count <= x
    for counts in product(range(x + 1), repeat=k):
        if sum(counts) == n:
            valid_outcomes.append(counts)

    # Calculate the cumulative probability
    cdf = 0.0
    for outcome in valid_outcomes:
        cdf += multinomial.pmf(outcome, n, p)

    return cdf


if __name__ == "__main__":
    # Example usage
    x_ = 100
    n_ = 200
    p_ = (0.2, 0.3, 0.5)

    start_time = time()
    result = multinomial_max_cdf(x_, n_, p_)
    end_time = time()
    print("Multinomial Max CDF:", result)
    print("Time taken:", end_time - start_time)
