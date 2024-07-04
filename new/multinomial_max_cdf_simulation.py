from time import time
from typing import Tuple

import numpy
from numpy import random, array, max, mean, sum


def multinomial_max_cdf_simulation(p: Tuple[float, ...], n: int, x: int, num_simulations: int) -> numpy.float64:
    """
    Calculate the CDF of the maximum value of a multinomial distribution using Monte Carlo simulation.

    Parameters:
    - p: List of probabilities for the multinomial distribution.
    - n: Number of trials.
    - x: Value at which to calculate the CDF.
    - num_simulations: Number of Monte Carlo simulations to perform (default is 100000).

    Returns:
    - cdf: Estimated CDF value at x.
    """
    # Ensure the probabilities sum to 1
    p = array(p)
    p /= sum(p)

    # Run Monte Carlo simulations
    max_values = []
    for _ in range(num_simulations):
        sample = random.multinomial(n, p)
        max_values.append(max(sample))

    # Calculate the CDF
    max_values = array(max_values)
    cdf = mean(max_values <= x)

    return cdf


def main():
    # Example usage
    x = 1000
    n = 2000
    p = (0.2, 0.3, 0.5)
    num_simulations = 100000

    start_time = time()
    result = multinomial_max_cdf_simulation(p=p, n=n, x=x, num_simulations=num_simulations)
    end_time = time()
    print("Multinomial Max CDF Simulation:", result)
    print("Time taken:", end_time - start_time)


if __name__ == "__main__":
    main()
