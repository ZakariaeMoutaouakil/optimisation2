from time import time
from typing import Tuple

import numpy as np

from new.multinomial_max_cdf import multinomial_max_cdf


def quantile_of_multinomial_max(p: Tuple[float, ...], n: int, alpha: float, num_simulations: int) -> int:
    """
    Computes the quantile of the maximum of the multinomial distribution of order alpha using simulation.

    Args:
    - p (Tuple[float, ...]): Probabilities for the multinomial distribution.
    - n (int): Number of trials.
    - alpha (float): Quantile level.
    - num_simulations (int): Number of simulations.

    Returns:
    - float: The quantile of the maximum of the multinomial distribution.
    """
    # Generate the simulations
    simulations = np.random.multinomial(n, p, size=num_simulations)

    # Find the maximum value in each simulation
    max_values = np.max(simulations, axis=1)

    # Determine the conservative quantile
    quantile = np.quantile(max_values, alpha, method='lower')

    return int(quantile) - 1


def main():
    # Example usage
    p = (0.2, 0.3, 0.5)
    n = 500
    alpha = 0.01
    num_simulations = 1000000

    start_time = time()
    result = quantile_of_multinomial_max(p, n, alpha, num_simulations)
    end_time = time()
    print("Quantile:", result)
    probability = multinomial_max_cdf(x=result, n=n, p=p)
    print("Probability of CDF <= alpha:", probability)
    print("Time taken:", end_time - start_time)
    assert probability <= alpha, "The CDF is not less than alpha"


if __name__ == "__main__":
    while True:
        main()
