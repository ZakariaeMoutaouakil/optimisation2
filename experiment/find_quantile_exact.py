from time import time
from typing import Tuple

from numpy import random, array, percentile, max, ndarray

from new.multinomial_max_cdf import multinomial_max_cdf


def multinomial_max_simulation(p: Tuple[float, ...], n: int, num_simulations: int) -> ndarray:
    """
    Simulate the maximum values of a multinomial distribution.

    Parameters:
    - p: List of probabilities for the multinomial distribution.
    - n: Number of trials.
    - num_simulations: Number of Monte Carlo simulations to perform.

    Returns:
    - max_values: Array of maximum values from the simulations.
    """
    # Ensure the probabilities sum to 1
    p = array(p)
    p /= sum(p)

    # Run Monte Carlo simulations in a vectorized manner
    samples = random.multinomial(n, p, size=num_simulations)
    max_values = max(samples, axis=1)

    return max_values


def find_quantile_exact(p: Tuple[float, ...], n: int, alpha: float, num_simulations: int) -> int:
    """
    Find the quantile value x such that the CDF is greater than 1 - alpha using simulation.

    Parameters:
    - p: List of probabilities for the multinomial distribution.
    - n: Number of trials.
    - alpha: Significance level.
    - num_simulations: Number of Monte Carlo simulations to perform.

    Returns:
    - x: The quantile value such that the CDF is greater than 1 - alpha.
    """
    # Simulate the maximum values
    max_values = multinomial_max_simulation(p, n, num_simulations)

    # Calculate the quantile
    quantile_value = percentile(max_values, (1 - alpha) * 100)
    result = int(quantile_value) - 1

    while True:
        if multinomial_max_cdf(x=result, n=n, p=p) > 1 - alpha:
            return result
        else:
            result += 1


def main():
    # Example usage
    alpha = 0.01
    n = 200
    p = (0.2, 0.3, 0.5)
    num_simulations = 100000

    start_time = time()
    x_value = find_quantile_exact(p=p, n=n, alpha=alpha, num_simulations=num_simulations)
    end_time = time()

    probability = multinomial_max_cdf(x=x_value, n=n, p=p)
    print("Probability of CDF > 1 - alpha:", probability)
    print("Quantile value x such that CDF > 1 - alpha:", x_value)
    print("Time taken:", end_time - start_time)
    assert probability > 1 - alpha, "The CDF is not greater than 1 - alpha"


if __name__ == "__main__":
    while True:
        main()
