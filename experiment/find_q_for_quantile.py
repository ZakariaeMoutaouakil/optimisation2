from time import time

from statsmodels.stats.proportion import proportion_confint

from experiment.find_quantile import find_quantile
from multinomial.generate_tuple import generate_tuple


def find_q_for_quantile(n: int, m: int, alpha: float, x: int, num_simulations: int, tolerance: float = 1e-5) -> float:
    """
    Find the value of q such that the quantile is strictly less than x.

    Parameters:
    - n: Number of trials.
    - m: Number of elements in the probability vector.
    - alpha: Significance level.
    - x: Target quantile value.
    - num_simulations: Number of Monte Carlo simulations to perform.
    - tolerance: The tolerance level for the bisection method (default is 1e-5).

    Returns:
    - q: The value of q such that the quantile is strictly less than x.
    """
    # Initialize the search range for q
    lower_bound = 1 / m
    upper_bound = 1

    while upper_bound - lower_bound > tolerance:
        # print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        mid_point = (lower_bound + upper_bound) / 2
        p = generate_tuple(m=m, q=mid_point)
        quantile_value = find_quantile(p=p, n=n, alpha=alpha, num_simulations=num_simulations)

        if quantile_value < x:
            lower_bound = mid_point
        else:
            upper_bound = mid_point

    return lower_bound


def main():
    # Example usage
    n = 1000
    m = 10
    alpha = 0.001
    x = 140  # Target quantile value
    num_simulations = 2000000

    start_time = time()
    q_value = find_q_for_quantile(n=n, m=m, alpha=alpha, x=x, num_simulations=num_simulations)
    end_time = time()

    cp = proportion_confint(x, n, alpha=2 * alpha, method="beta")[0]
    print("statsmodels p1:", cp)
    print("My p1:", q_value)
    print("Time taken:", end_time - start_time)


if __name__ == "__main__":
    while True:
        main()
