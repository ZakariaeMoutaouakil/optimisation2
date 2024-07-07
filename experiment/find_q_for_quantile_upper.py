from time import time

from statsmodels.stats.proportion import proportion_confint

from experiment.generate_tuple import generate_tuple
from experiment.multinomial_quantile import quantile_of_multinomial_max


def find_q_for_quantile_upper(n: int, m: int, alpha: float, x: int, num_simulations: int, tolerance: float = 1e-5) \
        -> float:
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
        quantile_value = quantile_of_multinomial_max(p=p, n=n, alpha=alpha, num_simulations=num_simulations)

        if quantile_value > x:
            upper_bound = mid_point
        else:
            lower_bound = mid_point

    return upper_bound


def main():
    # Example usage
    n = 600
    m = 3
    alpha = 0.001
    x = 205  # Target quantile value
    num_simulations = 10000000

    start_time = time()
    q_value = find_q_for_quantile_upper(n=n, m=m, alpha=alpha, x=x, num_simulations=num_simulations)
    end_time = time()

    cp = proportion_confint(x, n, alpha=2 * alpha, method="beta")[1]
    print("statsmodels p1:", cp)
    print("Value of q such that the quantile is strictly less than x:", q_value)
    print("Time taken:", end_time - start_time)


if __name__ == "__main__":
    while True:
        main()
