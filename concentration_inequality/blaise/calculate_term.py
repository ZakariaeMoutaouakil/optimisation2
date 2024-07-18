from math import log, sqrt
from random import random
from time import sleep

from numpy import ndarray, var

from concentration_inequality.generate_probability_matrix import generate_probability_matrix


def calculate_term(vector: ndarray, alpha: float) -> float:
    """
    Calculate the term given by the formula in the image.

    Args:
    Z (np.ndarray): Input vector
    alpha (float): Alpha value (should be between 0 and 1)

    Returns:
    float: Calculated term
    """
    n = len(vector)
    if n < 2:
        raise ValueError("Calculation requires at least two data points.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha should be strictly between 0 and 1.")

    # variance = sample_variance(vector)
    variance = var(vector, ddof=1)
    log_term = log(2 / alpha)

    term1 = (2 * variance * log_term) / n
    term2 = (7 * log_term) / (3 * (n - 1))

    result = sqrt(term1) + term2
    return result


def main():
    # Example usage
    # Generate a random vector
    vector_length = int(random() * 10 + 2)

    # Create a sample vector
    vector = generate_probability_matrix(num_rows=1, vector_length=vector_length)[0]

    alpha = 0.05  # Common significance level

    # Calculate the term
    result = calculate_term(vector, alpha)

    print(f"Sample vector: {vector}")
    print(f"Alpha: {alpha}")
    print(f"Calculated term: {result}")


# Example usage
if __name__ == "__main__":
    while True:
        main()
        sleep(5)
