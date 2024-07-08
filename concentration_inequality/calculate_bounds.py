from typing import Callable

from numpy import ndarray, zeros, vstack, array

from concentration_inequality.blaise.calculate_term import calculate_term
from concentration_inequality.column_means import column_means


def calculate_bounds(matrix: ndarray, shift: Callable[[ndarray, float], float], alpha: float, debug: bool = False) \
        -> ndarray:
    num_columns = len(matrix[0])
    upper_bounds = zeros(num_columns)
    lower_bounds = zeros(num_columns)
    means = column_means(matrix)
    for i in range(num_columns):
        vector = matrix[:, i]
        if debug:
            print("vector:", vector)
        mean = means[i]
        if debug:
            print("mean:", mean)
        upper_bounds[i] = mean + shift(vector, alpha)
        if debug:
            print("upper bound:", upper_bounds[i])
        lower_bounds[i] = mean - shift(vector, alpha)
        if debug:
            print("lower bound:", lower_bounds[i])

    return vstack((lower_bounds, means, upper_bounds))


def main():
    # Create a sample matrix
    sample_matrix = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    shift = calculate_term
    alpha = 0.05

    # Calculate bounds
    bounds = calculate_bounds(sample_matrix, shift, alpha=alpha, debug=True)

    print("Sample matrix:")
    print(sample_matrix)

    print("\nBounds:")
    print(bounds)


if __name__ == "__main__":
    main()
