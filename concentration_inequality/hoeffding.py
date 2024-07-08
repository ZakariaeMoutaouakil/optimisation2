from math import log, sqrt

from numpy import ndarray, zeros, vstack, sum
from statsmodels.stats.proportion import proportion_confint

from concentration_inequality.blaise.calculate_term import calculate_term
from concentration_inequality.calculate_bounds import calculate_bounds
from concentration_inequality.column_means import column_means
from concentration_inequality.count_column_maxima import count_column_maxima
from concentration_inequality.generate_probability_matrix_biased import generate_probability_matrix_biased


def hoeffding(matrix: ndarray, alpha: float, debug: bool = False) -> ndarray:
    num_columns = len(matrix[0])
    upper_bounds = zeros(num_columns)
    lower_bounds = zeros(num_columns)
    means = column_means(matrix)
    multinomial = count_column_maxima(matrix)
    if debug:
        print("multinomial:", multinomial)
    n = sum(multinomial)

    for i in range(num_columns):
        vector = matrix[:, i]
        if debug:
            print("vector:", vector)
        mean = means[i]
        if debug:
            print("mean:", mean)
        x = multinomial[i]
        if debug:
            print("x:", x)
        a, b = proportion_confint(x, n, alpha=alpha / 2, method="beta")
        if debug:
            print("a:", a)
            print("b:", b)
        shift = (b - a) * sqrt(- log(alpha / 2) / (2 * n))
        if debug:
            print("shift:", shift)
        upper_bounds[i] = mean + shift
        if debug:
            print("upper bound:", upper_bounds[i])
        lower_bounds[i] = mean - shift
        if debug:
            print("lower bound:", lower_bounds[i])

    return vstack((lower_bounds, means, upper_bounds))


def main():
    # Create a sample matrix
    # Generate a 3x4 probability matrix
    num_rows = 3000
    vector_length = 5
    high_prob_column = 0  # Index of the column to have higher probabilities

    sample_matrix = generate_probability_matrix_biased(num_rows, vector_length, high_prob_column)
    alpha = 0.05

    print("Sample matrix:")
    print(sample_matrix)

    # Calculate bounds
    my_bounds = hoeffding(sample_matrix, alpha=alpha, debug=False)

    # Calculate bounds
    bounds = calculate_bounds(sample_matrix, calculate_term, alpha=alpha, debug=False)

    print("\nMy Bounds:")
    print(my_bounds)

    print("\nBlaise Bounds:")
    print(bounds)


if __name__ == "__main__":
    main()
