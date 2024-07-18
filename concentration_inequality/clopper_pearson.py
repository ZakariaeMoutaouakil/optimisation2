from numpy import ndarray, sum, zeros, vstack, array
from statsmodels.stats.proportion import proportion_confint

from concentration_inequality.count_column_maxima import count_column_maxima


def clopper_pearson(matrix: ndarray, alpha: float, debug: bool = False) -> ndarray:
    """
    Computes the Clopper-Pearson confidence intervals for the column maxima counts.

    Args:
    matrix (np.ndarray): Input matrix (2D numpy array)
    alpha (float): Significance level for the confidence intervals

    Returns:
    np.ndarray: A matrix with lower bounds, means, and upper bounds of the confidence intervals for each column
    """
    counts = count_column_maxima(matrix)
    if debug:
        print("counts:", counts)
    n = sum(counts)
    m = len(counts)
    means = counts / n
    lower_bounds = zeros(m)
    upper_bounds = zeros(m)
    for i in range(m):
        lower_bounds[i], upper_bounds[i] = proportion_confint(counts[i], n, alpha=alpha / m, method="beta")
    return vstack((lower_bounds, means, upper_bounds))


def main():
    # Create an example matrix
    example_matrix = array([
        [5, 3, 1],
        [2, 8, 6],
        [7, 2, 4],
        [6, 3, 9]
    ])

    # Set the significance level
    alpha = 0.05

    # Compute the Clopper-Pearson confidence intervals
    result = clopper_pearson(example_matrix, alpha)

    # Print the result
    print("Lower bounds, means, and upper bounds for each column:")
    print(result)


# Example usage
if __name__ == "__main__":
    main()
