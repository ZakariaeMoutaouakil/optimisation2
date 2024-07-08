from numpy import ndarray, var, array


def sample_variance(vector: ndarray) -> float:
    """
    Calculate the sample variance of a numpy vector using the pairwise difference formula.

    Args:
    vector (np.ndarray): Input vector

    Returns:
    float: Sample variance
    """
    n = len(vector)
    if n < 2:
        raise ValueError("Sample variance requires at least two data points.")

    # Calculate the sum of squared differences
    sum_squared_diff = 0.
    for i in range(n):
        for j in range(i + 1, n):
            sum_squared_diff += (vector[i] - vector[j]) ** 2

    # Calculate the variance
    variance = sum_squared_diff / (n * (n - 1))

    return variance


def main():
    # Example usage
    vector = array([1, 2, 3, 4, 5])

    # Calculate the sample variance
    variance = sample_variance(vector)

    print(f"Sample vector: {vector}")
    print(f"Sample variance: {variance}")

    # Compare with numpy's var function (which uses a different formula)
    np_var = var(vector, ddof=1)
    print(f"NumPy's sample variance: {np_var}")


# Example usage
if __name__ == "__main__":
    main()
