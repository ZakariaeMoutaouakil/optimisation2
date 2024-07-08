from numpy import array, mean, ndarray


def column_means(matrix: ndarray) -> ndarray:
    """
    Calculate the mean of each column in the given matrix.

    Args:
    matrix (np.ndarray): Input matrix.

    Returns:
    np.ndarray: A vector containing the mean of each column.
    """
    return mean(matrix, axis=0)


def main():
    # Create a sample matrix
    sample_matrix = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    print("Sample matrix:")
    print(sample_matrix)

    # Calculate column means
    means = column_means(sample_matrix)

    print("\nColumn means:")
    print(means)


# Example usage
if __name__ == "__main__":
    main()
