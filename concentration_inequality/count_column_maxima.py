from numpy import argmax, bincount, array, ndarray


def count_column_maxima(matrix: ndarray) -> ndarray:
    """
    Takes a matrix and returns a vector counting how many times each column
    has the greatest element in a row.

    Args:
    matrix (np.ndarray): Input matrix (2D numpy array)

    Returns:
    np.ndarray: Vector with counts for each column
    """
    # Ensure the input is a 2D array
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    # Find the indices of the maximum values along each row
    max_indices = argmax(matrix, axis=1)

    # Count the occurrences of each column index
    num_columns = matrix.shape[1]
    counts = bincount(max_indices, minlength=num_columns)

    return counts


def main():
    # Create a sample matrix
    matrix = array([
        [1, 5, 3, 2],
        [4, 2, 6, 1],
        [3, 3, 3, 5],
        [2, 4, 1, 3]
    ])

    # Calculate the count vector
    result = count_column_maxima(matrix)

    print("Input matrix:")
    print(matrix)
    print("\nResulting count vector:")
    print(result)


# Example usage
if __name__ == "__main__":
    main()
