from numpy import square, sqrt, array, ndarray, any


def matrix_sqrt(matrix: ndarray) -> ndarray:
    """
    Takes a NumPy matrix and applies the square root to every coordinate.

    Args:
    matrix (np.ndarray): Input matrix

    Returns:
    np.ndarray: A new matrix with the square root applied to each element
    """
    # Check if the input is a NumPy array
    if not isinstance(matrix, ndarray):
        raise TypeError("Input must be a NumPy array")

    # Check if all elements are non-negative
    if any(matrix < 0):
        raise ValueError("All elements must be non-negative to compute real square roots")

    # Compute the element-wise square root
    return sqrt(matrix)


def main():
    # Create a sample matrix
    sample_matrix = array([
        [1, 4, 9],
        [16, 25, 36],
        [49, 64, 81]
    ])

    print("Original matrix:")
    print(sample_matrix)

    # Apply square root to the matrix
    sqrt_matrix = matrix_sqrt(sample_matrix)

    print("\nMatrix after applying square root:")
    print(sqrt_matrix)

    # Verify the result
    print("\nVerification (squaring the result):")
    print(square(sqrt_matrix))


# Example usage
if __name__ == "__main__":
    main()
