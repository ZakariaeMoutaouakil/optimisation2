from numpy import array, dot, ndarray


def vector_to_square_matrix(v: ndarray) -> ndarray:
    """
    Takes a numpy vector and returns the squared matrix which is the product of the vector and its transpose.

    Args:
    v (np.ndarray): Input vector (1D numpy array)

    Returns:
    np.ndarray: Squared matrix (2D numpy array)
    """
    # Ensure the input is a 1D array
    if v.ndim != 1:
        raise ValueError("Input must be a 1D numpy array")

    # Reshape v into a column vector
    v_col = v.reshape(-1, 1)

    # Compute the outer product
    squared_matrix = dot(v_col, v_col.T)

    return squared_matrix


def main():
    # Create a sample vector
    v = array([1, 2, 3])

    # Calculate the squared matrix
    result = vector_to_square_matrix(v)

    print("Input vector:")
    print(v)
    print("\nResulting squared matrix:")
    print(result)


# Example usage
if __name__ == "__main__":
    main()
