from time import sleep

from numpy import random, allclose, ndarray


def generate_probability_matrix(num_rows: int, vector_length: int) -> ndarray:
    """
    Generate a matrix where each row is a probability vector of the same length.

    Args:
    num_rows (int): Number of rows in the matrix (number of probability vectors).
    vector_length (int): Length of each probability vector.

    Returns:
    np.ndarray: A NumPy matrix where each row is a probability vector.
    """
    # Generate a random matrix
    matrix = random.rand(num_rows, vector_length)

    # Normalize each row so it sums to 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / row_sums

    return matrix


def main():
    # Generate a 3x4 probability matrix
    num_rows = 3
    vector_length = 4

    prob_matrix = generate_probability_matrix(num_rows, vector_length)

    print("Generated Probability Matrix:")
    print(prob_matrix)
    print(prob_matrix[:, 0])
    print(len(prob_matrix[:, 0]))

    # Verify that each row sums to 1 (allowing for small floating-point errors)
    row_sums = prob_matrix.sum(axis=1)
    print("\nRow sums:")
    print(row_sums)

    # Check if all row sums are close to 1
    assert allclose(row_sums, 1.0), "Not all rows sum to 1"
    print("\nAll rows sum to 1 (within floating-point precision)")


# Example usage
if __name__ == "__main__":
    while True:
        main()
        sleep(5)
