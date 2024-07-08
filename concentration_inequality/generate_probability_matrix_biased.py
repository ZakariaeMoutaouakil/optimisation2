from time import sleep

from numpy import random, allclose, ndarray


def generate_probability_matrix_biased(num_rows: int, vector_length: int, high_prob_column: int) -> ndarray:
    """
    Generate a matrix where each row is a probability vector of the same length,
    with one column having much higher probabilities.

    Args:
    num_rows (int): Number of rows in the matrix (number of probability vectors).
    vector_length (int): Length of each probability vector.
    high_prob_column (int): Index of the column to have higher probabilities.

    Returns:
    np.ndarray: A NumPy matrix where each row is a probability vector.
    """
    # Check if high_prob_column is valid
    if high_prob_column < 0 or high_prob_column >= vector_length:
        raise ValueError("high_prob_column must be between 0 and vector_length - 1")

    # Generate a random matrix
    matrix = random.rand(num_rows, vector_length)

    # Boost the specified column
    boost_factor = 10  # You can adjust this to control how much higher the probabilities are
    matrix[:, high_prob_column] *= boost_factor

    # Normalize each row so it sums to 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / row_sums

    return matrix


def main():
    # Generate a 3x4 probability matrix
    num_rows = 3
    vector_length = 4
    high_prob_column = 2  # Index of the column to have higher probabilities

    prob_matrix = generate_probability_matrix_biased(num_rows, vector_length, high_prob_column)

    print("Generated Probability Matrix:")
    print(prob_matrix)
    print(f"\nColumn {high_prob_column + 1} (higher probabilities):")
    print(prob_matrix[:, high_prob_column])

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
