from numpy import allclose, prod, sqrt, array, cos, sin, sum, ndarray, linalg, zeros, arctan2


def cartesian_to_spherical(matrix: ndarray) -> ndarray:
    """
    Convert a matrix of Cartesian coordinates on the n-sphere to spherical coordinates.

    Args:
    matrix (np.ndarray): Input matrix where each row is a unit vector in Cartesian coordinates

    Returns:
    np.ndarray: Matrix where each row is in spherical coordinates (excluding radius)
    """
    # Check if input is a NumPy array
    if not isinstance(matrix, ndarray):
        raise TypeError("Input must be a NumPy array")

    # Check if vectors are unit vectors (with some tolerance for floating-point arithmetic)
    if not allclose(linalg.norm(matrix, axis=1), 1, rtol=1e-5):
        raise ValueError("Input vectors must be unit vectors (on the n-sphere)")

    n = matrix.shape[1]  # Dimension of the space
    spherical = zeros((matrix.shape[0], n - 1))

    for i in range(n - 1):
        if i == n - 2:
            # Last angle
            spherical[:, i] = arctan2(matrix[:, -1], matrix[:, -2])
        else:
            # Other angles
            numerator = sqrt(sum(matrix[:, i + 1:] ** 2, axis=1))
            denominator = matrix[:, i]
            spherical[:, i] = arctan2(numerator, denominator)

    return spherical


# Example usage
if __name__ == "__main__":
    # Create some sample unit vectors
    cartesian_vectors = array([
        [sqrt(0.5), sqrt(0.5), 0.],
        [1, 0, 0],  # x-axis
        [0, 1, 0],  # y-axis
        [0, 0, 1],  # z-axis
        [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]  # Diagonal
    ])

    print("Cartesian coordinates:")
    print(cartesian_vectors)

    spherical_vectors = cartesian_to_spherical(cartesian_vectors)

    print("\nSpherical coordinates:")
    print(spherical_vectors)


    # Convert back to Cartesian for verification
    def spherical_to_cartesian(spherical):
        n = spherical.shape[1] + 1
        cartesian = zeros((spherical.shape[0], n))

        sin_phi = sin(spherical)
        cos_phi = cos(spherical)

        cartesian[:, 0] = cos_phi[:, 0]

        for i in range(1, n - 1):
            cartesian[:, i] = prod(sin_phi[:, :i], axis=1) * cos_phi[:, i]

        cartesian[:, -1] = prod(sin_phi, axis=1)

        return cartesian


    cartesian_again = spherical_to_cartesian(spherical_vectors)

    print("\nConverted back to Cartesian:")
    print(cartesian_again)

    print("\nAre the original and reconverted matrices equal?")
    print(allclose(cartesian_vectors, cartesian_again))
