from typing import Dict, Tuple

from mpmath import mp
from numpy import array, ndarray


def create_column_product_dict(y: ndarray, precision: int = 50) -> Dict[Tuple[int, ...], mp.mpf]:
    mp.dps = precision  # Set decimal precision
    k = y.shape[1]
    result: Dict[Tuple[int, ...], mp.mpf] = {}

    for j in range(k):
        e_j = tuple(1 if i == j else 0 for i in range(k))
        # Use mpmath's fprod for high-precision product calculation
        result[e_j] = mp.fprod(y[:, j])

    return result


def main():
    # Create a sample 3x3 numpy array
    y = array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

    # Call the function with the sample array
    result_dict = create_column_product_dict(y, precision=50)

    # Print the resulting dictionary
    for key, value in result_dict.items():
        print(f"e{list(key).index(1)}: {key} -> {value}")


# Example usage
if __name__ == "__main__":
    main()
