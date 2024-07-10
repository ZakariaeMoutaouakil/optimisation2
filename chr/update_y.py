from typing import Dict, Tuple

from numpy import array, ndarray

from chr.create_column_product_dict import create_column_product_dict
from chr.generate_vectors import generate_vectors


def update_y(y: ndarray, t: int) -> Dict[Tuple[int, ...], float]:
    assert 0 < t <= y.shape[0], "t must be a positive integer less than or equal to the number of trials"
    _K = y.shape[1]

    if t == 1:
        return create_column_product_dict(y)

    y_prev = update_y(y, t - 1)
    y_current: Dict[Tuple[int, ...], float] = {}

    for k in generate_vectors(_K, t):
        value = 0.
        for j in range(_K):
            if k[j] >= 1:
                k_minus_ej = tuple(k[i] - (1 if i == j else 0) for i in range(_K))
                value += y[t - 1, j] * y_prev.get(k_minus_ej, 0)
        y_current[k] = value

    return y_current


def main():
    # Create a sample 3x3 numpy array
    y = array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

    # Calculate y^2
    y_2 = update_y(y, 3)

    # Print the resulting dictionary
    for k, v in y_2.items():
        print(f"{k}: {v}")


# Example usage
if __name__ == "__main__":
    main()
