from typing import Dict, Tuple
from mpmath import mp
from numpy import array, ndarray

from chr.create_column_product_dict import create_column_product_dict
from chr.generate_vectors import generate_vectors


def update_y(y: ndarray, t: int, precision: int = 50) -> Dict[Tuple[int, ...], mp.mpf]:
    assert 0 < t <= y.shape[0], "t must be a positive integer less than or equal to the number of trials"
    mp.dps = precision  # Set decimal precision
    _K = y.shape[1]

    if t == 1:
        return create_column_product_dict(y, precision=precision)

    y_prev = update_y(y, t - 1, precision=precision)
    y_current: Dict[Tuple[int, ...], mp.mpf] = {}

    for k in generate_vectors(_K, t):
        value = mp.mpf(0.)  # Use mpmath floating point number for high precision
        for j in range(_K):
            if k[j] >= 1:
                k_minus_ej = tuple(k[i] - (1 if i == j else 0) for i in range(_K))
                value += y[t - 1, j] * y_prev.get(k_minus_ej, mp.mpf(0.))
        y_current[k] = value

    return y_current


def main():
    # Create a sample 3x3 numpy array
    y = array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

    # Calculate y^2
    y_2 = update_y(y, 2, precision=50)  # Notice I've changed this to calculate y^2 as t=3 is out of bounds for this y

    # Print the resulting dictionary
    for k, v in y_2.items():
        print(f"{k}: {v}")


# Example usage
if __name__ == "__main__":
    main()
