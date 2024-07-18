from typing import Dict, Tuple

from mpmath import mp
from numpy import array, ndarray

from chr.create_column_product_dict import create_column_product_dict
from chr.multivariate_beta import multivariate_beta
from chr.update_y import update_y
from concentration_inequality.dirichlet_sample import dirichlet_sample


def wealth(y: Dict[Tuple[int, ...], mp.mpf], o: ndarray, precision: int = 50) -> mp.mpf:
    mp.dps = precision  # Set decimal precision
    alpha = tuple(mp.mpf(1) / 2 for _ in o)  # Use mpmath for division
    denominator = multivariate_beta(alpha)

    numerator = mp.mpf(0.)
    for k, y_k in y.items():
        o_k = mp.fprod(o[i] ** k[i] for i in range(len(k)))  # High-precision product
        beta = multivariate_beta(tuple(k[i] + alpha[i] for i in range(len(k))))
        numerator += y_k * o_k * beta

    return numerator / denominator


def wealth_inverse(y: Dict[Tuple[int, ...], mp.mpf], o: ndarray, precision: int = 50) -> float:
    # Inverting all coordinates of o
    o_inverse = mp.matrix([mp.mpf(1) / o_i for o_i in o])
    # Call wealth function with the inverted o
    return wealth(y, o_inverse, precision)


def main():
    # Create a sample 3x3 numpy array
    y_matrix = array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # Calculate y^1
    y_dict = create_column_product_dict(y_matrix)

    # Define o vector
    o = (0.1, 0.2, 0.3)

    # Calculate wealth
    result = wealth(y_dict, array(o))

    print("y dictionary:")
    for k, v in y_dict.items():
        print(f"{k}: {v}")

    print(f"\no vector: {o}")
    print(f"\nWealth: {result}")


def test():
    alpha_asymmetric = (0.1, 1, 5)
    true_mean = array(alpha_asymmetric) / sum(alpha_asymmetric)
    size = 100
    y = dirichlet_sample(alpha_asymmetric, size=size)
    print("y:", y)
    # Use the same y_dict from previous example
    y_dict = update_y(y, y.shape[0])  # Your dictionary of y values
    for k, v in y_dict.items():
        print(k, "->", v)
    result = wealth(y_dict, true_mean)

    print("y dictionary:")
    for k, v in y_dict.items():
        print(f"{k}: {v}")

    print(f"\nWealth: {result}")


if __name__ == "__main__":
    # main()
    test()
