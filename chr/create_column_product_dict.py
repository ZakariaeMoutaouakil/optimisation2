from typing import Dict, Tuple

from numpy import prod, array, ndarray


def create_column_product_dict(y: ndarray) -> Dict[Tuple[int, ...], float]:
    k = y.shape[1]
    result: Dict[Tuple[int, ...], float] = {}

    for j in range(k):
        e_j = tuple(1 if i == j else 0 for i in range(k))
        result[e_j] = prod(y[:, j])

    return result


def main():
    # Create a sample 3x3 numpy array
    y = array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

    # Call the function with the sample array
    result_dict = create_column_product_dict(y)

    # Print the resulting dictionary
    for key, value in result_dict.items():
        print(f"e_{list(key).index(1)}: {key} -> {value}")


# Example usage
if __name__ == "__main__":
    main()
