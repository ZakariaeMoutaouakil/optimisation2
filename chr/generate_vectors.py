from typing import Tuple


def generate_vectors(dim: int, t: int) -> Tuple[Tuple[int, ...], ...]:
    if dim == 1:
        return ((t,),)
    vectors = []
    for i in range(t + 1):
        for v in generate_vectors(dim - 1, t - i):
            vectors.append((i,) + v)
    return tuple(vectors)


def main():
    # Define parameters for the multinomial distribution
    k = 3  # Number of categories
    t = 4  # Number of trials

    # Generate all possible vectors
    vectors = generate_vectors(k, t)

    # Print the generated vectors
    print(f"Generated vectors for K={k} and t={t}:")
    for vector in vectors:
        print(vector)


if __name__ == "__main__":
    # Run the example usage
    main()
