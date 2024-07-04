from time import time

from mpmath import mpf, binomial
from numpy import cumsum, array, ones, ndarray


def dynamic_programming_solution(n_trials: int, threshold: int, p: ndarray) -> float:
    n_classes = len(p)

    # precompute the cumulative probabilities
    cumprobas = cumsum(p)
    f = ones((n_trials + 1, n_classes + 1), dtype=object)
    f[threshold + 1:, 1] = 0

    # DP
    for k in range(2, n_classes + 1):
        for n in range(1, n_trials + 1):
            f[n, k] = mpf(0)
            pk = mpf(p[k - 1]) / mpf(cumprobas[k - 1])
            for j in range(threshold + 1):
                f[n, k] += f[n - j, k - 1] * binomial(n, j) * (pk ** j) * ((1 - pk) ** (n - j))

    return float(f[n_trials, n_classes])


# Example usage
def main():
    n_trials = 200
    threshold = 190
    p = array([0.1, 0.9])

    start_time = time()
    result_dynamic = dynamic_programming_solution(n_trials, threshold, p)
    time_dynamic = time() - start_time
    print("Dynamic Programming Solution:", result_dynamic)
    print("Time taken:", time_dynamic)


if __name__ == "__main__":
    main()
