from time import time

from numpy import array
from numpy import inf, cumsum, log, exp, ndarray, zeros
from scipy.special import comb, logsumexp


def dynamic_programming_solution(n_trials: int, threshold: int, p: ndarray) -> float:
    n_classes = len(p)

    # precompute the cumulative probabilities
    cumprobas = cumsum(p)
    f = zeros((n_trials + 1, n_classes + 1))
    f[:threshold + 1, 1] = 1

    # DP
    for k in range(2, n_classes + 1):
        pk = p[k - 1] / cumprobas[k - 1]
        log_pk = log(pk)
        log_1_minus_pk = log(1 - pk)
        for n in range(1, n_trials + 1):
            log_terms = []
            for j in range(min(threshold + 1, n + 1)):
                if f[n - j, k - 1] > 0:  # Only consider non-zero terms
                    log_comb = log(comb(n, j)) if n >= j else -inf
                    log_term = log(f[n - j, k - 1]) + log_comb + j * log_pk + (n - j) * log_1_minus_pk
                    log_terms.append(log_term)
            if log_terms:
                f[n, k] = exp(logsumexp(log_terms))

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
