from math import comb
from time import time

# from scipy.special import gammaincc
from mpmath import gammainc
from new.multinomial_max_cdf import multinomial_max_cdf


def closed_max_cdf(q: float, n: int, a: int, s: float) -> float:
    assert a <= n <= 2 * a, f"a={a} must be less than or equal to n={n}"
    assert 0.5 < q < 1, f"q={q} must be strictly between 0.5 and 1"

    # first_gamma_inc = gammainc(a + 1, s * q, regularized=True)
    # print("first_gamma_inc:", first_gamma_inc)
    # second_gamma_inc = gammainc(a + 1, s * (1 - q), regularized=True)
    # print("second_gamma_inc:", second_gamma_inc)

    second_factor = 0.
    for k in range(n - a, a + 1):
        second_factor += comb(n, k) * ((q / (1 - q)) ** k)

    print("second_factor:", second_factor)
    # return first_gamma_inc * second_gamma_inc * second_factor
    return second_factor * ((1 - q) ** n)


def main():
    q = 0.95
    n = 20
    s = 1
    a = 19
    p = (q, 1 - q)

    start_time = time()
    result = multinomial_max_cdf(x=a, n=n, p=p)
    end_time = time()
    print("Multinomial Max CDF:", result)
    print("Time taken:", end_time - start_time)

    start_time = time()
    result = closed_max_cdf(q=q, n=n, a=a, s=s)
    end_time = time()
    print("Closed Max CDF:", result)
    print("Time taken:", end_time - start_time)


if __name__ == "__main__":
    main()
