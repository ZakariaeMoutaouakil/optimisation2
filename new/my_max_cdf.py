from time import time

from mpmath import mp, mpf, factorial, exp, gammainc

from new.convolve_pmfs import convolve_pmfs
from new.debug_log import debug_log
from new.multinomial_max_cdf import multinomial_max_cdf
from new.multinomial_max_cdf_simulation import multinomial_max_cdf_simulation
from new.truncated_poisson_pmf import truncated_poisson_pmf


def my_max_cdf(q: float, n: int, a: int, s: float, debug: bool = False) -> float:
    assert 0.5 < q < 1, f"q={q} must be strictly between 0.5 and 1"
    assert n > 1, "n must be greater than 1"
    assert a <= n, "a must be less than or equal to n"
    assert n <= 2 * a, "n must be less than or equal to 2a"
    assert s > 0, "s must be positive"

    # Set the precision
    mp.dps = 50

    first_factor = factorial(n) / (mpf(s) ** n * exp(-s))
    debug_log(f"first factor: {first_factor}", debug)

    second_factor = gammainc(a + 1, s * q, regularized=True)
    debug_log(f"second factor: {second_factor}", debug)

    third_factor = gammainc(a + 1, s * (1 - q), regularized=True)
    debug_log(f"third factor: {third_factor}", debug)

    pmf1 = truncated_poisson_pmf(lam=s * q, b=a, precision=50)
    debug_log(f"pmf1: {pmf1}", debug)

    pmf2 = truncated_poisson_pmf(lam=s * (1 - q), b=a, precision=50)
    debug_log(f"pmf2: {pmf2}", debug)

    sum_pmf = convolve_pmfs(pmf1, pmf2)
    debug_log(f"sum_pmf: {sum_pmf}", debug)

    final_factor = sum_pmf.get(n, mpf(0))
    debug_log(f"final factor: {final_factor}", debug)

    result = first_factor * second_factor * third_factor * final_factor
    return float(result)


def main():
    q = 0.95
    n = 300  # limit 177
    s = 1
    a = 290
    start_time = time()
    result = my_max_cdf(q=q, n=n, s=s, a=a, debug=False)
    end_time = time()
    print("My Max CDF:", result)
    print("Time taken:", end_time - start_time)

    p = (q, 1 - q, 0)
    num_simulations = 100000
    start_time = time()
    result = multinomial_max_cdf_simulation(p=p, n=n, x=a, num_simulations=num_simulations)
    end_time = time()
    print("Multinomial Max CDF Simulation:", result)
    print("Time taken:", end_time - start_time)

    start_time = time()
    result = multinomial_max_cdf(x=a, n=n, p=p)
    end_time = time()
    print("My Max CDF:", result)
    print("Time taken:", end_time - start_time)


if __name__ == "__main__":
    main()
