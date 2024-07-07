from math import exp, factorial

from scipy.special import comb, gammainc

from new.convolve_pmf_at_point import convolve_pmf_at_point
from new.truncated_poisson_pmf import truncated_poisson_pmf


def conv(s: float, a: int, q: float, n: int) -> float:
    assert a <= n <= 2 * a
    first_factor = exp(-s) * ((s * (1 - q)) ** n) / factorial(n)
    second_factor = 0.
    for k in range(n - a, a + 1):
        second_factor += comb(n, k) * ((q / (1 - q)) ** k)

    return first_factor * second_factor


s = 0.1
a = 40
q = 0.9
lam1, lam2 = s * q, s * (1 - q)
pmf1 = truncated_poisson_pmf(lam1, a)
pmf2 = truncated_poisson_pmf(lam2, a)

n = 45
convolved_value = convolve_pmf_at_point(pmf1, pmf2, n)
my_value = conv(s, a, q, n)

print("Expected value:", convolved_value)
print("My value:", my_value)

first_gamma_inc = gammainc(a + 1, s * q)
second_gamma_inc = gammainc(a + 1, s * (1 - q))
product = first_gamma_inc * second_gamma_inc
print("Product:", product)
