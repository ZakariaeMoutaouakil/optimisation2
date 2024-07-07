from pickle import dump
from random import randint
from time import time

from numpy import array

from new.dynamic_programming_solution import dynamic_programming_solution
from new.multinomial_max_cdf import multinomial_max_cdf
from new.multinomial_max_cdf_simulation import multinomial_max_cdf_simulation
from new.my_max_cdf import my_max_cdf


def plot() -> None:
    q = 0.95
    s = 1

    num_trials = 50
    dp = False
    results = []

    for _ in range(num_trials):
        print(f"Trial {_ + 1}/{num_trials}")
        n = randint(100, 400) if dp else randint(1000, 5000)
        a = randint(n // 2, n)

        p = (q, 1 - q)
        num_simulations = 1000

        start_time = time()
        multinomial_max_cdf_simulation(p=p, n=n, x=a, num_simulations=num_simulations)
        time_simulation = time() - start_time

        start_time = time()
        multinomial_max_cdf(x=a, n=n, p=p)
        time_analytical = time() - start_time

        start_time = time()
        my_max_cdf(q=q, n=n, s=s, a=a, debug=False)
        time_my_max_cdf = time() - start_time

        if dp:
            start_time = time()
            dynamic_programming_solution(n_trials=n, threshold=a, p=array(p))
            time_dynamic_programming = time() - start_time

            results.append({
                'n': n,
                'a': a,
                'time_simulation': time_simulation,
                'time_analytical': time_analytical,
                'time_my_max_cdf': time_my_max_cdf,
                'time_dynamic_programming': time_dynamic_programming,
            })
        else:
            results.append({
                'n': n,
                'a': a,
                'time_simulation': time_simulation,
                'time_analytical': time_analytical,
                'time_my_max_cdf': time_my_max_cdf,
            })

    with open('results.pkl', 'wb') as f:
        dump(results, f)


if __name__ == "__main__":
    plot()
