from statsmodels.stats.proportion import proportion_confint

from experiment.find_q_for_quantile_exact import find_q_for_quantile_exact
from experiment.find_quantile_exact import find_quantile_exact
from multinomial.generate_tuple import generate_tuple

alpha = 0.01
m = 3
num_simulations = 100000
n = 50
x = 40  # 36 25
print("x:", x)
q = proportion_confint(x, n, alpha=2 * alpha, method="beta")[0]
print("q:", q)
p = generate_tuple(m=m, q=q)
print("p:", p)
quantile = find_quantile_exact(p=p, n=n, alpha=alpha, num_simulations=num_simulations)
print("quantile:", quantile)
q_value = find_q_for_quantile_exact(n=n, m=m, alpha=alpha, x=x, num_simulations=num_simulations)
print("q_value:", q_value)
