from numpy import isclose, arange
from statsmodels.stats.proportion import proportion_confint

alpha = 0.1
step = 0.01
x = (7, 10, 25)
n = sum(x)
m = len(x)
num_simulations = 1000000
p1 = proportion_confint(max(x), n, alpha=2 * alpha, method="beta")[0]
print("statsmodels p1:", p1)
p2 = 1 - p1
print("Pessimistic p2:", p2)


y = x[:len(x) - 1]
print("y:", y)
n_y = sum(y)
m_y = len(y)
optimal_q = 1.
for gamma in arange(0.001, alpha - 0.001, 0.0001):
    print("gamma:", gamma)
    beta = (alpha - gamma) / (1 - gamma)
    print("beta:", beta)
    assert isclose((1 - beta) * (1 - gamma), 1 - alpha)
    p1 = proportion_confint(max(x), n, alpha=2 * beta, method="beta")[0]
    print("My p1:", p1)
    p1_ = proportion_confint(max(y), n_y, alpha=2 * gamma, method="beta")[1]
    print("My p1':", p1_)
    q = p1_ * (1 - p1)
    print("My q:", q)
    if q < p2:
        print("q < p2")
    if q < optimal_q:
        optimal_q = q
print("optimal_q:", optimal_q)
