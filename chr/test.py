import numpy as np
from matplotlib import pyplot as plt
from mpmath import log
from numpy import array, ndarray

from chr.update_y import update_y
from chr.wealth import wealth_inverse
from concentration_inequality.dirichlet_sample import dirichlet_sample

# Example usage:
p: float = 0.3  # Probability of each row being [0, 1]
rows: int = 100  # Number of rows in the matrix
precision: int = 50000  # Precision of the matrix
alpha = [p, 1 - p]
matrix: ndarray = dirichlet_sample(tuple(alpha), rows)
print(matrix)

# Use the same y_dict from previous example
y_dict = update_y(matrix, matrix.shape[0], precision)  # Your dictionary of y values
for k, v in y_dict.items():
    print(k, "->", v)
result = wealth_inverse(y_dict, array(alpha), precision)

print(f"\nWealth: {result}")

# Vary p from 0 to 1
p_values = np.linspace(0.01, 0.99, 100)
for p in p_values:
    print(f"p = {p:.2f}, wealth = {log(wealth_inverse(y_dict, np.array([p, 1 - p]), precision))}")
wealth_results = [log(wealth_inverse(y_dict, np.array([p, 1 - p]), precision)) for p in p_values]

delta = 0.1
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(p_values, wealth_results, label='Wealth')
plt.axhline(y=log(1 / delta), color='r', linestyle='--', linewidth=2,
            label='Horizontal Line at y=150')  # Add horizontal line
plt.xlabel('p')
plt.ylabel('Wealth')
plt.title('Wealth vs p')
plt.legend()
plt.grid(True)
plt.show()
