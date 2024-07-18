from matplotlib import pyplot as plt
from numpy import random, array

from concentration_inequality.blaise.calculate_term import calculate_term
from concentration_inequality.calculate_bounds import calculate_bounds
from concentration_inequality.clopper_pearson import clopper_pearson
from concentration_inequality.dirichlet_sample import dirichlet_sample

# Set random seed for reproducibility
random.seed(42)
alpha_asymmetric = (0.1, 1, 1, 2)
size = 300
matrix = dirichlet_sample(alpha_asymmetric, size=size)
print("\nAsymmetric Dirichlet samples:")
print(matrix)
true_mean = array(alpha_asymmetric) / sum(alpha_asymmetric)
print("\nAsymmetric Dirichlet mean:" + str(true_mean))
empirical_mean = matrix.mean(axis=0)
print("Asymmetric Dirichlet empirical mean:" + str(empirical_mean))

alpha = 0.05
m = len(alpha_asymmetric)
blaise_bounds = calculate_bounds(matrix, calculate_term, alpha=alpha / (2 * m))
clopper_pearson_bounds = clopper_pearson(matrix, alpha=alpha, debug=True)
print("\nClopper-Pearson bounds:")
print(clopper_pearson_bounds)
print("\nBlaise bounds:")
print(blaise_bounds)

sizes = range(5000, 10000, 100)  # Example sizes from 100 to 1000

# Initialize storage for bounds
cp_bounds = {size: None for size in sizes}
blaise_bounds = {size: None for size in sizes}

# Compute bounds for each size
for size in sizes:
    matrix = dirichlet_sample(alpha_asymmetric, size=size)
    cp_bounds[size] = clopper_pearson(matrix, alpha=alpha, debug=True)
    blaise_bounds[size] = calculate_bounds(matrix, calculate_term, alpha=alpha / (2 * m))

# Setup plot
fig, axes = plt.subplots(1, m, figsize=(18, 6))  # 1 row, 3 columns

# Plotting the results
for i in range(m):
    ax = axes[i]  # Select the appropriate subplot
    cp_lower = [cp_bounds[size][0, i] for size in sizes]
    cp_mean = [cp_bounds[size][1, i] for size in sizes]
    cp_upper = [cp_bounds[size][2, i] for size in sizes]
    blaise_lower = [blaise_bounds[size][0, i] for size in sizes]
    blaise_mean = [blaise_bounds[size][1, i] for size in sizes]
    blaise_upper = [blaise_bounds[size][2, i] for size in sizes]

    ax.plot(sizes, cp_lower, 'r--', label='Clopper-Pearson Bounds')  # Red for Clopper-Pearson
    ax.plot(sizes, cp_mean, 'r-', label='Clopper-Pearson Mean')
    ax.plot(sizes, cp_upper, 'r--')  # Red for Clopper-Pearson
    ax.plot(sizes, blaise_lower, 'b--', label='Blaise Bounds')  # Blue for Blaise
    ax.plot(sizes, blaise_mean, 'b-', label='Blaise Mean')
    ax.plot(sizes, blaise_upper, 'b--')  # Blue for Blaise
    ax.axhline(y=float(true_mean[i]), color='k', linestyle='--', linewidth=2,
               label='True Mean = ' + str(true_mean[i]))

    ax.set_title(f'Bounds for {true_mean[i]}')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Bound Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
