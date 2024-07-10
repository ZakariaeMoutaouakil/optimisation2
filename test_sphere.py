from numpy import pi, copy, zeros, sin, cos, amin, prod, amax

from concentration_inequality.blaise.calculate_term import calculate_term
from concentration_inequality.calculate_bounds import calculate_bounds
from concentration_inequality.cartesian_to_spherical import cartesian_to_spherical
from concentration_inequality.generate_probability_matrix import generate_probability_matrix
from concentration_inequality.matrix_sqrt import matrix_sqrt

num_rows = 700
vector_length = 5
high_prob_column = 0  # Index of the column to have higher probabilities

sample_matrix = generate_probability_matrix(num_rows, vector_length)  # , high_prob_column)
alpha = 0.05

print("Sample matrix:")
print(sample_matrix)

# Calculate bounds
bounds = calculate_bounds(sample_matrix, calculate_term, alpha=alpha, debug=False)

print("\nBlaise Bounds:")
print(bounds)

sample_matrix_sqrt = matrix_sqrt(sample_matrix)
print("\nSample matrix (Sqrt):", sample_matrix_sqrt)

spherical_matrix = cartesian_to_spherical(sample_matrix_sqrt)
print("\nSpherical matrix:", spherical_matrix)

normalized_spherical_matrix = copy(spherical_matrix)  # make a copy of the original matrix to keep it intact
normalized_spherical_matrix[:, :-1] = normalized_spherical_matrix[:, :-1] / pi  # divide all columns but the last
normalized_spherical_matrix[:, -1] = normalized_spherical_matrix[:, -1] / (2 * pi)  # divide the last column

# Calculate bounds
spherical_bounds = calculate_bounds(normalized_spherical_matrix, calculate_term, alpha=alpha, debug=False)
final_spherical_bounds = copy(spherical_bounds)
final_spherical_bounds[:, :-1] = final_spherical_bounds[:, :-1] * pi  # multiply all columns but the last
final_spherical_bounds[:, -1] = final_spherical_bounds[:, -1] * (2 * pi)  # multiply the last column

print("\nBlaise Spherical Bounds:")
print(final_spherical_bounds)

n = final_spherical_bounds.shape[1] + 1
cartesian = zeros((2, n))
print("\nCartesian:", cartesian)

sin_phi = sin(final_spherical_bounds) ** 2
cos_phi = cos(final_spherical_bounds) ** 2
print("sin_phi:", sin_phi)
print("cos_phi:", cos_phi)

min_sin = amin(sin_phi, axis=0)
min_cos = amin(cos_phi, axis=0)
print("min_sin:", min_sin)
print("min_cos:", min_cos)

cartesian[0, 0] = min_cos[0]

for i in range(1, n - 1):
    cartesian[0, i] = prod(min_sin[:i]) * min_cos[i]

cartesian[0, -1] = prod(min_sin)
print("\nCartesian:", cartesian)

max_sin = amax(sin_phi, axis=0)
max_cos = amax(cos_phi, axis=0)
print("max_sin:", max_sin)
print("max_cos:", max_cos)

cartesian[1, 0] = max_cos[0]

for i in range(1, n - 1):
    cartesian[1, i] = prod(max_sin[:i]) * max_cos[i]

cartesian[1, -1] = prod(max_sin)
print("\nCartesian:", cartesian)

print("\nBlaise Bounds:")
print(bounds)
