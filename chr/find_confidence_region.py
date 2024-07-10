from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import minimize_scalar

from chr.update_y import update_y
from chr.wealth import wealth
from concentration_inequality.dirichlet_sample import dirichlet_sample


def find_confidence_region(y: Dict[Tuple[int, ...], float], delta: float,
                           num_points: int = 1000) -> List[Tuple[float, float]]:
    K = len(next(iter(y.keys())))  # Dimensionality of the problem

    def objective(p: float, fix_dim: int, other_probs: np.ndarray) -> float:
        m = np.zeros(K)
        m[fix_dim] = p
        m[:fix_dim] = other_probs[:fix_dim]
        m[fix_dim + 1:] = other_probs[fix_dim:]
        m /= np.sum(m)  # Normalize to ensure it's a probability vector
        return abs(wealth(y, 1 / m) - 1 / delta)

    bounds = []
    for i in range(K):
        other_probs = np.ones(K - 1) / (K - 1)

        # Find lower bound
        res_lower = minimize_scalar(lambda p: objective(p, i, other_probs),
                                    method='bounded', bounds=(1e-6, 1))
        lower = max(0, res_lower.x)

        # Find upper bound
        res_upper = minimize_scalar(lambda p: objective(p, i, other_probs),
                                    method='bounded', bounds=(lower, 1))
        upper = min(1, res_upper.x)

        bounds.append((lower, upper))

    return bounds


# Example usage
if __name__ == "__main__":
    alpha_asymmetric = (0.1, 1, 5)
    true_mean = np.array(alpha_asymmetric) / np.sum(alpha_asymmetric)
    size = 50
    y = dirichlet_sample(alpha_asymmetric, size=size)
    print("y:", y)
    # Use the same y_dict from previous example
    y_dict = update_y(y, y.shape[0])  # Your dictionary of y values
    for k, v in y_dict.items():
        print(k, "->", v)
    delta = 0.001

    confidence_region = find_confidence_region(y_dict, delta)
    print("True Mean:", true_mean)
    print("Confidence Region Bounds:")
    for i, (lower, upper) in enumerate(confidence_region):
        print(f"Dimension {i + 1}: [{lower:.4f}, {upper:.4f}]")
