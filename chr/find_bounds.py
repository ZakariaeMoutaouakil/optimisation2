from scipy.optimize import minimize, Bounds
import numpy as np
from typing import Dict, Tuple, List

from chr.update_y import update_y
from chr.wealth import wealth
from concentration_inequality.dirichlet_sample import dirichlet_sample


def objective(o: np.ndarray, y: Dict[Tuple[int, ...], float], delta: float) -> float:
    return wealth(y, o) - 1 / delta


def find_bounds(y: Dict[Tuple[int, ...], float], delta: float, o_initial: np.ndarray) -> List[Tuple[float, float]]:
    n = len(o_initial)
    bounds = Bounds(0, 1)
    results = []

    for i in range(n):
        def constraint_sum(o: np.ndarray) -> float:
            return np.sum(o) - 1

        # Minimize element i
        def objective_min(o: np.ndarray) -> float:
            return o[i]

        res_min = minimize(objective_min, o_initial, args=(), method='SLSQP', bounds=bounds,
                           constraints={'type': 'eq', 'fun': constraint_sum})
        if objective(res_min.x, y, delta) < 0:
            lower_bound = res_min.x[i]
        else:
            lower_bound = 0

        # Maximize element i
        def objective_max(o: np.ndarray) -> float:
            return -o[i]

        res_max = minimize(objective_max, o_initial, args=(), method='SLSQP', bounds=bounds,
                           constraints={'type': 'eq', 'fun': constraint_sum})
        if objective(res_max.x, y, delta) < 0:
            upper_bound = res_max.x[i]
        else:
            upper_bound = 1

        results.append((lower_bound, upper_bound))

    return results


def main() -> None:
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

    # Define initial o vector
    o_initial = np.array([0.1, 0.2, 0.3])

    # Define delta
    delta = 0.05

    # Calculate wealth
    result = wealth(y_dict, o_initial)

    print("y dictionary:")
    for k, v in y_dict.items():
        print(f"{k}: {v}")

    print(f"\no vector: {o_initial}")
    print(f"\nWealth: {result}")

    # Find bounds for o
    bounds = find_bounds(y_dict, delta, o_initial)
    print("\nBounds for o:")
    print("True Mean:", true_mean)
    for i, (lb, ub) in enumerate(bounds):
        print(f"o[{i}]: ({lb}, {ub})")

if __name__ == "__main__":
    main()
