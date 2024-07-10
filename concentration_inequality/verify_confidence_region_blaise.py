from typing import Tuple

from numpy import all, random, mean

from concentration_inequality.blaise.calculate_term import calculate_term
from concentration_inequality.calculate_bounds import calculate_bounds
from concentration_inequality.dirichlet_sample import dirichlet_sample


def verify_confidence_region(alpha: Tuple[float, ...], num_samples: int, num_tests: int, confidence_level: float) \
        -> float:
    # Generate Dirichlet samples
    samples = dirichlet_sample(alpha, size=num_samples)
    print(f"Samples: {samples}")

    # Calculate confidence region
    confidence_region = calculate_bounds(samples, calculate_term, alpha=1 - confidence_level, debug=False)
    print(f"Confidence region: {confidence_region}")

    test_samples = dirichlet_sample(alpha, size=num_tests)
    print(f"Test samples: {test_samples}")

    # Check how many samples fall within the confidence region
    within_region = all((test_samples >= confidence_region[0]) & (test_samples <= confidence_region[2]), axis=1)

    # Calculate the proportion of samples within the region
    proportion_within = mean(within_region)

    return proportion_within


def main():
    random.seed(42)  # For reproducibility

    # Dirichlet parameters
    alpha = (1, 1, 10)

    # Number of samples to generate
    num_samples = 10

    # Number of tests to run
    num_tests = 1000000

    # Confidence level
    confidence_level = 0.95

    proportion = verify_confidence_region(alpha, num_samples, num_tests, confidence_level)

    print(f"Proportion of samples within the confidence region: {proportion:.4f}")
    print(f"Expected proportion (confidence level): {confidence_level:.4f}")

    if proportion >= confidence_level:
        print("The confidence region appears to be correct.")
    else:
        print("The confidence region may not be accurate.")


# Example usage
if __name__ == "__main__":
    main()
