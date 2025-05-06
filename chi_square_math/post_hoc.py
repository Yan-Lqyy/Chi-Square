# chi_square_math/post_hoc.py

"""Functions for post-hoc analysis using residuals."""

import numpy as np
from typing import Optional, List, Tuple

# Assuming constants.py is in the same package directory
from .constants import RESIDUAL_SIGNIFICANCE_THRESHOLD

def calculate_adjusted_residuals(observed: np.ndarray, expected: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculates adjusted (standardized) residuals for contingency tables.

    Adjusted Residual = (Observed - Expected) / sqrt(Expected * (1 - row_margin_prop) * (1 - col_margin_prop))
    A simpler version is (Observed - Expected) / sqrt(Expected). Let's implement the simpler one first
    as it's less complex and often sufficient for indicating cell contributions.
    This is sometimes called the "standardized residual".

    Args:
        observed: NumPy array of observed frequencies (2D).
        expected: NumPy array of expected frequencies (2D).

    Returns:
        A NumPy array of adjusted residuals of the same shape, or None if inputs are invalid.
    """
    if observed.shape != expected.shape or observed.ndim != 2:
        # Input shape mismatch or not a 2D table
        return None

    # Avoid division by zero. Where expected is 0, the contribution is 0 if observed is also 0.
    # If observed > 0 and expected is 0, the Chi-Square contribution is infinite,
    # and the residual is also effectively infinite or undefined.
    # SciPy's chi2_contingency handles the main chi2 stat calculation with zeros.
    # For residuals, we can mask or handle carefully. A simple way is to use a small epsilon
    # or calculate only where expected > 0.

    residuals = np.zeros_like(observed, dtype=float)

    # Create a mask where expected counts are greater than a small epsilon to avoid div by zero
    valid_mask = expected > np.finfo(float).eps # Use machine epsilon

    # Calculate residuals only where the expected count is valid
    obs_valid = observed[valid_mask]
    exp_valid = expected[valid_mask]

    # Calculate (Observed - Expected) / sqrt(Expected)
    residuals[valid_mask] = (obs_valid - exp_valid) / np.sqrt(exp_valid)

    # Optional: Implement the full Adjusted Residual if needed
    # row_sums = observed.sum(axis=1, keepdims=True)
    # col_sums = observed.sum(axis=0, keepdims=True)
    # total_sum = observed.sum()
    # row_props = row_sums / total_sum
    # col_props = col_sums / total_sum
    #
    # # Need to broadcast row_props and col_props to match the table shape for calculation
    # row_props_table = np.repeat(row_props, observed.shape[1], axis=1)
    # col_props_table = np.repeat(col_props.T, observed.shape[0], axis=0)
    #
    # variance = expected * (1 - row_props_table) * (1 - col_props_table)
    # # Handle potential division by zero if variance is zero (e.g., row_prop or col_prop is 1)
    # adjusted_residuals = np.zeros_like(observed, dtype=float)
    # non_zero_variance_mask = variance > np.finfo(float).eps
    # adjusted_residuals[non_zero_variance_mask] = (observed[non_zero_variance_mask] - expected[non_zero_variance_mask]) / np.sqrt(variance[non_zero_variance_mask])
    # return adjusted_residuals

    return residuals


def find_significant_residuals(residuals: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """
    Identifies residuals whose absolute value exceeds a significance threshold.

    Args:
        residuals: NumPy array of adjusted residuals (2D).
        threshold: The significance threshold (absolute value, e.g., 1.96 for alpha=0.05).

    Returns:
        A list of tuples, where each tuple contains (row_index, column_index, residual_value)
        for significant residuals. Indices are 0-based.
    """
    significant = []
    if residuals is None or residuals.ndim != 2:
        return significant # Return empty list for invalid input

    rows, cols = residuals.shape
    # Use numpy indexing for efficiency
    significant_indices = np.where(np.abs(residuals) > threshold)

    # significant_indices is a tuple of arrays (row_indices, col_indices)
    for r, c in zip(significant_indices[0], significant_indices[1]):
        significant.append((int(r), int(c), float(residuals[r, c])))

    # Optional: Sort significant residuals by absolute value (descending)
    # significant.sort(key=lambda item: abs(item[2]), reverse=True)

    return significant