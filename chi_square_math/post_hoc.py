# chi_square_math/post_hoc.py

"""Functions for post-hoc analysis using residuals."""

import numpy as np
from typing import Optional, List, Tuple

from .constants import RESIDUAL_SIGNIFICANCE_THRESHOLD

def calculate_adjusted_residuals(observed: np.ndarray, expected: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculates adjusted (standardized) residuals for contingency tables.
    Adjusted Residual = (Observed - Expected) / sqrt(Expected)

    Args:
        observed: NumPy array of observed frequencies.
        expected: NumPy array of expected frequencies.

    Returns:
        A NumPy array of adjusted residuals, or None if shapes mismatch
        or expected contains zeros where division would occur.
    """
    if observed.shape != expected.shape:
        # This should ideally be caught earlier, but double-check
        return None

    # Avoid division by zero. Replace expected[i] with a small number or handle?
    # If expected[i] is 0, observed[i] must also be 0 for chi2 to be calculable by scipy.
    # In this 0/sqrt(0) case, the residual contribution is 0.
    residuals = np.zeros_like(observed, dtype=float)
    
    # Calculate residuals only where expected > 0
    valid_mask = expected > 1e-9 # Use a small epsilon to avoid floating point issues near zero
    
    obs_valid = observed[valid_mask]
    exp_valid = expected[valid_mask]
    
    residuals[valid_mask] = (obs_valid - exp_valid) / np.sqrt(exp_valid)

    return residuals


def find_significant_residuals(residuals: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """
    Identifies residuals whose absolute value exceeds a significance threshold.

    Args:
        residuals: NumPy array of adjusted residuals.
        threshold: The significance threshold (absolute value).

    Returns:
        A list of tuples, where each tuple contains (row_index, column_index, residual_value)
        for significant residuals.
    """
    significant = []
    if residuals is None:
        return significant

    rows, cols = residuals.shape
    for r in range(rows):
        for c in range(cols):
            if abs(residuals[r, c]) > threshold:
                significant.append((r, c, residuals[r, c]))
    return significant