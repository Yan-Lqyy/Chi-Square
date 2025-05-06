# chi_square_math/utils.py

"""Utility functions for input validation and array handling."""

import numpy as np
from typing import Union, List, Tuple

def safe_to_ndarray(data: Union[List, Tuple, np.ndarray], dtype=np.float64) -> np.ndarray:
    """
    Safely converts input data to a NumPy array, performing basic checks.

    Args:
        data: Input data (list, tuple, or existing ndarray).
        dtype: NumPy data type for the array.

    Returns:
        A NumPy array.

    Raises:
        ValueError: If data is empty, contains non-numeric types, or has negative values.
        TypeError: If the input type is not supported.
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f"Input data must be a list, tuple, or numpy array, got {type(data)}")

    try:
        arr = np.array(data, dtype=dtype)
    except ValueError as e:
        raise ValueError(f"Could not convert data to numeric array. Check for non-numeric values. Original error: {e}") from e

    if arr.size == 0:
        raise ValueError("Input data cannot be empty.")

    if np.any(arr < 0):
        raise ValueError("Input data contains negative values. Frequencies must be non-negative.")

    # Check for NaNs or Infs introduced during conversion
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Input data contains NaN or Infinite values after conversion.")

    return arr

def check_expected_frequencies(expected: np.ndarray, threshold: float) -> List[str]:
    """
    Checks if any expected frequencies fall below a specified threshold.

    Args:
        expected: NumPy array of expected frequencies.
        threshold: The minimum acceptable expected frequency.

    Returns:
        A list of warning messages, empty if all frequencies meet the threshold.
    """
    warnings = []
    if np.any(expected < threshold):
        count_below = np.sum(expected < threshold)
        total_cells = expected.size
        percentage_below = (count_below / total_cells) * 100
        warnings.append(
            f"Assumption Warning: {count_below} out of {total_cells} ({percentage_below:.1f}%) "
            f"expected frequencies are below the recommended threshold of {threshold}. "
            "Chi-Square results may be inaccurate."
        )
        # Suggest Fisher's Exact Test for 2x2 tables
        if expected.ndim == 2 and expected.shape == (2, 2):
             warnings.append("Consider using Fisher's Exact Test for 2x2 tables with low expected frequencies.")
    return warnings