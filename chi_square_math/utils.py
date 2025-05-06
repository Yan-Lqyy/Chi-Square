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

    # Handle empty input gracefully before converting to array
    if not data or (isinstance(data, (list, tuple)) and all(not item for item in data)):
         raise ValueError("Input data cannot be empty or contain only empty items.")

    try:
        arr = np.array(data, dtype=dtype)
    except ValueError as e:
        raise ValueError(f"Could not convert data to numeric array. Check for non-numeric values. Original error: {e}") from e
    except Exception as e:
        # Catch other potential numpy conversion issues
         raise ValueError(f"An error occurred during data conversion: {e}") from e


    if arr.size == 0:
        raise ValueError("Input data resulted in an empty array.")

    if np.any(arr < 0):
        raise ValueError("Input data contains negative values. Frequencies must be non-negative.")

    # Check for NaNs or Infs introduced during conversion (e.g., from non-finite input)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Input data contains NaN or Infinite values after conversion. Check input format.")

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
    if expected is None or expected.size == 0:
        # This case should ideally not happen if analysis ran, but defensive check
        return warnings

    low_expected_mask = expected < threshold
    if np.any(low_expected_mask):
        count_below = np.sum(low_expected_mask)
        total_cells = expected.size
        percentage_below = (count_below / total_cells) * 100

        warning_message = (
            f"{count_below} out of {total_cells} ({percentage_below:.1f}%) expected frequencies "
            f"are below the recommended threshold of {threshold}."
        )

        # Provide more specific guidance based on context/test type
        if expected.ndim == 1: # GoF test
             warning_message += " This may make Chi-Square results inaccurate, especially if many categories have low expected counts."
        elif expected.ndim == 2: # Contingency table test
             warning_message += " This violates a common assumption for the Chi-Square test."
             if expected.shape == (2, 2):
                 warning_message += " Consider using Fisher's Exact Test for 2x2 tables with low expected frequencies."
             else:
                  warning_message += " Results may be unreliable. Consider combining categories if theoretically sound, or using alternative tests (e.g., exact tests if computationally feasible)."


        warnings.append(warning_message)
    return warnings