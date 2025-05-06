# chi_square_math/effect_sizes.py

"""Functions to calculate effect sizes for Chi-Square tests."""

import numpy as np
import math
from typing import Optional

# Assuming results.py is in the same package directory
from .results import EffectSizeResult

def interpret_cramers_v(v: float, df_min: int) -> str:
    """
    Provides a rough interpretation of Cramer's V based on Cohen's guidelines,
    adapted for different degrees of freedom (df_min = min(R-1, C-1)).
    """
    # Cohen's benchmarks are often cited, but interpretations can vary.
    # These are common benchmarks.
    # For df_min = 1 (Phi): .10 (small), .30 (medium), .50 (large)
    # For df_min = 2: .07 (small), .21 (medium), .35 (large)
    # For df_min > 2: Benchmarks decrease. Using an approximation.

    v = abs(v) # Effect size is non-negative

    if df_min == 1: # Equivalent to Phi
        if v < 0.10: return "negligible"
        if v < 0.30: return "small"
        if v < 0.50: return "medium"
        return "large"
    elif df_min == 2:
        if v < 0.07: return "negligible"
        if v < 0.21: return "small"
        if v < 0.35: return "medium"
        return "large"
    # Simple general approximation for higher df_min
    elif df_min > 2:
        if v < 0.05: return "negligible"
        if v < 0.15: return "small"
        if v < 0.25: return "medium"
        return "large"
    else: # Should not happen with df_min >= 0
        return "unknown"


def calculate_cramers_v(chi2_stat: float, n: int, rows: int, cols: int) -> Optional[EffectSizeResult]:
    """
    Calculates Cramer's V effect size for contingency tables.

    Args:
        chi2_stat: The calculated Chi-Square statistic.
        n: Total number of observations.
        rows: Number of rows in the contingency table.
        cols: Number of columns in the contingency table.

    Returns:
        An EffectSizeResult object, or None if calculation is not possible
        (e.g., N<=0 or table is 1xN or Nx1).
    """
    if n <= 0 or rows <= 1 or cols <= 1:
        return None # Cannot calculate Cramer's V

    min_dim = min(rows - 1, cols - 1)
    if min_dim == 0:
         return None # Avoid division by zero if table is 1xN or Nx1 (df=0)

    try:
        # Phi-squared is chi2 / N
        # Cramer's V is sqrt(phi2 / min(R-1, C-1))
        v = math.sqrt(chi2_stat / (n * min_dim))
        # Ensure V is between 0 and 1 (due to potential floating point issues or specific edge cases)
        v = max(0.0, min(1.0, v))

        interpretation = interpret_cramers_v(v, min_dim)

        return EffectSizeResult(name="Cramér's V", value=v, interpretation=interpretation)
    except (ValueError, ZeroDivisionError):
        # This can happen if chi2_stat is negative (shouldn't per se, but float error),
        # or if N or min_dim is zero (handled above, but defensive).
        return None


def calculate_phi_coefficient(chi2_stat: float, n: int) -> Optional[EffectSizeResult]:
    """
    Calculates the Phi coefficient effect size for 2x2 tables.
    Note: For 2x2 tables, Phi is equivalent to Cramer's V with df_min=1.

    Args:
        chi2_stat: The calculated Chi-Square statistic.
        n: Total number of observations.

    Returns:
        An EffectSizeResult object, or None if calculation is not possible (N<=0).
    """
    if n <= 0:
        return None
    try:
        phi = math.sqrt(chi2_stat / n)
        # Ensure phi is between 0 and 1 for a 2x2 table (theoretically)
        phi = max(0.0, min(1.0, phi))

        interpretation = interpret_cramers_v(phi, df_min=1) # Use df_min=1 interpretation

        return EffectSizeResult(name="Phi Coefficient", value=phi, interpretation=interpretation)

    except (ValueError, ZeroDivisionError):
         # Handles sqrt of negative or division by zero
         return None

def calculate_chi_square_effect_size(chi2_stat: float, n: int, rows: int, cols: int) -> Optional[EffectSizeResult]:
    """
    Calculates the appropriate effect size (Phi for 2x2, Cramer's V otherwise).

    Args:
        chi2_stat: The calculated Chi-Square statistic.
        n: Total number of observations.
        rows: Number of rows in the contingency table.
        cols: Number of columns in the contingency table.

    Returns:
        An EffectSizeResult object, or None if not applicable/calculable.
    """
    if chi2_stat is None or chi2_stat < 0:
        # Cannot calculate effect size if Chi2 is invalid/negative
        return None

    if rows == 2 and cols == 2:
        return calculate_phi_coefficient(chi2_stat, n)
    elif rows > 1 and cols > 1:
        return calculate_cramers_v(chi2_stat, n, rows, cols)
    else:
        # Effect size (like V or Phi) is not typically calculated for 1xN or Nx1 tables
        # as df = 0, and the concept of association between two variables doesn't apply
        # in the same way. Goodness-of-Fit doesn't have these standard effect sizes.
        return None