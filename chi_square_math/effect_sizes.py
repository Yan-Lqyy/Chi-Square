# chi_square_math/effect_sizes.py

"""Functions to calculate effect sizes for Chi-Square tests."""

import numpy as np
import math
from typing import Optional

from .results import EffectSizeResult

def interpret_cramers_v(v: float, df: int) -> str:
    """Provides a rough interpretation of Cramer's V based on Cohen's guidelines (adapted)."""
    # Cohen's benchmarks (for V) depend on df* = min(R-1, C-1)
    # thresholds = {1: (0.10, 0.30), 2: (0.07, 0.21), 3: (0.06, 0.17), ...}
    # Simplified approach:
    if df == 1: # Equivalent to Phi
        if v < 0.10: return "negligible"
        if v < 0.30: return "small"
        if v < 0.50: return "medium"
        return "large"
    elif df == 2:
        if v < 0.07: return "negligible"
        if v < 0.21: return "small"
        if v < 0.35: return "medium"
        return "large"
    else: # General approximation for higher df
        if v < 0.05: return "negligible"
        if v < 0.15: return "small"
        if v < 0.25: return "medium"
        return "large"


def calculate_cramers_v(chi2_stat: float, n: int, rows: int, cols: int) -> Optional[EffectSizeResult]:
    """
    Calculates Cramer's V effect size for contingency tables.

    Args:
        chi2_stat: The calculated Chi-Square statistic.
        n: Total number of observations.
        rows: Number of rows in the contingency table.
        cols: Number of columns in the contingency table.

    Returns:
        An EffectSizeResult object, or None if calculation is not possible.
    """
    if n <= 0 or rows <= 1 or cols <= 1:
        return None # Cannot calculate

    min_dim = min(rows - 1, cols - 1)
    if min_dim == 0:
         return None # Avoid division by zero if table is 1xN or Nx1

    try:
        phi2 = chi2_stat / n
        v = math.sqrt(phi2 / min_dim)
        # Ensure V is between 0 and 1 (due to potential floating point issues)
        v = max(0.0, min(1.0, v))
        df_effect_size = min_dim
        interpretation = interpret_cramers_v(v, df_effect_size)
        return EffectSizeResult(name="Cramér's V", value=v, interpretation=interpretation)
    except (ValueError, ZeroDivisionError):
        return None # Handles sqrt of negative or division by zero


def calculate_phi_coefficient(chi2_stat: float, n: int) -> Optional[EffectSizeResult]:
    """
    Calculates the Phi coefficient effect size for 2x2 tables.
    Note: For 2x2 tables, Phi is equivalent to Cramer's V.

    Args:
        chi2_stat: The calculated Chi-Square statistic.
        n: Total number of observations.

    Returns:
        An EffectSizeResult object, or None if calculation is not possible.
    """
    if n <= 0:
        return None
    try:
        phi = math.sqrt(chi2_stat / n)
        # Ensure phi is between 0 and 1 (theoretically max is 1 for 2x2)
        phi = max(0.0, min(1.0, phi))
        interpretation = interpret_cramers_v(phi, df=1) # Use df=1 interpretation
        return EffectSizeResult(name="Phi Coefficient", value=phi, interpretation=interpretation)

    except (ValueError, ZeroDivisionError):
         return None

def calculate_chi_square_effect_size(chi2_stat: float, n: int, rows: int, cols: int) -> Optional[EffectSizeResult]:
    """
    Calculates the appropriate effect size (Phi or Cramer's V).

    Args:
        chi2_stat: The calculated Chi-Square statistic.
        n: Total number of observations.
        rows: Number of rows in the contingency table.
        cols: Number of columns in the contingency table.

    Returns:
        An EffectSizeResult object, or None if not applicable/calculable.
    """
    if rows == 2 and cols == 2:
        return calculate_phi_coefficient(chi2_stat, n)
    elif rows > 1 and cols > 1:
        return calculate_cramers_v(chi2_stat, n, rows, cols)
    else:
        # Effect size not typically calculated for 1xN or Nx1 tables
        return None