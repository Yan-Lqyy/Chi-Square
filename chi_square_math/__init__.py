# chi_square_math/__init__.py

"""
Chi-Square Math Module

Provides functions for performing Chi-Square tests (Goodness-of-Fit, Independence, Homogeneity)
along with effect sizes and post-hoc analyses using SciPy and NumPy.
"""

# Expose the main analysis functions directly when importing the package
from .analyzer import run_goodness_of_fit, run_contingency_test

# Expose result classes for type hinting and checking outside the package
from .results import ChiSquareResult, GoodnessOfFitResult, ContingencyTableResult, EffectSizeResult

# Optionally expose constants or specific sub-module functions if needed directly
from .constants import MIN_EXPECTED_FREQUENCY, RESIDUAL_SIGNIFICANCE_THRESHOLD
# from .effect_sizes import calculate_cramers_v
# from .post_hoc import calculate_adjusted_residuals

__all__ = [
    "run_goodness_of_fit",
    "run_contingency_test",
    "ChiSquareResult",
    "GoodnessOfFitResult",
    "ContingencyTableResult",
    "EffectSizeResult",
    "MIN_EXPECTED_FREQUENCY",
    "RESIDUAL_SIGNIFICANCE_THRESHOLD",
]