# chi_square_math/constants.py

"""Constants used in Chi-Square calculations."""

# Minimum recommended expected frequency per cell for Chi-Square validity
MIN_EXPECTED_FREQUENCY = 5.0

# Threshold for significance in post-hoc residual analysis (absolute value)
# Corresponds roughly to p < 0.05 (two-tailed) for a standard normal distribution
# A common threshold for adjusted residuals is Z > 1.96 for p < 0.05,
# or often > 2.0 or 3.0 are used for clearer indication of "strong" contributions.
# We'll use 1.96 as the statistical threshold.
RESIDUAL_SIGNIFICANCE_THRESHOLD = 1.96