# chi_square_math/constants.py

"""Constants used in Chi-Square calculations."""

# Minimum recommended expected frequency per cell for Chi-Square validity
MIN_EXPECTED_FREQUENCY = 5.0

# Threshold for significance in post-hoc residual analysis (absolute value)
# Corresponds roughly to p < 0.05 (two-tailed) for a standard normal distribution
RESIDUAL_SIGNIFICANCE_THRESHOLD = 1.96