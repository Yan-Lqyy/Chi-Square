# chi_square_math/analyzer.py

"""Main interface for performing Chi-Square tests."""

from typing import Optional, List, Union
import numpy as np
from scipy import stats

# Assuming other modules are in the same package directory
from . import utils
from . import effect_sizes
from . import post_hoc
from .constants import MIN_EXPECTED_FREQUENCY, RESIDUAL_SIGNIFICANCE_THRESHOLD
from .results import (
    ChiSquareResult, GoodnessOfFitResult, ContingencyTableResult, EffectSizeResult
)


def run_goodness_of_fit(
    observed_freqs: Union[List[int], np.ndarray],
    expected_freqs: Optional[Union[List[float], np.ndarray]] = None,
    expected_probs: Optional[Union[List[float], np.ndarray]] = None,
    alpha: float = 0.05 # Add alpha parameter
) -> GoodnessOfFitResult:
    """
    Performs a Chi-Square Goodness-of-Fit test.

    Tests if the observed frequencies of a single categorical variable match
    expected frequencies from a hypothesized distribution.

    Args:
        observed_freqs: A list or NumPy array of observed counts for each category.
                        Must contain non-negative integers.
        expected_freqs: Optional. A list or NumPy array of expected counts.
                        Must have the same length as observed_freqs. If sum differs,
                        expected frequencies will be scaled to match observed total.
        expected_probs: Optional. A list or NumPy array of expected probabilities/proportions.
                        Must have the same length as observed_freqs and sum to 1.0.
                        Cannot be used if expected_freqs is provided.
        alpha: The significance level (0 < alpha < 1) to use for determining significance.

    Returns:
        A GoodnessOfFitResult object containing test results and interpretation.

    Raises:
        ValueError: If inputs are invalid (e.g., wrong lengths, negative counts,
                    both expected_freqs and expected_probs provided, probabilities
                    don't sum to 1, invalid alpha).
        TypeError: If input types are incorrect.
    """
    # Input validation and parsing
    try:
        # Observed frequencies must be integers (counts)
        f_obs = utils.safe_to_ndarray(observed_freqs, dtype=np.int64)
        # Check if conversion truncated non-integers, though safe_to_ndarray should raise
        if not np.all(np.equal(f_obs, observed_freqs)):
             raise ValueError("Observed frequencies must be whole numbers (integers).")

    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid observed frequencies: {e}") from e

    n_obs = np.sum(f_obs)
    k = len(f_obs) # Number of categories

    if k <= 1:
        raise ValueError("Goodness of Fit test requires at least 2 categories.")

    # Validate alpha
    if not (0 < alpha < 1):
        raise ValueError("Significance level (alpha) must be between 0 and 1.")

    f_exp = None
    warnings = []

    # Determine expected frequencies
    if expected_freqs is not None and expected_probs is not None:
        raise ValueError("Provide either expected_freqs or expected_probs, not both.")

    elif expected_freqs is not None:
        try:
            # Expected can be floats
            f_exp = utils.safe_to_ndarray(expected_freqs, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid expected frequencies: {e}") from e

        if len(f_exp) != k:
            raise ValueError("Observed and expected frequencies must have the same number of categories.")

        # Option: Scale expected to match observed total if sums differ (common practice)
        sum_f_exp = np.sum(f_exp)
        if not np.isclose(sum_f_exp, n_obs):
             if np.isclose(sum_f_exp, 0):
                  # Avoid division by zero if expected sums to zero
                  raise ValueError("Sum of provided expected frequencies is zero, cannot scale.")
             warnings.append(f"Warning: Sum of expected frequencies ({sum_f_exp:.2f}) did not match sum of observed ({n_obs}). Scaling expected frequencies.")
             f_exp = (f_exp / sum_f_exp) * n_obs
             if np.any(np.isnan(f_exp)): # Should not happen if sum_f_exp was non-zero
                 raise ValueError("Scaling expected frequencies resulted in NaN values.")


    elif expected_probs is not None:
        try:
            probs = utils.safe_to_ndarray(expected_probs, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid expected probabilities: {e}") from e

        if len(probs) != k:
            raise ValueError("Observed frequencies and expected probabilities must have the same number of categories.")
        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError(f"Expected probabilities must sum to 1.0 (they sum to {np.sum(probs):.4f}).")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("Expected probabilities must be between 0 and 1.")

        f_exp = probs * n_obs

    else:
        # Default: Assume uniform distribution if no expected values are provided
        warnings.append("Assuming uniform distribution for expected frequencies as none were provided.")
        if n_obs == 0:
             f_exp = np.zeros(k, dtype=np.float64) # If no observations, expected are 0
        else:
             f_exp = np.full(k, fill_value=n_obs / k, dtype=np.float64)


    # Check for zero expected frequencies AFTER calculation/scaling
    # SciPy's chisquare handles cases where f_obs[i] > 0 and f_exp[i] = 0 by returning inf Chi2.
    # If f_obs[i] = 0 and f_exp[i] = 0, that category's contribution is 0 and SciPy handles it.
    zero_expected_mask = np.isclose(f_exp, 0)
    if np.any(zero_expected_mask):
         zero_indices = np.where(zero_expected_mask)[0]
         # Check if there are non-zero observed frequencies where expected are zero
         if np.any(f_obs[zero_indices] > 0):
              warnings.append(
                  "Warning: Found observed frequencies > 0 where expected frequencies are zero. "
                  "The Chi-Square statistic will be infinite. This likely indicates an error in your expected frequencies or data structure."
              )
         # If f_obs are also 0 where f_exp are 0, it's typically okay but still unusual.
         elif np.any(f_obs[zero_indices] == 0):
              warnings.append("Warning: Some expected frequencies are zero. Corresponding observed frequencies must also be zero for the test to be meaningful (which they are in this case).")


    # Perform Chi-Square calculation using SciPy
    # ddof=0 because we haven't estimated parameters from the data to get f_exp
    try:
        # SciPy handles filtering zero expected/observed cells internally for the calculation,
        # but we need the original f_exp for the assumption check.
        chi2, p = stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=0)
        # Ensure p-value is within [0, 1] in case of float issues
        p = max(0.0, min(1.0, p))

    except ValueError as e:
        # Catch errors from scipy (e.g., if total observed differs significantly from total expected after filtering)
        return GoodnessOfFitResult(
            observed=f_obs,
            expected=f_exp,
            alpha=alpha,
            assumption_warnings=warnings + [f"SciPy calculation error: {e}"],
            interpretation="Chi-Square calculation failed due to an internal error."
        )
    except Exception as e:
        # Catch any other unexpected errors during calculation
         return GoodnessOfFitResult(
            observed=f_obs,
            expected=f_exp,
            alpha=alpha,
            assumption_warnings=warnings + [f"An unexpected error occurred during calculation: {e}"],
            interpretation="Chi-Square calculation failed due to an unexpected error."
        )


    df = k - 1 # Degrees of freedom for GoF with no parameters estimated

    # Assumption Check: Expected Frequencies
    assumption_warnings = utils.check_expected_frequencies(f_exp, MIN_EXPECTED_FREQUENCY)
    warnings.extend(assumption_warnings)

    # Create result object
    result = GoodnessOfFitResult(
        chi2_statistic=float(chi2) if np.isfinite(chi2) else None, # Store as None if infinite/NaN
        p_value=float(p) if np.isfinite(p) else None,
        degrees_of_freedom=df if df >= 0 else None, # Ensure df is valid
        observed=f_obs,
        expected=f_exp,
        assumption_warnings=warnings,
        alpha=alpha # Pass alpha to result object
        # Interpretation is generated in __post_init__ based on alpha
    )

    # Refine interpretation if statistic/p-value are not finite
    if result.chi2_statistic is None or result.p_value is None:
         # Specific message if infinite Chi2 was likely the cause
         if np.isinf(chi2):
             result.interpretation = (
                 f"Chi-Square calculation resulted in an infinite statistic (χ² = inf). "
                 "This typically happens when an observed frequency is greater than 0 but the corresponding expected frequency is 0. "
                 "Check your data and expected values."
             )
         else:
              result.interpretation = "Chi-Square calculation resulted in undefined values. Check inputs and assumption warnings."


    return result


def run_contingency_test(
    contingency_table: Union[List[List[int]], np.ndarray],
    correction: bool = False, # Yates' correction for 2x2 tables
    test_type_hint: str = "Independence", # Or "Homogeneity" - affects interpretation wording
    alpha: float = 0.05 # Add alpha parameter
) -> ContingencyTableResult:
    """
    Performs a Chi-Square test of Independence or Homogeneity on a contingency table.

    Tests if two categorical variables are associated (Independence) or if the
    distribution of a variable is the same across populations (Homogeneity).

    Args:
        contingency_table: A 2D list or NumPy array representing the observed
                           frequencies (rows are one variable, columns the other).
                           Must contain non-negative integers.
        correction: Apply Yates' continuity correction (typically only for 2x2 tables).
                    Default is False.
        test_type_hint: Helps tailor the interpretation ('Independence' or 'Homogeneity').
        alpha: The significance level (0 < alpha < 1) to use for determining significance.

    Returns:
        A ContingencyTableResult object containing test results, effect size,
        post-hoc analysis (if applicable), and interpretation.

    Raises:
        ValueError: If inputs are invalid (e.g., not 2D, negative counts, table too small,
                    invalid alpha).
        TypeError: If input types are incorrect.
    """
    # Input validation and parsing
    try:
        # Observed frequencies must be integers (counts)
        observed_table = utils.safe_to_ndarray(contingency_table, dtype=np.int64)
        if not np.all(np.equal(observed_table, contingency_table)):
             raise ValueError("Observed frequencies in the table must be whole numbers (integers).")
        if observed_table.ndim != 2:
             raise ValueError("Contingency table must be 2-dimensional (a list of lists or 2D array).")
        if observed_table.shape[0] < 2 or observed_table.shape[1] < 2:
             raise ValueError("Contingency table must have at least 2 rows and 2 columns.")

    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid contingency table data: {e}") from e

    rows, cols = observed_table.shape
    n_total = np.sum(observed_table)

    # Validate alpha
    if not (0 < alpha < 1):
        raise ValueError("Significance level (alpha) must be between 0 and 1.")

    # Perform Chi-Square calculation using SciPy
    try:
        # chi2_contingency handles rows/cols summing to zero by returning NaN/Inf
        # lambda_='pearson' is the standard Chi-Square statistic
        chi2, p, df, expected = stats.chi2_contingency(
            observed=observed_table,
            correction=correction,
            lambda_='pearson' # Explicitly request Pearson Chi-Square
        )
        # Ensure p-value is within [0, 1]
        p = max(0.0, min(1.0, p))

    except ValueError as e:
         # SciPy can raise ValueError, e.g., if rows/columns sum to zero making expected NaNs/Infs
        return ContingencyTableResult(
            observed=observed_table,
            alpha=alpha,
            assumption_warnings=[f"SciPy calculation error during contingency test: {e}. Check if any rows or columns in your table sum to zero."],
            interpretation="Chi-Square calculation failed due to an internal error."
        )
    except Exception as e:
        # Catch any other unexpected errors during calculation
         return ContingencyTableResult(
            observed=observed_table,
            alpha=alpha,
            assumption_warnings=[f"An unexpected error occurred during calculation: {e}"],
            interpretation="Chi-Square calculation failed due to an unexpected error."
        )


    # Assumption Check: Expected Frequencies
    # Do this AFTER SciPy calculates expected counts under the null hypothesis
    warnings = utils.check_expected_frequencies(expected, MIN_EXPECTED_FREQUENCY)

    # Calculate Effect Size
    # Only calculate if Chi2 statistic is finite and non-negative
    effect_size_result = None
    if np.isfinite(chi2) and chi2 >= 0:
        effect_size_result = effect_sizes.calculate_chi_square_effect_size(
            chi2_stat=float(chi2), n=int(n_total), rows=rows, cols=cols # Ensure types are standard Python
        )
    elif not np.isfinite(chi2):
         warnings.append("Warning: Cannot calculate effect size because the Chi-Square statistic is undefined (NaN or Inf).")


    # Post-Hoc Analysis (Adjusted Residuals) - Run if the main test is significant at the chosen alpha
    # Also check if Chi2 is finite and df is valid (>0) as residuals aren't meaningful otherwise.
    residuals = None
    significant_residuals = []
    if df is not None and df > 0 and np.isfinite(chi2) and (p is not None and p < alpha):
        residuals = post_hoc.calculate_adjusted_residuals(observed_table, expected)
        if residuals is not None:
            significant_residuals = post_hoc.find_significant_residuals(
                residuals, RESIDUAL_SIGNIFICANCE_THRESHOLD
            )
        else:
            # This case implies calculate_adjusted_residuals returned None (e.g. shape mismatch, which shouldn't happen here)
            warnings.append("Warning: Could not calculate post-hoc residuals.")
    elif df is not None and df == 0:
         # df = 0 for 1xN or Nx1 tables, or if R=1, C=1 (caught by shape check)
         # Residuals aren't typically done for df=0 cases
         pass
    elif df is not None and df < 0:
         warnings.append(f"Warning: Invalid degrees of freedom ({df}), cannot perform post-hoc analysis.")
    elif not np.isfinite(chi2):
         warnings.append("Warning: Cannot perform post-hoc analysis because the Chi-Square statistic is undefined (NaN or Inf).")
    elif p is None or p >= alpha:
         # Test is not significant, residuals are exploratory
         # Calculate them anyway but the result object will note non-significance
         residuals = post_hoc.calculate_adjusted_residuals(observed_table, expected)
         if residuals is not None:
            significant_residuals = post_hoc.find_significant_residuals(
                residuals, RESIDUAL_SIGNIFICANCE_THRESHOLD # Use standard threshold regardless of overall alpha for consistency
            )
         else:
             warnings.append("Warning: Could not calculate post-hoc residuals.")


    # Create result object
    result = ContingencyTableResult(
        test_type=test_type_hint, # Use the hint for interpretation
        chi2_statistic=float(chi2) if np.isfinite(chi2) else None, # Store as None if infinite/NaN
        p_value=float(p) if np.isfinite(p) else None,
        degrees_of_freedom=df if df is not None and df >= 0 else None, # Ensure df is valid
        observed=observed_table,
        expected=expected, # expected from scipy is already a numpy array
        assumption_warnings=warnings,
        alpha=alpha, # Pass alpha to result object
        effect_size=effect_size_result,
        residuals=residuals, # numpy array or None
        significant_residuals=significant_residuals # list of tuples
        # Interpretation is generated/extended in __post_init__
    )

    # Refine interpretation if statistic/p-value are not finite
    if result.chi2_statistic is None or result.p_value is None:
         # Specific message if infinite Chi2 was likely the cause
         if np.isinf(chi2):
              result.interpretation = (
                 f"Chi-Square calculation resulted in an infinite statistic (χ² = inf). "
                 "This typically happens when an observed frequency is greater than 0 but the corresponding expected frequency is 0 in the contingency table. "
                 "Check your data for structural zeros (combinations that cannot logically occur) or sparse data."
             )
         else:
              result.interpretation = "Chi-Square calculation resulted in undefined values. Check inputs and assumption warnings."


    return result