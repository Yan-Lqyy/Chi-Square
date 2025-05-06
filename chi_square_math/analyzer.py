# chi_square_math/analyzer.py

"""Main interface for performing Chi-Square tests."""

from typing import Optional, List, Union

import numpy as np
from scipy import stats

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
) -> GoodnessOfFitResult:
    """
    Performs a Chi-Square Goodness-of-Fit test.

    Tests if the observed frequencies of a single categorical variable match
    expected frequencies from a hypothesized distribution.

    Args:
        observed_freqs: A list or NumPy array of observed counts for each category.
                        Must contain non-negative integers.
        expected_freqs: Optional. A list or NumPy array of expected counts.
                        Must have the same length as observed_freqs and sum to the
                        same total as observed_freqs.
        expected_probs: Optional. A list or NumPy array of expected probabilities/proportions.
                        Must have the same length as observed_freqs and sum to 1.0.
                        Cannot be used if expected_freqs is provided.

    Returns:
        A GoodnessOfFitResult object containing test results and interpretation.

    Raises:
        ValueError: If inputs are invalid (e.g., wrong lengths, negative counts,
                    both expected_freqs and expected_probs provided, probabilities
                    don't sum to 1).
        TypeError: If input types are incorrect.
    """
    try:
        f_obs = utils.safe_to_ndarray(observed_freqs, dtype=np.int_) # Observed should be counts
        if np.any(f_obs != observed_freqs): # Check if any were truncated/changed
             raise ValueError("Observed frequencies must be integers.")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid observed frequencies: {e}") from e

    n_obs = np.sum(f_obs)
    k = len(f_obs) # Number of categories

    if k <= 1:
        raise ValueError("Goodness of Fit test requires at least 2 categories.")

    f_exp = None
    warnings = []

    # Determine expected frequencies
    if expected_freqs is not None and expected_probs is not None:
        raise ValueError("Provide either expected_freqs or expected_probs, not both.")

    elif expected_freqs is not None:
        try:
            f_exp = utils.safe_to_ndarray(expected_freqs, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid expected frequencies: {e}") from e

        if len(f_exp) != k:
            raise ValueError("Observed and expected frequencies must have the same number of categories.")
        if not np.isclose(np.sum(f_exp), n_obs):
            # Option 1: Raise error
            # raise ValueError("Sum of expected frequencies must equal sum of observed frequencies.")
            # Option 2: Scale expected to match observed total (common practice)
            warnings.append("Warning: Sum of expected frequencies did not match sum of observed. Scaling expected frequencies.")
            f_exp = (f_exp / np.sum(f_exp)) * n_obs
            if np.any(np.isnan(f_exp)): # Check if scaling caused issues (e.g. sum(f_exp) was 0)
                raise ValueError("Could not scale expected frequencies (original sum might be zero).")


    elif expected_probs is not None:
        try:
            probs = utils.safe_to_ndarray(expected_probs, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid expected probabilities: {e}") from e

        if len(probs) != k:
            raise ValueError("Observed frequencies and expected probabilities must have the same number of categories.")
        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError("Expected probabilities must sum to 1.0.")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("Expected probabilities must be between 0 and 1.")

        f_exp = probs * n_obs

    else:
        # Default: Assume uniform distribution
        warnings.append("Assuming uniform distribution for expected frequencies as none were provided.")
        f_exp = np.full(k, fill_value=n_obs / k, dtype=np.float64)


    # Check for zero expected frequencies AFTER calculation/scaling
    if np.any(np.isclose(f_exp, 0)):
         zero_indices = np.where(np.isclose(f_exp, 0))[0]
         # SciPy chisquare might handle this, but good to warn or error
         # If f_obs[i] is also 0 where f_exp[i] is 0, the contribution is 0.
         # If f_obs[i] > 0 where f_exp[i] is 0, chi2 is infinite. SciPy handles this.
         if np.any(f_obs[zero_indices] > 0):
              warnings.append("Warning: Observed frequency > 0 exists where expected frequency is 0. Chi-square statistic will be infinite.")
         else:
              warnings.append("Warning: Some expected frequencies are zero. Corresponding observed frequencies must also be zero.")
              # We might need to filter these categories out before passing to scipy
              # For now, let scipy handle it but keep the warning.


    # Perform Chi-Square calculation using SciPy
    # ddof (delta degrees of freedom) is 0 because we haven't estimated parameters
    # from the data to *generate* the expected frequencies (they were given or uniform).
    # If, for example, we estimated the mean of a Poisson distribution from the data
    # to generate expected counts, ddof would be 1.
    try:
        # Filter out categories where expected is zero? Scipy handles infinite result if obs > 0.
        # Let's proceed carefully. If f_exp has zeros, chisquare might return inf/nan.
        chi2, p = stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=0)
    except ValueError as e:
        # Catch errors from scipy (e.g., if f_obs sum != f_exp sum after filtering,
        # or other internal issues)
        return GoodnessOfFitResult(
            observed=f_obs,
            expected=f_exp,
            assumption_warnings=warnings + [f"SciPy Error: {e}"],
            interpretation="Chi-Square calculation failed."
        )


    df = k - 1 - 0 # ddof is 0 here

    # Assumption Check: Expected Frequencies
    assumption_warnings = utils.check_expected_frequencies(f_exp, MIN_EXPECTED_FREQUENCY)
    warnings.extend(assumption_warnings)

    # Create result object
    result = GoodnessOfFitResult(
        chi2_statistic=float(chi2) if not np.isinf(chi2) and not np.isnan(chi2) else None, # Handle inf/nan
        p_value=float(p) if not np.isnan(p) else None,
        degrees_of_freedom=df,
        observed=f_obs,
        expected=f_exp,
        assumption_warnings=warnings
        # Interpretation is generated in __post_init__
    )

    # Add more specific interpretation if calculation failed
    if result.chi2_statistic is None or result.p_value is None:
         result.interpretation = "Chi-Square calculation resulted in undefined values (possibly due to zero expected frequencies with non-zero observed frequencies)."


    return result


def run_contingency_test(
    contingency_table: Union[List[List[int]], np.ndarray],
    correction: bool = False, # Yates' correction for 2x2 tables
    test_type_hint: str = "Independence" # Or "Homogeneity" - affects interpretation wording slightly
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

    Returns:
        A ContingencyTableResult object containing test results, effect size,
        post-hoc analysis (if applicable), and interpretation.

    Raises:
        ValueError: If inputs are invalid (e.g., not 2D, negative counts).
        TypeError: If input types are incorrect.
    """
    try:
        observed_table = utils.safe_to_ndarray(contingency_table, dtype=np.int_)
        if np.any(observed_table != contingency_table):
             raise ValueError("Observed frequencies in the table must be integers.")
        if observed_table.ndim != 2:
             raise ValueError("Contingency table must be 2-dimensional.")
        if observed_table.shape[0] < 2 or observed_table.shape[1] < 2:
             raise ValueError("Contingency table must have at least 2 rows and 2 columns.")

    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid contingency table: {e}") from e

    rows, cols = observed_table.shape
    n_total = np.sum(observed_table)

    # Perform Chi-Square calculation using SciPy
    try:
        # lambda_='likelihood' can be used for G-test instead
        chi2, p, df, expected = stats.chi2_contingency(
            observed=observed_table,
            correction=correction,
            lambda_=None
        )
    except ValueError as e:
         # This can happen if rows/columns sum to zero, leading to NaN expected values
        return ContingencyTableResult(
            test_type=test_type_hint,
            observed=observed_table,
            assumption_warnings=[f"SciPy Error during contingency calculation: {e}. Check for rows/columns summing to zero."],
            interpretation="Chi-Square calculation failed for contingency table."
        )

    # Assumption Check: Expected Frequencies
    warnings = utils.check_expected_frequencies(expected, MIN_EXPECTED_FREQUENCY)

    # Calculate Effect Size
    effect_size_result = effect_sizes.calculate_chi_square_effect_size(
        chi2_stat=chi2, n=n_total, rows=rows, cols=cols
    )

    # Post-Hoc Analysis (Adjusted Residuals) - Run if the main test is significant
    residuals = None
    significant_residuals = []
    if p < 0.05: # Only makes sense if the overall test is significant
        residuals = post_hoc.calculate_adjusted_residuals(observed_table, expected)
        if residuals is not None:
            significant_residuals = post_hoc.find_significant_residuals(
                residuals, RESIDUAL_SIGNIFICANCE_THRESHOLD
            )

    # Create result object
    result = ContingencyTableResult(
        test_type=test_type_hint,
        chi2_statistic=float(chi2),
        p_value=float(p),
        degrees_of_freedom=df,
        observed=observed_table,
        expected=expected,
        assumption_warnings=warnings,
        effect_size=effect_size_result,
        residuals=residuals,
        significant_residuals=significant_residuals
        # Interpretation is generated/extended in __post_init__
    )

    return result