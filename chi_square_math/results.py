# chi_square_math/results.py

"""Dataclasses for structuring Chi-Square analysis results."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

# Assuming effect_sizes.py and post_hoc.py are in the same package directory
from . import effect_sizes
from . import post_hoc


@dataclass
class EffectSizeResult:
    """Holds effect size information."""
    name: str
    value: float
    interpretation: str = "" # Optional textual interpretation (e.g., "small", "medium", "large")

@dataclass
class ChiSquareResult:
    """Base class for Chi-Square test results."""
    test_type: str
    alpha: float = 0.05 # Store the significance level used
    chi2_statistic: Optional[float] = None
    p_value: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    observed: Optional[np.ndarray] = field(default=None, repr=False) # Avoid printing large arrays in repr
    expected: Optional[np.ndarray] = field(default=None, repr=False)
    assumption_warnings: List[str] = field(default_factory=list)
    significant: Optional[bool] = None
    interpretation: str = ""

    def __post_init__(self):
        """Post-initialization logic, including determining significance and generating basic interpretation."""
        # Determine significance based on the provided alpha
        if self.p_value is not None and self.alpha is not None:
            self.significant = self.p_value < self.alpha
        # Generate interpretation after significance is determined
        self._generate_basic_interpretation()

    def _generate_basic_interpretation(self):
        """Generates a basic textual interpretation based on the results."""
        if self.significant is None:
            self.interpretation = "Result significance could not be determined (missing p-value or alpha)."
            return

        # Use self.alpha in the interpretation text for clarity
        significance_text = "statistically significant" if self.significant else "not statistically significant"
        p_value_text = f"p = {self.p_value:.4f}" if self.p_value is not None else "p-value not available"
        stat_text = f"χ²({self.degrees_of_freedom}) = {self.chi2_statistic:.3f}" if self.chi2_statistic is not None and self.degrees_of_freedom is not None else "statistic not available"
        alpha_text = f"at alpha = {self.alpha:.3f}"

        # Tailor the conclusion based on the test type and significance
        if self.test_type == "Goodness of Fit":
            conclusion = "observed frequencies differ significantly from the expected frequencies" if self.significant else "no statistically significant difference between observed and expected frequencies was detected"
        elif self.test_type in ["Independence", "Homogeneity", "Independence / Homogeneity"]:
            # Use the more specific hint if available, otherwise the base combined one
            test_label = self.test_type if self.test_type != "Independence / Homogeneity" else "Independence/Homogeneity"
            conclusion = f"there is a statistically significant association between the variables" if self.significant else f"no statistically significant association between the variables was detected" # Wording works for both independence and homogeneity interpretation
        else:
             conclusion = "the result interpretation is unclear for this test type" # Should not happen if test_type is set correctly

        self.interpretation = (
            f"The Chi-Square test for {self.test_type} was {significance_text} "
            f"({stat_text}, {p_value_text}, {alpha_text}). "
            f"This suggests that {conclusion}."
        )

        # Append assumption warnings clearly
        if self.assumption_warnings:
            # Convert list of strings to HTML list for better display
            warnings_html = "<ul>" + "".join([f"<li>{w}</li>" for w in self.assumption_warnings]) + "</ul>"
            self.interpretation += f"<br><strong>NOTE ON ASSUMPTIONS:</strong>{warnings_html}"


@dataclass
class GoodnessOfFitResult(ChiSquareResult):
    """Specific results for Goodness of Fit test."""
    test_type: str = "Goodness of Fit"
    # GoF doesn't typically have standard effect sizes like Cramer's V
    # or post-hoc residuals in the same way as contingency tables.


@dataclass
class ContingencyTableResult(ChiSquareResult):
    """Specific results for Independence or Homogeneity tests."""
    # Test type hint will be passed during creation
    test_type: str = field(default="Independence / Homogeneity") # Default, but should be overridden
    effect_size: Optional[EffectSizeResult] = None
    residuals: Optional[np.ndarray] = field(default=None, repr=False) # Adjusted or Standardized Residuals
    significant_residuals: Optional[List[Tuple[int, int, float]]] = field(default_factory=list) # (row, col, residual_value)

    def __post_init__(self):
        """Extend base interpretation for contingency tests."""
        super().__post_init__() # Call base __post_init__ first (sets significance, basic interpretation)

        # Add effect size interpretation if available
        if self.effect_size:
            self.interpretation += f"<br>The effect size ({self.effect_size.name} = {self.effect_size.value:.3f}) indicates a <strong>{self.effect_size.interpretation}</strong> effect."

        # Add post-hoc interpretation if applicable and significant
        # Residuals are only calculated if overall test is significant
        if self.significant and self.residuals is not None:
             residual_threshold_text = f"|{post_hoc.RESIDUAL_SIGNIFICANCE_THRESHOLD:.2f}|" # Use threshold from constants
             if self.significant_residuals:
                 self.interpretation += f"<br>Post-hoc analysis (adjusted residuals &gt; {residual_threshold_text}) identified significant contributions from cell(s): {self._format_sig_residuals()}."
             else:
                  self.interpretation += f"<br>Post-hoc analysis (adjusted residuals) did not identify individual cells exceeding the standard significance threshold ({residual_threshold_text})."
        elif not self.significant and self.residuals is not None:
             # If residuals were calculated despite not significant overall (analyzer logic)
             self.interpretation += f"<br>The overall test was not statistically significant at α = {self.alpha:.3f}. While adjusted residuals are shown, interpret individual cell contributions with caution as exploratory."


    def _format_sig_residuals(self) -> str:
        """Helper to format significant residual locations for interpretation string."""
        # Format as (Row i, Col j, Residual=val)
        return ", ".join([f"(Row {r+1}, Col {c+1}, Residual={res:.2f})" for r, c, res in self.significant_residuals])
        # Adding +1 to r and c for 1-based indexing in output