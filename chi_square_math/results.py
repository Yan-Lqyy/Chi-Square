# chi_square_math/results.py

"""Dataclasses for structuring Chi-Square analysis results."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

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
    chi2_statistic: Optional[float] = None
    p_value: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    observed: Optional[np.ndarray] = field(default=None, repr=False) # Avoid printing large arrays
    expected: Optional[np.ndarray] = field(default=None, repr=False)
    assumption_warnings: List[str] = field(default_factory=list)
    significant: Optional[bool] = None # Based on alpha=0.05
    interpretation: str = ""

    def __post_init__(self):
        if self.p_value is not None:
            self.significant = self.p_value < 0.05
        # Basic interpretation can be added here or in the analyzer
        self._generate_basic_interpretation()

    def _generate_basic_interpretation(self):
        if self.significant is None:
            self.interpretation = "Result significance could not be determined (missing p-value)."
            return

        significance_text = "statistically significant" if self.significant else "not statistically significant"
        p_value_text = f"p = {self.p_value:.4f}" if self.p_value is not None else "p-value not available"
        stat_text = f"χ²({self.degrees_of_freedom}) = {self.chi2_statistic:.3f}" if self.chi2_statistic is not None and self.degrees_of_freedom is not None else "statistic not available"

        if self.test_type == "Goodness of Fit":
            conclusion = "observed frequencies differ significantly from the expected frequencies" if self.significant else "no significant difference between observed and expected frequencies"
        elif self.test_type == "Independence" or self.test_type == "Homogeneity":
            conclusion = "there is a statistically significant association between the variables" if self.significant else "there is no statistically significant association between the variables"
        else:
             conclusion = "the result interpretation is unclear for this test type"

        self.interpretation = (
            f"The Chi-Square test for {self.test_type} was {significance_text} "
            f"({stat_text}, {p_value_text}). "
            f"This suggests that {conclusion} (at alpha = 0.05)."
        )
        if self.assumption_warnings:
            self.interpretation += " NOTE: " + " ".join(self.assumption_warnings)


@dataclass
class GoodnessOfFitResult(ChiSquareResult):
    """Specific results for Goodness of Fit test."""
    test_type: str = "Goodness of Fit"
    # GoF doesn't typically have standard effect sizes like Cramer's V
    # or post-hoc residuals in the same way as contingency tables.

@dataclass
class ContingencyTableResult(ChiSquareResult):
    """Specific results for Independence or Homogeneity tests."""
    test_type: str = "Independence / Homogeneity" # Specify more clearly in analyzer if possible
    effect_size: Optional[EffectSizeResult] = None
    residuals: Optional[np.ndarray] = field(default=None, repr=False) # Adjusted or Standardized Residuals
    significant_residuals: Optional[List[Tuple[int, int, float]]] = field(default_factory=list) # (row, col, residual_value)

    def __post_init__(self):
        """Override or extend base interpretation."""
        super().__post_init__() # Call base interpretation generation
        if self.effect_size:
            self.interpretation += f" The effect size ({self.effect_size.name} = {self.effect_size.value:.3f}) indicates a {self.effect_size.interpretation} effect."
        if self.significant and self.significant_residuals:
             self.interpretation += f" Post-hoc analysis (adjusted residuals) identified significant contributions from cell(s): {self._format_sig_residuals()}."

    def _format_sig_residuals(self) -> str:
        """Helper to format significant residual locations."""
        return ", ".join([f"(Row {r}, Col {c}, Residual={res:.2f})" for r, c, res in self.significant_residuals])