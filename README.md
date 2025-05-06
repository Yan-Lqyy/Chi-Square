# Chi-Square Inference Web Application

## Overview

This repository contains a web application built with Python and Flask designed for performing common Chi-Square (χ²) statistical tests. It provides an interactive interface for users to input categorical data and obtain detailed results, including the Chi-Square statistic, p-value, degrees of freedom, effect sizes, post-hoc analysis, and visualizations of the Chi-Square distribution. The application emphasizes statistical correctness and provides educational context for the tests performed.

## Mathematical Foundation

The application implements the following core statistical concepts and tests based on the Chi-Square distribution:

### 1. Chi-Square Goodness-of-Fit (GoF) Test
*   **Purpose:** Determines if the observed frequency distribution of a single categorical variable significantly differs from a hypothesized (expected) frequency distribution.
*   **Hypotheses:**
    *   *H₀:* The observed frequencies match the expected frequencies.
    *   *H₁:* The observed frequencies do not match the expected frequencies.
*   **Input:** A single vector of observed counts and optionally a corresponding vector of expected counts or probabilities. If expectations are omitted, a uniform distribution is assumed.

### 2. Chi-Square Test of Independence
*   **Purpose:** Determines if there is a statistically significant association between two categorical variables within a single population.
*   **Hypotheses:**
    *   *H₀:* The two variables are independent (no association).
    *   *H₁:* The two variables are dependent (there is an association).
*   **Input:** A 2D contingency table (cross-tabulation) of observed counts.

### 3. Chi-Square Test of Homogeneity
*   **Purpose:** Determines if the distribution of a single categorical variable is the same across two or more distinct populations or groups.
*   **Hypotheses:**
    *   *H₀:* The distribution of the variable is homogeneous across all groups.
    *   *H₁:* The distribution of the variable is not homogeneous across all groups.
*   **Input:** A 2D contingency table where rows typically represent groups and columns represent the categories of the variable being compared.
*   **Note:** Mathematically, the calculation is identical to the Test of Independence, but the study design and interpretation differ. The application allows specifying the context (Independence/Homogeneity) for clearer interpretation.

### Core Calculations
*   **Chi-Square Statistic (χ²):** Calculated using the formula `χ² = Σ [ (Observed - Expected)² / Expected ]`, summing over all categories (GoF) or cells (Contingency). This is primarily computed using `scipy.stats.chisquare` and `scipy.stats.chi2_contingency`.
*   **Degrees of Freedom (df):**
    *   GoF: `df = k - 1 - p` (k = number of categories, p = number of parameters estimated from data, usually 0 here).
    *   Independence/Homogeneity: `df = (rows - 1) * (cols - 1)`.
*   **P-value:** Determined by comparing the calculated χ² statistic to the Chi-Square distribution with the corresponding df. Calculated using `scipy.stats`.
*   **Expected Frequencies:** Calculated based on the null hypothesis (e.g., uniform distribution for GoF default, or based on marginal totals for contingency tests).

### Statistical Enhancements & Considerations
*   **Adjustable Significance Level (Alpha, α):** Users can select standard alpha levels (0.10, 0.05, 0.025, 0.01, 0.001) to define the threshold for statistical significance (p < α).
*   **Assumption Checks:** The application checks the critical assumption that expected cell frequencies are sufficiently large (typically ≥ 5), providing warnings if violated, as low expected counts can make the Chi-Square approximation inaccurate.
*   **Effect Sizes:**
    *   **Phi Coefficient (Φ):** Calculated for 2x2 contingency tables to measure the strength of association (`sqrt(χ² / N)`).
    *   **Cramér's V:** Calculated for contingency tables larger than 2x2 (`sqrt(χ² / (N * min(rows-1, cols-1)))`). Provides a measure of association strength normalized between 0 and 1. Interpretations (small, medium, large) are provided based on common benchmarks adapted for df.
*   **Post-Hoc Analysis (Adjusted Residuals):** For significant contingency table results, adjusted residuals (`(O - E) / sqrt(E * (1 - row_prop) * (1 - col_prop))`, or often the simpler standardized residual `(O - E) / sqrt(E)`) are calculated for each cell. Residuals with absolute values greater than a threshold (e.g., ~1.96 for α ≈ 0.05) indicate which specific cells contribute significantly to the overall result.
*   **Yates' Continuity Correction:** Offered as an option for 2x2 contingency tables, though its use is sometimes debated.

## Code Implementation and Design

The application follows a modular design pattern, separating the statistical logic from the web presentation layer.

### Technology Stack
*   **Backend:** Python 3
*   **Web Framework:** Flask
*   **Statistical Computation:** SciPy (`scipy.stats`), NumPy
*   **Plotting:** Plotly
*   **Templating:** Jinja2 (via Flask)
*   **Frontend:** HTML, CSS (no JavaScript framework)

### Project Structure
*   **`app.py`:** The main Flask application file. Handles routing, request parsing, calling the core logic, and rendering templates. Uses Flask `flash` for user feedback and a context processor to inject the current year.
*   **`plotting.py`:** Contains the `plot_chi_square_curve` function using Plotly to generate interactive visualizations of the Chi-Square distribution, critical value, and test statistic.
*   **`chi_square_math/` (Core Logic Package):**
    *   **`__init__.py`:** Makes the package importable and exposes key functions/classes.
    *   **`analyzer.py`:** Contains the main functions (`run_goodness_of_fit`, `run_contingency_test`) that orchestrate the calculations by calling utility, effect size, and post-hoc functions. Relies heavily on `scipy.stats` for core computations.
    *   **`results.py`:** Defines `dataclass` structures (`ChiSquareResult`, `GoodnessOfFitResult`, `ContingencyTableResult`, `EffectSizeResult`) to hold and organize the results from the analyses in a structured way. Includes `__post_init__` logic for basic interpretation generation based on results and alpha.
    *   **`utils.py`:** Provides helper functions for robust input parsing (converting string inputs to NumPy arrays) and common statistical checks (like the expected frequency assumption).
    *   **`effect_sizes.py`:** Implements functions (`calculate_phi_coefficient`, `calculate_cramers_v`, `interpret_cramers_v`) dedicated to calculating and interpreting effect sizes.
    *   **`post_hoc.py`:** Implements functions (`calculate_adjusted_residuals`, `find_significant_residuals`) for performing post-hoc analysis on contingency tables.
    *   **`constants.py`:** Defines shared constants like `MIN_EXPECTED_FREQUENCY` and `RESIDUAL_SIGNIFICANCE_THRESHOLD` to ensure consistency.
*   **`templates/`:** Contains Jinja2 HTML templates.
    *   `base.html`: Base template providing overall structure, navigation, CSS links, and Plotly JS CDN link. Uses template inheritance.
    *   Form templates (`gof_form.html`, `contingency_form.html`, `critical_value_form.html`): Input forms for users, repopulated on error.
    *   `results.html`: Unified template to display detailed results, interpretation, assumption warnings, tables, effect sizes, residuals, and the Plotly plot.
    *   `learn_more.html`: Static content page explaining Chi-Square concepts.
    *   `index.html`: Application home page.
*   **`static/`:** Contains static assets.
    *   `style.css`: CSS file for styling the application, aiming for a clean and modern interface using CSS variables and basic Flexbox layouts.

### Design Principles
*   **Separation of Concerns:** The statistical calculation logic (`chi_square_math/`) is kept separate from the Flask web handling (`app.py`, `plotting.py`, `templates/`).
*   **Modularity:** Specific calculations (effect sizes, post-hoc) are broken into dedicated modules.
*   **Data Structures:** Python `dataclasses` are used in `results.py` to provide clear, structured objects for passing results between the backend logic and the frontend templates.
*   **Reliance on Scientific Libraries:** Leverages the well-tested and robust implementations in `SciPy` and `NumPy` for core statistical computations and array handling.
*   **User Feedback:** Uses Flask's `flash` mechanism for error messages and warnings. Assumption warnings are clearly presented with results.
*   **Interactivity:** Uses Plotly for interactive data visualization embedded directly in the results page.

## Features Summary

*   Execution of Chi-Square Goodness-of-Fit test with user-defined or uniform expectations.
*   Execution of Chi-Square Test of Independence / Homogeneity from contingency table input.
*   User-selectable significance level (alpha).
*   Calculation and display of Chi-Square statistic, degrees of freedom, and p-value.
*   Clear interpretation of results based on chosen alpha.
*   Calculation and interpretation of effect sizes (Phi, Cramér's V).
*   Post-hoc analysis using adjusted residuals for significant contingency table results.
*   Highlighting of significant residuals in the output table.
*   Interactive Plotly visualization showing the relevant Chi-Square distribution, critical value, and calculated statistic.
*   Warnings for violations of the expected frequency assumption.
*   Dedicated tool for looking up Chi-Square critical values.
*   Educational "Learn More" page explaining the underlying concepts.
*   Responsive design for usability on different screen sizes.

## Limitations

*   Does not implement alternatives like Fisher's Exact Test for small expected frequencies (though warnings are provided).
*   Input parsing relies on specific delimiters (comma, newline/semicolon). More complex input formats are not supported.
*   Assumes independence of observations, which must be ensured by the user's study design.

---
