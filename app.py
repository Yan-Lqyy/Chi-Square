import os
from flask import Flask, render_template, request, flash, redirect, url_for
# Correct Markup import
from markupsafe import Markup
import numpy as np
# Need scipy.stats here for critical value lookup route
from scipy import stats
import logging # Import logging module
import datetime # Import the datetime module

# Import the chi_square_math package
try:
    from chi_square_math import (
        run_goodness_of_fit,
        run_contingency_test,
        GoodnessOfFitResult,
        ContingencyTableResult
    )
    # Import the new plotting function
    from plotting import plot_chi_square_curve
    CHI_MATH_LOADED = True
    # Set up basic logging for app.py
    logging.basicConfig(level=logging.INFO)
    app_logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import required libraries: {e}")
    print("Please ensure 'chi_square_math' directory is in the correct location and packages like Flask, numpy, scipy, plotly, markupsafe are installed (`pip install Flask numpy scipy plotly pandas markupsafe`).")
    CHI_MATH_LOADED = False
    app_logger = logging.getLogger(__name__) # Still get logger instance
    app_logger.critical("Calculation libraries failed to load.")
    # Define dummy classes/functions if import fails, so app can start but show error
    class MockResult:
        interpretation = "Calculation libraries failed to load. Functionality is unavailable."
        assumption_warnings = ["Calculation libraries failed to load."]
        degrees_of_freedom = None
        chi2_statistic = None
        alpha = 0.05
        p_value = None
        effect_size = None
        residuals = None
        significant_residuals = None
        observed = None
        expected = None

    run_goodness_of_fit = lambda **kwargs: MockResult()
    run_contingency_test = lambda **kwargs: MockResult()
    plot_chi_square_curve = lambda **kwargs: "<p class='flash danger'>Plotting library failed to load.</p>"
    GoodnessOfFitResult = MockResult
    ContingencyTableResult = MockResult


app = Flask(__name__)

# IMPORTANT: Set a secret key for Flask sessions (used for flashing messages)
# Use environment variable in production, generate a secure random key
app.config['SECRET_KEY'] = os.urandom(24).hex() # Generate a random key for session management

# --- Add enumerate to Jinja2 globals for loops in templates ---
app.jinja_env.globals['enumerate'] = enumerate

# --- Context processor to add current year to all templates ---
@app.context_processor
def inject_current_year():
    """Injects the current year into all templates."""
    return {'current_year': datetime.datetime.now().year}
# -----------------------------------------------------------

# --- Helper Functions ---
# Move parsing functions here as they are part of the Flask app's request handling
def parse_int_list(input_str: str, delimiter=',') -> list[int]:
    """Parses a delimited string into a list of integers."""
    if not input_str:
        return []
    try:
        return [int(x.strip()) for x in input_str.split(delimiter) if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid input: Could not convert all items separated by '{delimiter}' to integers.")

def parse_float_list(input_str: str, delimiter=',') -> list[float]:
    """Parses a delimited string into a list of floats."""
    if not input_str:
        return []
    try:
        return [float(x.strip()) for x in input_str.split(delimiter) if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid input: Could not convert all items separated by '{delimiter}' to numbers.")

def parse_contingency_table(input_str: str) -> list[list[int]]:
    """Parses a multi-line/semicolon-separated string into a 2D list of integers."""
    table = []
    if not input_str:
        return []
    # Handle both newline and semicolon as row separators
    lines = input_str.replace(';', '\n').splitlines()
    first_row_len = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue # Skip empty lines
        try:
            row = [int(x.strip()) for x in line.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid input in row {i+1}: Could not convert all items separated by ',' to integers.")

        if not row:
             # This means a line had delimiters but no valid numbers, e.g., "1,,"
             # Or a line was just commas. Treat as invalid.
             raise ValueError(f"Invalid input: Row {i+1} is empty or contains only delimiters after parsing.")

        if first_row_len is None:
            first_row_len = len(row)
        elif len(row) != first_row_len:
            raise ValueError(f"Invalid input: Row {i+1} has {len(row)} columns, but previous rows had {first_row_len}.")

        table.append(row)

    if not table:
         raise ValueError("Invalid input: No valid data rows found for contingency table.")

    return table

def parse_alpha(alpha_str: str) -> float:
    """Parses and validates the alpha value."""
    try:
        alpha = float(alpha_str)
        if not (0 < alpha < 1):
            raise ValueError("Significance level (alpha) must be between 0 and 1 (exclusive).")
        return alpha
    except (ValueError, TypeError):
        raise ValueError("Invalid significance level (alpha). Please select a valid option.") # Guide user to select from dropdown

# --- Routes ---

@app.route('/')
def index():
    """Home page linking to tests and resources."""
    if not CHI_MATH_LOADED:
        # If critical error on load, flash on index
        flash("Critical Error: Calculation libraries failed to load. Please check server logs.", "danger")
    return render_template('index.html')

@app.route('/learn-more')
def learn_more():
    """Displays explanations about Chi-Square tests."""
    return render_template('learn_more.html')

@app.route('/critical-value-lookup', methods=['GET', 'POST'])
def critical_value_lookup():
    """Page to look up Chi-Square critical values."""
    if not CHI_MATH_LOADED:
        flash("Calculation libraries failed to load. This feature is unavailable.", "danger")
        return redirect(url_for('index'))

    result_value = None
    form_data = {}
    if request.method == 'POST':
        form_data = request.form
        try:
            df_str = request.form.get('df')
            alpha_str = request.form.get('alpha', '0.05') # Default to 0.05 if not provided

            if not df_str:
                raise ValueError("Degrees of Freedom (df) is required.")

            try:
                df = int(df_str)
                if df <= 0:
                    raise ValueError("Degrees of Freedom (df) must be a positive integer.")
            except ValueError:
                 raise ValueError("Degrees of Freedom (df) must be a whole number.")


            alpha = parse_alpha(alpha_str)

            # Calculate critical value using scipy.stats
            # ppf(q, df) gives the value x such that P(X <= x) = q
            # We want P(X > critical_value) = alpha, which is P(X <= critical_value) = 1 - alpha
            try:
                critical_value = stats.chi2.ppf(1 - alpha, df)
                if not np.isfinite(critical_value):
                     raise ValueError("Calculated critical value is not a finite number. Check df and alpha.")
            except Exception as e:
                 raise ValueError(f"Error calculating critical value: {e}")


            result_value = {
                "df": df,
                "alpha": alpha,
                "critical_value": float(critical_value) # Ensure it's a standard float
            }
            # Flash a success message
            flash(f"Critical value found: {result_value['critical_value']:.4f}", "success")

        except ValueError as e:
            flash(f"Input Error: {e}", "danger")
            app_logger.warning(f"Critical Value Lookup Input Error: {e}")
        except Exception as e:
            flash("An unexpected error occurred during lookup.", "danger")
            app_logger.error(f"Critical Value Lookup Unexpected Error: {e}", exc_info=True)

    return render_template('critical_value_form.html', result_value=result_value, form_data=form_data)


@app.route('/goodness-of-fit', methods=['GET', 'POST'])
def goodness_of_fit_view():
    """Handles the Goodness of Fit test form and results."""
    if not CHI_MATH_LOADED:
        flash("Calculation libraries failed to load. This feature is unavailable.", "danger")
        return redirect(url_for('index'))

    form_data = {}
    if request.method == 'POST':
        form_data = request.form # Keep form data to refill on error
        result = None
        plot_div = None
        try:
            # --- Parse Inputs ---
            observed_str = request.form.get('observed_freqs', '').strip()
            expected_freqs_str = request.form.get('expected_freqs', '').strip()
            expected_probs_str = request.form.get('expected_probs', '').strip()
            alpha_str = request.form.get('alpha', '0.05') # Default alpha

            if not observed_str:
                raise ValueError("Observed frequencies are required.")

            observed = parse_int_list(observed_str)
            if not observed:
                 raise ValueError("Observed frequencies cannot be empty after parsing.")

            alpha = parse_alpha(alpha_str) # Validate alpha

            expected_freqs = None
            if expected_freqs_str:
                expected_freqs = parse_float_list(expected_freqs_str)
                if len(expected_freqs) != len(observed):
                     raise ValueError(f"Number of expected frequencies ({len(expected_freqs)}) must match number of observed frequencies ({len(observed)}).")

            expected_probs = None
            if expected_probs_str:
                if expected_freqs_str: # Check if expected_freqs_str was non-empty
                    raise ValueError("Provide either Expected Frequencies OR Expected Probabilities, not both.")
                expected_probs = parse_float_list(expected_probs_str)
                if len(expected_probs) != len(observed):
                    raise ValueError(f"Number of expected probabilities ({len(expected_probs)}) must match number of observed frequencies ({len(observed)}).")
                if not np.isclose(sum(expected_probs), 1.0):
                    flash(f"Warning: Expected probabilities sum to {sum(expected_probs):.4f}, which is not exactly 1.0. Proceeding with calculation, but please check.", "warning") # Allow slight float deviation but warn
                    # Alternatively, could raise ValueError("Expected probabilities must sum exactly to 1.0.")


            # --- Run Analysis (pass alpha) ---
            app_logger.info(f"Running GoF with Alpha: {alpha}, Observed Count: {len(observed)}")
            result = run_goodness_of_fit(
                observed_freqs=observed,
                expected_freqs=expected_freqs,
                expected_probs=expected_probs,
                alpha=alpha # Pass alpha here
            )
            app_logger.info("GoF analysis complete.")

            # --- Generate Plot ---
            plot_div = None # Initialize plot_div
            # Only generate plot if df is valid and calculation was successful
            if result.degrees_of_freedom is not None and result.degrees_of_freedom > 0 and result.chi2_statistic is not None:
                 plot_div = plot_chi_square_curve(
                     df=result.degrees_of_freedom,
                     chi2_statistic=result.chi2_statistic,
                     alpha=result.alpha, # Use alpha from result object
                     p_value=result.p_value
                 )
                 if plot_div is None:
                      flash("Warning: Could not generate Chi-Square distribution plot.", "warning")


            # --- Prepare for Template ---
            # Convert relevant numpy arrays/types in result to standard Python types for JSON/templating
            display_result = {
                "test_type": result.test_type,
                "chi2_statistic": float(result.chi2_statistic) if result.chi2_statistic is not None else None,
                "p_value": float(result.p_value) if result.p_value is not None else None,
                "degrees_of_freedom": int(result.degrees_of_freedom) if result.degrees_of_freedom is not None else None,
                "alpha": float(result.alpha),
                "observed": result.observed.tolist() if result.observed is not None else None,
                "expected": result.expected.tolist() if result.expected is not None else None,
                "assumption_warnings": result.assumption_warnings,
                "significant": result.significant,
                "interpretation": Markup(result.interpretation), # Use Markup for safety with HTML breaks
                # GoodnessOfFitResult doesn't have effect_size, residuals, significant_residuals
            }

            return render_template('results.html', result=display_result, plot_div=plot_div, title="Goodness of Fit Results")

        except ValueError as e:
            flash(f"Input Error: {e}", "danger")
            app_logger.warning(f"GoF Input Error: {e}")
            # Return the form, preserving user input
            return render_template('gof_form.html', form_data=form_data), 400 # Bad request
        except Exception as e:
            flash("An unexpected error occurred during analysis. Please check the inputs or contact support.", "danger")
            app_logger.error(f"GoF Unexpected Error: {e}", exc_info=True) # Log traceback
            # Return the form, preserving user input
            return render_template('gof_form.html', form_data=form_data), 500 # Internal server error

    # --- Handle GET Request ---
    return render_template('gof_form.html', form_data=None) # Show empty form


@app.route('/contingency-test', methods=['GET', 'POST'])
def contingency_test_view():
    """Handles the Contingency Table test form and results."""
    if not CHI_MATH_LOADED:
        flash("Calculation libraries failed to load. This feature is unavailable.", "danger")
        return redirect(url_for('index'))

    form_data = {}
    if request.method == 'POST':
        form_data = request.form
        result = None
        plot_div = None
        try:
            # --- Parse Inputs ---
            table_str = request.form.get('contingency_table', '').strip()
            correction_str = request.form.get('yates_correction', 'false') # Checkbox value might be 'on' or 'true' depending on HTML
            test_type_hint = request.form.get('test_type', 'Independence') # Optional hint from form
            alpha_str = request.form.get('alpha', '0.05') # Default alpha

            if not table_str:
                 raise ValueError("Contingency table data is required.")

            table_data = parse_contingency_table(table_str)
            # Basic validation (2D, min 2x2, integers) is done in parse_contingency_table

            alpha = parse_alpha(alpha_str) # Validate alpha

            use_correction = correction_str.lower() in ['on', 'true', 'yes', '1']
            if use_correction and (len(table_data) != 2 or len(table_data[0]) != 2):
                # This is just a warning as scipy can technically apply it elsewhere
                flash("Yates' correction is typically only applied to 2x2 tables. Applying it to larger tables is generally not recommended.", "warning")


            # --- Run Analysis (pass alpha) ---
            app_logger.info(f"Running Contingency Test with Alpha: {alpha}, Table Shape: ({len(table_data)}, {len(table_data[0]) if table_data else 0}), Correction: {use_correction}, Type Hint: {test_type_hint}")
            result = run_contingency_test(
                contingency_table=table_data,
                correction=use_correction,
                test_type_hint=test_type_hint,
                alpha=alpha # Pass alpha here
            )
            app_logger.info("Contingency analysis complete.")


            # --- Generate Plot ---
            plot_div = None # Initialize plot_div
            # Only generate plot if df is valid and calculation was successful
            if result.degrees_of_freedom is not None and result.degrees_of_freedom > 0 and result.chi2_statistic is not None:
                plot_div = plot_chi_square_curve(
                    df=result.degrees_of_freedom,
                    chi2_statistic=result.chi2_statistic,
                    alpha=result.alpha, # Use alpha from result object
                    p_value=result.p_value
                )
                if plot_div is None:
                      flash("Warning: Could not generate Chi-Square distribution plot.", "warning")


            # --- Prepare for Template ---
            # Convert relevant numpy arrays/types in result to standard Python types for JSON/templating
            display_result = {
                "test_type": result.test_type,
                "chi2_statistic": float(result.chi2_statistic) if result.chi2_statistic is not None else None,
                "p_value": float(result.p_value) if result.p_value is not None else None,
                "degrees_of_freedom": int(result.degrees_of_freedom) if result.degrees_of_freedom is not None else None,
                "alpha": float(result.alpha),
                "observed": result.observed.tolist() if result.observed is not None else None,
                "expected": result.expected.tolist() if result.expected is not None else None,
                "assumption_warnings": result.assumption_warnings,
                "significant": result.significant,
                "interpretation": Markup(result.interpretation), # Use Markup for safety
                "effect_size": { # Handle potential None effect size
                    "name": result.effect_size.name,
                    "value": float(result.effect_size.value),
                    "interpretation": result.effect_size.interpretation
                    } if result.effect_size else None,
                "residuals": result.residuals.tolist() if result.residuals is not None else None, # numpy array to list of lists
                "significant_residuals": result.significant_residuals # This is already a list of tuples/lists from chi_square_math
            }

            return render_template('results.html', result=display_result, plot_div=plot_div, title="Contingency Table Test Results")

        except ValueError as e:
            flash(f"Input Error: {e}", "danger")
            app_logger.warning(f"Contingency Input Error: {e}")
            return render_template('contingency_form.html', form_data=form_data), 400
        except Exception as e:
            flash("An unexpected error occurred during analysis. Please check the inputs or contact support.", "danger")
            app_logger.error(f"Contingency Unexpected Error: {e}", exc_info=True)
            return render_template('contingency_form.html', form_data=form_data), 500

    # --- Handle GET Request ---
    return render_template('contingency_form.html', form_data=None)

# --- Run Application ---
if __name__ == '__main__':
    # Logging is set up near the top after checking CHI_MATH_LOADED
    # Set debug=False for production!
    # Use a more robust server like Gunicorn or uWSGI in production.
    app.run(debug=True, host='0.0.0.0', port=5000)