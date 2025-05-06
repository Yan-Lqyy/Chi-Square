import os
from flask import Flask, render_template, request, flash, redirect, url_for
from markupsafe import Markup # Import Markup from markupsafe instead
import numpy as np


# Import the chi_square_math package functions and result classes
try:
    from chi_square_math import (
        run_goodness_of_fit,
        run_contingency_test,
        GoodnessOfFitResult,
        ContingencyTableResult
    )
    CHI_MATH_LOADED = True
except ImportError as e:
    print(f"ERROR: Could not import chi_square_math package: {e}")
    print("Please ensure the chi_square_math directory is in the correct location or installed.")
    CHI_MATH_LOADED = False
    # Define dummy classes/functions if import fails, so app can start but show error
    class MockResult:
        interpretation = "Math library failed to load."
        assumption_warnings = []
    run_goodness_of_fit = lambda **kwargs: MockResult()
    run_contingency_test = lambda **kwargs: MockResult()
    GoodnessOfFitResult = MockResult
    ContingencyTableResult = MockResult


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Use a secure random key for session management

# --- Add enumerate to Jinja2 globals ---
app.jinja_env.globals['enumerate'] = enumerate

# Helper function to safely parse string lists/tables
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
             raise ValueError(f"Invalid input: Row {i+1} is empty after parsing.")

        if first_row_len is None:
            first_row_len = len(row)
        elif len(row) != first_row_len:
            raise ValueError(f"Invalid input: Row {i+1} has {len(row)} columns, but previous rows had {first_row_len}.")

        table.append(row)

    if not table:
         raise ValueError("Invalid input: No valid data found for contingency table.")

    return table

# --- Routes ---

@app.route('/')
def index():
    """Home page linking to the tests."""
    if not CHI_MATH_LOADED:
        flash("Critical Error: The Chi-Square calculation library failed to load. Please check server logs.", "danger")
    return render_template('index.html')

@app.route('/goodness-of-fit', methods=['GET', 'POST'])
def goodness_of_fit_view():
    """Handles the Goodness of Fit test form and results."""
    if not CHI_MATH_LOADED:
        flash("Critical Error: The Chi-Square calculation library failed to load.", "danger")
        return redirect(url_for('index'))

    form_data = {} # Store form data to refill form on error
    if request.method == 'POST':
        form_data = request.form # Keep form data
        result = None
        try:
            # --- Parse Inputs ---
            observed_str = request.form.get('observed_freqs', '')
            expected_freqs_str = request.form.get('expected_freqs', None)
            expected_probs_str = request.form.get('expected_probs', None)

            if not observed_str:
                raise ValueError("Observed frequencies are required.")

            observed = parse_int_list(observed_str)
            if not observed:
                 raise ValueError("Observed frequencies cannot be empty after parsing.")

            expected_freqs = None
            if expected_freqs_str:
                expected_freqs = parse_float_list(expected_freqs_str)
                if len(expected_freqs) != len(observed):
                     raise ValueError("Number of observed frequencies must match number of expected frequencies.")

            expected_probs = None
            if expected_probs_str:
                if expected_freqs is not None:
                    raise ValueError("Provide either Expected Frequencies OR Expected Probabilities, not both.")
                expected_probs = parse_float_list(expected_probs_str)
                if len(expected_probs) != len(observed):
                    raise ValueError("Number of observed frequencies must match number of expected probabilities.")
                if not np.isclose(sum(expected_probs), 1.0):
                    raise ValueError(f"Expected probabilities must sum to 1.0 (they sum to {sum(expected_probs):.4f}).")

            # --- Run Analysis ---
            app.logger.info(f"Running GoF with Observed: {observed}, Expected Freqs: {expected_freqs}, Expected Probs: {expected_probs}")
            result = run_goodness_of_fit(
                observed_freqs=observed,
                expected_freqs=expected_freqs,
                expected_probs=expected_probs
            )
            app.logger.info("GoF analysis complete.")

            # --- Prepare for Template ---
            # Convert numpy arrays in result to lists for easier Jinja handling
            display_result = {
                "test_type": result.test_type,
                "chi2_statistic": result.chi2_statistic,
                "p_value": result.p_value,
                "degrees_of_freedom": result.degrees_of_freedom,
                "observed": result.observed.tolist() if result.observed is not None else None,
                "expected": result.expected.tolist() if result.expected is not None else None,
                "assumption_warnings": result.assumption_warnings,
                "significant": result.significant,
                "interpretation": Markup(result.interpretation.replace('\n', '<br>')), # Allow basic HTML like breaks
                # Add any other fields from GoodnessOfFitResult if needed
            }

            return render_template('results.html', result=display_result, title="Goodness of Fit Results")

        except ValueError as e:
            flash(f"Input Error: {e}", "danger")
            app.logger.warning(f"GoF Input Error: {e}")
            # Return the form, preserving user input
            return render_template('gof_form.html', form_data=form_data), 400 # Bad request
        except Exception as e:
            flash("An unexpected error occurred during analysis. Please check the inputs or contact support.", "danger")
            app.logger.error(f"GoF Unexpected Error: {e}", exc_info=True) # Log traceback
            # Return the form, preserving user input
            return render_template('gof_form.html', form_data=form_data), 500 # Internal server error

    # --- Handle GET Request ---
    return render_template('gof_form.html', form_data=None) # Show empty form

@app.route('/contingency-test', methods=['GET', 'POST'])
def contingency_test_view():
    """Handles the Contingency Table test form and results."""
    if not CHI_MATH_LOADED:
        flash("Critical Error: The Chi-Square calculation library failed to load.", "danger")
        return redirect(url_for('index'))

    form_data = {}
    if request.method == 'POST':
        form_data = request.form
        result = None
        try:
            # --- Parse Inputs ---
            table_str = request.form.get('contingency_table', '')
            correction_str = request.form.get('yates_correction', 'false') # Checkbox value might be 'on' or 'true' depending on HTML
            test_type_hint = request.form.get('test_type', 'Independence') # Optional hint from form

            if not table_str:
                 raise ValueError("Contingency table data is required.")

            table_data = parse_contingency_table(table_str)
            # Basic validation done in parse_contingency_table

            use_correction = correction_str.lower() in ['on', 'true', 'yes', '1']
            if use_correction and (len(table_data) != 2 or len(table_data[0]) != 2):
                flash("Yates' correction is typically only applied to 2x2 tables.", "warning")
                # We can still proceed, scipy allows it, but warn the user.

            # --- Run Analysis ---
            app.logger.info(f"Running Contingency Test with Table: {table_data}, Correction: {use_correction}, Type Hint: {test_type_hint}")
            result = run_contingency_test(
                contingency_table=table_data,
                correction=use_correction,
                test_type_hint=test_type_hint
            )
            app.logger.info("Contingency analysis complete.")

            # --- Prepare for Template ---
            display_result = {
                "test_type": result.test_type,
                "chi2_statistic": result.chi2_statistic,
                "p_value": result.p_value,
                "degrees_of_freedom": result.degrees_of_freedom,
                "observed": result.observed.tolist() if result.observed is not None else None,
                "expected": result.expected.tolist() if result.expected is not None else None,
                "assumption_warnings": result.assumption_warnings,
                "significant": result.significant,
                "interpretation": Markup(result.interpretation.replace('\n', '<br>')),
                "effect_size": { # Handle potential None effect size
                    "name": result.effect_size.name,
                    "value": result.effect_size.value,
                    "interpretation": result.effect_size.interpretation
                    } if result.effect_size else None,
                "residuals": result.residuals.tolist() if result.residuals is not None else None,
                "significant_residuals": result.significant_residuals # Already list of tuples
            }

            return render_template('results.html', result=display_result, title="Contingency Table Test Results")

        except ValueError as e:
            flash(f"Input Error: {e}", "danger")
            app.logger.warning(f"Contingency Input Error: {e}")
            return render_template('contingency_form.html', form_data=form_data), 400
        except Exception as e:
            flash("An unexpected error occurred during analysis. Please check the inputs or contact support.", "danger")
            app.logger.error(f"Contingency Unexpected Error: {e}", exc_info=True)
            return render_template('contingency_form.html', form_data=form_data), 500

    # --- Handle GET Request ---
    return render_template('contingency_form.html', form_data=None)

# --- Run Application ---
if __name__ == '__main__':
    # Enable logging for development
    import logging
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # Check if math lib loaded on startup
    if not CHI_MATH_LOADED:
         app.logger.critical("Chi-Square math library failed to load. Functionality will be limited.")

    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000)