# plotting.py
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from typing import Optional

def plot_chi_square_curve(
    df: int,
    chi2_statistic: Optional[float],
    alpha: float,
    p_value: Optional[float]
) -> Optional[str]:
    """
    Generates an interactive Plotly plot of the Chi-Square distribution.

    Args:
        df: Degrees of freedom. Must be a positive integer.
        chi2_statistic: The calculated Chi-Square statistic from the test (can be None).
        alpha: The significance level used (0 < alpha < 1).
        p_value: The calculated p-value from the test (can be None).

    Returns:
        A string containing the HTML div for the Plotly plot, or None if df is invalid or plot cannot be generated.
    """
    if df is None or not isinstance(df, int) or df <= 0:
        # Cannot plot Chi-square distribution for non-positive or invalid df
        return None

    # Calculate critical value for the given alpha
    # Use a try-except block as stats.chi2.ppf might raise errors for invalid inputs (though df should be checked)
    try:
        critical_value = stats.chi2.ppf(1 - alpha, df)
        # Ensure critical value is finite - sometimes can be inf for small df and tiny alpha
        if not np.isfinite(critical_value):
             critical_value = None # Don't plot if infinite
    except Exception:
        critical_value = None # Handle potential errors from ppf


    # Determine plot range (x-axis)
    # Need to comfortably include the mode, statistic, and critical value
    # Mode is at df - 2 for df > 2, 0 for df = 1, 0 for df = 2 but PDF is infinite at 0
    # Use percentiles to estimate a reasonable upper bound.
    try:
        x_range_upper = stats.chi2.ppf(0.999, df) # 99.9th percentile
        # Ensure the range includes the statistic and critical value if they are finite
        if chi2_statistic is not None and np.isfinite(chi2_statistic):
             x_range_upper = max(x_range_upper, chi2_statistic * 1.2) # Extend if statistic is far out
        if critical_value is not None and np.isfinite(critical_value):
             x_range_upper = max(x_range_upper, critical_value * 1.2) # Extend if critical value is far out

        # Ensure a minimum upper bound for visual clarity for very small dfs
        x_range_upper = max(x_range_upper, 10)

        # Generate x values, starting slightly above 0 for pdf definition
        x = np.linspace(1e-6, x_range_upper, 500)

        # Calculate PDF values
        pdf = stats.chi2.pdf(x, df)

    except Exception as e:
        print(f"Error generating Chi-Square plot range or PDF: {e}")
        return None # Cannot generate plot

    # Create Plotly Figure
    fig = go.Figure()

    # --- Plot the PDF Curve ---
    fig.add_trace(go.Scatter(
        x=x, y=pdf,
        mode='lines',
        name=f'Chi-Square PDF (df={df})',
        line=dict(color='blue', width=2),
        hovertemplate='<b>χ² Value:</b> %{x:.2f}<br><b>Density:</b> %{y:.4f}<extra></extra>'
    ))

    # --- Shade the rejection region (area to the right of critical value) ---
    if critical_value is not None and np.isfinite(critical_value):
        # Create points for filling: critical_value -> x_max along the curve, plus base line
        x_fill = np.linspace(critical_value, x[-1], 100) # Use the actual max x from linspace
        pdf_fill = stats.chi2.pdf(x_fill, df)

        fig.add_trace(go.Scatter(
            x=np.concatenate(([critical_value], x_fill, [x[-1]])),
            y=np.concatenate(([0], pdf_fill, [0])),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)', # Light red fill
            line=dict(width=0), # No border line for the fill
            name=f'Rejection Region (Area = α = {alpha:.3f})',
            hovertemplate='<b>Region:</b> Rejection<extra></extra>'
        ))

    # --- Add vertical line for Critical Value ---
    if critical_value is not None and np.isfinite(critical_value):
         # Find the PDF height at the critical value for the line
         critical_pdf_height = stats.chi2.pdf(critical_value, df)
         fig.add_trace(go.Scatter(
             x=[critical_value, critical_value],
             y=[0, critical_pdf_height],
             mode='lines',
             name=f'Critical Value = {critical_value:.3f}',
             line=dict(color='red', width=2, dash='dash'),
             hovertemplate='<b>Feature:</b> Critical Value<br><b>χ² Value:</b> %{x:.3f}<extra></extra>'
         ))

    # --- Add vertical line for the Test Statistic ---
    if chi2_statistic is not None and np.isfinite(chi2_statistic) and chi2_statistic >= 0:
        # Find the PDF height at the statistic
        stat_pdf_height = stats.chi2.pdf(chi2_statistic, df)
        fig.add_trace(go.Scatter(
            x=[chi2_statistic, chi2_statistic],
            y=[0, stat_pdf_height],
            mode='lines',
            name=f'χ² Statistic = {chi2_statistic:.3f}',
            line=dict(color='green', width=3), # Thicker green line
            hovertemplate='<b>Feature:</b> Test Statistic<br><b>χ² Value:</b> %{x:.3f}<extra></extra>'
        ))

    # --- Layout and Styling ---
    p_value_text = f"p={p_value:.4f}" if p_value is not None and np.isfinite(p_value) else "p=N/A"
    title_text = f'Chi-Square Distribution (df={df})<br><sup>α={alpha:.3f}, {p_value_text}</sup>'

    fig.update_layout(
        title={
            'text': title_text,
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title='Chi-Square Value (χ²)',
        yaxis_title='Probability Density Function (PDF)',
        yaxis=dict(range=[0, np.max(pdf) * 1.1]), # Ensure y-axis starts at 0 and has padding
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=40, r=20, t=80, b=40), # Adjust margins to fit title and labels
        hovermode="x unified", # Show hover info for features at the same x-value
        template="plotly_white" # Use a clean white background template
    )

    # Configure interactivity features
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="var(--font-family)"), # Match app font
        modebar_activecolor=var('--secondary-color'), # Customize hover color
        # Additional config options can be passed to fig.to_html via config={...}
        # e.g., config={'displayModeBar': True}
    )


    # Convert plot to HTML div
    # include_plotlyjs='cdn' ensures Plotly library is loaded from a CDN
    try:
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        return plot_html
    except Exception as e:
         print(f"Error converting Plotly figure to HTML: {e}")
         return None


# Example usage (for testing plotting.py alone)
if __name__ == '__main__':
    # Generate a sample plot
    df_test = 5
    chi2_stat_test = 11.07
    alpha_test = 0.05
    p_value_test = stats.chi2.sf(chi2_stat_test, df_test) # Survival function gives p-value (1-cdf)

    plot_html = plot_chi_square_curve(df_test, chi2_stat_test, alpha_test, p_value_test)

    if plot_html:
        # You can save this HTML to a file and open it in a browser to view the plot
        with open("test_plot.html", "w") as f:
            f.write(f"""
            <html>
            <head><title>Plotly Test Plot</title></head>
            <body>
            <h1>Test Chi-Square Plot (df={df_test})</h1>
            {plot_html}
            </body>
            </html>
            """)
        print("Test plot saved to test_plot.html")
    else:
        print("Failed to generate test plot.")