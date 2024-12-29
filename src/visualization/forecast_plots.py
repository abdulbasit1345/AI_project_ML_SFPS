import matplotlib.pyplot as plt
import seaborn as sns


class ForecastPlotter:
    def __init__(self, daily_revenue):
        sns.set_theme(style="whitegrid")
        self.daily_revenue = daily_revenue

    def create_forecast_plot(self, future_dates_dt,
                        future_revenue, last_date):
        """Create forecast visualization"""
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot historical data
        ax.plot(
            self.daily_revenue['purchase-date'],
            self.daily_revenue['revenue'],
            label="Observed Revenue",
            marker='o',
            color='blue',
            linewidth=2,
            markersize=6,
            linestyle='--'
        )

        # Plot forecast
        ax.plot(
            future_dates_dt,
            future_revenue,
            label="Forecasted Revenue (Next 30 Days)",
            marker='x',
            color='red',
            linewidth=2,
            markersize=6,
            linestyle='-.'
        )

        # Add vertical line at forecast start
        ax.axvline(last_date, color='green', linestyle='--',
                   linewidth=1.5, label="Forecast Start")

        # Formatting
        ax.set_title("Sales Forecasting: Observed vs. Predicted Revenue", fontsize=18)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Revenue ($)", fontsize=14)
        ax.legend(fontsize=12, loc="upper left")

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Add grid
        plt.grid(True, linestyle=':', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        return fig
