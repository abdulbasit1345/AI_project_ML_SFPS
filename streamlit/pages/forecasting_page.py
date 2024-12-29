import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
from src.forecasting.sales_forecast import SalesForecaster
from src.visualization import ForecastPlotter


def show_forecast_page(data):
    st.title("Sales Forecasting")

    try:
        # Generate forecast
        daily_revenue = data.get_daily_revenue()

        # Create forecaster instance
        forecaster = SalesForecaster(daily_revenue)
        future_dates_dt, future_revenue, last_date = forecaster.generate_forecast()

        # Create visualization
        forecast_df = forecaster.create_forecast_dataframe(
            future_dates_dt,
            future_revenue
        )

        # Plot
        plotter = ForecastPlotter(daily_revenue)
        fig = plotter.create_forecast_plot(
            future_dates_dt, future_revenue,
            last_date
        )

        # Display results
        st.pyplot(fig)
        st.subheader("Forecasted Revenue")
        st.dataframe(forecast_df)
    except Exception as e:
        st.error(f"Error in forecast generation: {e}")
