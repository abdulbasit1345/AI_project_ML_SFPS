import streamlit as st
from src.forecasting.sales_forecast import SalesForecaster
from src.visualization import ForecastPlotter


def show_forecast_page(data):
    st.title("Sales Forecasting")

    # Create forecaster instance
    forecaster = SalesForecaster(data)

    # Generate forecast
    future_dates_dt, future_revenue = forecaster.generate_forecast()

    # Create forecast DataFrame
    forecast_df = forecaster.create_forecast_dataframe(future_dates_dt, future_revenue)

    # Display forecast
    st.subheader("Forecasted Revenue for the Next 30 Days")
    st.write(forecast_df)

    # Create and display plot
    plotter = ForecastPlotter()
    fig = plotter.create_forecast_plot(
        historical_dates=data['purchase-date'],
        historical_revenue=data['revenue'],
        forecast_dates=future_dates_dt,
        forecast_revenue=future_revenue,
        last_date=data['purchase-date'].max()
    )

    st.subheader("Sales Forecast Visualization")
    st.pyplot(fig)

