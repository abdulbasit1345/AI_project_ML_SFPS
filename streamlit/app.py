import streamlit as st
from src.data.data_loader import SalesDataLoader
from src.forecasting.sales_forecast import SalesForecaster
from src.segmentation.product_segment import ProductSegmentation
from src.visualization.forecast_plots import ForecastPlotter
from src.visualization.segment_plots import SegmentationPlotter


def main():
    st.title("Sales Analytics Dashboard")

    # Load data
    data_loader = SalesDataLoader('data/cleaned_sales_data.xlsx')
    data = data_loader.prepare_for_segmentation()

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Sales Forecast", "Product Segmentation"]
    )

    if page == "Sales Forecast":
        show_forecast_page(data_loader)
    else:
        show_segmentation_page(data)


def show_forecast_page(data_loader):
    st.header("Sales Forecast")
    daily_revenue = data_loader.get_daily_revenue()

    # Your existing forecasting code here
    forecaster = SalesForecaster(daily_revenue)
    future_dates_dt, future_revenue, last_date = forecaster.generate_forecast()

    forecast_df = forecaster.create_forecast_dataframe(future_dates_dt, future_revenue)
    plotter = ForecastPlotter(daily_revenue)
    fig = plotter.create_forecast_plot(future_dates_dt, future_revenue, last_date)
    st.pyplot(fig)
    st.write(forecast_df)


def show_segmentation_page(data):
    st.header("Product Segmentation")

    # Add segmentation parameters
    n_clusters = st.slider("Number of segments", 2, 8, 4)

    # Perform segmentation
    segmentation = ProductSegmentation(data)
    segments = segmentation.segment_products(n_clusters=n_clusters)

    # Display results
    st.write("Product Segments:")
    st.write(segments)

    # Add visualization of segments
    plotter = SegmentationPlotter()
    fig = plotter.plot_clusters(segments)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
