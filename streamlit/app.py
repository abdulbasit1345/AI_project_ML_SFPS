# streamlit/app.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
from src.data.data_loader import SalesDataLoader
from src.forecasting.sales_forecast import SalesForecaster
from src.segmentation.product_segment import ProductSegmentation
from src.visualization.forecast_plots import ForecastPlotter
from src.visualization.segment_plots import SegmentationPlotter


def load_data():
    """Load and cache data"""

    @st.cache_data  # Cache the data loading
    def _load_data():
        try:
            data_loader = SalesDataLoader('data/cleaned_sales_data.xlsx')
            return data_loader
        except FileNotFoundError:
            st.error("Data file not found. Please check the file path.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    return _load_data()


def main():
    st.title("Sales Analytics Dashboard")

    # Load data
    data_loader = load_data()
    if data_loader is None:
        st.stop()

    try:
        data = data_loader.prepare_for_segmentation()
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        st.stop()

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

    try:
        daily_revenue = data_loader.get_daily_revenue()

        # Create forecast
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


def show_segmentation_page(data):
    st.header("Product Segmentation")

    try:
        # Add segmentation parameters
        n_clusters = st.slider("Number of segments", 2, 8, 4)

        # Perform segmentation
        segmentation = ProductSegmentation(data)
        segments = segmentation.segment_products(n_clusters=n_clusters)

        # Display results
        st.subheader("Product Segments")
        st.dataframe(segments)

        # Add visualization
        # Scatter Plot
        plotter = SegmentationPlotter()
        fig = plotter.plot_clusters(segments)
        st.pyplot(fig)

        # plot_elbow_method
        elbow_method = plotter.plot_elbow_method(data)
        st.pyplot(elbow_method)

        # Show segment statistics
        st.subheader("Segment Statistics")
        segment_stats = segments.groupby('segment_label').agg({
            'total_revenue': 'sum',
            'total_quantity': 'sum',
            'avg_price': 'mean',
            'sku': 'count'
        }).round(2)
        segment_stats.columns = ['Total Revenue', 'Total Quantity', 'Average Price', 'Number of Products']
        st.dataframe(segment_stats)

    except Exception as e:
        st.error(f"Error in segmentation process: {e}")


if __name__ == "__main__":
    main()
